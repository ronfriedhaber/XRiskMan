"""Clipped PPO + GAE, with bounded context and no gradients through rollouts."""
import argparse
from dataclasses import asdict, fields, replace
import json
import math
from pathlib import Path

import numpy as np
import torch

from pace import Config, Pace, cautious
from config import ExperimentConfig, ModelConfig, TrainingConfig, RewardConfig


def objective(cash, crashed, alpha=0.0, beta=0.0):
    """Apply to cash deltas + new crashes, or final cash + episode crash flags."""
    return (cash[:, 0] - alpha * cash[:, 1]) / 20 - beta * crashed


def advance(env, history, action, rng=None):
    """Step, optionally reset finished rows, then update the model's history."""
    observation, cash, done = env.step(np.column_stack((action.cpu().numpy(), np.zeros(env.n, dtype=np.int64))))
    crashed = env.crashed.copy()
    if rng is not None and done.any():
        observation = env.reset(rng.integers(0, 2**31, int(done.sum())), done)
    history.push(observation, action, cash[:, 0] / 20, done)
    return cash, done, crashed


def optimize(model, optimizer, loss, max_grad_norm):
    if not torch.isfinite(loss):
        raise FloatingPointError("nonfinite loss")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm, error_if_nonfinite=True)
    optimizer.step()


def advantages(reward, value, done, bootstrap, gamma=1.0, lam=0.95):
    result = torch.zeros_like(reward)
    carry = torch.zeros_like(bootstrap)
    for t in reversed(range(len(reward))):
        live = (~done[t]).float()
        delta = reward[t] + gamma * live * bootstrap - value[t]
        carry = delta + gamma * lam * live * carry
        result[t] = carry
        bootstrap = value[t]
    return result, result + value


@torch.no_grad()
def evaluate(model, config, episodes=16, seed=0, alpha=0.0, beta=0.0):
    # Evaluation seeds never occur in training's [0, 2**31) seed pool.
    rng = np.random.default_rng(seed)
    env = Pace(rng.integers(2**31, 2**32, episodes), config, legacy=model.config.input_dim == 23)
    history = History(env.observe(), model.config.context, next(model.parameters()).device)
    # Fixed evaluation draws, isolated from training's action-sampling RNG.
    generator = torch.Generator().manual_seed(seed)
    while not env.done.all():
        distribution, _ = model(history.tokens, history.lengths)
        action = torch.multinomial(distribution.probs.cpu(), 1, generator=generator).squeeze(-1).to(history.tokens.device)
        advance(env, history, action)
    scores = env.cash[:, 0]
    returns = objective(env.cash, env.crashed, alpha, beta)
    return {"win_fraction": float(((scores > env.cash[:, 1]) & ~env.crashed).mean()),
            "catastrophe_fraction": float(env.crashed.mean()),
            "cash_advantage_billion_mean": float((scores - env.cash[:, 1]).mean()),
            "cash_billion_mean": float(scores.mean()),
            "episode_return_mean": float(returns.mean())}


def imitate(model, config, updates, envs, seed, lr, max_grad_norm=0.5):
    """Teacher rollouts; minimize E[-log πθ(a_teacher | history)] before PPO."""
    rng = np.random.default_rng(seed)
    env = Pace(rng.integers(0, 2**31, envs), config, legacy=model.config.input_dim == 23)
    device = next(model.parameters()).device
    history = History(env.observe(), model.config.context, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0)
    for _ in range(updates):
        action = torch.as_tensor(cautious(env), device=device)
        distribution, _ = model(history.tokens, history.lengths)
        optimize(model, optimizer, -distribution.log_prob(action).mean(), max_grad_norm)
        advance(env, history, action, rng)


def load(path, device="cpu"):
    saved = torch.load(path, map_location=device, weights_only=True)
    model_config = {"input_dim": saved["model"]["embed.weight"].shape[1], **saved["model_config"]}
    model = Policy(ModelConfig(**model_config)).to(device)
    model.load_state_dict(saved["model"])
    return model.eval(), saved


def main(argv=None, on_checkpoint=None, tracker=None, *, config=None, settings=None, experiment=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="JSON overrides grouped by environment/model/training/reward")
    parser.add_argument("--evaluate", type=Path, help="evaluate a saved checkpoint")
    for kind in (TrainingConfig, ModelConfig, RewardConfig):
        for f in fields(kind):
            parser.add_argument("--" + f.name.replace("_", "-"), type=f.type, default=argparse.SUPPRESS)
    cli = parser.parse_args(argv)
    overrides = {**vars(cli), **(settings or {})}
    try:
        experiment = experiment or ExperimentConfig.read(cli.config)
        sections = {name: replace(getattr(experiment, name), **{
            f.name: overrides[f.name] for f in fields(getattr(experiment, name)) if f.name in overrides
        }) for name in ("training", "model", "reward")}
        experiment = replace(experiment, environment=config or experiment.environment, **sections)
    except (ValueError, TypeError, OSError) as error:
        parser.error(str(error))
    args = argparse.Namespace(**experiment.arguments(), evaluate=cli.evaluate)
    config, model_config = experiment.environment, experiment.model
    reward_config = asdict(experiment.reward)
    args.out = Path(args.out)
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    if args.evaluate:
        model, saved = load(args.evaluate, device)
        torch.manual_seed(args.seed)
        config = Config(**saved["config"])
        result = {"transitions": saved.get("transitions", 0),
                  **evaluate(model, config, args.eval_episodes, args.seed, **saved.get("reward_config", {}))}
        print(json.dumps(result), flush=True)
        return result

    model = Policy(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, eps=args.adam_eps, weight_decay=0)
    args.out.mkdir(parents=True, exist_ok=False)  # Never overwrite a previous run.
    metadata = {"config": asdict(config), "model_config": asdict(model.config), "reward_config": reward_config,
                "experiment": asdict(experiment),
                "torch": str(torch.__version__), "numpy": np.__version__}
    (args.out / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if tracker is not None:
        tracker.config.update(metadata)

    def record_evaluation(step):
        result = evaluate(model, config, args.eval_episodes, args.seed, **reward_config)
        record = {"transitions": step, **result}
        with (args.out / "evaluation.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps({"evaluation": record}), flush=True)
        if tracker is not None:
            tracker.log({"transitions": step, **{f"eval/{k}": v for k, v in result.items()}}, step=step)
        return record

    if args.imitation_updates:
        imitate(model, config, args.imitation_updates, args.envs, args.seed, args.imitation_lr, args.max_grad_norm)
        torch.save({"model": model.state_dict(), **metadata}, args.out / "imitation.pt")
        if on_checkpoint:
            on_checkpoint()
    result = record_evaluation(0)
    env = Pace(rng.integers(0, 2**31, args.envs), config, legacy=model.config.input_dim == 23)
    history = History(env.observe(), args.context, device)
    transitions = 0
    updates = math.ceil(args.steps / (args.envs * args.rollout))
    for update in range(updates):
        # Store the exact contexts used to choose actions; PPO re-evaluates them.
        rollout = []
        with torch.no_grad():
            for _ in range(args.rollout):
                distribution, value = model(history.tokens, history.lengths)
                action = distribution.sample()
                sample = (history.tokens.clone(), history.lengths.clone(), action, distribution.log_prob(action), value)
                cash, done, crashed = advance(env, history, action, rng)
                reward = torch.as_tensor(objective(cash, crashed & done, **reward_config), dtype=torch.float32, device=device)
                rollout.append((*sample, reward, torch.as_tensor(done, device=device)))
            contexts, lengths, actions, old_logp, values, rewards, dones = (torch.stack(v) for v in zip(*rollout))
            _, bootstrap = model(history.tokens, history.lengths)
            advantage, returns = advantages(rewards, values, dones, bootstrap, gamma=args.gamma, lam=args.gae_lambda)
            advantage = advantage.flatten()
            advantage = (advantage - advantage.mean()) / advantage.std(unbiased=False).clamp_min(1e-8)
            returns = returns.flatten()
        contexts, lengths, actions, old_logp = (v.flatten(0, 1) for v in (contexts, lengths, actions, old_logp))
        count = len(actions)
        stop = False
        for _ in range(args.epochs):
            for indices in torch.randperm(count, device=device).split(args.batch):
                distribution, value = model(contexts[indices], lengths[indices])
                log_ratio = distribution.log_prob(actions[indices]) - old_logp[indices]
                ratio = log_ratio.exp()
                kl = (ratio - 1 - log_ratio).mean()
                if not torch.isfinite(kl):
                    raise FloatingPointError("nonfinite policy ratio")
                if kl.item() > args.target_kl:
                    stop = True
                    break
                a = advantage[indices]
                actor = -torch.minimum(ratio * a, ratio.clamp(1 - args.clip_ratio, 1 + args.clip_ratio) * a).mean()
                critic = args.value_coef * (value - returns[indices]).square().mean()
                entropy = distribution.entropy().mean()
                loss = actor + critic - args.entropy_coef * entropy
                optimize(model, optimizer, loss, args.max_grad_norm)
            if stop:
                break
        transitions += count
        if (update + 1) % args.eval_every == 0 or update + 1 == updates:
            result = record_evaluation(transitions)
            checkpoint = {"model": model.state_dict(), **metadata, "transitions": transitions}
            temporary = args.out / "checkpoint.tmp"
            torch.save(checkpoint, temporary)
            temporary.replace(args.out / "checkpoint.pt")
            if on_checkpoint:
                on_checkpoint()
    return result


if __name__ == "__main__":
    main()
