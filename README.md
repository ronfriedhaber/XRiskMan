# XRiskMan: Teaching an Autoregressive Agent to “Pace the Frontier”

![PACE game illustration](./pic0.png)

[Paradigm](https://www.paradigm.xyz/) recently released an article and accompanying game, [“The Game Theory of AI Pacing”](https://www.paradigm.xyz/research/pace/), inspired by [Drew Fudenberg and Andrew Koh’s paper](https://arxiv.org/abs/2609.28291). In the game, each of *k* players chooses whether to accelerate and whether to share information; Paradigm’s version omits the sharing choice.

Inspired by the rationalist literature, this project trains an autoregressive agent with imitation learning and reinforcement learning (PPO) to navigate the trade-off between capability and safety. The results are preliminary.

## Model

Input: **[BS, 32, 16]** — the current step and the previous 31 steps, each represented
by 16 numeric features (these are not linguistic tokens):

| Indices | Features, in order |
|---|---|
| 0–3 | Time remaining, safety, own deployment − safety, opponent deployment − safety |
| 4–7 | Own research, speed, deployment, cash |
| 8–11 | Opponent deployment, cash, visible research, research-visible flag |
| 12–15 | Own research-on flag, own sharing, opponent sharing, own stopping distance |

## Training: Imitation

The agent first imitates a cautious hand-coded policy, which provides a stable
initial policy for reinforcement learning.

## Training: PPO

The initialized policy is then optimized with Proximal Policy Optimization (PPO)
against the environment reward.

## Building

| Section | Options (default) |
|---|---|
| Environment | `dt=1/60`, `decision_dt=.1`, `duration=90`, `delay=2`, `acceleration=.62`, `deceleration=.62`, `initial_safety=12`, `safety_flat_until=18`, `safety_plateaus=true`, `max_hazard=.24` |
| Model | `context=32`, `width=64`, `heads=4`, `layers=2`, `input_dim=16` |
| Run | `steps=8388608`, `seed=0`, `out=runs/pace`, `device=auto`, `threads=4`, `envs=32` |
| Imitation | `imitation_updates=5000`, `imitation_lr=3e-4` |
| PPO | `rollout=1024`, `epochs=4`, `batch=256`, `lr=3e-4`, `adam_eps=1e-5`, `gamma=1`, `gae_lambda=1`, `clip_ratio=.2`, `value_coef=.5`, `entropy_coef=.01`, `max_grad_norm=.5`, `target_kl=.03` |
| Evaluation | `eval_episodes=128`, `eval_every=5` |
| Reward | `alpha=0`, `beta=0` |

```sh
./train_and_eval_full.py --config default.json --alpha 0.5 --beta 0.1
```
