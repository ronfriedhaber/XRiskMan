# XRiskMan: Teaching an autoregressive agent to "Pace The Frontier"

Recentely, the great venture capital firm Paradigm released an article and acompannying game, "The Game Theory of AI Pacing". It is inspired by a paper authored by Drew Fudenberg and Andrew Ko.
In one sentence, there exists k-players, each player has a choice of whether to Accelerate or not, and whether to publicize or not (the latter is removed by Paradigm's version).

Inspired by general rationalist litreture I set out to train an autoregressive agent, using a combination of imitation training and RL (PPO) to pareto-optimally navigate the P(doom) landscape. The following is a preliminary result.

## Model

Input: **[BS, 32, 16]** — the current step and 31 previous steps, each represented
by 16 numbers (Note, not a linguistic token):

| Indices | Features, in order |
|---|---|
| 0–3 | Time remaining, safety, own deployment − safety, opponent deployment − safety |
| 4–7 | Own research, speed, deployment, cash |
| 8–11 | Opponent deployment, cash, visible research, research-visible flag |
| 12–15 | Own research-on flag, own sharing, opponent sharing, own stopping distance |

## Training - Imitation
## Training - PPO

## Building
