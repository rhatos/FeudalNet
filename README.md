# Does Hierarchy Help? Few-Shot Adaptation of Feudal and Flat Policies in Real-Time Strategy Games

Conor Karl McKeag, University of Cape Town, 2026. Honours project.

Paper: [`paper/Does_Hierarchy_Help.pdf`](paper/Does_Hierarchy_Help.pdf) (LaTeX source in [`paper/`](paper/)).

## Abstract

Hierarchical reinforcement learning (HRL) promises the temporal abstraction, sample efficiency, and transfer ability that real-time strategy (RTS) games demand, yet no HRL method has been evaluated in a full-game RTS setting. We address this gap with FeudalNet, a hierarchical Gym-µRTS agent retaining the Manager/Worker division of FeUdal Networks but replacing its recurrent memory with dual-timescale transformers and adding a dedicated opponent-modelling stream. We evaluate FeudalNet alongside TransformerNet, a state-of-the-art flat policy, on few-shot adaptation: a fully-trained agent trains against an unseen opponent until a 90% win-rate or a 1M-step cap (versus 300M for training), and the change in win-rate is measured. FeudalNet defeats all fourteen scripted opponents in round-robin play and exceeds TransformerNet on the standard evaluation suite, including an 88.2% win-rate against coacAI. Under adaptation, FeudalNet improved significantly against all four unseen opponents, lifting its weakest matchup from 42.8% to 95.5% via a discovered counter-strategy, while TransformerNet improved against one, held level against two, and regressed against one; FeudalNet's adaptation score was significantly higher on three of four. These results answer the research question affirmatively and suggest the hierarchy's advantage lies not only in how much it adapts but in how reliably, requiring less intervention than the flat baseline.

## Research question

*Given a limited adaptation budget, does the hierarchical FeudalNet agent improve its win-rate against unseen opponent strategies in Gym-µRTS by more than the flat TransformerNet agent?*

## Main result

Win-rate before and after adaptation against the four unseen opponents (1000 games per condition, 16x16 map).

| Opponent | FeudalNet pre | FeudalNet post | TransformerNet pre | TransformerNet post |
|---|---|---|---|---|
| izanagi | 42.8% | 95.5% | 71.7% | 74.7% |
| mixedBot | 77.8% | 96.9% | 93.9% | 95.5% |
| naiveMCTSAI | 79.0% | 87.1% | 10.7% | 24.6% |
| tiamat | 88.6% | 92.8% | 99.1% | 94.6% |

All four FeudalNet improvements are significant at the 5% level. For TransformerNet only the naiveMCTSAI gain is significant, and the tiamat change is a significant regression. FeudalNet's adaptation score is significantly higher on izanagi, mixedBot and tiamat; TransformerNet's is higher on naiveMCTSAI, where its larger raw gain recovers a smaller fraction of its headroom.

## Repository layout

| Path | Contents |
|---|---|
| [`paper/`](paper/) | The paper as PDF and LaTeX source with its figures. |
| [`RESULTS.md`](RESULTS.md) | Cross-reference from every section, table and figure of the paper to the result files that back it. |
| [`FeudalNet/`](FeudalNet/) | Our agent. Numbered result folders in the paper's order (round-robin, 16x16 baseline, 8x8 baseline, adaptation evaluations, adaptation training runs, training history), the model checkpoints, and [`Code/`](FeudalNet/Code/) with the model, training, adaptation, evaluation, tournament and plotting scripts. |
| [`TransformerNet/`](TransformerNet/) | The flat baseline. Result folders (16x16 baseline, 8x8 baseline, adaptation evaluations, adaptation training runs), the post-adaptation checkpoints, and [`Code/`](TransformerNet/Code/) with the evaluation and adaptation scripts written for this project plus a compatibility patch. The agent itself comes from Zwingenberger's repository, https://github.com/NiklasZ/transformers-for-variable-action-envs, and is not included. |
| [`Figures/`](Figures/) | The illustrative figures (map sizes, architecture diagram). |

Each `Code/` folder has its own README describing the scripts and how the runs were launched.

## Environment

Both agents run on Gym-µRTS (`gym-microrts` 0.6, https://github.com/Farama-Foundation/MicroRTS-Py), which needs Java and must be installed separately. Python dependencies are listed in `FeudalNet/Code/requirements.txt` and `TransformerNet/Code/requirements.txt`. Training and adaptation runs log to Weights & Biases.

## Summary of the method

- **FeudalNet** keeps the Manager/Worker structure of FeUdal Networks. A shared IMPALA-style CNN encodes the observation; the Manager sets a goal direction every 25 ticks from a transformer over 160 Manager states; the Worker acts every tick from a transformer over the last 100 embeddings and a ConvTranspose decoder producing per-cell masked action logits. An enemy-encoder LSTM feeds opponent behaviour into the Manager. Both levels have separate critics, with a PPG auxiliary phase for the Worker's value head.
- **Training**: 300M environment steps on the 16x16 map against coacAI, workerRushAI, lightRushAI and randomBiasedAI, with shaped rewards (±20 win/loss).
- **Few-shot adaptation**: a trained agent trains against one unseen opponent for up to 1M steps; every 100k steps 50 games are screened and training stops at a 90% win-rate. Adaptation score is the change in 1000-game win-rate, tested with a two-proportion z-test.
- **Opponents**: fourteen scripted bots shipped with µRTS. Adaptation opponents (izanagi, mixedBot, naiveMCTSAI, tiamat) are the weakest FeudalNet matchups outside the training set.

## Citation

```bibtex
@misc{mckeag2026hierarchy,
  title  = {Does Hierarchy Help? Few-Shot Adaptation of Feudal and Flat Policies in Real-Time Strategy Games},
  author = {McKeag, Conor Karl},
  year   = {2026},
  school = {University of Cape Town},
  url    = {https://github.com/rhatos/FeudalNet}
}
```
