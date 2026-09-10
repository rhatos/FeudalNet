# Paper results: cross-reference

Results files behind *Does Hierarchy Help? Few-Shot Adaptation of Feudal and Flat Policies in Real-Time Strategy Games* (`paper/main.tex`), collected from the two working folders in which the experiments were run:

- `feudalnet/` (FeudalNet, our agent) → `FeudalNet/`
- `transformers-for-variable-action-envs/` (TransformerNet, Zwingenberger's agent, original repository https://github.com/NiklasZ/transformers-for-variable-action-envs) → `TransformerNet/`

The "Original location" column gives the path inside those working folders.

Every number quoted in the paper can be recomputed from the files here. Caveats about provenance are listed under **Caveats**.

File formats: `*_summary.csv` is one column per opponent with rows for games, wins, losses, draws, win_pct and step statistics. `*_games.csv` is one row per game (opponent, repeat, result, steps, per-component reward, checkpoint). The 1000-game evaluations are four repeats of 250 games (`repeat` column 1–4, seed 42).

## FeudalNet

| Paper location | What is quoted | Files | Original location |
|---|---|---|---|
| §4.1, Figure 3 | Round-robin tournament, 16x16, 10 games per pairing. FeudalNet 126.5 pts, coacAI 115.5, izanagi 102.5; FeudalNet vs izanagi 5.5 pts; adaptation opponents 3rd to 11th | `1_round_robin/*_standings.csv`, `*_matrix.csv` (points), `*_matrix_winrate.csv`, `*_matches.csv` (every game). `*_matrix.png` is Figure 3 | `feudalnet/results/round-robin-feudal/`, `feudalnet/results/tournament_*.png` |
| §4.2, Figure 4, Table 2 (WR pre), §4.3.3, §4.3.4, Table 4 (before) | Baseline 16x16, 1000 games per opponent: mayari 99.2%, coacAI 88.2%, izanagi 42.8% (428/556/16), mixedBot 77.8%, naiveMCTSAI 79.0% (790/30/180), tiamat 88.6%. Izanagi steps 824.9±251.9 (n=428) / 1306.5±248.5 (n=556). naiveMCTSAI steps-to-win 935.1±229.1 | `2_baseline_16x16/feudal_full_summary.csv` (all 14 opponents), `feudal_full_games.csv` (per-game rows), `feudal_full_livechart.png`; `eval_16x16_feudalnet.pdf` is Figure 4 | `feudalnet/results/`, `feudalnet/eval_16x16_feudalnet.pdf` |
| Appendix B, Figure 6; §3.5 | Baseline 8x8, 1000 games per opponent (justifies using only 16x16 for adaptation) | `3_baseline_8x8/eval_8x8-good-100percent_basesWorkers8x8_20260831_124751_{games,summary}.csv`; `eval_8x8_feudal.pdf` is Figure 6 | `feudalnet/results/`, `feudalnet/eval_8x8_feudal.pdf` |
| §4.3, Tables 2, 3, 4 (after) | Post-adaptation, 1000 games each: izanagi 95.5% (955/19/26, steps 358.3±82.5 / 994.0±231.6), mixedBot 96.9% (969/29/2), naiveMCTSAI 87.1% (871/19/110, steps-to-win 934.2±221.5), tiamat 92.8% (928/66/6). Table 3 z-tests are computed from these and the baseline | `4_adaptation_eval/feudal_post_<opponent>_{games,summary}.csv` | `feudalnet/results/` |
| Not in paper | Pre/post comparison plots of win-rate and game length | `4_adaptation_eval/supplementary_figures/` | `feudalnet/adapt_baseline_16x16/`, `fig/`, `results/feudal-adapt.pdf` |
| §3.5, Table 1, §4.3.6, Table 7 | Adaptation training runs. Steps at which the 90% screen fired: izanagi 400k, mixedBot 100k, naiveMCTSAI 400k, tiamat 100k. Table 1 values (lr 3e-4, linear anneal, max_steps 1e6, n_pi 2, entropy 0.015) | `5_adaptation_training_runs/run-*-<id>/` (W&B run folders: `files/config.yaml`, `files/wandb-summary.json` with `global_step` and last `eval/winrate`, `files/output.log`) and `configs/*_config.csv`. Run ids: izanagi `npbqmut4`, mixedBot `g09rqp8w`, naiveMCTSAI `91k0pnqa`, tiamat `zs1phv1c` | `feudalnet/wandb/`, `feudalnet/models/configs/` |
| §3.3, Table 10 | 300M-step training run `tqlgxayp`: exported W&B history (reward, episode length, losses, learning rate per step) and the reward and episode-length curves | `6_training/export_tqlgxayp.csv`, `tqlgxayp_episode_total_reward.png`, `tqlgxayp_episode_length.png` | `feudalnet/` |
| all of the above | Source code: model, training, adaptation, evaluation, tournament and plotting scripts, each described in `Code/README.md` | `Code/` | `feudalnet/` |
| all of the above | Checkpoints that produced the results: `tqlgxayp_GOOD.pt` (16x16 baseline), `8x8-good-100percent.pt` (8x8), `tqlgxayp_GOOD-izanagi_step=400800.pt`, `tqlgxayp_GOOD-mixedBot.pt`, `tqlgxayp_GOOD-naiveMCTSAI.pt`, `tqlgxayp_GOOD-tiamat.pt` (post-adaptation) | `checkpoints/` (51 MB each) | `feudalnet/models/`, `feudalnet/8x8-good-100percent.pt` |

## TransformerNet

| Paper location | What is quoted | Files | Original location |
|---|---|---|---|
| §4.2, Figure 5, Table 5 (WR pre), §4.4 | Baseline 16x16, 1000 games per opponent: coacAI 87.4%, mayari 88.1%, naiveMCTSAI 10.7% (107/0/893), izanagi 71.7% (717/135/148; draws 14.8%, losses 13.5%), mixedBot 93.9%, tiamat 99.1%. Pre-adaptation steps: izanagi win 1114 / lose 1300, tiamat 811 / 1033, naiveMCTSAI win 1403 | `1_baseline_16x16/transformer_full_summary.csv`, `transformer_full_games.csv`, `transformer_coacAI_{games,summary}.csv` (1000-game coacAI re-run that the summary's 87.4% comes from); `eval_16x16_transformernet.pdf` is Figure 5 | `transformers-for-variable-action-envs/results/` (`transformer_full_games.csv` was named `feudal_full_games.csv` there; it is TransformerNet data, checkpoint `agent.pt`), `feudalnet/eval_16x16_transformernet.pdf` |
| Appendix B, Figure 7 | Baseline 8x8, 1000 games per opponent; 0% against mayari | `2_baseline_8x8/eval_agent_basesWorkers8x8_20260831_144851_{games,summary}.csv`; `eval_8x8_transformer.pdf` is Figure 7 | `transformers-for-variable-action-envs/results/`, `feudalnet/eval_8x8_transformer.pdf` |
| §4.4, Tables 5, 6 | Post-adaptation, 1000 games each: izanagi 74.7% (747/150/103; draws 10.3%, losses 15.0%; steps 720±278 / 1217), mixedBot 95.5% (955/36/9), naiveMCTSAI 24.6% (246/24/730; draws 73.0%; steps-to-win 1230), tiamat 94.6% (946/53/1; steps 627 / 1147). Table 6 z-tests are computed from these and the baseline | `3_adaptation_eval/transformer_post_<opponent>_{games,summary}.csv` | `transformers-for-variable-action-envs/results/eval_agent-<opponent>_basesWorkers16x16_20260830_*_games.csv` and `transformernet_post_<opponent>.csv` |
| §3.5.2, §4.4.5, Table 7 | Adaptation training runs under the final protocol (lr 2.5e-4, linear anneal, win/loss reward weight 20, screen every 100k). Halted at: izanagi 1M (cap), mixedBot 100k (baseline already above 90%), naiveMCTSAI 1M (cap), tiamat 100k (baseline already above 90%) | `4_adaptation_training_runs/run-*-<id>/`. Run ids: izanagi `vee88nm8`, mixedBot `8box7dxl`, naiveMCTSAI `bx8r54so`, tiamat `r3m9uta8` | `transformers-for-variable-action-envs/wandb/` |
| all of the above | Evaluation and adaptation scripts written for this project, plus a patch for the original agent package; described in `Code/README.md`. The original repository code is not included | `Code/` | `transformers-for-variable-action-envs/` |
| all of the above | Checkpoints: `agent_16x16_baseline.pt` and `agent_8x8_baseline.pt` (Zwingenberger's published models, used as-is), `agent-mixedBot.pt`, `agent-naiveMCTSAI.pt`, `agent-tiamat.pt` (post-adaptation). No post-adaptation izanagi checkpoint survives, see below | `checkpoints/` | `transformers-for-variable-action-envs/example_models/`, `models/` |

## Cross-agent tables

Table 8 (difference of adaptation scores) and the "answering the research question" section use only the win-rates above.

## Illustrative figures

| Paper location | Files | Original location |
|---|---|---|
| Figure 1 (8x8 and 16x16 maps) | `Figures/8x8v16x16.pdf`; composed from the screenshots `Figures/8x8.png` and `Figures/micro-rts-example-1.png` | paper folder |
| Figure 2 (FeudalNet architecture) | `Figures/feudalnet_diagram.pdf` | paper folder |

## Caveats

1. **FeudalNet izanagi checkpoint.** `models/tqlgxayp_GOOD-izanagi.pt` was overwritten on 29 Aug by a later run (`q77hulh5`). The checkpoint saved by the run that produced the paper's 95.5% result (`npbqmut4`, stopped at 400,800 steps on 27 Aug) is `tqlgxayp_GOOD-izanagi_step=400800.pt`, which is the one copied here.
2. **TransformerNet izanagi checkpoint.** The 74.7% evaluation (30 Aug, 09:33) came from run `vee88nm8`, but every `agent-izanagi*.pt` file in `models/` was overwritten by a later run on 31 Aug (`ru6egcvu`, learning rate 2.5e-5, which evaluated at 67.7% and 86.6%, not used in the paper). Only the evaluation CSVs and the W&B run folder remain for this matchup.