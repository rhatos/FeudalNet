# FeudalNet code

The scripts that produced the FeudalNet results in the paper, taken from `feudalnet/`.

## Model (§3.2.1, Figure 2)

| File | Role |
|---|---|
| `feudalnet.py` | The FeudalNet agent: IMPALA-style perception CNN, Manager and Worker heads, ConvTranspose actor, per-level critics, the feudal loss and the PPG auxiliary loss. |
| `transformer.py` | `WorkerTransformer` and `ManagerTransformer`, the two dual-timescale pre-norm transformers used in place of FuN's recurrent memory. |
| `enemy_encoder.py` | The opponent-modelling stream: per-cell enemy type embedding and stacked LSTM fused into the Manager. |
| `storage.py` | Rollout storage. |
| `run_config.py` | Writes each run's arguments to a `*_config.csv` (the files in `../5_adaptation_training_runs/configs/`). |

## Training (§3.3, Table 10)

| File | Role |
|---|---|
| `main-16x16.py` | Trains the 16x16 agent (`tqlgxayp_GOOD.pt`) against coacAI, workerRushAI, lightRushAI and randomBiasedAI with 48 parallel environments. |
| `main.py` | Same trainer with 8x8 defaults; trained the 8x8 agent (`8x8-good-100percent.pt`) used in Appendix B. |

## Adaptation (§3.5, Table 1, Table 7)

| File | Role |
|---|---|
| `finetune_opponent_norender.py` | The adaptation procedure: loads a checkpoint, trains against one unseen opponent for up to 1M steps, screens 50 games every 100k steps and stops at 90%. Produced the four runs in `../5_adaptation_training_runs/`. |

## Evaluation (§4.2, §4.3, Tables 2-4, Appendix B)

| File | Role |
|---|---|
| `evaluate_agent_chart.py` | Evaluates a checkpoint against any set of opponents in repeats of 250 games and writes the `*_games.csv` and `*_summary.csv` files in `../2_baseline_16x16/`, `../3_baseline_8x8/` and `../4_adaptation_eval/`. |
| `live_winrate.py` | Live win-rate chart used by the evaluator (`*_livechart.png`). |

## Round-robin tournament (§4.1, Figure 3)

| File | Role |
|---|---|
| `tournament.py` | Plays every participant against every other and writes the `_matches`, `_standings`, `_matrix` and `_matrix_winrate` CSVs in `../1_round_robin/`. |
| `plot_tournament.py` | Draws the points heat-map (Figure 3) from `*_matrix.csv`. |
| `plot_standings.py` | Draws the standings bar chart from `*_standings.csv`. |

## Plotting and export (Figures 4, 6; training curves; supplementary figures)

| File | Role |
|---|---|
| `plot_evaluation.py` | Ranked win-rate bar chart from `*_summary.csv` files (Figures 4 and 6). |
| `plot_winrate_comparison.py` | Pre- versus post-adaptation win-rate comparison (`supplementary_figures/comparewin.pdf`, `win-rate-comparison.pdf`). |
| `plot_steps_comparison.py` | Pre- versus post-adaptation game-length comparison (`supplementary_figures/comparesteps.pdf`, `comparestepsloss.pdf`). |
| `export_runs.py` | Exports a W&B run history to CSV (`../6_training/export_tqlgxayp.csv`). |
| `plot_reward_curve.py` | Plots a logged metric from a W&B run (`../6_training/*.png`). |

## Environment

`requirements.txt` and `environment.yml`. The agent runs on Gym-µRTS (`gym-microrts`), which must be installed separately.
