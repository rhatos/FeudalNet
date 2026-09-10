# TransformerNet code

Only the scripts written for this project are included here. The agent itself, its training scripts and the published checkpoints (`example_models/`) are Zwingenberger's and are **not** included; they come from the original repository, https://github.com/NiklasZ/transformers-for-variable-action-envs, which accompanies the paper "Transformers as Policies for Variable Action Environments" (Zwingenberger, 2023, https://arxiv.org/abs/2301.03679). The scripts below are placed in the root of a checkout of that repository, next to its `transformer_agent/` package, which they import.

## Adaptation (§3.5.2, Table 7)

| File | Role |
|---|---|
| `finetune_transformer.py` | The adaptation procedure for TransformerNet, mirroring the FeudalNet protocol: loads `example_models/16x16/agent.pt`, trains against one unseen opponent for up to 1M steps with PPO, screens 50 games every 100k steps and stops at 90%. Produced the four runs in `../4_adaptation_training_runs/`. |
| `run_config.py` | Writes each run's arguments to a `*_config.csv`. |

The four paper runs were launched as

```
python finetune_transformer.py --checkpoint example_models/16x16/agent.pt --opponent <izanagi|mixedBot|naiveMCTSAI|tiamat> \
    --agent-type embedded --embed-size 64 --map-h 16 --map-w 16 \
    --max-steps 1000000 --num-workers 24 --num-steps 256 --n-minibatch 4 --update-epochs 4 \
    --lr 2.5e-4 --anneal-lr True --ent-coef 0.01 --vf-coef 0.5 \
    --reward-weights 20 1 1 0.2 1 4 \
    --eval-every 100000 --eval-games 50 --eval-winrate 0.90 --eval-workers 32 --eval-record False
```

which keeps the original training values except for the step budget and the win/loss reward weight (10 to 20), as described in §3.5.2.

## Evaluation (§4.2, §4.4, Tables 5-6, Appendix B)

| File | Role |
|---|---|
| `evaluate_agent_chart.py` | Evaluates a checkpoint against any set of opponents in repeats of 250 games and writes the `*_games.csv` and `*_summary.csv` files in `../1_baseline_16x16/`, `../2_baseline_8x8/` and `../3_adaptation_eval/`. Same output format as the FeudalNet evaluator, so the two agents' results are directly comparable. |
| `live_winrate.py` | Live win-rate chart used by the evaluator. |

## Patch to the original package

`transformer_agent_gym-microrts-0.6.patch` is a unified diff against the original repository's `transformer_agent/` package. The original was written for the pre-0.4 gym-microrts API; the patch makes it run on gym-microrts 0.6 (flattened action space, `get_action_mask()`, `map_paths` list, explicit `partial_obs` flag) while leaving the network unchanged. Apply it in the repository root with

```
patch -p0 < transformer_agent_gym-microrts-0.6.patch
```

## Environment

`requirements.txt` is the environment the experiments ran in (CUDA 12.1 build of PyTorch). gym-microrts must be installed separately.
