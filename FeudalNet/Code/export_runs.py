"""
export_runs.py

Exports wandb run histories to CSV files for offline analysis.

Usage:
    python export_runs.py --runs z21ti97y abc123xy --project fun-microrts
    python export_runs.py --runs z21ti97y --entity rhatos-university-of-cape-town

Produces one CSV per run: export_<run_id>.csv
Upload the CSVs (and optionally this script's console output) for analysis.
"""

import argparse
import wandb
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--runs',    nargs='+', required=True,
                    help='One or more wandb run ids')
parser.add_argument('--project', type=str, default='fun-microrts')
parser.add_argument('--entity',  type=str, default=None,
                    help='wandb entity (username/team); default = your login')
parser.add_argument('--samples', type=int, default=5000,
                    help='Max history rows per run (wandb downsamples above this)')
args = parser.parse_args()

METRICS = [
    "global_step",
    "eval/winrate",
    "episode/total_reward",
    "episode/length",
    "worker/entropy",
    "worker/intrinsic_reward",
    "worker/advantage",
    "manager/cosines",
    "manager/advantage",
    "loss/total",
    "loss/worker",
    "loss/manager",
    "loss/value_worker",
    "loss/value_manager",
    "ppg/aux_value_worker",
    "charts/learning_rate",
    "charts/steps_per_sec",
    "charts/episode_reward/WinLossRewardFunction",
    "charts/episode_reward/ResourceGatherRewardFunction",
    "charts/episode_reward/ProduceWorkerRewardFunction",
    "charts/episode_reward/ProduceBuildingRewardFunction",
    "charts/episode_reward/AttackRewardFunction",
    "charts/episode_reward/ProduceCombatUnitRewardFunction",
]

api = wandb.Api()

for run_id in args.runs:
    path = (f"{args.entity}/{args.project}/{run_id}" if args.entity
            else f"{args.project}/{run_id}")
    print(f"Fetching {path} ...")
    run = api.run(path)

    # Config summary — printed so it can be shared alongside the CSV
    print(f"  name:   {run.name}")
    print(f"  state:  {run.state}")
    keep = ["hidden_dim", "num_workers", "num_steps", "time_horizon", "lr",
            "gamma_w", "gamma_m", "alpha", "gae_lambda", "entropy_coef",
            "value_coef", "T_W", "T_M", "worker_layers", "manager_layers",
            "n_pi", "e_aux", "map_h", "map_w", "opponents"]
    for k in keep:
        if k in run.config:
            print(f"  {k}: {run.config[k]}")

    hist = run.history(samples=args.samples)
    cols = [c for c in METRICS if c in hist.columns]
    hist = hist[cols]

    out = f"export_{run_id}.csv"
    hist.to_csv(out, index=False)
    print(f"  → {out}  ({len(hist)} rows, {len(cols)} metrics)\n")

print("Done. Upload the CSVs for analysis.")
