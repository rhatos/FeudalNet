"""
run_config.py

Write every parameter of a training run to a CSV at startup, so each run is
reproducible from its own record rather than from shell history.

The file captures more than argparse: it also records the resolved device, the
git commit, the model size, the wandb run id, and the host — the things you
need to explain a result months later but never think to save at the time.

Used by the training scripts via:

    from run_config import write_run_config
    write_run_config(args, run=run, extra={"model/n_params": n_params})
"""

import os
import csv
import sys
import socket
import getpass
import platform
import datetime
import subprocess


def _git_info():
    """Current commit and whether the tree was dirty, if this is a git repo."""
    out = {}
    try:
        root = subprocess.run(["git", "rev-parse", "--show-toplevel"],
                              capture_output=True, text=True, timeout=5)
        if root.returncode != 0:
            return out
        out["git/commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            timeout=5).stdout.strip()
        out["git/branch"] = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True,
            text=True, timeout=5).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain"],
                                capture_output=True, text=True,
                                timeout=5).stdout.strip()
        out["git/dirty"] = bool(status)
        if status:
            # Which files differ from the commit — the usual reason a rerun
            # of the "same" commit does not reproduce. Porcelain lines are
            # "XY path", but the status field width varies between git
            # versions, so split rather than slicing a fixed offset.
            files = []
            for line in status.splitlines()[:10]:
                parts = line.strip().split(maxsplit=1)
                files.append(parts[-1] if parts else line.strip())
            out["git/modified"] = "; ".join(files)
    except (OSError, subprocess.SubprocessError):
        pass
    return out


def _env_info():
    """Versions and hardware — the parts of a result that argparse misses."""
    info = {
        "env/python":   platform.python_version(),
        "env/platform": platform.platform(),
        "env/host":     socket.gethostname(),
        "env/user":     getpass.getuser(),
        "env/cwd":      os.getcwd(),
        "env/command":  " ".join([os.path.basename(sys.argv[0])] + sys.argv[1:]),
    }
    for var in ("SLURM_JOB_ID", "SLURM_ARRAY_TASK_ID", "SLURM_JOB_NODELIST",
                "CUDA_VISIBLE_DEVICES"):
        if os.environ.get(var):
            info[f"env/{var}"] = os.environ[var]
    try:
        import torch
        info["env/torch"] = torch.__version__
        info["env/cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["env/cuda"] = torch.version.cuda
            info["env/gpu"] = torch.cuda.get_device_name(0)
            info["env/gpu_count"] = torch.cuda.device_count()
            props = torch.cuda.get_device_properties(0)
            info["env/gpu_mem_gb"] = round(props.total_memory / 1024 ** 3, 1)
    except Exception:                                        # noqa: BLE001
        pass
    try:
        import numpy
        info["env/numpy"] = numpy.__version__
    except Exception:                                        # noqa: BLE001
        pass
    try:
        import gym_microrts
        info["env/gym_microrts"] = getattr(gym_microrts, "__version__",
                                           "unknown")
    except Exception:                                        # noqa: BLE001
        pass
    return info


def _fmt(v):
    """Render a value so the CSV round-trips as text."""
    if isinstance(v, (list, tuple)):
        return " ".join(str(x) for x in v)
    if isinstance(v, dict):
        return "; ".join(f"{k}={x}" for k, x in v.items())
    return str(v)


def write_run_config(args, run=None, extra=None, path=None, quiet=False):
    """
    Write every run parameter to a CSV.

    args   : the argparse Namespace (all of it, whatever the script defines)
    run    : the wandb run, if one exists — records id, name, url, project
    extra  : any extra key/value pairs to record (e.g. model/n_params)
    path   : output path. Defaults to
             <args.save_dir>/configs/<run name or timestamp>_config.csv

    Returns the path written. Never raises: a run should not die because its
    config file could not be saved.
    """
    try:
        rows = {}

        # 1. Every argparse parameter, whatever the script happens to define
        for k, v in sorted(vars(args).items()):
            rows[f"arg/{k}"] = _fmt(v)

        # 2. wandb identifiers, so the CSV and the dashboard can be matched up
        if run is not None:
            rows["wandb/id"] = getattr(run, "id", "")
            rows["wandb/name"] = getattr(run, "name", "")
            rows["wandb/project"] = getattr(run, "project", "")
            rows["wandb/entity"] = getattr(run, "entity", "")
            try:
                rows["wandb/url"] = run.get_url()
            except Exception:                                # noqa: BLE001
                pass

        # 3. Environment, versions, hardware, git state
        rows.update({k: _fmt(v) for k, v in _env_info().items()})
        rows.update({k: _fmt(v) for k, v in _git_info().items()})
        rows["run/started"] = datetime.datetime.now().isoformat(timespec="seconds")

        if extra:
            rows.update({k: _fmt(v) for k, v in extra.items()})

        # --- Resolve the output path ---
        if path is None:
            stem = (getattr(run, "name", None)
                    or getattr(args, "run_name", None)
                    or datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S"))
            stem = str(stem).replace("/", "_").replace(" ", "_")
            rid = getattr(run, "id", None)
            if rid:
                stem = f"{stem}_{rid}"
            base = getattr(args, "save_dir", ".") or "."
            path = os.path.join(base, "configs", f"{stem}_config.csv")

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        # Long format (one parameter per row) rather than one wide row: it
        # stays readable for ~80 parameters and diffs cleanly between runs.
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["parameter", "value"])
            for k in sorted(rows):
                w.writerow([k, rows[k]])

        if not quiet:
            print(f"Run config ({len(rows)} parameters) → {path}")

        # Attach to the wandb run so it is preserved alongside the metrics
        if run is not None:
            try:
                import wandb
                wandb.save(path)
            except Exception:                                # noqa: BLE001
                pass

        return path

    except Exception as e:                                   # noqa: BLE001
        # A failed config dump must never take the training run down with it
        print(f"WARNING: could not write the run config CSV ({e})")
        return None


# ---------------------------------------------------------------------------
# CLI: compare the config CSVs of two runs
# ---------------------------------------------------------------------------

def _load(path):
    with open(path, newline="") as f:
        return {r["parameter"]: r["value"] for r in csv.DictReader(f)}


def _main():
    import argparse
    p = argparse.ArgumentParser(
        description="Show a run config, or diff two of them")
    p.add_argument("csv", nargs="+", help="One config CSV to show, or two to diff")
    p.add_argument("--all", action="store_true",
                   help="When diffing, also list identical parameters")
    a = p.parse_args()

    if len(a.csv) == 1:
        cfg = _load(a.csv[0])
        width = max(len(k) for k in cfg)
        for k in sorted(cfg):
            print(f"{k:<{width}}  {cfg[k]}")
        return

    left, right = _load(a.csv[0]), _load(a.csv[1])
    keys = sorted(set(left) | set(right))
    width = max(len(k) for k in keys)
    n_diff = 0
    print(f"{'parameter':<{width}}  {os.path.basename(a.csv[0])}"
          f"  |  {os.path.basename(a.csv[1])}\n" + "-" * (width + 40))
    for k in keys:
        lv, rv = left.get(k, "<absent>"), right.get(k, "<absent>")
        if lv != rv:
            n_diff += 1
            print(f"{k:<{width}}  {lv}  |  {rv}")
        elif a.all:
            print(f"{k:<{width}}  {lv}  (same)")
    print("-" * (width + 40))
    print(f"{n_diff} parameter(s) differ of {len(keys)}")


if __name__ == "__main__":
    _main()
