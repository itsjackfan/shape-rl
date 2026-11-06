"""SLURM launcher. Humans only, from the synced worktree on a login node:
`uv run --env-file .env python cluster/launch.py train|eval <hydra overrides> [--gpus N] [--rtx] [--time D-HH:MM:SS]`"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir

from shape_rl import paths

ROOT = Path(__file__).resolve().parents[1]
GPUS = ["nvidia_h200", "nvidia_h100_80gb_hbm3", "nvidia_a100-sxm4-80gb", "nvidia_rtx_pro_6000_blackwell_server_edition"]  # >=80GB only
QUEUES = [("kempner_requeue", "kempner_ydu_lab"), ("gpu_requeue", "ydu_lab")]  # gpu_requeue denies kempner_* accounts


def queues(rtx):
    qs = [(p, a, g) for p, a in QUEUES for g in GPUS]
    return qs + [("kempner_rtx", "kempner_ydu_lab", GPUS[-1])] if rtx else qs


def best_queue(gpus, rtx, time="2-00:00:00"):
    """(partition, account, gpu type) with the earliest `sbatch --test-only` start (ISO timestamps sort as strings)."""
    def start(partition, account, gpu):
        err = subprocess.run(["sbatch", "--test-only", f"--partition={partition}", f"--account={account}", f"--gres=gpu:{gpu}:{gpus}",
                              f"--time={time}", str(ROOT / "cluster/run.sbatch")], cwd=ROOT, capture_output=True, text=True).stderr
        m = re.search(r"to start at (\S+)", err)
        return m.group(1) if m else "9999"  # partition lacks this GPU type
    return min(queues(rtx), key=lambda q: start(*q))


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("entry", choices=["train", "eval"])
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--rtx", action="store_true", help="also allow kempner_rtx (keep to a minimum)")
    parser.add_argument("--time", default="2-00:00:00")
    args, overrides = parser.parse_known_args(argv)
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        cfg = compose(args.entry, overrides=overrides)
    paths.check_logged(cfg.log_path, cfg.exp, cfg.ablation)
    logs = Path(cfg.output_root) / cfg.exp / "slurm"
    logs.mkdir(parents=True, exist_ok=True)
    partition, account, gpu = best_queue(args.gpus, args.rtx, args.time)
    subprocess.run(["sbatch", f"--job-name={cfg.exp}/{cfg.ablation}", f"--output={logs}/{cfg.ablation}-{args.entry}-%j.out",
                    f"--partition={partition}", f"--account={account}", f"--gres=gpu:{gpu}:{args.gpus}", f"--time={args.time}",
                    str(ROOT / "cluster/run.sbatch"), args.entry, *overrides], cwd=ROOT, check=True)


if __name__ == "__main__":
    main(sys.argv[1:])
