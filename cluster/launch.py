"""SLURM launcher, from the synced worktree on a login node:
`uv run --env-file .env python cluster/launch.py train|eval|rollout|encode|ssl <hydra overrides> [--gpus N] [--rtx] [--time D-HH:MM:SS]
[--queues partition:account[:gpu],...] [--shards N]`
Queues: --queues, else $SHAPE_RL_QUEUES (same format, set in .env), else QUEUES x GPUS. Every job gets exp=<branch> and
code=<code hash> (a requeued job keeps its experiment name and refuses changed code)."""
import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from hydra import compose, initialize_config_dir

from shape_rl import paths

ROOT = Path(__file__).resolve().parents[1]
GPUS = ["nvidia_h200", "nvidia_h100_80gb_hbm3", "nvidia_a100-sxm4-80gb", "nvidia_rtx_pro_6000_blackwell_server_edition"]  # >=80GB only
QUEUES = [("kempner_requeue", "kempner_ydu_lab"), ("gpu_requeue", "ydu_lab")]  # gpu_requeue denies kempner_* accounts


def queues(rtx):
    qs = [(p, a, g) for p, a in QUEUES for g in GPUS]
    return qs + [("kempner_rtx", "kempner_ydu_lab", GPUS[-1])] if rtx else qs


def parse_queues(spec):
    """'kempner_requeue:kempner_krajan_lab:nvidia_h200,...' -> [(partition, account, gpu)]; no gpu = every type in GPUS."""
    out = []
    for q in spec.split(","):
        p, a, *g = q.split(":")
        out += [(p, a, x) for x in (g or GPUS)]
    return out


def gres(gpu, gpus):
    """GPU job: gres + 16 CPUs and 160G per GPU. --gpus 0 (CPU-only): 16 CPUs, 128G; run.sbatch sees SLURM_GPUS_ON_NODE unset."""
    return [f"--gres=gpu:{gpu}:{gpus}", "--cpus-per-gpu=16", "--mem-per-gpu=160G"] if gpus else ["--cpus-per-task=16", "--mem=128G"]


def best_queue(gpus, rtx, time="2-00:00:00", qs=None, extra=()):
    """(partition, account, gpu type) with the earliest `sbatch --test-only` start (ISO timestamps sort as strings)."""
    def start(partition, account, gpu):
        err = subprocess.run(["sbatch", "--test-only", f"--partition={partition}", f"--account={account}", *gres(gpu, gpus),
                              f"--time={time}", *extra, str(ROOT / "cluster/run.sbatch")], cwd=ROOT, capture_output=True, text=True).stderr
        m = re.search(r"to start at (\S+)", err)
        return m.group(1) if m else "9999"  # partition lacks this GPU type
    return min(qs or queues(rtx), key=lambda q: start(*q))


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("entry", choices=["train", "eval", "rollout", "encode", "ssl"])
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--rtx", action="store_true", help="also allow kempner_rtx (keep to a minimum)")
    parser.add_argument("--time", default="2-00:00:00")
    parser.add_argument("--queues", default=os.environ.get("SHAPE_RL_QUEUES") or None,
                        help="partition:account[:gpu],... (default: $SHAPE_RL_QUEUES, else QUEUES x GPUS)")
    parser.add_argument("--shards", type=int, default=0, help="rollout/encode: submit N jobs with shard=i num_shards=N")
    parser.add_argument("--sbatch", default="", help="extra sbatch flags, e.g. '--cpus-per-gpu=8 --mem-per-gpu=64G'")
    args, overrides = parser.parse_known_args(argv)
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        cfg = compose(args.entry, overrides=overrides)
    paths.check_logged(cfg.log_path, cfg.exp, cfg.ablation)
    overrides = [o for o in overrides if not o.startswith(("exp=", "code="))] + [f"exp={cfg.exp}", f"code={paths.code_hash()}"]
    logs = Path(cfg.output_root) / cfg.exp / "slurm"
    logs.mkdir(parents=True, exist_ok=True)
    qs = parse_queues(args.queues) if args.queues else None
    for extra in ([[f"shard={i}", f"num_shards={args.shards}"] for i in range(args.shards)] or [[]]):
        partition, account, gpu = best_queue(args.gpus, args.rtx, args.time, qs, args.sbatch.split())
        subprocess.run(["sbatch", f"--job-name={cfg.exp}/{cfg.ablation}", f"--output={logs}/{cfg.ablation}-{args.entry}-%j.out",
                        f"--partition={partition}", f"--account={account}", *gres(gpu, args.gpus), f"--time={args.time}", *args.sbatch.split(),
                        str(ROOT / "cluster/run.sbatch"), args.entry, *overrides, *extra], cwd=ROOT, check=True)
        time.sleep(1)


if __name__ == "__main__":
    main(sys.argv[1:])
