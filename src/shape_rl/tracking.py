"""Single WandB interface. Every metric goes through here so keys are identical across experiments.

Namespaces: train/* (loss, reward, mean_score, <suite>/mean_score), test/* (same, held-out + lm-eval),
optim/* (lr, grad_norm), progress/* (step, epoch), perf/* (tokens_per_sec, steps_per_sec),
policy/* (everything else TRL logs, plus ablation-specific metrics).
"""
import time
from collections import defaultdict

import wandb
from omegaconf import OmegaConf
from transformers import TrainerCallback

FIXED = {"learning_rate": "optim/lr", "grad_norm": "optim/grad_norm", "epoch": "progress/epoch"}
SCORE = ("loss", "reward", "reward_std", "mean_score")


def rename(key: str) -> str:
    if key in FIXED:
        return FIXED[key]
    split = "test" if key.startswith("eval_") else "train"
    key = key.removeprefix("eval_")
    if key in SCORE or key.endswith("/mean_score") or key.startswith("rewards/"):
        return f"{split}/{key}"
    return f"policy/{key}" if split == "train" else f"test/policy/{key}"


def eval_metrics(results: dict, benches: dict) -> dict:
    """lm-eval results -> test/<suite>/<bench>, test/<suite>/mean_score, test/mean_score."""
    m, by_suite = {}, defaultdict(list)
    for name, b in benches.items():
        score = results[b["task"]][b["metric"]]
        m[f"test/{b['suite']}/{name}"] = score
        by_suite[b["suite"]].append(score)
    for suite, scores in by_suite.items():
        m[f"test/{suite}/mean_score"] = sum(scores) / len(scores)
    m["test/mean_score"] = sum(m[f"test/{s}/mean_score"] for s in by_suite) / len(by_suite)
    return m


class WandbLogger(TrainerCallback):
    def __init__(self, cfg, run_dir):
        id_file = run_dir / "wandb_id"
        run_id = id_file.read_text() if id_file.exists() else None
        self.run = wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity, mode=cfg.wandb.mode, group=cfg.exp,
                              name=f"{cfg.exp}/{cfg.ablation}", id=run_id, resume="allow", dir=str(run_dir),
                              config=OmegaConf.to_container(cfg, resolve=True))
        self.run.define_metric("test/*", step_metric="test/step")
        id_file.write_text(self.run.id)
        self.last = None  # (time, step, num_tokens)

    def log(self, metrics: dict):
        self.run.log(metrics)

    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = dict(logs)
        now, tokens = time.perf_counter(), logs.pop("num_tokens", None)
        m = {rename(k): v for k, v in logs.items()}
        m["progress/step"] = state.global_step
        if self.last is not None:
            t0, s0, n0 = self.last
            m["perf/steps_per_sec"] = (state.global_step - s0) / (now - t0)
            if tokens is not None:
                m["perf/tokens_per_sec"] = (tokens - n0) / (now - t0)
        self.last = (now, state.global_step, tokens)
        self.log(m)

    def finish(self):
        self.run.finish()
