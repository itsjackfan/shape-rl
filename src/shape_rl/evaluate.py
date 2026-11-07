import json
import shutil

import hydra
import lm_eval
from lm_eval.tasks import TaskManager

from shape_rl import paths
from shape_rl.trainer import last_complete_checkpoint
from shape_rl.tracking import WandbLogger, eval_metrics


def eval_dir(cfg):
    return paths.run_dir(cfg) / "eval" / (cfg.ckpt or "base")


def ckpt_progress(cfg) -> dict:
    """test/step and test/flops of the evaluated checkpoint (flops.json, written by ShapeTrainer); 0 for the base model."""
    if not cfg.ckpt:
        return {"test/step": 0, "test/flops": 0.0}
    f = paths.run_dir(cfg) / cfg.ckpt / "flops.json"
    p = json.loads(f.read_text()) if f.exists() else {"step": int(cfg.ckpt.split("-")[-1]), "flops": float("nan")}
    return {"test/step": p["step"], "test/flops": p["flops"]}


def prune_checkpoints(run):
    """Disk policy: delete each evaluated checkpoint (eval/<ckpt>/results.json exists), except the newest complete one,
    from which a requeued train job resumes."""
    keep = last_complete_checkpoint(run)
    for d in run.glob("checkpoint-*"):
        if str(d) != keep and (run / "eval" / d.name / "results.json").exists():
            shutil.rmtree(d)


def run(cfg):
    """Evaluate model.path (or <run_dir>/<ckpt>) with lm-eval; logs test/* into the ablation's eval run."""
    paths.guard(cfg)
    out = paths.job_dir(eval_dir(cfg))
    pretrained = paths.run_dir(cfg) / cfg.ckpt if cfg.ckpt else cfg.model.path
    benches = cfg.data.eval
    model_args = ",".join(a for a in (f"pretrained={pretrained}", cfg.eval.model_args, cfg.eval.backend_args) if a)
    res = lm_eval.simple_evaluate(model=cfg.eval.backend, model_args=model_args,
                                  tasks=sorted({b.task for b in benches.values()}), batch_size=cfg.eval.batch_size,
                                  limit=cfg.eval.limit, gen_kwargs=cfg.eval.gen_kwargs, apply_chat_template=True,
                                  confirm_run_unsafe_code=True, random_seed=cfg.seed,
                                  task_manager=TaskManager(include_path=str(paths.ROOT / "configs" / "lm_eval")))
    metrics = eval_metrics(res["results"], benches)
    logger = WandbLogger(cfg, paths.run_dir(cfg), job="eval")
    logger.log({**metrics, **ckpt_progress(cfg)})
    logger.finish()
    (out / "results.json").write_text(json.dumps(res["results"], indent=2, default=str))  # last: marks the eval complete
    if cfg.ckpt and cfg.eval.delete_evaluated:
        prune_checkpoints(paths.run_dir(cfg))
    return metrics


@hydra.main(config_path="../../configs", config_name="eval", version_base=None)
def main(cfg):
    run(cfg)


if __name__ == "__main__":
    main()
