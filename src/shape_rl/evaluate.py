import json

import hydra
import lm_eval

from shape_rl import paths
from shape_rl.tracking import WandbLogger, eval_metrics


def eval_dir(cfg):
    return paths.run_dir(cfg) / "eval" / (cfg.ckpt or "base")


def run(cfg):
    """Evaluate model.path (or <run_dir>/<ckpt>) with lm-eval; logs test/* into the ablation's WandB run."""
    paths.check_logged(cfg.log_path, cfg.exp, cfg.ablation)
    out = paths.fresh_dir(eval_dir(cfg))
    pretrained = paths.run_dir(cfg) / cfg.ckpt if cfg.ckpt else cfg.model.path
    benches = cfg.data.eval
    res = lm_eval.simple_evaluate(model=cfg.eval.backend, model_args=f"pretrained={pretrained},dtype=bfloat16",
                                  tasks=sorted({b.task for b in benches.values()}), batch_size=cfg.eval.batch_size,
                                  limit=cfg.eval.limit, gen_kwargs=cfg.eval.gen_kwargs, apply_chat_template=True,
                                  confirm_run_unsafe_code=True, random_seed=cfg.seed)
    (out / "results.json").write_text(json.dumps(res["results"], indent=2, default=str))

    metrics = eval_metrics(res["results"], benches)
    logger = WandbLogger(cfg, paths.run_dir(cfg))
    logger.log({**metrics, "test/step": int(cfg.ckpt.split("-")[-1]) if cfg.ckpt else 0})
    logger.finish()
    return metrics


@hydra.main(config_path="../../configs", config_name="eval", version_base=None)
def main(cfg):
    run(cfg)


if __name__ == "__main__":
    main()
