import hydra
from accelerate import PartialState
from hydra.utils import instantiate
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from trl import GRPOConfig

from shape_rl import data, paths
from shape_rl.tracking import WandbLogger
from shape_rl.trainer import ShapeTrainer, last_complete_checkpoint


def grpo_kwargs(cfg) -> dict:
    """GRPOConfig kwargs as plain containers (GRPOConfig serializes them to JSON). Later sources win:
    model.trl (vLLM sizing) < train < objective.trl."""
    c = lambda x: OmegaConf.to_container(x, resolve=True) if x is not None else {}
    return {**c(cfg.model.get("trl")), **c(cfg.train), **c(cfg.objective.trl)}


def train_dataset(cfg):
    """Train prompts whose chat-templated length is <= max_prompt_len tokens (the rule of rollout.py)."""
    tok = AutoTokenizer.from_pretrained(cfg.model.path)
    fits = lambda x: len(data.chat_prompt_ids(tok, x["prompt"], cfg.model.chat_template_kwargs)) <= cfg.max_prompt_len
    return data.load_train(cfg.data.train).filter(fits)


def run(cfg):
    paths.guard(cfg)
    out, state = paths.run_dir(cfg), PartialState()
    if state.is_main_process:
        paths.job_dir(out)
        OmegaConf.save(cfg, out / "config.yaml", resolve=True)
    state.wait_for_everyone()

    args = GRPOConfig(output_dir=str(out), run_name=f"{cfg.exp}/{cfg.ablation}", report_to="none", seed=cfg.seed, **grpo_kwargs(cfg))
    budget = OmegaConf.to_container(cfg.budget, resolve=True)
    trainer = ShapeTrainer(model=cfg.model.path, reward_funcs=[instantiate(r) for r in cfg.objective.rewards.values()], args=args,
                           train_dataset=train_dataset(cfg), estimator=instantiate(cfg.objective.estimator),
                           dynamic_sampling=cfg.objective.dynamic_sampling, budget=budget)
    logger = WandbLogger(cfg, out) if state.is_main_process else None
    if logger:
        trainer.add_callback(logger)
    trainer.train(resume_from_checkpoint=last_complete_checkpoint(out))  # non-None only after a requeue
    if logger:
        logger.finish()
    return trainer


@hydra.main(config_path="../../configs", config_name="train", version_base=None)
def main(cfg):
    run(cfg)


if __name__ == "__main__":
    main()
