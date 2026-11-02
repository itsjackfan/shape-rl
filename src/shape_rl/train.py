import hydra
from accelerate import PartialState
from hydra.utils import instantiate
from omegaconf import OmegaConf
from trl import GRPOConfig

from shape_rl import data, paths
from shape_rl.rewards import correctness
from shape_rl.tracking import WandbLogger
from shape_rl.trainer import ShapeTrainer


def run(cfg):
    paths.check_logged(cfg.log_path, cfg.exp, cfg.ablation)
    out, state = paths.run_dir(cfg), PartialState()
    if state.is_main_process:
        paths.fresh_dir(out)
        OmegaConf.save(cfg, out / "config.yaml", resolve=True)
    state.wait_for_everyone()

    args = GRPOConfig(output_dir=str(out), run_name=f"{cfg.exp}/{cfg.ablation}", report_to="none", seed=cfg.seed,
                      **cfg.objective.trl, **cfg.train)
    trainer = ShapeTrainer(model=cfg.model.path, reward_funcs=[correctness], args=args,
                           train_dataset=data.load_train(cfg.data.train), estimator=instantiate(cfg.objective.estimator))
    logger = WandbLogger(cfg, out) if state.is_main_process else None
    if logger:
        trainer.add_callback(logger)
    trainer.train()
    if logger:
        logger.finish()
    return trainer


@hydra.main(config_path="../../configs", config_name="train", version_base=None)
def main(cfg):
    run(cfg)


if __name__ == "__main__":
    main()
