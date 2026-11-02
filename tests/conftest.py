from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

CONFIGS = str(Path(__file__).parents[1] / "configs")


@pytest.fixture
def cfg_factory(tmp_path, monkeypatch):
    """Compose a config with outputs + run log redirected to tmp_path and wandb disabled."""
    monkeypatch.setenv("WANDB_MODE", "disabled")

    def make(name="train", overrides=(), logged=True):
        log = tmp_path / "EXPERIMENT_LOG.md"
        overrides = ["exp=test-exp", "ablation=abl", f"output_root={tmp_path / 'outputs'}", f"log_path={log}", "wandb.mode=disabled", *overrides]
        with initialize_config_dir(config_dir=CONFIGS, version_base=None):
            cfg = compose(name, overrides=overrides)
        if logged:
            log.write_text(f"## {cfg.exp} (branch: {cfg.exp})\nq\n- 2026-10-31 12:00 EST — {cfg.ablation}: test\n")
        return cfg

    return make
