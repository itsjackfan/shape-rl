import importlib.util
import subprocess
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("launch", Path(__file__).parents[1] / "cluster" / "launch.py")
launch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launch)

STARTS = {("gpu_requeue", "nvidia_h200"): "2026-11-05T12:00:00", ("kempner_requeue", "nvidia_h100_80gb_hbm3"): "2026-11-05T09:00:00"}


@pytest.fixture
def fake_sbatch(monkeypatch):
    submitted = []

    def run(cmd, **kw):
        if "--test-only" in cmd:
            part = next(a.split("=")[1] for a in cmd if a.startswith("--partition="))
            gpu = next(a.split("=")[1].split(":")[1] for a in cmd if a.startswith("--gres="))
            return subprocess.CompletedProcess(cmd, 0, "", f"sbatch: Job 1 to start at {STARTS.get((part, gpu), '2026-11-06T00:00:00')} using 1 processors")
        submitted.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "Submitted batch job 7", "")

    monkeypatch.setattr(subprocess, "run", run)
    return submitted


def test_best_queue_picks_earliest(fake_sbatch):
    part, acct, gpu = launch.best_queue(gpus=1, rtx=False)
    assert (part, acct, gpu) == ("kempner_requeue", "kempner_ydu_lab", "nvidia_h100_80gb_hbm3")
    assert all(q[0] != "kempner_rtx" for q in launch.queues(rtx=False))
    assert any(q[0] == "kempner_rtx" for q in launch.queues(rtx=True))


def test_launch_submits_logged(cfg_factory, fake_sbatch):
    cfg = cfg_factory()
    launch.main(["train", "exp=test-exp", "ablation=abl", f"output_root={cfg.output_root}", f"log_path={cfg.log_path}", "--gpus", "2"])
    (cmd,) = fake_sbatch
    assert "--gres=gpu:nvidia_h100_80gb_hbm3:2" in cmd and "--account=kempner_ydu_lab" in cmd
    assert cmd[-5:] == ["train", "exp=test-exp", "ablation=abl", f"output_root={cfg.output_root}", f"log_path={cfg.log_path}"]


def test_launch_refuses_unlogged(cfg_factory, fake_sbatch):
    cfg = cfg_factory(logged=False)
    with pytest.raises(RuntimeError):
        launch.main(["train", "exp=test-exp", "ablation=abl", f"output_root={cfg.output_root}", f"log_path={cfg.log_path}"])
    assert not fake_sbatch
