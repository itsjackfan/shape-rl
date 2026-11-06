import subprocess

import pytest

from shape_rl import paths


def test_exp_name_from_branch(monkeypatch):
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **k: "feat/geo-adv\n")
    assert paths.exp_name() == "feat-geo-adv"


def test_fresh_dir_refuses_overwrite(tmp_path):
    d = paths.fresh_dir(tmp_path / "exp" / "abl")
    assert d.is_dir()
    with pytest.raises(FileExistsError):
        paths.fresh_dir(d)


LOG = """## other (branch: other)
x
- 2026-10-01 10:00 EST — abl: wrong section

## my-exp (branch: my-exp)
Question. Hypothesis.
- 2026-10-31 12:00 EST — abl: tests one thing
"""


def test_check_logged(tmp_path):
    log = tmp_path / "EXPERIMENT_LOG.md"
    log.write_text(LOG)
    paths.check_logged(log, "my-exp", "abl")
    with pytest.raises(RuntimeError):
        paths.check_logged(log, "my-exp", "missing")
    with pytest.raises(RuntimeError):
        paths.check_logged(log, "other", "abl2")
    with pytest.raises(RuntimeError):
        paths.check_logged(log, "nonexistent", "abl")


def test_exp_name_from_gitinfo(tmp_path, monkeypatch):
    (tmp_path / ".gitinfo").write_text("geo-adv abc123\n")
    monkeypatch.setattr(paths, "ROOT", tmp_path)
    assert paths.exp_name() == "geo-adv"


def test_job_dir_resumes_on_requeue(tmp_path, monkeypatch):
    d = paths.job_dir(tmp_path / "abl")
    assert d.is_dir()
    with pytest.raises(FileExistsError):
        paths.job_dir(d)
    monkeypatch.setenv("SLURM_RESTART_COUNT", "1")
    assert paths.job_dir(d) == d


def test_job_dir_requeue_before_first_start(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_RESTART_COUNT", "2")
    assert paths.job_dir(tmp_path / "abl").is_dir()


def test_last_complete_checkpoint(tmp_path):
    from shape_rl.trainer import STATE_FILE, last_complete_checkpoint

    assert last_complete_checkpoint(tmp_path) is None
    for step, complete in ((5, True), (10, True), (20, False)):  # 20: preempted while saving
        (tmp_path / f"checkpoint-{step}").mkdir()
        if complete:
            (tmp_path / f"checkpoint-{step}" / STATE_FILE).touch()
    assert last_complete_checkpoint(tmp_path) == str(tmp_path / "checkpoint-10")
