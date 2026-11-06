import hashlib
import os
import subprocess
from pathlib import Path

from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[2]


def exp_name() -> str:
    if (ROOT / ".gitinfo").exists() and not (ROOT / ".git").exists():  # cluster copy synced by cluster/hooks/post-commit
        return (ROOT / ".gitinfo").read_text().split()[0]
    return subprocess.check_output(["git", "symbolic-ref", "--short", "HEAD"], cwd=ROOT, text=True).strip().replace("/", "-")


def code_hash() -> str:
    """sha1 of the code a job runs (src/ and configs/), committed or not."""
    h = hashlib.sha1()
    for f in sorted([*(ROOT / "src").rglob("*.py"), *(ROOT / "configs").rglob("*.yaml"), *(ROOT / "configs").rglob("*.py")]):
        h.update(str(f.relative_to(ROOT)).encode() + b"\0" + f.read_bytes())
    return h.hexdigest()[:12]


OmegaConf.register_new_resolver("git_branch", exp_name, replace=True)
OmegaConf.register_new_resolver("add", lambda *x: sum(x), replace=True)


def run_dir(cfg) -> Path:
    return Path(cfg.output_root) / cfg.exp / cfg.ablation


def fresh_dir(path) -> Path:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} exists; outputs are never overwritten. Pick a new ablation name.")
    path.mkdir(parents=True)
    return path


def job_dir(path) -> Path:
    """fresh_dir, except a requeued SLURM job reuses its own dir and resumes."""
    if int(os.environ.get("SLURM_RESTART_COUNT", 0)) > 0:
        return Path(path)
    return fresh_dir(path)


def check_logged(log_path, exp: str, ablation: str):
    """Run guard: EXPERIMENT_LOG.md must have a `- ... — <ablation>: ...` line under `## <exp>` before any run."""
    text = Path(log_path).read_text() if Path(log_path).exists() else ""
    sections = ("\n" + text).split("\n## ")
    section = next((s for s in sections if s.split()[:1] == [exp]), "")
    if not any(line.startswith("- ") and f" {ablation}:" in line for line in section.splitlines()):
        raise RuntimeError(f"Log this ablation first: add `- YYYY-MM-DD HH:MM EST — {ablation}: <what it tests>` under `## {exp}` in {log_path}.")


def check_code(code):
    """A job launched by cluster/launch.py carries the code hash of launch time; a requeued job refuses changed code."""
    if code and code != code_hash():
        raise RuntimeError(f"The code changed since launch ({code} -> {code_hash()}); relaunch under a new ablation name.")


def guard(cfg):
    check_logged(cfg.log_path, cfg.exp, cfg.ablation)
    check_code(cfg.get("code"))
