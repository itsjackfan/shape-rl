import subprocess
from pathlib import Path

from omegaconf import OmegaConf


def exp_name() -> str:
    return subprocess.check_output(["git", "symbolic-ref", "--short", "HEAD"], text=True).strip().replace("/", "-")


OmegaConf.register_new_resolver("git_branch", exp_name, replace=True)


def run_dir(cfg) -> Path:
    return Path(cfg.output_root) / cfg.exp / cfg.ablation


def fresh_dir(path) -> Path:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} exists; outputs are never overwritten. Pick a new ablation name.")
    path.mkdir(parents=True)
    return path


def check_logged(log_path, exp: str, ablation: str):
    """Run guard: EXPERIMENT_LOG.md must have a `- ... — <ablation>: ...` line under `## <exp>` before any run."""
    text = Path(log_path).read_text() if Path(log_path).exists() else ""
    sections = ("\n" + text).split("\n## ")
    section = next((s for s in sections if s.split()[:1] == [exp]), "")
    if not any(line.startswith("- ") and f" {ablation}:" in line for line in section.splitlines()):
        raise RuntimeError(f"Log this ablation first: add `- YYYY-MM-DD HH:MM EST — {ablation}: <what it tests>` under `## {exp}` in {log_path}.")
