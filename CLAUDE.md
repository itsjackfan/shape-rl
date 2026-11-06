# shape-rl protocol

Geometry-based RL / self-supervised advantage estimation for LLM post-training (see `CS2823r_Formal_Proposal.pdf`).
Built on TRL `GRPOTrainer` (pinned dependency, subclassed — not vendored). Runs on Harvard FASRC (SLURM).

## Hard constraints (every agent, every subagent)
- NEVER start runs (`cluster/launch.py`, `sbatch`, `srun`, `salloc`, train, eval), even if asked. `sbatch --test-only` is fine. The user starts every run.
- NEVER write experiment documentation (hypotheses, expectations, code-change rationale, results). Prompt the user to write it.
- ALWAYS propose code changes and wait for approval before editing, even in auto mode.
- Tests are near-immutable. Change a test only with explicit user approval.

## Layout
```
configs/              Hydra. train.yaml, eval.yaml share base.yaml
  model/              one yaml per model (path for from_pretrained)
  objective/          estimator (_target_ or null = TRL built-in) + trl: GRPOConfig kwargs
  data/train/         one yaml per train set (_target_ loader); enable via data/train=[a,b]
  data/eval/          one yaml per benchmark: lm-eval task, suite, primary metric
src/shape_rl/
  paths.py            exp = git branch; run dir; never-overwrite; run guard
  tracking.py         the only WandB interface; fixed metric namespaces
  data.py             train loaders -> {prompt, answer, suite}
  rewards.py          reward funcs (correctness = mean_score)
  advantage.py        AdvantageEstimator base; one subclass per ablation
  trainer.py          ShapeTrainer(GRPOTrainer): estimator hook + per-suite metrics
  train.py, evaluate.py  Hydra entrypoints
cluster/              launch.py (log gate, queue choice, sbatch), run.sbatch, zero3.yaml, setup.sh, hooks/post-commit
tests/                shape + smoke tests (mocked hub/lm-eval; `-m slow` runs a tiny real GRPO step)
scripts/tmp_<exp>/    one-off scripts; delete once results are documented (ask first)
```

## Idioms
- Minimal code, Torch/HF idioms, no defensive checks, comment only non-obvious info.
- Tensor dims: Shazeer suffixes (`rewards_N`, `x_BLD`) with a one-docstring legend at the top of each file. Prefer einsum/einops.
- Hot-swap by config, not flags: new estimator = subclass in `advantage.py` + `configs/objective/<name>.yaml`; new train set = function in `data.py` + `configs/data/train/<name>.yaml`; new benchmark = `configs/data/eval/<name>.yaml`; new model = `configs/model/<name>.yaml`.
- All metrics go through `tracking.py`. Namespaces: `train/*`, `test/*`, `optim/*`, `progress/*`, `perf/*`, `policy/*` (ablation-specific). Don't invent new top-level namespaces.
- Tests: shape + smoke only. `uv run pytest -m "not slow"` before handing over any change; `uv run pytest` for trainer changes.

## Experiments
`<exp>` = short descriptive name of the hypothesis. One change per ablation.
1. `git worktree add ../shape-rl-<exp> -b <exp>` (exp is read from the branch name).
2. Propose changes; edit only after approval.
3. Hand the user the launch command (run from the synced worktree on a FASRC login node), e.g.
   `uv run --env-file .env python cluster/launch.py train ablation=<abl> objective=dr_grpo --gpus 4`
   `uv run --env-file .env python cluster/launch.py eval ablation=<abl> ckpt=checkpoint-500`
   One job per launch; each ablation needs its own log line, dir and run.
4. Run guard: train/eval refuse to start unless `EXPERIMENT_LOG.md` has `- YYYY-MM-DD HH:MM EST — <abl>: <what it tests>` under `## <exp>`. The user writes it.
5. Outputs: `$SHAPE_RL_OUTPUT_ROOT/<exp>/<ablation>/` (default `outputs/`), never overwritten. WandB run `<exp>/<ablation>`, group `<exp>`; eval resumes the same run and logs `test/*` against `test/step`.
6. Smoke/failed/test runs not needed for a writeup: once the user has processed results, remind them to delete; delete on approval.
7. Done: success -> merge to main. Failure -> keep branch, push; the user adds an entry in main's log (branch, code changes, why it failed).

## Cluster
- Queues: always `kempner_requeue` (account `kempner_ydu_lab`) or `gpu_requeue` (account `ydu_lab`; denies kempner accounts), whichever has the earlier `sbatch --test-only` start. `kempner_rtx` only when needed; keep it to a minimum.
- `cluster/launch.py` picks the earliest (partition, account, GPU type) by `sbatch --test-only`; GPU types are >=80GB only (H200, H100, A100-80GB, RTX PRO 6000). `--rtx` adds `kempner_rtx`.
- Requeue partitions preempt: a requeued job reuses its run dir (`paths.job_dir`) and resumes from the last checkpoint; up to `save_steps` of work is lost. SLURM logs append to `<output_root>/<exp>/slurm/`.
- Code lives in git locally only. Every commit rsyncs the worktree to `fasrc:/n/holylabs/ydu_lab/Lab/jfan425/shape-rl/<branch>/` (`cluster/hooks/post-commit`, needs an open `ssh fasrc` ControlMaster; enable with `git config core.hooksPath cluster/hooks`). It writes `.gitinfo` (branch + commit), which gives `exp` on the cluster. Never edit code on the cluster.
- Once per synced worktree, and after dependency changes: `bash cluster/setup.sh` (uv sync with the `cluster` extra = vLLM + DeepSpeed; downloads `configs/model/*` weights). Jobs use `uv run --no-sync`.
- `.env` (cluster only, never committed; see `.env.example`): WandB/HF tokens, `SHAPE_RL_OUTPUT_ROOT` on netscratch (purged: copy keepers to holylabs), `HF_HOME` = the shared lab cache `/n/netscratch/ydu_lab/Lab/jfan425/hf_cache` (same as dec-mas).
- Jobs set `train.use_vllm=true` / `eval.backend=vllm`. Full-param RL on 27B/35B needs several GPUs (~430GB with ZeRO-3); eval fits on 1.
