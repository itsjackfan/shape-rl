# shape-rl protocol

Geometry-based RL / self-supervised advantage estimation for LLM post-training (see `CS2823r_Formal_Proposal.pdf`).
Built on TRL `GRPOTrainer` (pinned dependency, subclassed — not vendored). Runs on Harvard FASRC (SLURM).

   - Exception (Kaden, 2026-11-06): the autonomous agent running `prompts/SSL_GOAL_PROMPT.md` in its own cluster clone edits code without waiting for approval, launches its own jobs on the Kempner Krajan lab account, and writes its own run-guard lines tagged `[agent]`. The other rules above still bind it.

## Layout
```
configs/              Hydra. train.yaml, eval.yaml share base.yaml
  model/              one yaml per model (path for from_pretrained)
  objective/          estimator (_target_ or null = TRL built-in), rewards, dynamic_sampling + trl: GRPOConfig kwargs
                      (_base.yaml shared; grpo, dr_grpo, dapo, noop; hvalue = the method, hcritic = MLP critic, hcluster = GiGPO/BiPACE-style)
  lm_eval/            local lm-eval tasks (math500: minerva_math500 without its antlr4 4.11 check)
  data/train/         one yaml per train set (_target_ loader); enable via data/train=[a,b]
  data/eval/          one yaml per benchmark: lm-eval task, suite, primary metric
  rollout/, estimator/  offline dataset settings; one yaml per offline estimator (B0-B3, C0-C3; _head.yaml shared)
src/shape_rl/
  paths.py            exp = git branch; run dir; never-overwrite; run guard
  tracking.py         the only WandB interface; fixed metric namespaces
  data.py             train loaders -> {prompt, answer, suite}
  rewards.py          grade (last \boxed of a finished response) + Correctness (RL reward = grade; truncated = 0)
  advantage.py        AdvantageEstimator base (observe hidden states, forward -> (n,) or (n, T)); one subclass per ablation
  trainer.py          ShapeTrainer(GRPOTrainer): estimator + hidden-state hook, DAPO dynamic sampling, FLOP count/budget, per-suite metrics
  flops.py            model FLOPs of the training process (the compute-matching axis)
  train.py, evaluate.py  Hydra entrypoints (RL, lm-eval)
  offline estimator study (Hydra entrypoints rollout, encode, ssl):
  rollout.py          counterfactual rollouts (vLLM): traces -> branch points -> MC labels of V(s_t), Q(s_t, a)
  hidden.py, encode.py  last-layer states at the branch points
  ssl.py              offline estimators (heads exported for RL); contrastive.py C0-C3 projections; ssl_metrics.py metrics, ceilings
cluster/              launch.py (log gate, queue choice, sbatch), run.sbatch, zero3.yaml, setup.sh, hooks/post-commit
tests/                shape + smoke tests (mocked hub/lm-eval; `-m slow` runs a tiny real GRPO step)
scripts/tmp_<exp>/    one-off scripts; delete once results are documented (ask first)
```

## Idioms
- Minimal code, Torch/HF idioms, no defensive checks, comment only non-obvious info.
- Tensor dims: Shazeer suffixes (`rewards_N`, `x_BLD`) with a one-docstring legend at the top of each file. Prefer einsum/einops.
- Hot-swap by config, not flags: new estimator = subclass in `advantage.py` + `configs/objective/<name>.yaml`; new train set = function in `data.py` + `configs/data/train/<name>.yaml`; new benchmark = `configs/data/eval/<name>.yaml`; new model = `configs/model/<name>.yaml`.
- All metrics go through `tracking.py`. Namespaces: `train/*`, `test/*`, `optim/*`, `progress/*`, `perf/*`, `policy/*` (ablation-specific). Don't invent new top-level namespaces.
- Tests: shape + smoke, plus exhaustive/property checks of pure functions (`tests/test_properties.py`). `uv run pytest -m "not slow"` before handing over any change; `uv run pytest` for trainer changes.

## Experiments
`<exp>` = short descriptive name of the hypothesis. One change per ablation.
1. `git worktree add ../shape-rl-<exp> -b <exp>` (exp is read from the branch name).
2. Propose changes; edit only after approval.
3. Hand the user the launch command (run from the synced worktree on a FASRC login node), e.g.
   `uv run --env-file .env python cluster/launch.py train ablation=<abl> objective=dr_grpo --gpus 4`
   `uv run --env-file .env python cluster/launch.py eval ablation=<abl> ckpt=checkpoint-500`
   One job per launch; each ablation needs its own log line, dir and run.
4. Run guard: train/eval refuse to start unless `EXPERIMENT_LOG.md` has `- YYYY-MM-DD HH:MM EST — <abl>: <what it tests>` under `## <exp>`. The user writes it.
5. Outputs: `$SHAPE_RL_OUTPUT_ROOT/<exp>/<ablation>/` (default `outputs/`), never overwritten. WandB run `<exp>/<ablation>`, group `<exp>`; evals log `test/*` into a second run `<exp>/<ablation>/eval` (same group) against `test/flops` (and `test/step`). Compute matching = equal `progress/flops` (training process only: generation, scoring and training passes, estimator; dataset building excluded); use `budget.max_flops` / `budget.save_every_flops`.
   Disk (about 60 GB per 4B checkpoint): evaluate each checkpoint. A checkpoint eval then deletes every evaluated checkpoint
   except the newest complete one, which a requeue needs (`eval.delete_evaluated`; set false to keep them).
6. Smoke/failed/test runs not needed for a writeup: once the user has processed results, remind them to delete; delete on approval.
7. Done: success -> merge to main. Failure -> keep branch, push; the user adds an entry in main's log (branch, code changes, why it failed).

## Cluster
- Queues: always `kempner_requeue` (account `kempner_ydu_lab`) or `gpu_requeue` (account `ydu_lab`; denies kempner accounts), whichever has the earlier `sbatch --test-only` start. Other users set `SHAPE_RL_QUEUES` in `.env` (for example `kempner_requeue:kempner_krajan_lab`). `kempner_rtx` only when needed; keep it to a minimum.
- `cluster/launch.py` picks the earliest (partition, account, GPU type) by `sbatch --test-only`; GPU types are >=80GB only (H200, H100, A100-80GB, RTX PRO 6000). `--rtx` adds `kempner_rtx`.
- Requeue partitions preempt: a requeued job reuses its run dir (`paths.job_dir`) and resumes from the last checkpoint (with the FLOP count and the online estimator head: `shape_state.pt`); up to `save_steps` of work is lost. launch.py passes `exp=` and `code=<hash>`: a requeued job refuses changed code. SLURM logs append to `<output_root>/<exp>/slurm/`.
- Code lives in git locally only. Every commit rsyncs the worktree to `fasrc:/n/holylabs/ydu_lab/Lab/jfan425/shape-rl/<branch>/` (`cluster/hooks/post-commit`, needs an open `ssh fasrc` ControlMaster; enable with `git config core.hooksPath cluster/hooks`). It writes `.gitinfo` (branch + commit), which gives `exp` on the cluster. Never edit code in these rsync mirrors. Exception: an agent running on the cluster works only in its own git clone (worktrees beside it, hook off) and pushes only when asked.
- Once per synced worktree, and after dependency changes: `bash cluster/setup.sh` (uv sync with the `cluster` extra = vLLM + DeepSpeed; downloads `configs/model/*` weights). Jobs use `uv run --no-sync`.
- `.env` (cluster only, never committed; see `.env.example`): WandB/HF tokens, `SHAPE_RL_OUTPUT_ROOT` on netscratch (purged: copy keepers to holylabs), `HF_HOME` = the shared lab cache `/n/netscratch/ydu_lab/Lab/jfan425/hf_cache` (same as dec-mas).
- Jobs set `train.use_vllm=true` / `eval.backend=vllm`. Full-param RL on 27B/35B needs several GPUs (~430GB with ZeRO-3); eval fits on 1.
