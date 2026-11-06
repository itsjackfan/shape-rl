# Goal: a first offline test of contrastive advantage estimation, in 7.5 hours

The repo is infrastructure only: a TRL 1.17.1 `GRPOTrainer` subclass with a swappable advantage-estimator hook, Hydra configs, WandB tracking, lm-eval and a FASRC launcher. Your goal: within **7.5 hours of starting**, deliver a first offline answer to the proposal's two hypotheses on a small model. Build a counterfactual dataset (token swaps at high-entropy positions, labelled by continuation rollouts, with last-layer hidden states), then compare contrastive estimators against baselines on held-out prompts. Verify only the parts of the existing code this result depends on. RL with the new estimator is the *next* goal.

`CLAUDE.md` is binding, including its exception for you, the autonomous agent running this prompt: you edit code without waiting for approval, launch your own jobs, and write your own run-guard lines tagged `[agent]`. You still never write hypotheses, expectations or results (the user writes those), never change an existing test, delete outputs or push without the user's approval, and never touch Jack's rsync mirrors. `CLAUDE.md` points to `CS2823r_Formal_Proposal.pdf`; read it if it's in your clone's root, otherwise work from the context below.

## Focus and time budget
- Run `date` when you start and put the elapsed time in every report. The final report is due by 7.5 h.
- Work only on the critical path: trustworthy labels → counterfactual dataset → estimators → final report. No benchmark evals, GRPO/RL runs, 27B or thinking-mode runs, hyperparameter sweeps, refactors of existing code, or fixes that the result doesn't need. The only extra runs allowed are one short smoke (≤10 min) before each new job type.
- Anything off the critical path, including review findings you don't need to fix, goes on a deferred list in the final report with its evidence.
- Milestones: pipeline coded, tested and smoke-passed by 1.5 h; full data generation running by 2 h; dataset complete by 4.5 h; all estimators evaluated by 6.5 h; final report by 7.5 h.
- If you fall behind, cut in this order: interpretability, C0, seeds (3 → 1), dataset size. Never cut B2, B3, C2 or the noise ceiling.

## Autonomy and compute
- Don't stop to ask. When a choice has a reasonable default, take it, say so in your next report, and keep going. Anything that needs the user's approval (a test change, a deletion, a push) goes on an approval list and never blocks other work. End your turn only when every remaining task is blocked on something only the user can provide (a credential, an access problem, a decision with no reasonable default), and say exactly what it is.
- GPUs: launch jobs yourself on the Kempner Krajan lab account, as much as the work needs to meet the deadline. First verify the exact account name and its partitions (`sacctmgr show assoc where user=$USER`, `sinfo`), then extend `cluster/launch.py` to submit there, choosing partitions by earliest `sbatch --test-only` start. Shard long work across GPUs and make it resumable (skip finished shards), since requeue partitions preempt. Cancel failed or stalled jobs, and keep a list of job IDs and what each is for.
- Register every run before launching it: under `## <exp>` in `EXPERIMENT_LOG.md`, add `- YYYY-MM-DD HH:MM EST — <ablation>: [agent] <what it tests>`. Leave the question, hypothesis and results to the user.
- While jobs run, work on the next step; check on them with `squeue`, `sacct` and their SLURM logs.
- At each milestone, report briefly in chat: elapsed time, what ran (job IDs), key numbers, defaults taken, what's next.

## Where you run
A FASRC login node: shared and CPU-only. Do only git, tests, SLURM commands and light analysis there; anything heavier (rollouts, hidden-state extraction, SSL training, analysis of large files, GPU tests) runs as a SLURM job.
- Work only in your own git clone of `itsjackfan/shape-rl` (the directory you were started in), on branch `ssl-goal`: branch your worktrees from `ssl-goal` and merge finished work back into it, never into `main`. Keep `core.hooksPath` unset.
- Never touch `/n/holylabs/ydu_lab/Lab/jfan425/shape-rl/<branch>/`: those are rsync mirrors of Jack's commits, overwritten with `--delete` on every sync.
- Each worktree needs the `.env` from your clone's root (copy it; never commit or print it) and `uv sync --extra cluster`. Download only the models you use (`uv run --env-file .env hf download <model>`); don't run `cluster/setup.sh`, which also pulls the 27B and 35B weights. If `.env` has no WandB key, run with `wandb.mode=offline`.
- Results are under `$SHAPE_RL_OUTPUT_ROOT/<exp>/`, SLURM logs under its `slurm/`.

## Research context
- H1: reasoning traces contain crucial token positions where swapping the token changes the rollout's expected success (a learnable advantage difference).
- H2: a contrastive embedding of the policy's last-layer hidden states, read out by clustering or a small FCNN, estimates advantages better than a learned value head (critic) and far cheaper than Monte-Carlo (MC) rollouts.
- Data: hidden states of on-policy ("oracle") traces and of token-replaced (counterfactual) prefixes, labelled by rollouts continued from those prefixes.

## Defaults (use them; list them in your first report)
1. Model: `Qwen/Qwen3.5-4B` (same family as Qwen3.6). No 27B runs in this goal.
2. Thinking off, set explicitly; a generation cap sized from the smoke so that under 5% of traces truncate.
3. "Oracle" traces: on-policy samples from π, correct and incorrect.
4. Prompts: GSM8K + MATH train, split by prompt into train/val/test.

## Step 1: verify only what the result depends on (≤45 min)
An initial review found the following. Confirm or refute each with evidence (file:line plus a snippet that reproduces it). Fix 1, 2 and 6 now, because they corrupt labels; put the rest on the deferred list unfixed.
1. **Thinking is on implicitly.** Qwen3.5/3.6 chat templates open `<think>` unless `enable_thinking=false`; nothing sets `chat_template_kwargs`, and `max_completion_length=2048`.
2. **The reward grades raw text.** `ShapeTrainer` passes no `tools`, so TRL never calls `add_response_schema`; unless the hub tokenizer ships a `response_template`, completions are decoded raw, thinking included. math-verify grades whatever it extracts: a truncated, unboxed trace whose last number equals the gold scores 1.0 (reproduced). Your rollout code must grade only finished answers and flag truncation.
3. Eval under-reports thinking models: lm-eval's default `max_gen_toks` is 256 on vLLM, and thinking is unset. (Deferred: no evals in this goal.)
4. The estimator sees only scalar rewards (`AdvantageEstimator.forward(rewards_N, G)`) and nothing captures hidden states. Offline you need the hidden-state utility (Step 2); the RL estimator interface is deferred. For later: TRL 1.17.1's loss already accepts `(B, T)` advantages, and `_generate_and_score_completions` already runs a no-grad forward over prompt + completion when vLLM is used.
5. `test_train_config_composes` (12 cases) builds `GRPOConfig(bf16=True)` and fails on CPU-only machines, login nodes included. Run the test suite in a short GPU job; don't edit the test.
6. `data.math` does `parse(solution)[-1]`; a solution math-verify can't parse raises IndexError at load.
7. With a custom estimator, TRL still logs its own group-normalised advantages: logged ≠ used. (Deferred.)
8. Evaluating a checkpoint while its training job runs resumes the same WandB run from two processes. (Deferred.)
9. The main/smoke run has no logged results. (Deferred: note what `$SHAPE_RL_OUTPUT_ROOT/main/` shows, nothing more.)

Exit: on ~20 real outputs of the model in your chosen mode, the reward agrees with a manual check and the truncation rate is known.

## Step 2: build the pipeline (by 1.5 h)
Tests first, minimal (shape tests plus one tiny-model smoke per entrypoint). Reusable code goes in `src/shape_rl/`; analysis and plots go in `scripts/tmp_<exp>/`.
1. Thinking and length control in the model config, used by generation.
2. A config for `Qwen/Qwen3.5-4B`.
3. One hidden-state utility: last-layer (post-norm) states at requested positions via a forward hook, not `output_hidden_states=True` (that returns every layer). Tested on the tiny model for equality with `output_hidden_states`.
4. A dataset format: sharded safetensors of bf16 states at selected positions only, plus one parquet index (prompt_id, split, trace_id, pos, token ids, entropy, logprob, MC estimates with counts, final reward, truncated flag, suite), under `$SHAPE_RL_OUTPUT_ROOT/<exp>/<ablation>/`, never overwritten.
5. Hydra entrypoints and launcher entries `rollout` (traces, branch points, continuations via vLLM; sharded, resumable), `encode` (states at branch points) and `ssl` (train and evaluate one estimator per config), each with a config group and the run guard.

## Step 3: counterfactual dataset (by 4.5 h)
- Prompts: keep those whose pass rate over 8 samples is strictly between 0 and 1. Branch on up to 2 correct and 2 incorrect traces per prompt.
- Branch points: 8 positions per trace sampled from its top-20% entropy positions, plus 8 random positions as a control (≈512 continuations per trace with the labels below).
- Labels: at branch point t, M = 8 continuations each from (a) the free prefix → V̂(s_t); (b) the prefix forced with the sampled token a_t; (c) the prefix forced with each of the k = 2 next most likely tokens a′. Q̂(s_t, a) is the success rate of (b)/(c); Â = Q̂ − V̂. Flag truncated continuations; never count them silently as failures.
- States: h(s_t) = last-layer state at position t−1 (predicts V); h(s_t, a) = state at position t with input token a (predicts Q). Put this indexing in the docstring.
- Smoke on ~50 prompts. From it, measure throughput, cost and label noise (the split-half reliability of Q̂ is the ceiling for every estimator). Then size the full set so it finishes by 4.5 h, sharded across as many GPUs as that takes.

## Step 4: estimators (by 6.5 h)
One experiment worktree; one change per ablation, each with its own log line, output dir and run. Same splits, readout and metrics everywhere; 3 seeds each. Baselines first:
- B0 per-prompt mean pass rate (GRPO's baseline); B1 entropy only; B2 linear probe on frozen h; B3 MLP value head regressed on Q̂ (the learned critic H2 must beat).

Contrastive variants: a small FCNN projection z = g(h) on frozen h, read out by one fixed head (kNN or linear) so only the representation changes:
- C0 label-free: trajectory InfoNCE (positives = nearby positions in the same trace, negatives = other prompts).
- C1 supervised contrastive: positives = same-prompt states in the same Q̂ bucket.
- C2 counterfactual pairs: anchor = h(s_t, a_t); positives = replacements with |ΔQ̂| ≤ δ; hard negatives = replacements with |ΔQ̂| > δ.
- C3 clustering: k-means on the best z, cluster-mean value as the estimator (the proposal's).

Metrics on held-out prompts, mean ± std over seeds, each against the noise ceiling: within-prompt AUROC of success from prefix; Spearman of predicted vs MC Â; sign accuracy of Q̂(a′) − Q̂(a_t) on counterfactual pairs; calibration. Control for confounds: probe for position, trace length and truncation, and report results with them regressed out. Interpretability, if time allows: cluster purity vs outcome and one 2-D projection.

## Final report (by 7.5 h; the goal is met when it's delivered)
In chat, numbers only; the user writes the interpretation:
1. Results table: every estimator × metric on held-out prompts, mean ± std, with the noise ceiling.
2. H1 check: the distribution of |Q̂(a′) − Q̂(a_t)| at high-entropy vs random branch points.
3. Plot paths; dataset stats (prompts, traces, branch points, continuations, tokens, GPU-hours); job IDs; defaults taken; cuts made.
4. The deferred list (with evidence) and the approval list.
5. Three lines proposing the next goal: the best estimator as `configs/objective/<name>.yaml`, producing token-level advantages from the hidden states of the existing forward pass, compared with GRPO / Dr. GRPO / DAPO at matched compute.

## Protocol for every step
1. Worktree and branch per `CLAUDE.md`. 2. Tests first, then code; `uv run pytest -m "not slow"` (in a GPU job when you need it fully green). 3. Register the run, then launch it. 4. Analyse with scripts in `scripts/tmp_<exp>/`, report, and take the next step. 5. Merge working branches into `ssl-goal` in your clone; never push or delete outputs without approval.
