# Experiment log

Written by humans only. Format (the run guard checks for the ablation line under its `## <exp>` heading):

```
## <exp> (branch: <exp>)
<2 sentences: question + hypothesis>
- YYYY-MM-DD HH:MM EST — <ablation>: <one-liner: what this ablation tests>
```

Failed experiments: add an entry here on main with branch, code changes, and why it failed.

## main (branch: main)
- 2026-11-05 19;34 - smoke: setting up all of the infrastructure and running an initial RL smoke run on Qwen 3.6 27B/35B; using TRL and DeepSpeed optimiser on the FASRC cluster
    - Changes: all of the above
    - Expected: W
    - Results: 

## cf-data (branch: cf-data)
- 2026-11-06 03:40 EST — smoke50: [agent] rollout smoke on 50 prompts (Qwen3.5-4B, thinking off, cap 4096): grader check on real outputs, truncation rate, throughput, split-half label noise
- 2026-11-06 04:20 EST — smoke50: [agent] encode smoke on the smoke50 rollouts: last-layer states at branch points, throughput
- 2026-11-06 04:20 EST — smoke50-ssl: [agent] ssl smoke: one estimator, one seed on the smoke50 dataset (pipeline check, not a result)
- 2026-11-06 04:40 EST — full: [agent] counterfactual dataset (rollout + encode): all GSM8K+MATH train prompts x 8 samples, cap 16384, branch on prompts with 0 < pass rate < 1; 32 shards on kempner_requeue
- 2026-11-06 05:09 EST — smoke-ck: [agent] smoke of the checkpointed vLLM engine loop (stage A + one branch chunk), 40 prompts, cap 2048, MIG slice

## ssl-est (branch: ssl-est)
- 2026-11-06 04:48 EST — b0: [agent] baseline B0: per-prompt pass rate over the 8 oracle samples (GRPO's baseline) as the value of every prefix; dataset cf-data/full, 3 seeds
- 2026-11-06 04:48 EST — b1: [agent] baseline B1: logistic regression on entropy and logprob of the prefix's last token; dataset cf-data/full, 3 seeds
- 2026-11-06 04:48 EST — b2: [agent] baseline B2: linear (logistic) probe on frozen last-layer h; dataset cf-data/full, 3 seeds
- 2026-11-06 04:48 EST — b3: [agent] baseline B3: MLP value head on frozen h regressed on MC success rates (the learned critic); dataset cf-data/full, 3 seeds
- 2026-11-06 04:48 EST — c0: [agent] C0: label-free trajectory InfoNCE projection z = g(h), linear readout; dataset cf-data/full, 3 seeds
- 2026-11-06 04:48 EST — c1: [agent] C1: supervised contrastive projection (same prompt, same success-rate bucket), linear readout; dataset cf-data/full, 3 seeds
- 2026-11-06 04:48 EST — c2: [agent] C2: counterfactual-pair contrastive projection (|dQ| <= 0.25 positives, > 0.25 hard negatives), linear readout; dataset cf-data/full, 3 seeds
- 2026-11-06 04:48 EST — c3: [agent] C3: k-means (64 clusters) on the best contrastive z, cluster-mean value; dataset cf-data/full, 3 seeds
