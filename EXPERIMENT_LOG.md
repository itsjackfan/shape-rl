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
