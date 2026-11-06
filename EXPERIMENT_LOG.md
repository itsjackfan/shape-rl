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