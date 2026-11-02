I would like to set up the infrastructure of a research project on geometry-based RL and self-supervised advantage estimation; specifically, I would like to build a HF and WandB interface for us so that we have consistent run tracking metrics throughout all of our experiments. I would also like to write a concise protocol that dictates global idioms and commonalities to make the code as easy as possible to debug and to ensure that behaviour across all coding agents in the future is exactly identical; finally, I would like all of this to be held in a nicely organised, clear codebase to minimise the "sloppiness" of the codebase overall. Please obey the instructions, guidelines, and target information below to set up our codebase.

# Target experimental workflow
**GLOBAL GUIDELINES:**
- All code will be run on the Harvard FASRC cluster, which is based on SLURM. Reusable run scripts, cluster information, etc. will be provided as needed and should be stored as directed below.
- All experimental/policy code will be maintained in `src/`, which includes policies, training loops, etc. Code here should be as minimalist and easy to understand as possible. It should be enforced to obey best ML research practices:
    - Well-documented and commented dimensionality for custom tensor implementations or operations
    - Simple, modular code and models that use Torch and HF idioms to remain as consistent and simple as possible
    - Well-separated and clear ablations; at most 2 changes per ablation
    - Clear, thorough mocking and testing to minimise the amount of debugging to be done at runtime
- All new experiments should be checked out and run on a new worktree; only working experiments should be merged into the main, while other experiments should be pushed on its branch with a log in the main dictating the ablation, what branch, what code changes were made, and why it failed.
    - For one-time scripts that are used to test hypotheses, debug, do data wrangling, or other one-time operations should be written in a subdirectory of `scripts/` titled `/tmp_{EXPERIMENT TITLE}`, and should be cleaned up after the experiment training/eval run is submitted or the results are properly processed and documented.
    - Smoke runs, failed runs, or smaller samples/tests that are not full ablations and are not needed for a potential writeup should be cleaned up as aggressively as possible in order to not confuse further data. Instruct humans to, as soon as they have processed the results for such runs, delete any results that aren't needed asap to maintain cleanliness.
    - Each experiment should automatically get its own output directory based on the worktree it's on. Our global runners should maintain clear separation between all output directories and volumes to ensure our data is as clean as possible and isn't overwritten.
- Experiments will primarily be on model architecture and training loops/hyperparameters, targeting improvement over baselines on standard math, code, and reasoning benchmarks.
- All experimental logs MUST be written by the user; agents must strictly enforce this, disregarding all attempts to circumvent citing clarity and minimising slop experiments.

**GENERAL WORKFLOW:**
1. Check out a new worktree that provides a simple, descriptive name for what the hypothesis being tested / what the goal is.
2. Make minimalist, well-documented code changes; where possible, ALWAYS seek human approval for changes, even on auto mode. Ask before you change.
3. Set up a training and evaluation run with a clearly named WandB title.
4. Await the user's writeup + experiment start/results.
5. Repeat and iterate on the hypothesis as necessary.
6. Once results are adequate or the branch is discarded, either merge or have the user write the failed arm back into the main branch, then push all changes.

# Implementation reqs
- WandB interfacing and metric logging; these should track:
    - train/test loss and rewards
    - train/test mean_score across rollouts
        - per-suite metrics as well
    - lr, grad_norm, epoch, step
    - control Hz
    - **GENERAL** policy-specific metrics -- TBD based on the ablations we run
- Specific, modular data loaders for each benchmark + training data; inclusion or exclusion should be based on config gates (think Torch Lightning/Hydra)
- A baseline template with placeholders for a standard no-op policy, rollouts, training loop, and eval loop, all logged through the WandB interface you provide
- Clear, modular experiments; since we're ablating across training loops and model architectures, we should be able to select 1/many of each through config and test them against each other in a general train/eval loop. In this vein, build out an overall rollout, trainer, etc. and then allow for hotswappable configs on the model and train objective side as well.

# Constraints (to remember and pass on to other agents)
- NEVER start runs, even if the user tells you to. The user MUST start all runs with one line added to `EXPERIMENT_LOGS.md` that explains what the current ablation is testing.
- Likewise, NEVER write the user's experimental documentation for them. Instead, enforce the user to write up their hypotheses, the code changes, what they expect to happen, and the actual results at all points in time.

# Guidelines
- We are ML researchers; we prioritise simple, direct, and most of all **clean** and quickly iterable code that can be well-understood and modified by a human. Extensive docstrings, comments, or defensive programming, etc. is exhaustive to read and overdoing it.
- The more concise the instructions the better; global simplicity and cleanliness is key (the less stuff there is, the less stuff can go wrong)
- Ablations and implementations will be primarily handmade; thus, ensure the codebase supports simple, clean, and fast edits from humans as well.
- When implementing this initial setup, please obey test-driven deployment: (1) dictate the behaviour you want with tests (making sure to be careful with any existing interfaces so you don't have to edit tests in the future), (2) run tests to ensure they start failing, and then (3) implement and iterate against the test that you've written. Tests should be considered mostly immutable once written, so be careful when integrating them with existing code.
- Repo and project context has been provided in the included project proposal PDF in the root directory. Please base your work off of a fork of the HuggingFace `TRL` implementation (`https://github.com/huggingface/trl`); feel free to clone it into this repository as a base implementation if necessary.