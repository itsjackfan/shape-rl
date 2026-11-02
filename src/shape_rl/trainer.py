"""N = num completions in the global batch, G = generations per prompt."""
import torch
from accelerate.utils import gather_object
from trl import GRPOTrainer


class ShapeTrainer(GRPOTrainer):
    """GRPOTrainer + per-suite mean_score logging + optional swappable AdvantageEstimator."""

    def __init__(self, *args, estimator=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.estimator = estimator

    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        self._rewards_NF = super()._calculate_rewards(inputs, prompts, completions, completion_ids_list)  # gathered
        return self._rewards_NF

    def _generate_and_score_completions(self, inputs):
        out = super()._generate_and_score_completions(inputs)
        mode = "train" if self.model.training else "eval"
        rewards_NF = self._rewards_NF
        score_N = rewards_NF[:, self.reward_func_names.index("correctness")]
        suites = gather_object([x["suite"] for x in inputs])
        self._metrics[mode]["mean_score"].append(score_N.nanmean().item())
        for s in set(suites):
            self._metrics[mode][f"{s}/mean_score"].append(score_N[[x == s for x in suites]].nanmean().item())

        if self.estimator is not None:
            G = self.num_generations if mode == "train" else self.num_generations_eval
            rewards_N = (rewards_NF * self.reward_weights.to(rewards_NF.device)).nansum(dim=1)
            adv_N = self.estimator(rewards_N, G)
            i, n = self.accelerator.process_index, len(out["advantages"])
            out["advantages"] = adv_N[i * n:(i + 1) * n]
        return out
