"""ShapeTrainer = TRL GRPOTrainer + swappable advantage estimator (with hidden-state capture), DAPO dynamic sampling,
FLOP accounting with a FLOP budget, and per-suite mean_score.

Shapes: N = completions in the global batch (prompt-major, process p holds rows p*n .. p*n+n-1), n = this process's rows,
G = generations per prompt, F = reward functions, P = padded prompt width, T = completion width, D = hidden size.
"""
import json
import os

import torch
from accelerate.utils import gather_object
from transformers import TrainerCallback
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from trl import GRPOTrainer
from trl.models.utils import disable_gradient_checkpointing

from shape_rl.flops import FlopModel
from shape_rl.hidden import final_norm

PROMPT_KEYS = {"prompt_ids": "pad", "prompt_mask": 0}  # left-padded per generation round
ROW_KEYS = {"advantages"}  # one value per row


class FlopBudget(TrainerCallback):
    """Stop at budget.max_flops; save a checkpoint each time the count crosses a multiple of budget.save_every_flops."""

    def __init__(self, trainer, max_flops=None, save_every_flops=None):
        self.t, self.max, self.every = trainer, max_flops, save_every_flops

    def on_step_end(self, args, state, control, **kwargs):
        done = self.t.flops_total
        if self.every and done // self.every > self.t.flops_saved_mark:
            self.t.flops_saved_mark = done // self.every
            control.should_save = True
        if self.max and done >= self.max:
            control.should_training_stop = control.should_save = True
        return control


class ShapeTrainer(GRPOTrainer):
    def __init__(self, *args, estimator=None, dynamic_sampling=None, budget=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.estimator, self.dynamic_sampling = estimator, dynamic_sampling
        assert not (estimator is not None and dynamic_sampling), "dynamic sampling is a DAPO (group-relative) option"
        if estimator is not None:
            estimator.gather = self.accelerator.gather
        self.flop_model = FlopModel(self.accelerator.unwrap_model(self.model).config._name_or_path)
        self.flops_total, self.flops_saved_mark, self._flops_local = 0.0, 0, 0.0
        self._capture, self._scoring = None, False
        budget = budget or {}
        self.add_callback(FlopBudget(self, budget.get("max_flops"), budget.get("save_every_flops")))

    # ---- FLOPs: scoring forwards (old logps, reference) are counted where they run; generation and training per batch

    def _get_per_token_logps_and_entropies(self, model, input_ids, attention_mask, logits_to_keep, batch_size=None, **kw):
        if not self._scoring:  # the training forward: counted in _count_batch
            return super()._get_per_token_logps_and_entropies(model, input_ids, attention_mask, logits_to_keep, batch_size, **kw)
        self._flops_local += self.flop_model.forward(attention_mask.sum(1))
        if self._capture is None or model is not self.model:
            return super()._get_per_token_logps_and_entropies(model, input_ids, attention_mask, logits_to_keep, batch_size, **kw)
        # hidden-state capture: each sub-batch forward hits the hook once, rows in order
        P, rows = input_ids.shape[1] - logits_to_keep, [0]

        def hook(module, args, out):
            b = out.shape[0]
            r = slice(rows[0], rows[0] + b)
            self._capture.append(self.estimator.observe(out[:, P - 1:P - 1 + logits_to_keep].detach(),
                                                        attention_mask[r, P:], input_ids[r, P:]))
            rows[0] += b

        handle = final_norm(self.accelerator.unwrap_model(self.model)).register_forward_hook(hook)
        try:
            res = super()._get_per_token_logps_and_entropies(model, input_ids, attention_mask, logits_to_keep, batch_size, **kw)
        finally:
            handle.remove()
        assert rows[0] == len(input_ids), (rows[0], len(input_ids))
        return res

    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        self._rewards_NF = super()._calculate_rewards(inputs, prompts, completions, completion_ids_list)  # gathered
        self._gen_lengths_n = torch.tensor([len(c) for c in completion_ids_list], device=self.accelerator.device)
        return self._rewards_NF

    def _score_round(self, inputs):
        """One TRL generation + scoring round, with state capture and FLOP counts. -> (output, rewards_N, correct_N)."""
        needs = self.estimator is not None and self.estimator.needs_states
        self._capture, self._scoring = ([] if needs else None), True
        try:
            out = super()._generate_and_score_completions(inputs)
            if needs and not self._capture:  # no scoring forward ran (no vLLM IS correction): run one for the states
                ids = torch.cat([out["prompt_ids"], out["completion_ids"]], 1)
                mask = torch.cat([out["prompt_mask"], out["completion_mask"]], 1)
                with torch.no_grad(), disable_gradient_checkpointing(self.model, self.args.gradient_checkpointing_kwargs):
                    self._get_per_token_logps_and_entropies(self.model, ids, mask, out["completion_ids"].shape[1],
                                                            batch_size=self.args.per_device_train_batch_size)
        finally:
            self._scoring = False
        prompt_len = out["prompt_mask"].sum(1)
        self._flops_local += self.flop_model.forward(prompt_len + self._gen_lengths_n)  # generation (prefill + decode)
        rewards_NF = self._rewards_NF
        rewards_N = (rewards_NF * self.reward_weights.to(rewards_NF.device)).nansum(dim=1)
        rewards_N[torch.isnan(rewards_NF).all(dim=1)] = torch.nan  # TRL's unscorable rule
        correct_N = rewards_NF[:, self.reward_func_names.index("correctness")]
        return out, rewards_N, correct_N

    def _count_batch(self, out):
        """Training FLOPs of the final batch: forward + backward per pass over its real tokens; reduce the step's count."""
        L = out["prompt_mask"].sum(1) + out["completion_mask"].sum(1)
        self._flops_local += 3 * self.num_iterations * self.flop_model.forward(L)
        if self.estimator is not None:
            self._flops_local += self.estimator.flops
            self.estimator.flops = 0.0
        step = torch.tensor(self._flops_local, dtype=torch.float64, device=self.accelerator.device)
        self.flops_total += float(self.accelerator.reduce(step, reduction="sum"))
        self._flops_local = 0.0

    # ---- one generation batch

    def _generate_and_score_completions(self, inputs):
        mode = "train" if self.model.training else "eval"
        out, rewards_N, correct_N = self._score_round(inputs)
        if self.dynamic_sampling and mode == "train":
            out, correct_N = self._dynamic_sample(inputs, out, rewards_N, correct_N)
        suites = gather_object([x["suite"] for x in inputs])
        if len(suites) == len(correct_N):  # per-suite scores of the scored batch (first round under dynamic sampling)
            self._metrics[mode]["mean_score"].append(correct_N.nanmean().item())
            for s in set(suites):
                self._metrics[mode][f"{s}/mean_score"].append(correct_N[[x == s for x in suites]].nanmean().item())

        if self.estimator is not None:
            G = self.num_generations if mode == "train" else self.num_generations_eval
            n = len(out["advantages"])
            local = slice(self.accelerator.process_index * n, (self.accelerator.process_index + 1) * n)
            obs = {k: torch.cat([c[k] for c in self._capture]) for k in self._capture[0]} if self._capture else {}
            adv = self.estimator(rewards_N, G, local, obs)
            assert adv.shape[0] == n and (adv.dim() == 1 or adv.shape[1] == out["completion_ids"].shape[1]), adv.shape
            out["advantages"] = adv
            self._capture = None
            # the completions table shows the advantages used (TRL logged its own): per-row mean over real tokens
            m = out["completion_mask"].float()
            row = adv if adv.dim() == 1 else (adv * m).sum(1) / m.sum(1).clamp(min=1)
            log = self._logs["advantages"]  # a deque (maxlen = generation batch): drop TRL's N values, add the used ones
            for _ in range(min(len(rewards_N), len(log))):
                log.pop()
            log.extend(self.accelerator.gather(row).tolist())
            for k, v in self.estimator.metrics.items():
                self._metrics[mode][f"estimator/{k}"].append(
                    self.accelerator.gather(torch.tensor([float(v)], device=self.accelerator.device)).mean().item())
        if mode == "train":
            self._count_batch(out)
            self._metrics[mode]["flops"] = [self.flops_total]
        return out

    # ---- DAPO dynamic sampling

    def _fresh_inputs(self, inputs, round_):
        """New prompts for a refill round: the same global draw on every process (seeded by seed, step and round), laid out
        as TRL's RepeatSampler does (each prompt G times in a row, process p takes rows p*n .. p*n+n-1)."""
        G, n, p = self.num_generations, len(inputs), self.accelerator.process_index
        n_prompts = n * self.accelerator.num_processes // G
        gen = torch.Generator().manual_seed(self.args.seed * 1_000_003 + self.state.global_step * 1009 + round_)
        idx = torch.randint(len(self.train_dataset), (n_prompts,), generator=gen).repeat_interleave(G)
        return [self.train_dataset[int(i)] for i in idx[p * n:(p + 1) * n]]

    def _dynamic_sample(self, inputs, out, rewards_N, correct_N):
        """Keep groups whose correctness is not all equal (0 < accuracy < 1); refill from fresh prompts for up to
        max_rounds generation rounds; the batch = the first N/G kept groups in round order, then the first degenerate
        groups if too few were kept. Rows move between processes through a gather of every round."""
        G = self.num_generations
        n_groups = len(rewards_N) // G
        rounds = [(out, correct_N)]

        def kept(c):
            c = c.view(-1, G)
            return ~torch.isclose(c.max(1).values, c.min(1).values)

        while sum(int(kept(c).sum()) for _, c in rounds) < n_groups and len(rounds) < self.dynamic_sampling["max_rounds"]:
            o, _, c = self._score_round(self._fresh_inputs(inputs, len(rounds)))
            rounds.append((o, c))
        pick = [(r, g) for r, (_, c) in enumerate(rounds) for g in kept(c).nonzero()[:, 0].tolist()]
        pick += [(r, g) for r, (_, c) in enumerate(rounds) for g in (~kept(c)).nonzero()[:, 0].tolist()]
        pick = pick[:n_groups]
        rows = [(r, g * G + j) for r, g in pick for j in range(G)]
        full = self._gather_rounds([o for o, _ in rounds])
        n, p = len(out["advantages"]), self.accelerator.process_index
        mine = rows[p * n:(p + 1) * n]
        new = {}
        for k, v in full[0].items():
            if not torch.is_tensor(v) or v.dim() == 0:
                continue
            new[k] = torch.stack([full[r][k][i] for r, i in mine])
        new["num_items_in_batch"] = self.accelerator.gather(new["completion_mask"].sum()).sum()
        self._metrics["train"]["dynamic_sampling/rounds"].append(float(len(rounds)))
        self._metrics["train"]["dynamic_sampling/kept_frac"].append(
            sum(1 for r, g in pick if kept(rounds[r][1])[g]) / n_groups)
        correct = torch.stack([rounds[r][1][i] for r, i in rows])
        return new, correct

    def _gather_rounds(self, outs):
        """Every round's per-row tensors from all processes, padded to common widths (prompt keys on the left,
        completion keys on the right): [{key: (N, ...)}] per round."""
        pad_id = self._tokenizer.pad_token_id
        P = max(int(o["prompt_ids"].shape[1]) for o in outs)
        T = max(int(o["completion_ids"].shape[1]) for o in outs)
        P = int(self.accelerator.gather(torch.tensor([P], device=self.accelerator.device)).max())
        T = int(self.accelerator.gather(torch.tensor([T], device=self.accelerator.device)).max())
        res = []
        for o in outs:
            g = {}
            for k, v in o.items():
                if not torch.is_tensor(v) or v.dim() == 0:
                    continue
                if k in PROMPT_KEYS:
                    fill = pad_id if PROMPT_KEYS[k] == "pad" else 0
                    v = torch.nn.functional.pad(v, (P - v.shape[1], 0), value=fill)
                elif k not in ROW_KEYS and v.dim() >= 2 and v.shape[1] == o["completion_ids"].shape[1] and v.shape[1] > 1:
                    fill = pad_id if k == "completion_ids" else 0
                    v = torch.nn.functional.pad(v, (0, 0) * (v.dim() - 2) + (0, T - v.shape[1]), value=fill)
                g[k] = self.accelerator.gather(v.contiguous())
            res.append(g)
        return res

    # ---- requeue: FLOP count and estimator state travel with each checkpoint

    def _save_checkpoint(self, model, trial):
        super()._save_checkpoint(model, trial)
        if self.args.should_save:
            d = os.path.join(self._get_output_dir(trial=trial), f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}")
            state = {"flops_total": self.flops_total, "flops_saved_mark": self.flops_saved_mark,
                     "estimator": None if self.estimator is None else self.estimator.extra_state()}
            torch.save(state, os.path.join(d, "shape_state.pt"))
            with open(os.path.join(d, "flops.json"), "w") as f:
                json.dump({"flops": self.flops_total, "num_tokens": self.state.num_input_tokens_seen,
                           "step": self.state.global_step}, f)

    def _load_optimizer_and_scheduler(self, checkpoint):
        super()._load_optimizer_and_scheduler(checkpoint)
        if checkpoint is not None and os.path.exists(f := os.path.join(checkpoint, "shape_state.pt")):
            state = torch.load(f, map_location="cpu", weights_only=False)
            self.flops_total, self.flops_saved_mark = state["flops_total"], state["flops_saved_mark"]
            if self.estimator is not None and state["estimator"] is not None:
                self.estimator.load_extra_state(state["estimator"])
