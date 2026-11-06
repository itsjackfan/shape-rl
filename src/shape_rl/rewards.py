"""Graders. `grade` outcome codes: 1 / 0 = finished response graded by its last \\boxed answer (no box = 0); -1 = truncated
(outcome unknown). The RL reward (`Correctness`) and the offline labels (rollout.py) both use `grade`."""
import re
from functools import lru_cache

from math_verify import parse, verify


def last_boxed(text: str) -> str | None:
    """Argument of the last \\boxed: the brace-matched {...} group, or for `\\boxed 5` (no brace) the next run of characters
    up to white space or `$`, without trailing `.,;`. None if there is no \\boxed, the group is not closed or `\\boxed`
    is part of a longer macro name."""
    i = text.rfind("\\boxed")
    if i < 0:
        return None
    j = i + len("\\boxed")
    if text[j:j + 1].isalpha():
        return None
    k = j + len(text[j:]) - len(text[j:].lstrip())
    if text[k:k + 1] == "{":
        depth = 0
        for e in range(k, len(text)):
            depth += {"{": 1, "}": -1}.get(text[e], 0)
            if depth == 0:
                return text[k + 1:e]
        return None
    if k == j:  # `\boxed` followed by a non-space, non-brace character (for example `\boxed}`): no argument
        return None
    arg = re.match(r"[^\s$]*", text[k:]).group(0).rstrip(".,;")
    return arg or None


@lru_cache(maxsize=1 << 16)
def _match(gold: str, pred: str) -> bool:
    return verify(parse(f"${gold}$"), parse(f"${pred}$"))


def grade(text: str, finished: bool, gold: str) -> int:
    """1/0 for a finished response by its last \\boxed answer (no box = 0); -1 = truncated (outcome unknown)."""
    if not finished:
        return -1
    pred = last_boxed(text)
    return int(pred is not None and _match(gold, pred))


def stop_ids(tok, model_path) -> set:
    """Token ids that end a response, as TRL and vLLM use them: the tokenizer's eos + the eos ids of generation_config.json,
    or of the model config when that file is missing (Qwen3.5-4B: <|im_end|> + <|endoftext|>)."""
    from transformers import AutoConfig, GenerationConfig

    try:
        eos = GenerationConfig.from_pretrained(model_path).eos_token_id
    except OSError:  # no generation_config.json: transformers and vLLM fall back to the model config
        eos = AutoConfig.from_pretrained(model_path).get_text_config().eos_token_id
    return {tok.eos_token_id, *(eos if isinstance(eos, list) else [eos])} - {None}


class Correctness:
    """RL reward: 1 if `grade` = 1, else 0 (wrong, no box, or truncated = the completion does not end with a stop id).
    TRL passes the decoded completion (special tokens skipped) and its token ids."""

    __name__ = "correctness"

    def __init__(self, model_path: str):
        from transformers import AutoTokenizer

        self.stops = stop_ids(AutoTokenizer.from_pretrained(model_path), model_path)

    def __call__(self, completions, answer, completion_ids, **kwargs) -> list[float]:
        return [float(grade(c[-1]["content"], len(ids) > 0 and ids[-1] in self.stops, a) == 1)
                for c, a, ids in zip(completions, answer, completion_ids)]
