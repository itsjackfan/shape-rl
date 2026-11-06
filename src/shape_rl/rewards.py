from functools import lru_cache

from math_verify import parse, verify


def correctness(completions, answer, **kwargs) -> list[float]:
    return [float(verify(parse(a if "\\boxed" in a else f"${a}$"), parse(c[-1]["content"]))) for c, a in zip(completions, answer)]


def last_boxed(text: str) -> str | None:
    """Content of the last \\boxed{...} (brace-matched), or None."""
    i = text.rfind("\\boxed")
    j = text.find("{", i)
    if i < 0 or j < 0:
        return None
    depth = 0
    for k in range(j, len(text)):
        depth += {"{": 1, "}": -1}.get(text[k], 0)
        if depth == 0:
            return text[j + 1:k]
    return None


@lru_cache(maxsize=1 << 16)
def _match(gold: str, pred: str) -> bool:
    return verify(parse(f"${gold}$"), parse(f"${pred}$"))


def grade(text: str, finished: bool, gold: str) -> int:
    """1/0 for a finished response by its last \\boxed{} answer (no box = 0); -1 = truncated (outcome unknown)."""
    if not finished:
        return -1
    pred = last_boxed(text)
    return int(pred is not None and _match(gold, pred))
