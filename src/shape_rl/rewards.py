from math_verify import parse, verify


def correctness(completions, answer, **kwargs) -> list[float]:
    return [float(verify(parse(a if "\\boxed" in a else f"${a}$"), parse(c[-1]["content"]))) for c, a in zip(completions, answer)]
