"""Train loaders: one function per dataset -> Dataset with columns {prompt (chat messages), answer (str), suite (str)}."""
import datasets
from hydra.utils import instantiate

from shape_rl.rewards import last_boxed

MATH_PROMPT = "{q}\nPlease reason step by step, and put your final answer within \\boxed{{}}."


def _chat(q):
    return [{"role": "user", "content": MATH_PROMPT.format(q=q)}]


def gsm8k(split="train"):
    ds = datasets.load_dataset("openai/gsm8k", "main", split=split)
    return ds.map(lambda x: {"prompt": _chat(x["question"]), "answer": x["answer"].split("####")[-1].strip(), "suite": "math"},
                  remove_columns=ds.column_names)


def math(split="train"):
    """Gold = last \\boxed{} of the solution; rows without one are dropped."""
    ds = datasets.load_dataset("DigitalLearningGmbH/MATH-lighteval", split=split)
    ds = ds.map(lambda x: {"prompt": _chat(x["problem"]), "answer": last_boxed(x["solution"]), "suite": "math"},
                remove_columns=ds.column_names)
    return ds.filter(lambda x: x["answer"] is not None)


def load_train(train_cfg) -> datasets.Dataset:
    return datasets.concatenate_datasets([instantiate(c) for c in train_cfg.values()])
