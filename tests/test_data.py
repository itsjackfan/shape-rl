import datasets
import pytest
from hydra.utils import instantiate

from shape_rl import data

RAW = {
    "openai/gsm8k": {"question": ["1+1?", "2+2?"], "answer": ["so #### 2", "so #### 4"]},
    "DigitalLearningGmbH/MATH-lighteval": {"problem": ["x?"], "solution": ["thus $\\boxed{\\frac{1}{2}}$."]},
}


@pytest.fixture(autouse=True)
def fake_hub(monkeypatch):
    monkeypatch.setattr(datasets, "load_dataset", lambda path, *a, **k: datasets.Dataset.from_dict(RAW[path]))


@pytest.mark.parametrize("name,answer", [("gsm8k", "2"), ("math", "\\frac{1}{2}")])
def test_loader_schema(cfg_factory, name, answer):
    ds = instantiate(cfg_factory(overrides=[f"data/train=[{name}]"]).data.train[name])
    assert set(ds.column_names) == {"prompt", "answer", "suite"}
    row = ds[0]
    assert row["prompt"][-1]["role"] == "user"
    assert row["answer"] == answer and row["suite"] == "math"


def test_load_train_gates(cfg_factory):
    assert len(data.load_train(cfg_factory(overrides=["data/train=[gsm8k,math]"]).data.train)) == 3
    assert len(data.load_train(cfg_factory(overrides=["data/train=[gsm8k]"]).data.train)) == 2
