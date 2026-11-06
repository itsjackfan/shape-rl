import datasets
import pytest

from shape_rl import evaluate, train


def test_unlogged_run_refused(cfg_factory):
    with pytest.raises(RuntimeError):
        train.run(cfg_factory(logged=False))
    with pytest.raises(RuntimeError):
        evaluate.run(cfg_factory("eval", logged=False))


def test_evaluate_mocked(cfg_factory, monkeypatch):
    import lm_eval

    calls = {}

    def fake_simple_evaluate(**kw):
        calls.update(kw)
        return {"results": {t: {"exact_match,flexible-extract": 1.0, "exact_match,none": 1.0, "math_verify,none": 1.0,
                                "pass@1,create_test": 1.0, "pass_at_1,extract_code": 1.0, "exact_match,remove_whitespace": 1.0}
                            for t in kw["tasks"]}}

    monkeypatch.setattr(lm_eval, "simple_evaluate", fake_simple_evaluate)
    cfg = cfg_factory("eval", overrides=["model=tiny"])
    metrics = evaluate.run(cfg)
    assert metrics["test/mean_score"] == 1.0
    assert "pretrained=trl-internal-testing" in calls["model_args"]
    assert (evaluate.eval_dir(cfg) / "results.json").exists()


SMOKE = {"noop": [], "dapo": [], "hcluster": [],
         "hvalue": ["objective.estimator.init=null", "+objective.estimator.kind=mlp", "+objective.estimator.hidden=8",
                    "objective.estimator.train=all"]}


@pytest.mark.slow
@pytest.mark.parametrize("objective", list(SMOKE))
def test_train_smoke(cfg_factory, monkeypatch, objective):
    rows = {"prompt": [[{"role": "user", "content": f"{i}+1?"}] for i in range(4)], "answer": ["1"] * 4, "suite": ["math"] * 4}
    monkeypatch.setattr("shape_rl.data.load_train", lambda _: datasets.Dataset.from_dict(rows))
    cfg = cfg_factory(overrides=["model=tiny", f"objective={objective}", "train.max_steps=1", "train.per_device_train_batch_size=4",
                                 "train.gradient_accumulation_steps=1", "train.num_generations=2", "train.bf16=false",
                                 "train.gradient_checkpointing=false", *SMOKE[objective]])
    trainer = train.run(cfg)
    hist = trainer.state.log_history
    assert trainer.state.global_step == 1 and trainer.flops_total > 0
    assert any("math/mean_score" in h for h in hist) and any("flops" in h for h in hist)
