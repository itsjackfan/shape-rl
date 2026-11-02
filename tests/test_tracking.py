import pytest

from shape_rl import tracking


@pytest.mark.parametrize(
    "key,expected",
    [
        ("loss", "train/loss"),
        ("reward", "train/reward"),
        ("reward_std", "train/reward_std"),
        ("mean_score", "train/mean_score"),
        ("math/mean_score", "train/math/mean_score"),
        ("rewards/correctness/mean", "train/rewards/correctness/mean"),
        ("learning_rate", "optim/lr"),
        ("grad_norm", "optim/grad_norm"),
        ("epoch", "progress/epoch"),
        ("eval_loss", "test/loss"),
        ("eval_reward", "test/reward"),
        ("eval_math/mean_score", "test/math/mean_score"),
        ("kl", "policy/kl"),
        ("entropy", "policy/entropy"),
        ("completions/mean_length", "policy/completions/mean_length"),
    ],
)
def test_rename(key, expected):
    assert tracking.rename(key) == expected


def test_logger_on_log(cfg_factory, tmp_path, monkeypatch):
    logged = []
    logger = tracking.WandbLogger(cfg_factory(), tmp_path)
    monkeypatch.setattr(logger, "log", lambda m: logged.append(m))

    class State:
        global_step = 2

    logger.on_log(None, State(), None, logs={"loss": 1.0, "learning_rate": 1e-6, "num_tokens": 100.0, "kl": 0.1})
    State.global_step = 4
    logger.on_log(None, State(), None, logs={"loss": 0.5, "num_tokens": 300.0})
    first, second = logged
    assert first["train/loss"] == 1.0 and first["optim/lr"] == 1e-6 and first["policy/kl"] == 0.1
    assert first["progress/step"] == 2
    assert "num_tokens" not in first and "policy/num_tokens" not in first
    assert second["perf/tokens_per_sec"] > 0 and second["perf/steps_per_sec"] > 0
    assert (tmp_path / "wandb_id").exists()
    logger.finish()


def test_eval_metrics():
    results = {
        "gsm8k_cot_zeroshot": {"exact_match,flexible-extract": 0.5, "exact_match_stderr,flexible-extract": 0.1},
        "aime24": {"exact_match,none": 0.1},
        "mbpp_instruct": {"pass_at_1,extract_code": 0.4},
    }
    benches = {
        "gsm8k": {"task": "gsm8k_cot_zeroshot", "suite": "math", "metric": "exact_match,flexible-extract"},
        "aime24": {"task": "aime24", "suite": "math", "metric": "exact_match,none"},
        "mbpp": {"task": "mbpp_instruct", "suite": "code", "metric": "pass_at_1,extract_code"},
    }
    m = tracking.eval_metrics(results, benches)
    assert m["test/math/gsm8k"] == 0.5
    assert m["test/math/mean_score"] == pytest.approx(0.3)
    assert m["test/code/mean_score"] == pytest.approx(0.4)
    assert m["test/mean_score"] == pytest.approx(0.35)
