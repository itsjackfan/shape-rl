"""Step 1 checks that need the venv, the model files and the train data (CPU only)."""
import inspect, re, sys
from pathlib import Path

import datasets
import transformers
import trl
from math_verify import parse
from transformers import AutoConfig, AutoTokenizer

M = "Qwen/Qwen3.5-4B"
tok = AutoTokenizer.from_pretrained(M)
msgs = [{"role": "user", "content": "What is 1+1?"}]
print("== item 1: chat template tail")
print("default        :", repr(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)[-80:]))
print("thinking=False :", repr(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)[-80:]))
print("thinking=True  :", repr(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True)[-80:]))
print("== item 2: tokenizer response schema/template attrs:",
      {a: getattr(tok, a, None) is not None for a in ["response_schema", "response_template"]})
print("eos:", tok.eos_token, tok.eos_token_id, "pad:", tok.pad_token, tok.pad_token_id)
cfg = AutoConfig.from_pretrained(M)
print("== model config:", cfg.architectures, type(cfg).__name__)
tc = getattr(cfg, "text_config", cfg)
print({k: getattr(tc, k, None) for k in ["hidden_size", "num_hidden_layers", "vocab_size", "layer_types", "max_position_embeddings", "tie_word_embeddings"]})
print("generation_config.json in repo:", transformers.utils.hub.cached_file(M, "generation_config.json", _raise_exceptions_for_missing_entries=False))

src = Path(trl.__file__).parent / "trainer" / "grpo_trainer.py"
txt = src.read_text().splitlines()
print("== TRL", trl.__version__, src)
for pat in ["add_response_schema", "chat_template_kwargs", "response_schema", 'self._logs["advantages"]', "advantages.dim()", "unsqueeze(1)", "batch_decode", "parse_response"]:
    hits = [f"{i+1}: {l.strip()}" for i, l in enumerate(txt) if pat in l]
    print(f"-- {pat!r}: {len(hits)} hits"); print("\n".join(hits[:8]))
try:
    import lm_eval.models.vllm_causallms as vm
    s = Path(vm.__file__).read_text().splitlines()
    print("== item 3 lm_eval vllm max_gen_toks:", [f"{i+1}: {l.strip()}" for i, l in enumerate(s) if "max_gen_toks" in l][:6])
except Exception as e:
    print("lm_eval vllm import failed:", repr(e))

print("== item 6: MATH-lighteval parse failures")
ds = datasets.load_dataset("DigitalLearningGmbH/MATH-lighteval", split="train")
bad = [i for i, s in enumerate(ds["solution"]) if len(parse(s)) == 0]
print(f"{len(bad)} / {len(ds)} solutions give parse(...) == [] -> IndexError; first ids {bad[:5]}")
if bad:
    print("example:", repr(ds[bad[0]]["solution"][-200:]))
print("gsm8k train rows:", len(datasets.load_dataset("openai/gsm8k", "main", split="train")))

print("== item 2 repro: correctness() on a truncated, unboxed trace whose last number equals the gold")
from shape_rl.rewards import correctness
trunc = [[{"role": "assistant", "content": "<think>\nLet me compute. 3 apples plus 4 apples gives 7. Hmm, wait, maybe I should check whether the answer is 7 or"}]]
print("truncated unboxed, gold 7 ->", correctness(trunc, answer=["7"]))
