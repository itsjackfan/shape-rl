"""Counterfactual rollouts (vLLM): oracle traces -> branch points -> continuation labels. Sharded, resumable.

Indexing. Response tokens r_0..r_{T-1} follow the prompt; `t` indexes the response, `pos = prompt_len + t` is absolute.
  s_t = prompt + r_<t is the state at branch point t; a_t = r_t is the sampled token;
  a'  = the n_alt most likely tokens != a_t under pi(.|s_t) (stop tokens excluded); actions = [a_t, a'_1, ..., a'_k].
  V(s_t)    <- M continuations sampled from s_t            (v_out).
  Q(s_t, a) <- M continuations sampled from s_t + a        (q_out, one block of M per action).
Outcome codes: 1 / 0 = finished response graded by its last \\boxed{} (no box = 0); -1 = truncated (unknown, never a failure).
Shapes: M = continuations per (state, action), A = 1 + n_alt actions, K = top-k logprobs kept per position.
"""
import hashlib
import time
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from hydra.utils import instantiate
from omegaconf import OmegaConf

from shape_rl import paths
from shape_rl.rewards import grade


def prompt_id(messages) -> str:
    return hashlib.sha1(messages[-1]["content"].encode()).hexdigest()[:16]


def split_of(pid: str, split) -> str:
    h = int(pid, 16) % 100
    return "train" if h < split[0] else "val" if h < split[0] + split[1] else "test"


def topk_entropy(logprobs_K: np.ndarray) -> float:
    """Entropy of pi(.|s) from its top-K logprobs, the leftover mass as one outcome (a lower bound)."""
    p_K = np.exp(logprobs_K)
    rest = max(1.0 - p_K.sum(), 0.0)
    return float(-(p_K * logprobs_K).sum() - (rest * np.log(rest) if rest > 0 else 0.0))


def branch_points(H_T: np.ndarray, rng, n_entropy: int, n_random: int, q: float):
    """t in [1, T-2] (skips the prompt-only state and the final stop token): n_entropy sampled from the
    top-(1-q) entropy positions, then n_random uniform over the rest. Returns (entropy_ts, random_ts)."""
    cand = np.arange(1, len(H_T) - 1)
    if len(cand) == 0:
        return np.array([], int), np.array([], int)
    hi = cand[H_T[cand] >= np.quantile(H_T[cand], q)]
    ent = np.sort(rng.choice(hi, min(n_entropy, len(hi)), replace=False))
    rest = np.setdiff1d(cand, ent)
    rnd = np.sort(rng.choice(rest, min(n_random, len(rest)), replace=False))
    return ent, rnd


def alternatives(top: dict, a_t: int, stop_ids, k: int) -> list[int]:
    """The k most likely tokens != a_t that are not stop tokens. top: {token_id: (logprob, rank)}."""
    ranked = sorted(top.items(), key=lambda kv: kv[1][1])
    return [tid for tid, _ in ranked if tid != a_t and tid not in stop_ids][:k]


def select_traces(rewards, max_correct: int, max_incorrect: int) -> list[int]:
    """Indices of up to max_correct correct and max_incorrect incorrect finished samples, in sample order."""
    c = [i for i, r in enumerate(rewards) if r == 1][:max_correct]
    w = [i for i, r in enumerate(rewards) if r == 0][:max_incorrect]
    return c + w


class Engine:
    """vLLM on token ids. sample -> per prompt, n (token_ids, finished); topk -> per sequence, per position {id: (logprob, rank)}.
    Sampling is pure pi (temperature only, no top-p/top-k); stops = tokenizer eos + generation_config eos ids."""

    def __init__(self, cfg, stops):
        from vllm import LLM

        R, self.stops = cfg.rollout, sorted(stops)
        self.llm = LLM(model=cfg.model.path, dtype="bfloat16", seed=cfg.seed, generation_config="vllm",
                       gpu_memory_utilization=R.gpu_memory_utilization, max_model_len=R.max_prompt_len + cfg.model.max_new_tokens + 1,
                       enable_prefix_caching=R.prefix_caching, max_num_seqs=R.max_num_seqs,
                       limit_mm_per_prompt={"image": 0, "video": 0})

    def sample(self, prompts, n, max_tokens, temperature):
        from vllm import SamplingParams, TokensPrompt

        sps = [SamplingParams(n=n, temperature=temperature, top_p=1.0, max_tokens=m, stop_token_ids=self.stops, detokenize=False)
               for m in max_tokens]
        outs = self.llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts], sps, use_tqdm=False)
        return [[(list(o.token_ids), o.finish_reason == "stop") for o in out.outputs] for out in outs]

    def topk(self, seqs, k):
        from vllm import SamplingParams, TokensPrompt

        outs = self.llm.generate([TokensPrompt(prompt_token_ids=s) for s in seqs], SamplingParams(max_tokens=1, prompt_logprobs=k, detokenize=False),
                                 use_tqdm=False)
        return [[None if d is None else {t: (lp.logprob, lp.rank) for t, lp in d.items()} for d in out.prompt_logprobs] for out in outs]


def load_prompts(cfg) -> pd.DataFrame:
    """Seeded shuffle of all train prompts, first num_prompts. Columns: prompt_id, source, suite, messages, answer, split."""
    rows = [{"prompt_id": prompt_id(x["prompt"]), "source": name, "suite": x["suite"], "messages": x["prompt"], "answer": x["answer"]}
            for name, c in cfg.data.train.items() for x in instantiate(c)]
    df = pd.DataFrame(rows).drop_duplicates("prompt_id").sample(frac=1.0, random_state=cfg.seed).head(cfg.rollout.num_prompts)
    df["split"] = [split_of(p, cfg.rollout.split) for p in df.prompt_id]
    return df.reset_index(drop=True)


def write(df: pd.DataFrame, f: Path):
    tmp = f.with_suffix(".tmp")
    df.to_parquet(tmp)
    tmp.rename(f)


def stop_ids(tok, cfg) -> set:
    from transformers import GenerationConfig

    try:
        eos = GenerationConfig.from_pretrained(cfg.model.path).eos_token_id
    except OSError:  # no generation_config.json
        eos = None
    return {tok.eos_token_id, *(eos if isinstance(eos, list) else [eos])} - {None}


def sample_prompts(eng, tok, df, cfg) -> pd.DataFrame:
    """Stage A: `samples` traces per prompt at the policy temperature, graded; pass_rate over finished traces."""
    R, cap = cfg.rollout, cfg.model.max_new_tokens
    kw = OmegaConf.to_container(cfg.model.get("chat_template_kwargs", {}))
    ids = [tok(tok.apply_chat_template(list(m), tokenize=False, add_generation_prompt=True, **kw), add_special_tokens=False).input_ids
           for m in df.messages]
    df = df.assign(prompt_ids=ids)[[len(i) <= R.max_prompt_len for i in ids]].reset_index(drop=True)
    outs = eng.sample(list(df.prompt_ids), R.samples, [cap] * len(df), R.temperature)
    df["sample_ids"] = [[o[0] for o in out] for out in outs]
    df["sample_rewards"] = [[grade(tok.decode(i, skip_special_tokens=True), fin, a) for i, fin in out] for out, a in zip(outs, df.answer)]
    df["pass_rate"] = [np.mean([r for r in rs if r >= 0]) if any(r >= 0 for r in rs) else np.nan for rs in df.sample_rewards]
    return df.drop(columns=["messages"])


def branch_chunk(eng, tok, chunk: pd.DataFrame, cfg, stops) -> pd.DataFrame:
    """Stages B + C for one chunk of kept prompts -> (one row per branch point, one row per branched trace)."""
    R, cap = cfg.rollout, cfg.model.max_new_tokens
    traces = [(p, k, p.sample_ids[k].tolist()) for p in chunk.itertuples()
              for k in select_traces(p.sample_rewards, R.max_correct, R.max_incorrect)]
    tops = eng.topk([p.prompt_ids.tolist() + r for p, _, r in traces], R.top_logprobs)
    rows, reqs = [], []
    for (p, k, r), top in zip(traces, tops):
        P = len(p.prompt_ids)
        top_T = top[P:P + len(r)]  # top_T[t]: pi(.|s_t), includes r_t
        H_T = np.array([topk_entropy(np.array([lp for lp, rank in d.values() if rank <= R.top_logprobs])) for d in top_T])
        rng = np.random.default_rng([cfg.seed, int(p.prompt_id, 16) % 2**32, k])
        for kind, ts in zip(("entropy", "random"), branch_points(H_T, rng, R.n_entropy, R.n_random, R.entropy_quantile)):
            for t in map(int, ts):  # numpy ints do not serialize into vLLM requests
                acts = [r[t]] + alternatives(top_T[t], r[t], stops, R.n_alt)
                if len(acts) < 1 + R.n_alt:
                    continue
                base = p.prompt_ids.tolist() + r[:t]
                reqs += [(base, cap - t)] + [(base + [a], cap - t - 1) for a in acts]
                rows.append({"prompt_id": p.prompt_id, "source": p.source, "suite": p.suite, "split": p.split, "answer": p.answer,
                             "pass_rate": p.pass_rate, "trace_id": f"{p.prompt_id}-{k}", "sample_idx": k, "prompt_len": P,
                             "trace_len": len(r), "reward": p.sample_rewards[k], "t": int(t), "pos": P + int(t), "kind": kind,
                             "entropy": H_T[t], "entropy_prev": H_T[t - 1], "logprob_prev": top_T[t - 1][r[t - 1]][0],
                             "actions": acts, "action_logprobs": [top_T[t][a][0] for a in acts], "trace_ids": p.prompt_ids.tolist() + r})
    outs = eng.sample([q for q, _ in reqs], R.M, [m for _, m in reqs], R.temperature)
    A = 1 + R.n_alt
    for i, row in enumerate(rows):
        P, t, ids = row["prompt_len"], row["t"], row["trace_ids"]
        heads = [ids[P:P + t]] + [ids[P:P + t] + [a] for a in row["actions"]]
        codes = [[grade(tok.decode(h + c, skip_special_tokens=True), fin, row["answer"]) for c, fin in outs[i * (1 + A) + j]]
                 for j, h in enumerate(heads)]
        row["v_out"], row["q_out"] = codes[0], sum(codes[1:], [])
        row["gen_tokens"] = sum(len(c) for j in range(1 + A) for c, _ in outs[i * (1 + A) + j])
    df = pd.DataFrame(rows)
    return df.drop(columns=["trace_ids"]), pd.DataFrame({"trace_id": [f"{p.prompt_id}-{k}" for p, k, _ in traces],
                                                       "token_ids": [p.prompt_ids.tolist() + r for p, _, r in traces],
                                                       "prompt_len": [len(p.prompt_ids) for p, _, _ in traces]})


def check_config(out: Path, cfg):
    """A dataset dir holds one config: shards and restarts must match it (outputs are never overwritten)."""
    c = OmegaConf.to_container(cfg, resolve=True)
    c.pop("shard")
    f = out / "config.yaml"
    if f.exists():
        if OmegaConf.to_container(OmegaConf.load(f)) != c:
            raise FileExistsError(f"{out} holds a different config; pick a new ablation name.")
    else:
        OmegaConf.save(c, f)


def run(cfg, engine=None):
    paths.check_logged(cfg.log_path, cfg.exp, cfg.ablation)
    from transformers import AutoTokenizer

    out = paths.run_dir(cfg)
    (out / "rollout").mkdir(parents=True, exist_ok=True)
    check_config(out, cfg)
    tok, R = AutoTokenizer.from_pretrained(cfg.model.path), cfg.rollout
    stops, eng = stop_ids(tok, cfg), engine
    pf = out / "rollout" / f"s{cfg.shard:03d}.prompts.parquet"
    if not pf.exists():
        eng = eng or Engine(cfg, stops)
        t0 = time.time()
        write(sample_prompts(eng, tok, load_prompts(cfg).iloc[cfg.shard::cfg.num_shards], cfg), pf)
        print(f"[rollout] shard {cfg.shard}: stage A {time.time() - t0:.0f}s", flush=True)
    prompts = pd.read_parquet(pf)
    kept = prompts[(prompts.pass_rate > 0) & (prompts.pass_rate < 1)].reset_index(drop=True)
    n_chunks = -(-len(kept) // R.chunk_prompts)
    for c in range(n_chunks):
        f = out / "rollout" / f"s{cfg.shard:03d}_c{c:03d}.parquet"
        if f.exists():
            continue
        eng = eng or Engine(cfg, stops)
        t0 = time.time()
        rows, traces = branch_chunk(eng, tok, kept.iloc[c * R.chunk_prompts:(c + 1) * R.chunk_prompts], cfg, stops)
        write(traces, f.with_suffix(".traces.parquet"))
        write(rows, f)
        dt = time.time() - t0
        print(f"[rollout] shard {cfg.shard} chunk {c + 1}/{n_chunks}: {len(rows)} branch points, {rows.gen_tokens.sum()} tokens, "
              f"{dt:.0f}s, {rows.gen_tokens.sum() / dt:.0f} tok/s", flush=True)
    (out / "rollout" / f"s{cfg.shard:03d}.done").touch()


@hydra.main(config_path="../../configs", config_name="rollout", version_base=None)
def main(cfg):
    run(cfg)


if __name__ == "__main__":
    main()
