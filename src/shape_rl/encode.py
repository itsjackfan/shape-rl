"""Hidden states at branch points -> one safetensors file per rollout chunk, plus one parquet index. Sharded, resumable.

For a branch row (pos = prompt_len + t, see rollout.py):
  h_v[i]    = state at pos - 1 of the trace = h(s_t): it predicts pi(.|s_t), so it is the input for V(s_t).
  h_q[i, j] = state at pos with input token actions[j] = h(s_t, a) (j = 0: sampled a_t, j >= 1: a'_j): input for Q(s_t, a).
States are post-norm last-layer, bf16 (hidden.states_at). N = branch rows, A = actions, D = hidden size, B/L/P as in hidden.py.
"""
import os
import time
import zlib
from collections import defaultdict

import hydra
import pandas as pd
import torch
from omegaconf import OmegaConf
from safetensors.torch import save_file

from shape_rl import paths
from shape_rl.hidden import final_norm, load_model, states_at

CHUNKS = "s[0-9][0-9][0-9]_c[0-9][0-9][0-9].parquet"


def requests(rows: pd.DataFrame, traces: pd.DataFrame):
    """Forward passes for one chunk: each trace once (h_v, h_q[:, 0]) and each (row, a') once (h_q[:, j>=1]).
    -> [(token_ids, [(position, row, j)])], j = -1 marks h_v."""
    seq = dict(zip(traces.trace_id, traces.token_ids))
    at_trace, reqs = defaultdict(list), []
    for i, r in enumerate(rows.itertuples()):
        at_trace[r.trace_id] += [(r.pos - 1, i, -1), (r.pos, i, 0)]
        reqs += [(seq[r.trace_id][:r.pos].tolist() + [int(a)], [(r.pos, i, j)]) for j, a in enumerate(r.actions[1:], 1)]
    return reqs + [(seq[tid].tolist(), ps) for tid, ps in at_trace.items()]


def chunk_states(model, rows, traces, batch_tokens, device):
    """-> h_v [N, D], h_q [N, A, D] (bf16, CPU). Length-sorted, right-padded batches of <= batch_tokens tokens."""
    reqs = requests(rows, traces)
    N, A, D = len(rows), len(rows.actions.iloc[0]), final_norm(model).weight.shape[0]
    h_v_ND, h_q_NAD = torch.empty(N, D, dtype=torch.bfloat16), torch.empty(N, A, D, dtype=torch.bfloat16)

    def flush(batch):
        L, P = max(len(reqs[k][0]) for k in batch), max(len(reqs[k][1]) for k in batch)
        ids_BL, mask_BL, pos_BP = (torch.zeros(len(batch), n, dtype=torch.long) for n in (L, L, P))
        for b, k in enumerate(batch):
            toks, sinks = reqs[k]
            ids_BL[b, :len(toks)], mask_BL[b, :len(toks)] = torch.tensor(toks), 1
            pos_BP[b, :len(sinks)] = torch.tensor([s[0] for s in sinks])
        x_BPD = states_at(model, ids_BL.to(device), mask_BL.to(device), pos_BP.to(device)).to("cpu", torch.bfloat16)
        for b, k in enumerate(batch):
            for p, (_, i, j) in enumerate(reqs[k][1]):
                if j < 0:
                    h_v_ND[i] = x_BPD[b, p]
                else:
                    h_q_NAD[i, j] = x_BPD[b, p]

    batch = []
    for k in sorted(range(len(reqs)), key=lambda k: len(reqs[k][0])):
        if batch and (len(batch) + 1) * len(reqs[k][0]) > batch_tokens:
            flush(batch)
            batch = []
        batch.append(k)
    flush(batch)
    return h_v_ND, h_q_NAD


def write_index(out, partial=False):
    """index.parquet = every branch row + (states file, row offset), once all rollout shards and their states exist.
    partial=True: write it now from the chunks that have states (a dataset cut at a deadline)."""
    if (out / "index.parquet").exists():
        return
    chunks = sorted((out / "rollout").glob(CHUNKS))
    has_states = [(out / "states" / f"{f.stem}.safetensors").exists() for f in chunks]
    if partial:
        chunks = [f for f, h in zip(chunks, has_states) if h]
    elif len(list((out / "rollout").glob("s*.done"))) < OmegaConf.load(out / "config.yaml").num_shards or not all(has_states):
        return
    dfs = [pd.read_parquet(f).assign(states=f"states/{f.stem}.safetensors", row=lambda d: range(len(d))) for f in chunks]
    tmp = out / "index.tmp"
    pd.concat(dfs, ignore_index=True).to_parquet(tmp)
    tmp.rename(out / "index.parquet")


def run(cfg):
    paths.check_logged(cfg.log_path, cfg.exp, cfg.ablation)
    out = paths.run_dir(cfg)
    assert OmegaConf.load(out / "config.yaml").model.path == cfg.model.path, "encode with the rollout's model"
    (out / "states").mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = None
    # a chunk's encode shard is fixed by its name, so jobs started while the rollout still adds chunks never overlap
    for f in [f for f in sorted((out / "rollout").glob(CHUNKS)) if zlib.crc32(f.name.encode()) % cfg.num_shards == cfg.shard]:
        dst = out / "states" / f"{f.stem}.safetensors"
        if dst.exists():
            continue
        model = model or load_model(cfg.model.path).to(device)
        t0 = time.time()
        h_v, h_q = chunk_states(model, pd.read_parquet(f), pd.read_parquet(f.with_suffix(".traces.parquet")), cfg.encode.batch_tokens, device)
        tmp = dst.with_name(f"{dst.stem}.{os.getpid()}.tmp")
        save_file({"h_v": h_v, "h_q": h_q}, tmp)
        tmp.rename(dst)
        print(f"[encode] {f.name}: {len(h_v)} rows, {time.time() - t0:.0f}s", flush=True)
    write_index(out, cfg.encode.partial)


@hydra.main(config_path="../../configs", config_name="encode", version_base=None)
def main(cfg):
    run(cfg)


if __name__ == "__main__":
    main()
