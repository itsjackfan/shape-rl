"""Hidden states at branch points -> one safetensors file per rollout chunk, plus one parquet index. Any number of jobs
share the chunks through claim files (states/<chunk>.claim); resumable.

For a branch row (pos = prompt_len + t, see rollout.py):
  h_v[i]    = state at pos - 1 of the trace = h(s_t): it predicts pi(.|s_t), so it is the input for V(s_t).
  h_q[i, j] = state at pos with input token actions[j] = h(s_t, a) (j = 0: sampled a_t, j >= 1: a'_j): input for Q(s_t, a).
States are post-norm last-layer, bf16 (hidden.states_at). N = branch rows, A = actions, D = hidden size, B/L/P as in hidden.py.
"""
import os
import time
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


def chunk_states(model, rows, traces, batch_tokens, device, beat=lambda: None):
    """-> h_v [N, D], h_q [N, A, D] (bf16, CPU). Length-sorted, right-padded batches of <= batch_tokens tokens; beat() per batch."""
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
        beat()

    batch = []
    for k in sorted(range(len(reqs)), key=lambda k: len(reqs[k][0])):
        if batch and (len(batch) + 1) * len(reqs[k][0]) > batch_tokens:
            flush(batch)
            batch = []
        batch.append(k)
    flush(batch)
    return h_v_ND, h_q_NAD


def write_index(out):
    """index.parquet = every branch row + (states file, row offset). Written once the chunk set is final (every rollout shard
    done, or rollout/STOP: the dataset was cut at a deadline after the rollout jobs ended) and every chunk has states."""
    chunks = sorted((out / "rollout").glob(CHUNKS))
    final = (out / "rollout" / "STOP").exists() or len(list((out / "rollout").glob("s*.done"))) >= OmegaConf.load(out / "config.yaml").num_shards
    if (out / "index.parquet").exists() or not final or not all((out / "states" / f"{f.stem}.safetensors").exists() for f in chunks):
        return
    dfs = [pd.read_parquet(f).assign(states=f"states/{f.stem}.safetensors", row=lambda d: range(len(d))) for f in chunks]
    tmp = out / f"index.{os.getpid()}.tmp"
    pd.concat(dfs, ignore_index=True).to_parquet(tmp)
    tmp.rename(out / "index.parquet")


def claim(path, stale_s=1200):
    """Exclusive claim on a chunk; a claim older than stale_s (a preempted job) is taken over."""
    try:
        os.close(os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        return True
    except FileExistsError:
        if time.time() - path.stat().st_mtime < stale_s:
            return False
        path.unlink(missing_ok=True)
        return claim(path, stale_s)


def encode_pending(cfg, out, model, device):
    """Encode every chunk that has no states and no live claim. -> model, number done."""
    n = 0
    for f in sorted((out / "rollout").glob(CHUNKS)):
        dst, lock = out / "states" / f"{f.stem}.safetensors", out / "states" / f"{f.stem}.claim"
        if dst.exists() or not claim(lock):
            continue
        model = model or load_model(cfg.model.path).to(device)
        t0 = time.time()
        h_v, h_q = chunk_states(model, pd.read_parquet(f), pd.read_parquet(f.with_suffix(".traces.parquet")), cfg.encode.batch_tokens, device,
                                beat=lambda: os.utime(lock))
        tmp = dst.with_name(f"{dst.stem}.{os.getpid()}.tmp")
        save_file({"h_v": h_v, "h_q": h_q}, tmp)
        tmp.rename(dst)
        n += 1
        print(f"[encode] {f.name}: {len(h_v)} rows, {time.time() - t0:.0f}s", flush=True)
    return model, n


def run(cfg):
    """encode.follow: keep encoding new chunks while the rollout runs, until rollout/STOP exists (then a last pass)."""
    paths.check_logged(cfg.log_path, cfg.exp, cfg.ablation)
    out = paths.run_dir(cfg)
    assert OmegaConf.load(out / "config.yaml").model.path == cfg.model.path, "encode with the rollout's model"
    (out / "states").mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = None
    while True:
        stop = (out / "rollout" / "STOP").exists()
        model, n = encode_pending(cfg, out, model, device)
        if not cfg.encode.follow or stop:
            break
        if n == 0:
            time.sleep(60)
    write_index(out)


@hydra.main(config_path="../../configs", config_name="encode", version_base=None)
def main(cfg):
    run(cfg)


if __name__ == "__main__":
    main()
