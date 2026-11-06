"""Preliminary estimator check on the chunks encoded so far (before the dataset cut): real split by prompt, one seed,
test + val metrics. Not a result; it checks training and metrics at scale.
Writes <dataset>/analysis/prelim-<HHMM>/{index.parquet, states -> ../../states}. Usage: python prelim.py <exp>/<ablation> [est ...]"""
import sys
import time
from pathlib import Path

import pandas as pd
from hydra import compose, initialize_config_dir

from shape_rl import ssl
from shape_rl.encode import CHUNKS

with initialize_config_dir(config_dir=str(ssl.paths.ROOT / "configs"), version_base=None):
    root = Path(compose("ssl").output_root)
d = root / sys.argv[1]
chunks = [f for f in sorted((d / "rollout").glob(CHUNKS)) if (d / "states" / f"{f.stem}.safetensors").exists()]
idx = pd.concat([pd.read_parquet(f).assign(states=f"states/{f.stem}.safetensors", row=lambda x: range(len(x))) for f in chunks],
                ignore_index=True)
pre = d / "analysis" / f"prelim-{time.strftime('%H%M')}"
pre.mkdir(parents=True)
(pre / "states").symlink_to(d / "states")
idx.to_parquet(pre / "index.parquet")
print(f"chunks {len(chunks)}, rows {len(idx)}, prompts {idx.prompt_id.nunique()}", idx.groupby("split").prompt_id.nunique().to_dict(), flush=True)
for est in sys.argv[2:] or ["b0", "b1", "b2", "b3", "c0", "c1", "c2", "c3"]:
    with initialize_config_dir(config_dir=str(ssl.paths.ROOT / "configs"), version_base=None):
        cfg = compose("ssl", overrides=[f"estimator={est}", f"ssl.data={pre.relative_to(root)}", "ssl.seeds=[0]"])
    t0 = time.time()
    data = ssl.prepare(cfg)
    test, val = ((data["idx"].split == s).values for s in ("test", "val"))
    p = ssl.estimate(data, cfg.estimator, 0)
    print(est, "test", {k: round(v, 3) for k, v in ssl.metrics(p, data, test).items()}, "| val",
          {k: round(v, 3) for k, v in ssl.metrics(p, data, val).items()}, f"{time.time() - t0:.0f}s", flush=True)
print("ceiling(test)", {k: round(v, 3) for k, v in ssl.ceiling(data, test).items()})
