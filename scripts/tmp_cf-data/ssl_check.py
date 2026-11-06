"""ssl pipeline check on a real (tiny) dataset: every estimator, one seed. The smoke50 prompts are all in train, so this
check re-splits rows by trace (check only; real runs split by prompt). Usage: python ssl_check.py <exp>/<ablation>"""
import sys
import time

import numpy as np
import pandas as pd
from hydra import compose, initialize_config_dir

from shape_rl import ssl

orig = ssl.load


def load(d):
    idx, h, s, f, o = orig(d)
    t = pd.factorize(idx.trace_id)[0]
    return idx.assign(split=np.where(t % 3 == 0, "test", np.where(t % 3 == 1, "val", "train"))), h, s, f, o


ssl.load = load
for est in ["b0", "b1", "b2", "b3", "c0", "c1", "c2", "c3"]:
    small = [] if est[0] == "b" else ["estimator.proj.steps=50", "estimator.proj.batch=8"]
    with initialize_config_dir(config_dir=str(ssl.paths.ROOT / "configs"), version_base=None):
        cfg = compose("ssl", overrides=[f"estimator={est}", f"ssl.data={sys.argv[1]}", "ssl.seeds=[0]", "exp=check", "ablation=check", *small])
    t0 = time.time()
    data = ssl.prepare(cfg)
    test = (data["idx"].split == "test").values
    p = ssl.estimate(data, cfg.estimator, 0)
    print(est, {k: round(v, 3) for k, v in ssl.metrics(p, data, test).items()}, f"{time.time() - t0:.0f}s", flush=True)
print("X", tuple(data["X"].shape), "ceiling", {k: round(v, 3) for k, v in ssl.ceiling(data, test).items()})
