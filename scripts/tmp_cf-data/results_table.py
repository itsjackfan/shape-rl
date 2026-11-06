"""Results table: every estimator x metric on held-out prompts, mean ± std over seeds, plus the noise ceiling row.
Usage: python scripts/tmp_cf-data/results_table.py <output_root>/<exp> [ablation ...]"""
import json
import sys
from pathlib import Path

d = Path(sys.argv[1])
abls = sys.argv[2:] or sorted(p.parent.name for p in d.glob("*/metrics.json"))
cols = ["auroc", "succ_pcorr", "succ_pcorr_resid", "spearman_A", "spearman_A_resid", "sign_acc", "ece"]
print("| estimator | n seeds | " + " | ".join(cols) + " |")
print("|" + "---|" * (len(cols) + 2))
ceil = None
for a in abls:
    m = json.loads((d / a / "metrics.json").read_text())
    ceil = ceil or m["ceiling"]
    cells = [f"{m['mean'][c]:.3f} ± {m['std'][c]:.3f}" for c in cols]
    print(f"| {a} | {len(m['seeds'])} | " + " | ".join(cells) + " |")
if ceil:
    row = {"auroc": f"{ceil['auroc']:.3f} (MC V, M=8)", "succ_pcorr": f"{ceil.get('succ_pcorr', float('nan')):.3f} (MC V)",
           "spearman_A": f"{ceil['spearman_A']:.3f} (sqrt rel)",
           "sign_acc": f"{ceil['sign_acc_halfM']:.3f} (M/2 labels)"}
    print("| noise ceiling | - | " + " | ".join(row.get(c, "-") for c in cols) + " |")
    print("\nsplit-half reliability (Spearman-Brown to M):", {k: round(v, 3) for k, v in ceil.items()})
