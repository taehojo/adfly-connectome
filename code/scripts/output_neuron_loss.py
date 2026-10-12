"""Do all-or-none output failures follow removal of the output neuron itself?

Pathology is deterministic given a fly's seed, so the set of removed neurons in
every fly of the 384-fly series can be rebuilt exactly and set against its readouts.

  python scripts/output_neuron_loss.py

Writes results/paper/output_neuron_loss.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "results").is_dir())
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adfly import pathology  # noqa: E402
from adfly.model import W_SYN, Connectome  # noqa: E402

cx = Connectome()
d = pd.read_csv(ROOT / "results/battery/stage_curve_x4.csv")
d = d[(d.scope == "pan") & (d.stage > 0)]
gfL, gfR = cx.idx(["DNp01"], ("L",)), cx.idx(["DNp01"], ("R",))
mn9 = cx.idx(["MN9"])
elig = ~cx.sensory
rows = []
for _, r in d.iterrows():
    rng = np.random.default_rng(10_000 + int(r.seed))      # same draw as build_brains
    _, _, dead, _ = pathology.apply(pathology.Stage(s=r.stage), rng, cx.row_ptr, cx.dst,
                                    cx.signed, W_SYN, cx.vuln, cx.is_kc, elig)
    assert abs(dead.mean() - r.dead_frac) < 1e-9, "reconstruction does not match the run"
    rows.append({"stage": r.stage, "gf_fail": bool(r.GF_loom_hz < 100),
                 "gf_left_removed": bool(dead[gfL].any()), "gf_right_removed": bool(dead[gfR].any()),
                 "mn9_silent": bool(r.MN9_sugar_hz == 0), "mn9_removed": int(dead[mn9].sum())})
x = pd.DataFrame(rows)


def tab(a, b):
    t = pd.crosstab(a, b)
    return {f"{i}|{j}": int(t.loc[i, j]) for i in t.index for j in t.columns}


out = {
    "n_flies": int(len(x)),
    "reconstruction_exact": True,
    "looming_side": "left",
    "vulnerability_percentile": {"DNp01_L": float(cx.vuln[gfL][0]), "DNp01_R": float(cx.vuln[gfR][0]),
                                 "MN9": [float(v) for v in cx.vuln[mn9]]},
    "gf_fail_by_left_removed": tab(x.gf_left_removed, x.gf_fail),
    "gf_fail_by_left_right": tab(x.gf_left_removed.astype(str) + "/" + x.gf_right_removed.astype(str),
                                 x.gf_fail),
    "mn9_silent_by_n_removed": tab(x.mn9_removed, x.mn9_silent),
}
(ROOT / "results/paper/output_neuron_loss.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
