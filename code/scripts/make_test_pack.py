"""Build a small synthetic connectome pack to smoke-test the experiment-1 code without MaleCNS.

  python scripts/make_test_pack.py --out /tmp/testpack

The pack has the file layout and annotation columns of the real one and contains every
cell type the protocol stimulates or reads (odour projection neurons, Kenyon cells, APL,
MBONs, PAM and PPL1 neurons, lateral horn neurons, LB3c/LB3b, MN9, LPLC2, LC4, DNp01),
wired so that each pathway carries activity. It also writes a stand-in severity-series
table (stage_curve_x4.csv) whose dead_frac values are computed with the real pathology
code, so the reconstruction check passes. Its dynamics mean nothing biologically.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "results").is_dir())
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-other", type=int, default=1500)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    rows = []

    def add(n, type_, cls, superclass, nt, side=None):
        for i in range(n):
            s = side or ("L" if i % 2 == 0 else "R")
            rows.append(dict(type=type_, cls=cls, superclass=superclass, nt=nt, side=s))

    gl = {"DM1": "DM1_lPN", "DM4": "DM4_adPN", "VA2": "VA2_adPN", "DL5": "DL5_adPN", "DM2": "DM2_lPN",
          "VM2": "VM2_adPN"}
    for g in gl:
        add(10, f"ORN_{g}", "olfactory", "sensory", "acetylcholine")
    for g, t in gl.items():
        add(6, t, "ALPN", "central", "acetylcholine")
    add(20, "LN_AL", "ALLN", "central", "gaba")
    add(300, "KCg-m", "Kenyon_Cell", "central", "acetylcholine")
    add(200, "KCab", "Kenyon_Cell", "central", "acetylcholine")
    add(100, "KCapbp", "Kenyon_Cell", "central", "acetylcholine")
    add(2, "APL", "MBIN", "central", "gaba")
    for k in range(1, 13):
        add(2, f"MBON{k:02d}", "MBON", "central", "glutamate" if k <= 6 else ("gaba" if k <= 9 else "acetylcholine"))
    for k in range(1, 11):
        add(2, f"PAM{k:02d}", "DAN", "central", "dopamine")
    for k in range(1, 4):
        add(2, f"PPL1{k:02d}", "DAN", "central", "dopamine")
    for k in range(1, 9):
        add(10, f"LHAV{k}", "LHN", "central", "acetylcholine" if k <= 6 else "gaba")
    add(10, "LB3c", "gustatory", "sensory", "acetylcholine")
    add(10, "LB3b", "gustatory", "sensory", "acetylcholine")
    add(40, "SEZ_feed", "SEZ", "central", "acetylcholine")
    add(2, "MN9", "motor", "vnc_motor", "acetylcholine")
    add(20, "LPLC2", "VPN", "visual_projection", "acetylcholine", "L")
    add(20, "LPLC2", "VPN", "visual_projection", "acetylcholine", "R")
    add(20, "LC4", "VPN", "visual_projection", "acetylcholine", "L")
    add(20, "LC4", "VPN", "visual_projection", "acetylcholine", "R")
    add(1, "DNp01", "descending", "descending", "acetylcholine", "L")
    add(1, "DNp01", "descending", "descending", "acetylcholine", "R")
    for k in range(a.n_other):
        typ = f"OTH{k % 150:03d}" if k % 7 else ""
        add(1, typ, ["CX", "SMP", "AOTU", "LAL", ""][k % 5], "central",
            ["acetylcholine", "gaba", "glutamate", "acetylcholine", "unknown"][k % 5])
    # pathology.hub_vulnerability needs the last neuron to have an outgoing signed edge, as in MaleCNS
    add(1, "OTH_last", "CX", "central", "acetylcholine")
    d = pd.DataFrame(rows)
    N = len(d)
    T = d.type.to_numpy()
    ix = lambda cond: np.flatnonzero(cond)  # noqa: E731
    edges = []

    def connect(pre, post, n_each, lo, hi, per_post=True):
        """Each post neuron gets n_each inputs from pre (or each pre gets n_each targets)."""
        if len(pre) == 0 or len(post) == 0:
            return
        if per_post:
            for j in post:
                for i in rng.choice(pre, min(n_each, len(pre)), replace=False):
                    if i != j:
                        edges.append((i, j, rng.integers(lo, hi + 1)))
        else:
            for i in pre:
                for j in rng.choice(post, min(n_each, len(post)), replace=False):
                    if i != j:
                        edges.append((i, j, rng.integers(lo, hi + 1)))

    kc = ix(pd.Series(T).str.startswith("KC"))
    pn = ix(d.cls == "ALPN")
    mbon = ix(pd.Series(T).str.startswith("MBON"))
    lh = ix(d.cls == "LHN")
    apl = ix(T == "APL")
    oth = ix(pd.Series(T).str.startswith("OTH") | (d.cls.isin(["CX", "SMP", "AOTU", "LAL"]) & (T == "")))
    for g, t in gl.items():
        connect(ix(T == f"ORN_{g}"), ix(T == t), 6, 2, 5)
    connect(ix(d.cls == "ALLN"), pn, 3, 1, 3)
    connect(pn, kc, 6, 8, 14)
    connect(kc, apl, 400, 1, 2)
    connect(apl, kc, 1, 2, 4, per_post=True)
    connect(kc, mbon, 250, 2, 5)
    connect(pn, lh, 6, 8, 14)
    connect(ix((T == "LB3c") | (T == "LB3b")), ix(T == "SEZ_feed"), 6, 12, 20)
    connect(ix(T == "SEZ_feed"), ix(T == "MN9"), 30, 10, 20)
    for s in ("L", "R"):
        connect(ix(((T == "LPLC2") | (T == "LC4")) & (d.side == s)), ix((T == "DNp01") & (d.side == s)), 30, 6, 12)
    connect(oth, oth, 15, 1, 4, per_post=False)
    connect(mbon, oth, 20, 1, 4, per_post=False)
    connect(lh, oth, 10, 1, 4, per_post=False)
    connect(oth, np.concatenate([kc, mbon, lh, ix(T == "SEZ_feed")]), 2, 1, 2, per_post=True)

    e = pd.DataFrame(edges, columns=["pre", "post", "w"]).groupby(["pre", "post"], as_index=False).w.sum()
    sign_of = {"acetylcholine": 1, "gaba": -1, "glutamate": -1}
    sg = d.nt.map(sign_of).fillna(0).astype(int).to_numpy()
    e["s"] = sg[e.pre.to_numpy()]
    signed = e[e.s != 0].sort_values(["pre", "post"])
    pre, post = signed.pre.to_numpy(), signed.post.to_numpy()
    row_ptr = np.zeros(N + 1, np.int64)
    np.add.at(row_ptr, pre + 1, 1)
    row_ptr = np.cumsum(row_ptr)
    np.save(out / "row_ptr.npy", row_ptr)
    np.save(out / "dst.npy", post.astype(np.int32))
    np.save(out / "signed_w.npy", (signed.s * signed.w).to_numpy().astype(np.int32))
    body = np.arange(10_000, 10_000 + N, dtype=np.int64)
    np.save(out / "neuron_ids.npy", body)
    neu = pd.DataFrame({"bodyId": body, "type": d.type.replace("", None), "instance": d.type,
                        "superclass": d.superclass, "class": d.cls.replace("", None), "subclass": None,
                        "somaSide": d.side, "rootSide": d.side, "flywireType": None, "hemibrainType": None,
                        "somaLocation": None, "consensus_nt": d.nt})
    neu.to_parquet(out / "neurons.parquet")
    # dopaminergic -> MBON edges by type (unsigned, so absent from the signed pack)
    mb = []
    for k in range(1, 11):
        for m in (f"MBON{(k % 6) + 1:02d}", f"MBON{((k + 2) % 12) + 1:02d}"):
            mb.append({"pre_t": f"PAM{k:02d}", "post_t": m, "weight": int(rng.integers(20, 80))})
    for k in range(1, 4):
        mb.append({"pre_t": f"PPL1{k:02d}", "post_t": f"MBON{k + 6:02d}", "weight": 50})
    pd.DataFrame(mb).to_parquet(out / "mb_edges_raw.parquet")
    # stand-in severity series with exact dead fractions
    from adfly import counterfactual as cf
    from adfly.model import Connectome
    cx = Connectome(out)
    rows = []
    for k, (stage, seed) in enumerate([(0.0, 1000), (0.0, 1001)] + [(0.5, 1100 + i) for i in range(4)]
                                      + [(0.3, 1200 + i) for i in range(2)]):
        dmg = cf.fly_damage(cx, seed, stage)
        rows.append({"condition": f"s{stage:g}", "stage": stage, "scope": "pan", "seed": seed,
                     "dead_frac": float(dmg.dead.mean())})
    pd.DataFrame(rows).to_csv(out / "stage_curve_x4.csv", index=False)
    print(f"test pack: {N} neurons, {len(signed)} signed edges -> {out}")


if __name__ == "__main__":
    main()
