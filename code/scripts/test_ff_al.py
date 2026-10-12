"""Feedforward antennal-lobe approximation for odor coding in the full-CNS model.

Odors enter as Poisson drive to uniglomerular PNs; chemical synapses among AL
neurons (ALPN, ALLN, ALIN, ALON and AL sensory neurons) are removed so the AL
cannot form the self-sustained attractor seen with ORN input. Everything
downstream of PNs (calyx, lateral horn, MB lobes, rest of CNS) is unchanged.

Reports KC population sparseness, odor specificity (Jaccard overlap of the KC
sets for two odors) and persistence after stimulus offset.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "results").is_dir())
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adfly.lif import PACK, Brains, Params, load_pack, poisson_events  # noqa: E402

neu = pd.read_parquet(PACK / "neurons.parquet")
t = neu.type.fillna("")
cls = neu["class"].fillna("")
row_ptr, dst, signed = load_pack()
src = np.repeat(np.arange(len(neu)), np.diff(row_ptr))
al_sens = t.str.match(r"^(ORN_|HRN_|TRN_)").to_numpy()
al = cls.isin(["ALPN", "ALLN", "ALIN", "ALON"]).to_numpy() | al_sens
is_kc = t.str.startswith("KC").to_numpy()
al_edge = al[src] & al[dst]
kckc = is_kc[src] & is_kc[dst]
sensory = neu.superclass.str.contains("sensory").to_numpy()
kc = np.flatnonzero(is_kc)
mbon = np.flatnonzero(t.str.startswith("MBON").to_numpy())

ODORS = {"A": ["DM1", "DM4", "VA2"], "B": ["DL5", "DM2", "VM2"]}


def upn(gloms):
    m = (cls == "ALPN") & t.str.match(r"^(%s)_" % "|".join(gloms))
    return np.flatnonzero(m.to_numpy())


def run(stim, ws, da, rate=100.0, B=8, seed=0):
    br = Brains(B, params=Params(tau_a=1000.0), wsets=ws[None, :])
    br.da[:, ~sensory] = da
    br.set_groups(np.arange(br.N), br.N)
    br.no_refractory(stim)
    si = np.tile(stim.astype(np.int32), (B, 1))
    rng = np.random.default_rng(seed)
    on = br.run(3000, si, poisson_events(rng, np.full(si.shape, rate), 3000, 0.1), bin_steps=1000)
    off = br.run(7000, si, np.zeros((B, 7000, stim.size), bool), bin_steps=1000)
    return np.concatenate([on, off], 1) * 10  # (B, 10, N) Hz


out = []
for label, mask, da in [("ffAL", al_edge, 0.0), ("ffAL", al_edge, 1.0),
                        ("ffAL-KCKC", al_edge | kckc, 0.0), ("ffAL-KCKC", al_edge | kckc, 1.0)]:
    ws = (signed * np.float32(0.275)).astype(np.float32)
    ws[mask] = 0
    res = {}
    for o, g in ODORS.items():
        r = run(upn(g), ws, da)
        kc_on = (r[:, :3][:, :, kc] > 0).any(1)   # (B, nKC)
        res[o] = {"r": r, "kc_on": kc_on}
    a, b = res["A"]["kc_on"], res["B"]["kc_on"]
    jac = (a & b).sum(1) / np.maximum((a | b).sum(1), 1)
    row = {"variant": label, "da": da, "n_PN_A": int(upn(ODORS["A"]).size), "n_PN_B": int(upn(ODORS["B"]).size),
           "KC_frac_A": round(float(a.mean()), 4), "KC_frac_B": round(float(b.mean()), 4),
           "KC_jaccard_AB": round(float(jac.mean()), 3),
           "active_on_A": int((res["A"]["r"][:, :3] > 0).any(1).sum(1).mean()),
           "active_last200_A": int((res["A"]["r"][:, -2:] > 0).any(1).sum(1).mean()),
           "active_last200_B": int((res["B"]["r"][:, -2:] > 0).any(1).sum(1).mean()),
           "MBON_hz_A": round(float(res["A"]["r"][:, :3][:, :, mbon].mean()), 2),
           "MBON_hz_B": round(float(res["B"]["r"][:, :3][:, :, mbon].mean()), 2)}
    out.append(row)
    print(row, flush=True)
(ROOT / "results/calibration/ff_al_test.json").write_text(json.dumps(out, indent=1))
