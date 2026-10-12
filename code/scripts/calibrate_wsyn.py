"""Calibrate w_syn for MaleCNS against the FlyWire-fitted Shiu et al. value.

Shiu et al. fitted w_syn = 0.275 mV on FlyWire, whose synapse detection yields
fewer synapses per connection than MaleCNS. For cell types matched between the
two datasets (MaleCNS `flywireType` == FlyWire `cell_type`), we compare the
number of input synapses per neuron and use the median per-type ratio to
rescale w_syn so that a matched neuron receives the same total drive.

Also builds the FlyWire v783 pack and tests odor-evoked persistence there.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pyarrow import ipc

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "results").is_dir())
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adfly.lif import PACK, Brains, poisson_events  # noqa: E402

OUT = ROOT / "results/calibration"
OUT.mkdir(parents=True, exist_ok=True)
FW = ROOT / "data/pack/flywire_783"

# ---- FlyWire v783 pack (Shiu et al. convention: GABA/Glu -, all else +)
comp = pd.read_csv(ROOT / "data/shiu2024/Completeness_783.csv", index_col=0)
con = pd.read_parquet(ROOT / "data/shiu2024/Connectivity_783.parquet")
if not (FW / "row_ptr.npy").exists():
    FW.mkdir(parents=True, exist_ok=True)
    pre = con.Presynaptic_Index.to_numpy(np.int64)
    post = con.Postsynaptic_Index.to_numpy(np.int32)
    w = con["Excitatory x Connectivity"].to_numpy(np.int32)
    o = np.lexsort((post, pre))
    rp = np.zeros(len(comp) + 1, np.int64)
    np.add.at(rp, pre[o] + 1, 1)
    np.save(FW / "row_ptr.npy", np.cumsum(rp))
    np.save(FW / "dst.npy", post[o])
    np.save(FW / "signed_w.npy", w[o])
    np.save(FW / "neuron_ids.npy", comp.index.to_numpy(np.int64))

# ---- input synapses per neuron, FlyWire (all partners)
fw_in = con.groupby("Postsynaptic_ID")["Connectivity"].sum()
ann = pd.read_csv(ROOT / "data/flywire/fw_neuron_annotations.tsv", sep="\t",
                  usecols=["root_id", "cell_type", "super_class"], low_memory=False)
ann = ann[ann.root_id.isin(comp.index)]
ann["in_syn"] = fw_in.reindex(ann.root_id).fillna(0).to_numpy()
fw_type = ann.dropna(subset=["cell_type"]).groupby("cell_type")["in_syn"].agg(["mean", "size"])

# ---- input synapses per neuron, MaleCNS (all partners among selected neurons)
neu = pd.read_parquet(PACK / "neurons.parquet")
ids = neu.bodyId.to_numpy(np.int64)
in_syn = np.zeros(ids.size, np.int64)
reader = ipc.open_file(ROOT / "data/malecns_v1.0/connectome-weights-male-cns-v1.0-minconf-0.5.feather")
for b in range(reader.num_record_batches):
    bt = reader.get_batch(b)
    pre = bt.column("body_pre").to_numpy().astype(np.int64)
    post = bt.column("body_post").to_numpy().astype(np.int64)
    w = bt.column("weight").to_numpy().astype(np.int64)
    pi = np.minimum(np.searchsorted(ids, pre), ids.size - 1)
    qi = np.minimum(np.searchsorted(ids, post), ids.size - 1)
    ok = (ids[pi] == pre) & (ids[qi] == post)
    np.add.at(in_syn, qi[ok], w[ok])
neu["in_syn"] = in_syn
brain = ~neu.superclass.str.startswith(("vnc_", "ascending", "sensory_ascending", "efferent_ascending"))
mc_type = neu[brain].dropna(subset=["flywireType"]).groupby("flywireType")["in_syn"].agg(["mean", "size"])

m = fw_type.join(mc_type, lsuffix="_fw", rsuffix="_mc", how="inner")
m = m[(m.mean_fw >= 50) & (m.mean_mc >= 50)]
m["ratio"] = m.mean_mc / m.mean_fw
w_cal = 0.275 / float(m.ratio.median())
res = {"n_matched_types": int(len(m)), "ratio_median": float(m.ratio.median()),
       "ratio_iqr": [float(m.ratio.quantile(.25)), float(m.ratio.quantile(.75))],
       "w_syn_calibrated_mV": w_cal,
       "fw_mean_in_syn_per_neuron": float(ann.in_syn.mean()),
       "mc_brain_mean_in_syn_per_neuron": float(neu[brain].in_syn.mean())}
m.to_csv(OUT / "matched_type_input_synapses.csv")
print(json.dumps(res, indent=1))


def persistence(pack, stim, w_syn, B=4):
    from adfly.lif import Params
    br = Brains(B, params=Params(w_syn=w_syn), pack=pack)
    br.set_groups(np.arange(br.N), br.N)
    br.no_refractory(stim)
    si = np.tile(stim.astype(np.int32), (B, 1))
    rng = np.random.default_rng(0)
    on = br.run(3000, si, poisson_events(rng, np.full(si.shape, 150.0), 3000, 0.1), bin_steps=1000)
    off = br.run(7000, si, np.zeros((B, 7000, stim.size), bool), bin_steps=1000)
    r = np.concatenate([on, off], 1).mean(0) * 10
    return r


# ---- odor persistence in FlyWire v783 (left DM1/DM4/VA2 ORNs)
fa = pd.read_csv(ROOT / "data/flywire/fw_neuron_annotations.tsv", sep="\t",
                 usecols=["root_id", "cell_type", "side"], low_memory=False)
fpos = {f: i for i, f in enumerate(comp.index)}
orn = fa[fa.cell_type.isin(["ORN_DM1", "ORN_DM4", "ORN_VA2"]) & (fa.side == "left")]
fstim = np.array([fpos[f] for f in orn.root_id if f in fpos])
r = persistence(FW, fstim, 0.275)
res["flywire_odor_active_per_100ms"] = [int(x) for x in (r > 0).sum(1)]
kc_fw = np.array([fpos[f] for f in fa[fa.cell_type.fillna("").str.startswith("KC")].root_id if f in fpos])
res["flywire_odor_KC_active_on_vs_last"] = [int((r[:3, kc_fw] > 0).any(0).sum()), int((r[-2:, kc_fw] > 0).any(0).sum()), int(kc_fw.size)]

# ---- MaleCNS with calibrated w_syn
neu["side"] = neu.somaSide.where(neu.somaSide.isin(["L", "R"]), neu.rootSide)
mstim = np.flatnonzero((neu.type.isin(["ORN_DM1", "ORN_DM4", "ORN_VA2"]) & (neu.side == "L")).to_numpy())
kc_mc = np.flatnonzero(neu.type.fillna("").str.startswith("KC").to_numpy())
for w in (0.275, w_cal):
    r = persistence(PACK, mstim, w)
    res[f"malecns_odor_w{w:.3f}_active_per_100ms"] = [int(x) for x in (r > 0).sum(1)]
    res[f"malecns_odor_w{w:.3f}_KC_active_on_vs_last"] = [int((r[:3, kc_mc] > 0).any(0).sum()), int((r[-2:, kc_mc] > 0).any(0).sum()), int(kc_mc.size)]
sug = np.flatnonzero((neu.type.isin(["LB3c", "LB3b"]) & (neu.side == "R")).to_numpy())
mn9 = np.flatnonzero((neu.type == "MN9").to_numpy())
r = persistence(PACK, sug, w_cal, B=16)
res["malecns_sugar_wcal_MN9_on_hz"] = r[:3][:, mn9].mean(0).round(1).tolist()
res["malecns_sugar_wcal_active_per_100ms"] = [int(x) for x in (r > 0).sum(1)]
print(json.dumps(res, indent=1))
(OUT / "calibration.json").write_text(json.dumps(res, indent=1))
