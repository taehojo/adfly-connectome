"""Compile MaleCNS v1.0 flat-connectome tables into a source-major CSR pack.

Selection rules follow the `male-cns-v1.0-superclass-non-null-known-nt` pack of
https://github.com/Kisame76/drosophila-brain-mlx (MIT), so results are comparable:

  nodes  annotation rows whose `superclass` is non-null, ascending bodyId
  edges  both endpoints are selected nodes and the presynaptic `consensus_nt`
         is acetylcholine (+1), GABA (-1) or glutamate (-1)

Neurons with modulatory or unknown transmitters keep their node (they receive
input) but contribute no outgoing edges.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pyarrow import feather, ipc

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "results").is_dir())
RAW = ROOT / "data/malecns_v1.0"
OUT = ROOT / "data/pack/malecns_v1.0"
F_ANN = RAW / "body-annotations-male-cns-v1.0-minconf-0.5.feather"
F_NT = RAW / "body-neurotransmitters-male-cns-v1.0.feather"
F_W = RAW / "connectome-weights-male-cns-v1.0-minconf-0.5.feather"
SIGN = {"acetylcholine": 1, "gaba": -1, "glutamate": -1}
ANN_COLS = ["bodyId", "type", "instance", "superclass", "class", "subclass",
            "somaSide", "rootSide", "flywireType", "hemibrainType", "somaLocation"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ann = feather.read_table(F_ANN).to_pandas()
    ann = ann[ann.superclass.notna()].sort_values("bodyId").reset_index(drop=True)
    ids = ann.bodyId.to_numpy(np.int64)
    assert np.all(np.diff(ids) > 0)

    nt = feather.read_table(F_NT, columns=["body", "consensus_nt"]).to_pandas()
    nt = nt.drop_duplicates("body").set_index("body")["consensus_nt"]
    ann["consensus_nt"] = nt.reindex(ids).to_numpy()
    sign = ann.consensus_nt.map(SIGN).fillna(0).astype(np.int8).to_numpy()

    reader = ipc.open_file(F_W)
    pre_l, post_l, w_l = [], [], []
    n_raw = n_endpoint = n_unsigned = 0
    for b in range(reader.num_record_batches):
        batch = reader.get_batch(b)
        pre = batch.column("body_pre").to_numpy().astype(np.int64)
        post = batch.column("body_post").to_numpy().astype(np.int64)
        w = batch.column("weight").to_numpy().astype(np.int64)
        n_raw += pre.size
        pi = np.minimum(np.searchsorted(ids, pre), ids.size - 1)
        qi = np.minimum(np.searchsorted(ids, post), ids.size - 1)
        ok = (ids[pi] == pre) & (ids[qi] == post)
        n_endpoint += int((~ok).sum())
        s = np.where(ok, sign[pi], 0)
        keep = ok & (s != 0) & (w > 0)
        n_unsigned += int((ok & (s == 0)).sum())
        pre_l.append(pi[keep].astype(np.int32))
        post_l.append(qi[keep].astype(np.int32))
        w_l.append((s[keep] * w[keep]).astype(np.int32))
    pre = np.concatenate(pre_l); post = np.concatenate(post_l); w = np.concatenate(w_l)

    order = np.lexsort((post, pre))
    pre, post, w = pre[order], post[order], w[order]
    row_ptr = np.zeros(ids.size + 1, np.int64)
    np.add.at(row_ptr, pre.astype(np.int64) + 1, 1)
    row_ptr = np.cumsum(row_ptr)

    np.save(OUT / "neuron_ids.npy", ids)
    np.save(OUT / "row_ptr.npy", row_ptr)
    np.save(OUT / "dst.npy", post)
    np.save(OUT / "signed_w.npy", w)
    ann.to_parquet(OUT / "neurons.parquet")

    manifest = {
        "dataset": "male-cns-v1.0-superclass-non-null-known-nt",
        "license": "CC BY 4.0 (FlyEM HHMI Janelia, Univ. Cambridge, MRC LMB, Google Research)",
        "neurons": int(ids.size),
        "edges": int(pre.size),
        "synapses_abs": int(np.abs(w).sum()),
        "raw_rows": int(n_raw),
        "dropped_endpoint_not_selected": int(n_endpoint),
        "dropped_unsigned_presynaptic": int(n_unsigned),
        "sign_counts": {str(int(k)): int(v) for k, v in
                        zip(*np.unique(sign, return_counts=True))},
        "sources": {p.name: sha256(p) for p in (F_ANN, F_NT, F_W)},
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(json.dumps(manifest, indent=2, default=str))


if __name__ == "__main__":
    main()
