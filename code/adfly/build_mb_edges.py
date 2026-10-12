"""Write mb_edges_raw.parquet for the MaleCNS pack: the unsigned edges among mushroom-body
neurons (Kenyon cells, MBONs and the PAM and PPL1 dopaminergic neurons).

model.Connectome reads this table to assign MBON types to dopaminergic compartments: an MBON
type belongs to a dopaminergic type's compartment if it receives at least 10% of that type's
synapses onto MBONs. Dopaminergic neurons carry no sign, so their edges are absent from the
signed pack written by build_pack.py and are read here from the source weights table.

Rows are the rows of the source weights table whose two bodies are both neurons of the pack
with a type name starting with KC, MBON, PAM or PPL1, in the order of the source table, with
the two type names added (columns body_pre, body_post, weight, pre_t, post_t). The index is
the row number within the record batch of the source file.

  python code/adfly/build_pack.py        # first: writes neurons.parquet
  python code/adfly/build_mb_edges.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from pyarrow import ipc

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "results").is_dir())
PACK = ROOT / "data/pack/malecns_v1.0"
F_W = ROOT / "data/malecns_v1.0/connectome-weights-male-cns-v1.0-minconf-0.5.feather"
MB_TYPES = r"^(KC|MBON|PAM|PPL1)"


def main(pack=PACK, weights=F_W):
    neu = pd.read_parquet(Path(pack) / "neurons.parquet")
    ids = neu.bodyId.to_numpy(np.int64)
    t = neu.type.fillna("")
    is_mb = t.str.match(MB_TYPES).to_numpy()
    reader = ipc.open_file(weights)
    parts = []
    for b in range(reader.num_record_batches):
        df = reader.get_batch(b).to_pandas()
        pre = df.body_pre.to_numpy(np.int64)
        post = df.body_post.to_numpy(np.int64)
        pi = np.minimum(np.searchsorted(ids, pre), ids.size - 1)
        qi = np.minimum(np.searchsorted(ids, post), ids.size - 1)
        keep = (ids[pi] == pre) & (ids[qi] == post) & is_mb[pi] & is_mb[qi]
        if keep.any():
            d = df[keep].copy()
            d["pre_t"] = t.iloc[pi[keep]].to_numpy()
            d["post_t"] = t.iloc[qi[keep]].to_numpy()
            parts.append(d)
    e = pd.concat(parts)
    e.to_parquet(Path(pack) / "mb_edges_raw.parquet")
    dan = e.pre_t.str.match(r"^(PAM|PPL1)") & e.post_t.str.startswith("MBON")
    print(f"{len(e)} edges among mushroom-body neurons, {int(dan.sum())} of them dopaminergic to MBON "
          f"({int(e.weight[dan].sum())} synapses) -> {Path(pack) / 'mb_edges_raw.parquet'}")


if __name__ == "__main__":
    main()
