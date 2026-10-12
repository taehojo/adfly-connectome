"""Write the hub rank u of every neuron to results/validation/hub_vulnerability_u.npz.

u is the percentile rank of a neuron's total synapse count in the signed network
(pathology.hub_vulnerability, i.e. Connectome.vuln); ids are the MaleCNS body identifiers in
the order of the neuron table. The fact scripts read this file so that they do not need the
connectome.

  python code/scripts/export_hub_rank.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "results").is_dir())
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adfly.model import Connectome  # noqa: E402

cx = Connectome()
out = ROOT / "results/validation/hub_vulnerability_u.npz"
out.parent.mkdir(parents=True, exist_ok=True)
np.savez(out, u=np.asarray(cx.vuln, np.float64), ids=cx.neu.bodyId.to_numpy(np.int64))
print(f"wrote {out}: {cx.N} neurons")
