"""AD-fly base model: connectome, neuron sets, mushroom-body learning, pathology.

Base model = Shiu et al. (2024) LIF on MaleCNS v1.0 with two documented
extensions (see results/calibration):
  1. spike-frequency adaptation (tau_a = 1 s, da = 1 mV, non-sensory neurons),
  2. feedforward antennal lobe: odors drive uniglomerular PNs and chemical
     synapses among AL neurons are removed (with ORN input the AL forms an
     odor-independent self-sustained attractor in both FlyWire and MaleCNS).
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import pathology
from .lif import PACK, Brains, Params, load_pack

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "results").is_dir())
TAU_A, DA = 1000.0, 1.0
W_SYN = 0.275
ODORS = {"A": ["DM1", "DM4", "VA2"], "B": ["DL5", "DM2", "VM2"]}


class Connectome:
    def __init__(self, pack=PACK):
        self.pack = Path(pack)
        self.neu = pd.read_parquet(pack / "neurons.parquet")
        n = self.neu
        n["side"] = n.somaSide.where(n.somaSide.isin(["L", "R"]), n.rootSide)
        self.t = n.type.fillna("")
        self.cls = n["class"].fillna("")
        self.row_ptr, self.dst, self.signed = load_pack(pack)
        self.N = len(n)
        self.src = np.repeat(np.arange(self.N), np.diff(self.row_ptr)).astype(np.int32)
        self.sensory = n.superclass.str.contains("sensory").to_numpy()
        al_sens = self.t.str.match(r"^(ORN_|HRN_|TRN_)").to_numpy()
        al = self.cls.isin(["ALPN", "ALLN", "ALIN", "ALON"]).to_numpy() | al_sens
        self.al_edge = al[self.src] & al[self.dst]
        self.is_kc = self.t.str.startswith("KC").to_numpy()
        self.vuln = pathology.hub_vulnerability(self.row_ptr, self.dst, self.signed)
        self._mb()

    def idx(self, types, sides=("L", "R")):
        return np.flatnonzero((self.t.isin(types) & self.neu.side.isin(sides)).to_numpy())

    def upn(self, gloms):
        m = (self.cls == "ALPN") & self.t.str.match(r"^(%s)_" % "|".join(gloms))
        return np.flatnonzero(m.to_numpy())

    def _mb(self):
        t = self.t
        self.mbon = np.flatnonzero(t.str.startswith("MBON").to_numpy())
        nt = self.neu.consensus_nt.to_numpy()
        # Aso et al. 2014: glutamatergic MBONs drive avoidance, GABA/ACh MBONs approach
        self.mbon_sign = np.where(nt[self.mbon] == "glutamate", -1.0, 1.0)
        self.pam = np.flatnonzero(t.str.startswith("PAM").to_numpy())
        # compartment map: DAN type -> MBON types receiving >= 10% of its MBON synapses
        e = pd.read_parquet(self.pack / "mb_edges_raw.parquet")
        dm = e[e.pre_t.str.match(r"^(PAM|PPL1)") & e.post_t.str.startswith("MBON")]
        tab = dm.groupby(["pre_t", "post_t"]).weight.sum().reset_index()
        tab["frac"] = tab.weight / tab.groupby("pre_t").weight.transform("sum")
        self.comp = tab[tab.frac >= 0.10].groupby("pre_t").post_t.apply(set).to_dict()
        # KC -> MBON edges
        m = self.is_kc[self.src] & np.isin(self.dst, self.mbon)
        self.kcmb_e = np.flatnonzero(m)
        self.kcmb_pre = self.src[self.kcmb_e]
        self.kcmb_post_t = t.to_numpy()[self.dst[self.kcmb_e]]


@dataclass
class Fly:
    fly_id: int
    stage: float = 0.0
    scope: str = "pan"
    seed: int = 0
    kw: dict = field(default_factory=dict)      # per-fly pathology overrides (controls)
    load: np.ndarray = field(default=None, repr=False)
    dead: np.ndarray = field(default=None, repr=False)


def build_brains(cx: Connectome, flies, stage_kw=None):
    """One weight set per fly (pathology differs), feedforward AL, adaptation."""
    stage_kw = stage_kw or {}
    B = len(flies)
    wsets = np.empty((B, cx.signed.size), np.float32)
    vth_shift = np.zeros((B, cx.N), np.float32)
    dead = np.zeros((B, cx.N), bool)
    eligible = ~cx.sensory
    for b, f in enumerate(flies):
        rng = np.random.default_rng(10_000 + f.seed)
        st = pathology.Stage(s=f.stage, scope=f.scope, **{**stage_kw, **f.kw})
        w, dth, dd, load = pathology.apply(st, rng, cx.row_ptr, cx.dst, cx.signed, W_SYN,
                                           cx.vuln, cx.is_kc, eligible)
        w[cx.al_edge] = 0.0
        wsets[b] = w
        vth_shift[b] = dth
        dead[b] = dd
        f.load, f.dead = load, dd
    br = Brains(B, params=Params(tau_a=TAU_A, w_syn=W_SYN), wsets=wsets,
                wset_of=np.arange(B))
    br.da[:, ~cx.sensory] = DA
    br.vth -= vth_shift
    br.vth[dead] = np.inf
    return br


def learn(cx: Connectome, br: Brains, kc_spikes, dan_rate_by_type, eta=0.6, dan_on_hz=5.0):
    """Hige et al. (2015)-style depression of KC->MBON synapses.

    kc_spikes: (B, N) spike counts during the odor + dopamine pairing.
    dan_rate_by_type: list (per fly) of {DAN type: mean rate Hz}.
    A KC->MBON synapse onto an MBON in a compartment whose DAN type was active
    is scaled by (1 - eta) if the KC fired during the pairing.
    """
    for b in range(br.B):
        active_types = [d for d, r in dan_rate_by_type[b].items() if r >= dan_on_hz]
        targets = set().union(*[cx.comp.get(d, set()) for d in active_types]) if active_types else set()
        if not targets:
            continue
        in_comp = np.isin(cx.kcmb_post_t, list(targets))
        kc_fired = kc_spikes[b, cx.kcmb_pre] > 0
        e = cx.kcmb_e[in_comp & kc_fired]
        br.wsets[b, e] *= (1.0 - eta)


def valence(cx: Connectome, rates):
    """MBON ensemble valence: approach minus avoidance MBON rates (Hz)."""
    return (rates[:, cx.mbon] * cx.mbon_sign[None, :]).sum(1)
