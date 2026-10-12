"""Counterfactual rescue in single simulated flies (experiment 1).

A simulated fly's pathology is fixed by its seed (model.build_brains draws it from
default_rng(10_000 + seed)), so the same individual can be simulated again with part of
its damage undone: removed neurons restored, thinned synapses restored, or the threshold
shift removed. Comparing the damaged fly with these counterfactual copies measures how
much of that fly's own deficit each lost class, cell type or neuron accounts for.

Two design choices make the comparisons exact and cheap.

* Common random numbers. Every variant of one fly, in every batch, receives the same
  Poisson input spikes, so a difference between two variants is caused by the
  counterfactual alone and needs no replicates to be resolved.
* Shared weights. Restoring a removed neuron changes only its threshold, so all
  variants of one fly that differ only in which neurons are restored share one weight
  vector; a batch of many variants costs little more memory than one fly. Variants that
  restore synapses use a second, healthy weight vector. Learning rescales weights per
  brain, so the full protocol (which measures the learning index) gives every brain its
  own copy instead.

Nothing here is observed data; every number is a property of the model.
"""
from dataclasses import dataclass

import numpy as np
from scipy import sparse

from . import pathology
from .lif import Brains, Params, poisson_events
from .model import DA, ODORS, TAU_A, W_SYN, learn, valence

DT = 0.1
SIGN = {"acetylcholine": 1, "gaba": -1, "glutamate": -1}
READOUTS_SHORT = ["brain_spikes_A", "LH_hz_A", "PN_hz_A", "KC_hz_A", "KC_frac_A", "MBON_hz_A",
                  "V_pre_A", "MN9_sugar_hz", "GF_loom_hz", "GF_latency_ms"]
READOUTS_FULL = READOUTS_SHORT + ["LI"]
# readouts used to choose what to screen further, and the neurons each one is read from
SELECT = ["brain_spikes_A", "LH_hz_A", "KC_hz_A", "MBON_hz_A", "MN9_sugar_hz", "GF_loom_hz"]


# ---------------------------------------------------------------- one fly's damage
@dataclass
class Damage:
    seed: int
    stage: float
    w: np.ndarray        # damaged weights, float32 [E], antennal-lobe edges zeroed
    dth: np.ndarray      # threshold shift, float32 [N]
    dead: np.ndarray     # removed neurons, bool [N]


def fly_damage(cx, seed, stage, stage_kw=None):
    """Rebuild one fly's pathology exactly as model.build_brains draws it."""
    rng = np.random.default_rng(10_000 + int(seed))
    st = pathology.Stage(s=float(stage), **(stage_kw or {}))
    w, dth, dead, _ = pathology.apply(st, rng, cx.row_ptr, cx.dst, cx.signed, W_SYN, cx.vuln, cx.is_kc,
                                      ~cx.sensory)
    w[cx.al_edge] = 0.0
    return Damage(int(seed), float(stage), w, dth.astype(np.float32), dead)


def healthy_weights(cx):
    w = cx.signed.astype(np.float32) * np.float32(W_SYN)
    w[cx.al_edge] = 0.0
    return w


def scale_weights(cx, w, w_scale=1.0, glu_scale=1.0):
    """Model variants for robustness: all weights scaled, or glutamatergic inhibition scaled."""
    if w_scale == 1.0 and glu_scale == 1.0:
        return w
    w = w * np.float32(w_scale)
    if glu_scale != 1.0:
        glu = (cx.neu.consensus_nt.to_numpy() == "glutamate")[cx.src]
        w[glu] *= np.float32(glu_scale)
    return w


# ---------------------------------------------------------------- units of rescue
def unit_labels(cx):
    """Class and cell-type label of every neuron; unlabelled neurons are grouped by superclass/class."""
    sc = cx.neu.superclass.fillna("").astype(str).to_numpy()
    cls = cx.cls.astype(str).to_numpy()
    cls_u = np.where(cls != "", cls, np.char.add("sc:", sc.astype(str)))
    typ = cx.t.astype(str).to_numpy()
    typ_u = np.where(typ != "", typ, np.char.add("untyped:", cls_u.astype(str)))
    return {"class": cls_u.astype(object), "type": typ_u.astype(object)}


def group(indices, labels):
    """{label: neuron indices} for the given neurons."""
    out = {}
    for i, lab in zip(indices, labels[indices]):
        out.setdefault(lab, []).append(i)
    return {k: np.asarray(v, np.int64) for k, v in out.items()}


# ---------------------------------------------------------------- variants of one fly
@dataclass
class Variant:
    label: str                       # what was undone
    level: str                       # baseline | class | type | neuron | subset | cumulative | lesion
    unit: str
    wset: int = 0                    # 0 damaged weights, 1 healthy weights
    restore: np.ndarray = None       # removed neurons to bring back
    shift: bool = True               # apply the threshold shift
    dead: bool = True                # apply the fly's neuron removal
    lesion: np.ndarray = None        # neurons to remove in addition (sufficiency tests)


def baseline_variants(dmg):
    dead_idx = np.flatnonzero(dmg.dead)
    return [Variant("damaged", "baseline", "damaged"),
            Variant("restore_dead_all", "baseline", "restore_dead_all", restore=dead_idx),
            Variant("restore_syn_all", "baseline", "restore_syn_all", wset=1),
            Variant("no_shift", "baseline", "no_shift", shift=False),
            Variant("healthy", "baseline", "healthy", wset=1, shift=False, dead=False)]


def vth_vector(v, dmg, v_th):
    vth = np.full(dmg.dead.size, v_th, np.float32)
    if v.shift:
        vth -= dmg.dth
    if v.dead:
        dead = dmg.dead.copy()
        if v.restore is not None and len(v.restore):
            dead[v.restore] = False
        vth[dead] = np.inf
    if v.lesion is not None and len(v.lesion):
        vth[v.lesion] = np.inf
    return vth


# ---------------------------------------------------------------- the simulation runner
class Runner:
    """One batch of B brains on a shared connectivity structure, reused across batches.

    protocol "short": odour A 1 s | gap 1 s | sugar 0.5 s | gap 1 s | looming 0.3 s (3.8 s);
    protocol "full": the assay battery of the paper, including training and the memory
    test (15.8 s), which adds the learning index.
    """

    def __init__(self, cx, B, protocol="short", sparse=False, da_scale=1.0):
        assert protocol in ("short", "full")
        self.cx, self.B, self.protocol = cx, int(B), protocol
        E = cx.signed.size
        self.br = Brains(self.B, params=Params(tau_a=TAU_A, w_syn=W_SYN), pack=cx.pack,
                         wsets=np.zeros((1, E), np.float32), wset_of=np.zeros(self.B, np.int64),
                         sparse=sparse)
        self.br.da[:, ~cx.sensory] = DA * da_scale
        t = cx.t
        self.pnA, self.pnB, self.pam = cx.upn(ODORS["A"]), cx.upn(ODORS["B"]), cx.pam
        self.grn = cx.idx(["LB3c", "LB3b"])
        self.loom = cx.idx(["LPLC2", "LC4"], ("L",))
        self.br.no_refractory(np.concatenate([self.pnA, self.pnB, self.pam, self.grn, self.loom]))
        self.kc = np.flatnonzero(cx.is_kc)
        self.pn = np.flatnonzero((cx.cls == "ALPN").to_numpy())
        self.lh = np.flatnonzero(t.str.startswith("LH").to_numpy())
        self.mn9, self.gf = cx.idx(["MN9"]), cx.idx(["DNp01"])
        self.g_mn9 = np.full(cx.N, -1, np.int32)
        self.g_mn9[self.mn9] = 0
        self.g_gf = np.full(cx.N, -1, np.int32)
        self.g_gf[self.gf] = 0
        self.g_all = np.arange(cx.N, dtype=np.int32)
        self.g_none = np.full(cx.N, -1, np.int32)
        self.dan_types = sorted(set(t.to_numpy()[self.pam]))
        self.dan_masks = {d: (t == d).to_numpy() for d in self.dan_types}
        self.readouts = READOUTS_FULL if protocol == "full" else READOUTS_SHORT

    # -- one segment, identical input spikes for every brain of the batch
    def _seg(self, rng, idx, rates, ms, grp, n_grp, bin_ms=None):
        n = int(round(ms / DT))
        br, B = self.br, self.B
        br.set_groups(grp, n_grp)
        bs = None if bin_ms is None else int(round(bin_ms / DT))
        if idx is None or len(idx) == 0:
            return br.run(n, np.zeros((B, 0), np.int32), np.zeros((B, n, 0), bool), bin_steps=bs)
        si = np.tile(np.asarray(idx, np.int32), (B, 1))
        ev = poisson_events(rng, np.asarray(rates, float)[None, :], n, DT)      # (1, n, M)
        return br.run(n, si, np.ascontiguousarray(np.repeat(ev, B, axis=0)), bin_steps=bs)

    def _gap(self, rng, ms):
        return self._seg(rng, None, None, ms, self.g_none, 1)

    def run(self, sets, variants, dmg, input_seed):
        """Simulate up to B variants of one fly; returns one dict of readouts per variant."""
        n = len(variants)
        assert 0 < n <= self.B
        pad = list(variants) + [variants[0]] * (self.B - n)
        br = self.br
        if self.protocol == "full":
            br.wsets = np.stack([sets[v.wset] for v in pad])          # own copy per brain
            br.wset_of = np.arange(self.B, dtype=np.int64)
        else:
            br.wsets = np.ascontiguousarray(np.stack(sets))
            br.wset_of = np.array([v.wset for v in pad], np.int64)
        br.vth = np.stack([vth_vector(v, dmg, br.p.v_th) for v in pad])
        br.reset_state()
        out = self._protocol(np.random.default_rng(int(input_seed)))
        return [{k: float(out[k][b]) for k in self.readouts} for b in range(n)]

    def _protocol(self, rng):
        cx, hz = self.cx, (lambda idx: np.full(len(idx), 150.0))
        preA = self._seg(rng, self.pnA, hz(self.pnA), 1000, self.g_all, cx.N)[:, 0, :].astype(np.float64)
        self._gap(rng, 1000)
        if self.protocol == "full":
            preB = self._seg(rng, self.pnB, hz(self.pnB), 1000, self.g_all, cx.N)[:, 0, :].astype(np.float64)
            self._gap(rng, 1000)
        sug = self._seg(rng, self.grn, hz(self.grn), 500, self.g_mn9, 1)[:, 0, 0] / 0.5 / max(len(self.mn9), 1)
        self._gap(rng, 1000)
        gfb = self._seg(rng, self.loom, hz(self.loom), 300, self.g_gf, 1, bin_ms=1.0)[:, :, 0]
        out = {"brain_spikes_A": preA.sum(1),
               "LH_hz_A": preA[:, self.lh].mean(1) if len(self.lh) else np.full(self.B, np.nan),
               "PN_hz_A": preA[:, self.pn].mean(1),
               "KC_hz_A": preA[:, self.kc].mean(1),
               "KC_frac_A": (preA[:, self.kc] > 0).mean(1),
               "MBON_hz_A": preA[:, cx.mbon].mean(1),
               "V_pre_A": valence(cx, preA),
               "MN9_sugar_hz": sug.astype(np.float64),
               "GF_loom_hz": gfb.sum(1) / 0.3 / max(len(self.gf), 1),
               "GF_latency_ms": np.array([float(np.flatnonzero(r)[0]) if r.any() else np.nan for r in gfb])}
        if self.protocol == "full":
            self._gap(rng, 1000)
            pair_idx = np.concatenate([self.pnA, self.pam])
            rates = np.concatenate([hz(self.pnA), np.full(len(self.pam), 100.0)])
            pair = self._seg(rng, pair_idx, rates, 1000, self.g_all, cx.N)[:, 0, :]
            self._gap(rng, 2000)
            self._seg(rng, self.pnB, hz(self.pnB), 1000, self.g_none, 1)
            self._gap(rng, 2000)
            dan_rate = [{d: float(pair[b, m].mean()) for d, m in self.dan_masks.items()} for b in range(self.B)]
            learn(cx, self.br, pair, dan_rate)
            postA = self._seg(rng, self.pnA, hz(self.pnA), 1000, self.g_all, cx.N)[:, 0, :].astype(np.float64)
            self._gap(rng, 1000)
            postB = self._seg(rng, self.pnB, hz(self.pnB), 1000, self.g_all, cx.N)[:, 0, :].astype(np.float64)
            vpa, vpb, vqa, vqb = (valence(cx, r) for r in (preA, preB, postA, postB))
            out["LI"] = (vqa - vpa) - (vqb - vpb)
        return out


# ---------------------------------------------------------------- structure-only predictors
class GraphPredictors:
    """What the wiring alone predicts about the importance of a set of lost neurons.

    For each readout, the target set is the neurons the readout is read from. For a lost
    neuron i, `direct` is its contact count onto the target set and `two_step` the sum over
    intermediate neurons k of contacts(i, k) x contacts(k, target set). A unit's predictor is
    the sum over its lost neurons. If counterfactual importance were explained by these, the
    simulation would add nothing beyond the graph.
    """

    def __init__(self, cx, runner):
        n = cx.N
        absw = np.abs(cx.signed).astype(np.float64)
        self.A = sparse.csr_matrix((absw, cx.dst, cx.row_ptr), shape=(n, n))
        out_syn = np.asarray(self.A.sum(1)).ravel()
        in_syn = np.asarray(self.A.sum(0)).ravel()
        self.syn = out_syn + in_syn
        self.vuln = cx.vuln
        nt = cx.neu.consensus_nt.map(SIGN).fillna(0).to_numpy()
        self.inhib = nt < 0
        targets = {"brain_spikes_A": None, "LH_hz_A": runner.lh, "KC_hz_A": runner.kc,
                   "MBON_hz_A": cx.mbon, "MN9_sugar_hz": runner.mn9, "GF_loom_hz": runner.gf}
        self.direct, self.two = {}, {}
        for r, tg in targets.items():
            if tg is None:                           # whole brain: every neuron is a target
                d1 = out_syn
            else:
                t = np.zeros(n)
                t[tg] = 1.0
                d1 = self.A @ t
            self.direct[r] = d1
            self.two[r] = self.A @ d1

    def unit_row(self, idx):
        row = {"n_lost": int(len(idx)), "syn_lost": float(self.syn[idx].sum()),
               "frac_inhib": float(self.inhib[idx].mean()) if len(idx) else np.nan,
               "mean_rank": float(self.vuln[idx].mean()) if len(idx) else np.nan}
        for r in self.direct:
            row[f"direct_{r}"] = float(self.direct[r][idx].sum())
            row[f"twostep_{r}"] = float(self.two[r][idx].sum())
        return row
