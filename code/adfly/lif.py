"""Batched leaky integrate-and-fire engine for whole-CNS connectome simulation.

Reimplements the Shiu et al. (2024, Nature) Brian2 model with exact
(`linear`) integration at dt = 0.1 ms and Brian2's update order
(state update -> threshold -> synaptic delivery / Poisson input -> reset):

    dv/dt = (v_0 - v + g) / t_mbr      (frozen while refractory)
    dg/dt = -g / tau                   (frozen while refractory)
    spike: v > v_th  ->  v = v_rst, g = 0, refractory for t_rfc
    synapse: g_post += w after t_dly, w = signed synapse count * w_syn
    Poisson input: v += w_syn * f_poi per event, no refractoriness for targets

Extension (off by default, da = 0 reproduces Shiu et al. exactly): a
spike-frequency adaptation current a, dv/dt = (v_0 - v + g - a) / t_mbr,
da/dt = -a / tau_a, a += da_i at each spike.

Each fly in the batch has its own state, its own synaptic weight set (so
pathology and plasticity can differ between flies) and its own per-neuron
threshold (v_th = +inf removes a neuron).
"""
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numba import njit, prange

PACK = next(p for p in Path(__file__).resolve().parents if (p / "results").is_dir()) / "data/pack/malecns_v1.0"


@dataclass(frozen=True)
class Params:
    v_0: float = -52.0      # mV, Kakaria & de Bivort 2017
    v_rst: float = -52.0    # mV
    v_th: float = -45.0     # mV
    t_mbr: float = 20.0     # ms
    tau: float = 5.0        # ms, Jurgensen et al. 2021
    t_rfc: float = 2.2      # ms, Lazar et al. 2021
    t_dly: float = 1.8      # ms, Paul et al. 2015
    w_syn: float = 0.275    # mV per synapse (free parameter in Shiu et al.)
    f_poi: float = 250.0    # Poisson synapse scaling
    dt: float = 0.1         # ms
    tau_a: float = 1000.0   # ms, spike-frequency adaptation (extension; off when da = 0)

    @property
    def delay_steps(self):
        return int(round(self.t_dly / self.dt))

    @property
    def rfc_steps(self):
        return int(round(self.t_rfc / self.dt))


def load_pack(path=PACK):
    row_ptr = np.load(path / "row_ptr.npy")
    dst = np.load(path / "dst.npy")
    signed = np.load(path / "signed_w.npy")
    return row_ptr, dst, signed


@njit(parallel=True, fastmath=False, cache=True)
def _run(v, g, a, ref, buf, t0, n_steps, row_ptr, dst, wsets, wset_of, vth, rfc, da,
         v0, v_rst, em, es, cg, ea, ca, delay, stim_idx, stim_ev, stim_w,
         grp, grp_counts, bin_steps, spk_scratch, gate):
    B, N = v.shape
    D = buf.shape[1]
    for b in prange(B):
        W = wsets[wset_of[b]]
        for k in range(n_steps):
            t = t0 + k
            slot = t % D
            nspk = 0
            for i in range(N):
                r = ref[b, i]
                if r > 0:
                    r -= 1
                    ref[b, i] = r
                ai = a[b, i]
                if r == 0:
                    vi = v[b, i]
                    gi = g[b, i]
                    vi = v0 + (vi - v0) * em + gi * cg - ai * ca
                    v[b, i] = vi
                    gi = gi * es
                    if -1e-20 < gi < 1e-20:
                        gi = 0.0
                    g[b, i] = gi
                    if vi > vth[b, i]:
                        spk_scratch[b, nspk] = i
                        nspk += 1
                if ai != 0.0:
                    ai = ai * ea
                    if -1e-20 < ai < 1e-20:
                        ai = 0.0
                    a[b, i] = ai
                x = buf[b, slot, i]
                if x != 0.0:
                    # Brian2 semantics: `g (unless refractory)` also blocks
                    # synaptic writes while the neuron is refractory
                    if not (gate and r > 0):
                        g[b, i] += x
                    buf[b, slot, i] = 0.0
            for m in range(stim_idx.shape[1]):
                j = stim_idx[b, m]
                if j >= 0 and stim_ev[b, k, m]:
                    v[b, j] += stim_w
            tslot = (t + delay) % D
            for s in range(nspk):
                i = spk_scratch[b, s]
                for e in range(row_ptr[i], row_ptr[i + 1]):
                    buf[b, tslot, dst[e]] += W[e]
                v[b, i] = v_rst
                g[b, i] = 0.0
                a[b, i] += da[b, i]
                ref[b, i] = rfc[i]
                gi_ = grp[i]
                if gi_ >= 0:
                    grp_counts[b, k // bin_steps, gi_] += 1


@njit(parallel=True, fastmath=False, cache=True)
def _run_sparse(v, g, a, ref, buf, t0, n_steps, row_ptr, dst, wsets, wset_of, vth, rfc, da,
                v0, v_rst, em, es, cg, ea, ca, delay, stim_idx, stim_ev, stim_w,
                grp, grp_counts, bin_steps, spk_scratch, gate, act, n_act, in_act, eps, eps_a):
    """Same update as _run, but only visits neurons that are away from rest.

    A neuron at rest (v == v_0, g == 0, a == 0, not refractory, no pending
    input) is a fixed point of the update, so skipping it is exact. Neurons
    join the active list when they receive input or a stimulus and leave it
    once |v - v_0| and |g| fall below eps and |a| below eps_a (then snapped
    to rest; with eps = 1e-4 mV and eps_a = 1e-2 mV the snap is far below the
    7 mV threshold gap).
    """
    B, N = v.shape
    D = buf.shape[1]
    for b in prange(B):
        W = wsets[wset_of[b]]
        for k in range(n_steps):
            t = t0 + k
            slot = t % D
            nspk = 0
            # stimulated neurons must be visited this step
            for m in range(stim_idx.shape[1]):
                j = stim_idx[b, m]
                if j >= 0 and stim_ev[b, k, m] and not in_act[b, j]:
                    in_act[b, j] = True
                    act[b, n_act[b]] = j
                    n_act[b] += 1
            na = n_act[b]
            q = 0
            for p in range(na):
                i = act[b, p]
                r = ref[b, i]
                if r > 0:
                    r -= 1
                    ref[b, i] = r
                ai = a[b, i]
                if r == 0:
                    vi = v[b, i]
                    gi = g[b, i]
                    vi = v0 + (vi - v0) * em + gi * cg - ai * ca
                    v[b, i] = vi
                    gi = gi * es
                    if -1e-20 < gi < 1e-20:
                        gi = 0.0
                    g[b, i] = gi
                    if vi > vth[b, i]:
                        spk_scratch[b, nspk] = i
                        nspk += 1
                if ai != 0.0:
                    ai = ai * ea
                    if -1e-20 < ai < 1e-20:
                        ai = 0.0
                    a[b, i] = ai
                x = buf[b, slot, i]
                if x != 0.0:
                    if not (gate and r > 0):
                        g[b, i] += x
                    buf[b, slot, i] = 0.0
                # keep or drop from the active list
                pending = False
                for dd in range(D):
                    if buf[b, dd, i] != 0.0:
                        pending = True
                        break
                quiet = (r == 0 and abs(v[b, i] - v0) < eps and abs(g[b, i]) < eps
                         and abs(a[b, i]) < eps_a and not pending)
                if quiet:
                    v[b, i] = v0
                    g[b, i] = 0.0
                    a[b, i] = 0.0
                    in_act[b, i] = False
                else:
                    act[b, q] = i
                    q += 1
            n_act[b] = q
            for m in range(stim_idx.shape[1]):
                j = stim_idx[b, m]
                if j >= 0 and stim_ev[b, k, m]:
                    v[b, j] += stim_w
                    if not in_act[b, j]:
                        in_act[b, j] = True
                        act[b, n_act[b]] = j
                        n_act[b] += 1
            tslot = (t + delay) % D
            for s in range(nspk):
                i = spk_scratch[b, s]
                for e in range(row_ptr[i], row_ptr[i + 1]):
                    j = dst[e]
                    buf[b, tslot, j] += W[e]
                    if not in_act[b, j]:
                        in_act[b, j] = True
                        act[b, n_act[b]] = j
                        n_act[b] += 1
                v[b, i] = v_rst
                g[b, i] = 0.0
                a[b, i] += da[b, i]
                ref[b, i] = rfc[i]
                if not in_act[b, i]:
                    in_act[b, i] = True
                    act[b, n_act[b]] = i
                    n_act[b] += 1
                gi_ = grp[i]
                if gi_ >= 0:
                    grp_counts[b, k // bin_steps, gi_] += 1


class Brains:
    """A batch of B flies sharing one connectome topology."""

    def __init__(self, B, params=Params(), pack=PACK, wsets=None, wset_of=None,
                 gate_refractory_input=True, sparse=False, eps=1e-4, eps_a=1e-2):
        self.p = params
        self.sparse = bool(sparse)
        self.eps = float(eps)
        self.eps_a = float(eps_a)
        self.gate = bool(gate_refractory_input)
        self.row_ptr, self.dst, signed = load_pack(pack)
        self.N = self.row_ptr.size - 1
        self.B = B
        base = (signed.astype(np.float32) * np.float32(params.w_syn))
        self.wsets = base[None, :] if wsets is None else wsets
        self.wset_of = np.zeros(B, np.int64) if wset_of is None else np.asarray(wset_of, np.int64)
        p = params
        self.em = math.exp(-p.dt / p.t_mbr)
        self.es = math.exp(-p.dt / p.tau)
        # exact solution of the linear (v, g) system over one step
        self.cg = (p.tau / (p.tau - p.t_mbr)) * (self.es - self.em)
        # adaptation current a enters as dv/dt = (v_0 - v + g - a) / t_mbr
        self.ea = math.exp(-p.dt / p.tau_a)
        self.ca = (p.tau_a / (p.tau_a - p.t_mbr)) * (self.ea - self.em)
        self.da = np.zeros((B, self.N), np.float32)
        self.D = p.delay_steps + 1
        self.vth = np.full((B, self.N), p.v_th, np.float32)
        self.rfc = np.full(self.N, p.rfc_steps, np.int32)
        self.grp = np.full(self.N, -1, np.int32)
        self.n_grp = 1
        self.reset_state()

    def reset_state(self):
        B, N = self.B, self.N
        self.v = np.full((B, N), self.p.v_0, np.float32)
        self.g = np.zeros((B, N), np.float32)
        self.a = np.zeros((B, N), np.float32)
        self.ref = np.zeros((B, N), np.int32)
        self.buf = np.zeros((B, self.D, N), np.float32)
        self.t = 0
        self.spk_scratch = np.zeros((self.B, self.N), np.int32)
        self.act = np.zeros((B, N), np.int32)
        self.n_act = np.zeros(B, np.int64)
        self.in_act = np.zeros((B, N), np.bool_)

    def set_groups(self, grp, n_grp):
        self.grp = np.asarray(grp, np.int32)
        self.n_grp = int(n_grp)

    def no_refractory(self, idx):
        """Poisson targets have no refractory period, as in Shiu et al."""
        self.rfc[np.asarray(idx)] = 0

    def run(self, n_steps, stim_idx, stim_ev, bin_steps=None):
        """Advance n_steps. stim_idx: (B, M) int32 (-1 pads); stim_ev: (B, n_steps, M) bool.

        Returns spike counts per group in bins of bin_steps (default: one bin),
        shape (B, ceil(n_steps / bin_steps), n_grp).
        """
        bin_steps = n_steps if bin_steps is None else int(bin_steps)
        counts = np.zeros((self.B, -(-n_steps // bin_steps), self.n_grp), np.int32)
        args = (self.v, self.g, self.a, self.ref, self.buf, self.t, n_steps, self.row_ptr, self.dst,
                self.wsets, self.wset_of, self.vth, self.rfc, self.da,
                np.float32(self.p.v_0), np.float32(self.p.v_rst), np.float32(self.em),
                np.float32(self.es), np.float32(self.cg), np.float32(self.ea), np.float32(self.ca),
                self.p.delay_steps,
                stim_idx, stim_ev, np.float32(self.p.w_syn * self.p.f_poi),
                self.grp, counts, bin_steps, self.spk_scratch, self.gate)
        if self.sparse:
            _run_sparse(*args, self.act, self.n_act, self.in_act, np.float32(self.eps), np.float32(self.eps_a))
        else:
            _run(*args)
        self.t += n_steps
        return counts


def poisson_events(rng, rates_hz, n_steps, dt_ms):
    """Bernoulli approximation of Poisson events per step. rates_hz: (B, M)."""
    p = np.clip(np.asarray(rates_hz) * dt_ms * 1e-3, 0, 1)
    return rng.random((p.shape[0], n_steps, p.shape[1])) < p[:, None, :]
