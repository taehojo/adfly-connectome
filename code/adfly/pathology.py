"""Aβ42-inspired pathology applied to a connectome LIF brain.

Each neuron i gets a pathology load l_i in [0, 1] that grows with disease
stage s in [0, 1]. Three effects, each tied to findings in Aβ42 flies:

  hyperexcitability   Aβ42 degrades the A-type K+ channel Kv4 and raises
                      neuronal excitability (Ping et al. 2015, PLoS Genet).
                      Modelled as a threshold shift v_th,i = v_th - l_i * dth_max.
  synapse loss        Aβ42 reduces synapses (e.g. Mhatre et al. 2014). Each
                      output synapse of neuron i is kept with probability
                      1 - l_i * syn_loss_max (binomial thinning of contact counts).
  neuron loss         progressive neurodegeneration (Iijima et al. 2004).
                      Neuron i is removed with probability l_i * death_max
                      (v_th = +inf, so it no longer fires).

Vulnerability follows the hub hypothesis of amyloid deposition (Buckner et
al. 2009): u_i is the percentile rank of a neuron's total synapse count
(in + out). Load l_i = s * u_i for the pan-neuronal (elav-GAL4-like)
scenario; for the mushroom-body (OK107-like) scenario only Kenyon cells carry
load. All draws use a seeded generator, so each simulated fly is a
reproducible in silico individual. Nothing here is observed data.

Controls for the paper: any effect can be switched off by setting its
maximum to 0, and the vulnerability map can be "hub" (default), "shuffled"
(the same load values randomly reassigned among eligible neurons, per fly)
or "antihub" (low-degree neurons first).
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Stage:
    s: float                    # disease stage, 0 = healthy, 1 = end stage
    dth_max: float = 2.0        # mV threshold drop at full load
    syn_loss_max: float = 0.6   # fraction of output synapses lost at full load
    death_max: float = 0.5      # probability of neuron loss at full load
    scope: str = "pan"          # "pan" (elav-like) or "mb" (Kenyon cells only)
    vuln: str = "hub"           # "hub", "shuffled" or "antihub"


def hub_vulnerability(row_ptr, dst, signed):
    """Percentile rank (0..1] of total synapse count per neuron."""
    n = row_ptr.size - 1
    syn = np.abs(signed).astype(np.int64)
    out_deg = np.add.reduceat(syn, row_ptr[:-1]) if syn.size else np.zeros(n)
    out_deg[np.diff(row_ptr) == 0] = 0
    in_deg = np.bincount(dst, weights=syn, minlength=n)
    tot = out_deg + in_deg
    order = tot.argsort(kind="stable")
    rank = np.empty(n)
    rank[order] = np.arange(1, n + 1) / n
    return rank


def apply(stage, rng, row_ptr, dst, signed, w_syn, vuln, kc_mask, eligible):
    """Return (weights float32[E] in mV, threshold shift float32[N], dead bool[N], load[N]).

    eligible: neurons that can carry pathology (sensory neurons are excluded:
    their activity is set by the stimulus).
    """
    n = row_ptr.size - 1
    u = np.asarray(vuln, float)
    # "shuffled" and "antihub" reassign the SAME multiset of load values among the
    # eligible neurons that "hub" uses, so all three carry an identical expected
    # damage burden and differ only in which neurons receive which load. Reversing
    # the global rank instead would give the antihub map a smaller burden, because
    # sensory neurons are ineligible and sit toward the low-degree end.
    if stage.vuln in ("antihub", "shuffled"):
        idx = np.flatnonzero(eligible)
        u = u.copy()
        if stage.vuln == "shuffled":
            u[idx] = u[idx][rng.permutation(idx.size)]
        else:
            order = np.argsort(u[idx], kind="stable")      # eligible, ascending by degree
            u[idx[order]] = np.sort(u[idx])[::-1]          # least connected gets most load
    elif stage.vuln != "hub":
        raise ValueError(stage.vuln)
    load = stage.s * u
    if stage.scope == "mb":
        load = np.where(kc_mask, load, 0.0)
    load = np.where(eligible, load, 0.0)
    src = np.repeat(np.arange(n), np.diff(row_ptr))
    keep_p = 1.0 - load[src] * stage.syn_loss_max
    counts = np.abs(signed).astype(np.int64)
    kept = rng.binomial(counts, np.clip(keep_p, 0.0, 1.0))
    w = (np.sign(signed) * kept).astype(np.float32) * np.float32(w_syn)
    dead = rng.random(n) < load * stage.death_max
    dth = (load * stage.dth_max).astype(np.float32)
    return w, dth, dead, load
