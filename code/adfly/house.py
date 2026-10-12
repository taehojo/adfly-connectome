"""Fly house: healthy and AD-stage connectome flies in one closed-loop arena.

Every fly runs its own whole-CNS LIF brain (adfly.model). Each 10 ms body
step the arena sets sensory drive, the brain runs 100 steps of 0.1 ms, and
brain outputs set behaviour:

  food patch    sugar GRNs (LB3c, LB3b) -> MN9 rate > feed_hz -> stop and feed
  looming       LPLC2 + LC4 on the side facing the shadow -> giant fiber
                (DNp01) spikes -> take off away from the shadow
  odor A / B    uniglomerular PNs, rate ~ concentration -> MBON ensemble
                valence (Aso et al. 2014) -> steer up (valence > 0) or down
                (valence < 0) the local odor gradient

Walking itself is a body-level correlated random walk (not read from the
brain), hunger is a body-level variable that scales sugar drive, and the
odor gradient direction is given to the body controller; the sign and
strength of odor steering, feeding and escape come from the connectome.
"""
from dataclasses import dataclass, field

import numpy as np

from .lif import poisson_events
from .model import ODORS, Connectome, Fly, build_brains, learn, valence

DT_BODY = 10.0          # ms
STEPS = 100             # brain steps per body step (0.1 ms each)


@dataclass
class Arena:
    R: float = 50.0                                     # mm
    food: tuple = ((0.0, 27.0, 8.0), (0.0, -27.0, 8.0))
    odor_src: dict = field(default_factory=lambda: {"A": (-21.0, 0.0), "B": (21.0, 0.0)})
    sigma: float = 14.0                                 # odor plume width, mm
    shadow_dur: float = 0.6                             # s, looming expansion time
    shadow_rmax: float = 24.0                           # mm


@dataclass
class Body:
    walk_speed: float = 9.0       # mm/s
    turn_sd: float = 2.2          # rad/sqrt(s), heading diffusion
    odor_gain: float = 6.0        # rad/s at saturating valence
    v_scale: float = 30.0         # Hz, learned-valence saturation scale
    feed_hz: float = 10.0         # MN9 rate that triggers feeding (half the healthy mean response)
    gf_spikes: int = 2            # giant-fiber spikes in 30 ms that trigger take-off
    jump_mm: float = 22.0
    air_s: float = 0.35
    hunger_feed_rate: float = 0.25   # hunger drop per s of feeding
    hunger_recover: float = 0.03     # hunger recovery per s


class House:
    def __init__(self, cx: Connectome, flies, arena=Arena(), body=Body(), seed=0):
        self.cx, self.flies, self.a, self.body = cx, flies, arena, body
        self.rng = np.random.default_rng(seed)
        self.br = build_brains(cx, flies)
        self.B = len(flies)
        t = cx.t
        self.grn = cx.idx(["LB3c", "LB3b"])
        self.pn = {o: cx.upn(g) for o, g in ODORS.items()}
        self.loom = {s: cx.idx(["LPLC2", "LC4"], (s,)) for s in ("L", "R")}
        self.pam = cx.pam
        # stimulus columns: one fixed index list for all flies
        self.cols = np.concatenate([self.grn, self.pn["A"], self.pn["B"],
                                    self.loom["L"], self.loom["R"], self.pam]).astype(np.int32)
        o = np.cumsum([0, self.grn.size, self.pn["A"].size, self.pn["B"].size,
                       self.loom["L"].size, self.loom["R"].size, self.pam.size])
        self.sl = {"grn": slice(o[0], o[1]), "A": slice(o[1], o[2]), "B": slice(o[2], o[3]),
                   "L": slice(o[3], o[4]), "R": slice(o[4], o[5]), "pam": slice(o[5], o[6])}
        self.br.no_refractory(self.cols)
        # output groups: 0 MN9, 1 GF, 2.. MBON neurons, then display groups
        grp = np.full(cx.N, -1, np.int32)
        grp[cx.idx(["MN9"])] = 0
        grp[cx.idx(["DNp01"])] = 1
        grp[cx.mbon] = 2 + np.arange(cx.mbon.size)
        self.disp_names = ["KC", "PN", "MBON", "DAN", "LPLC2/LC4", "GF", "MN9", "GRN", "VNC", "other"]
        base = 2 + cx.mbon.size
        dmap = {
            "KC": cx.is_kc, "PN": (cx.cls == "ALPN").to_numpy(),
            "DAN": t.str.match(r"^(PAM|PPL1)").to_numpy(),
            "LPLC2/LC4": t.isin(["LPLC2", "LC4"]).to_numpy(),
            "GRN": t.isin(["LB3c", "LB3b"]).to_numpy(),
            "VNC": cx.neu.superclass.str.startswith("vnc").to_numpy(),
        }
        self.disp_grp = np.full(cx.N, len(self.disp_names) - 1, np.int32)
        for k, m in dmap.items():
            self.disp_grp[m] = self.disp_names.index(k)
        self.disp_grp[cx.mbon] = self.disp_names.index("MBON")
        self.disp_grp[cx.idx(["DNp01"])] = self.disp_names.index("GF")
        self.disp_grp[cx.idx(["MN9"])] = self.disp_names.index("MN9")
        self.n_disp = np.bincount(self.disp_grp, minlength=len(self.disp_names))
        # combined grouping: outputs take priority, display groups for the rest
        g2 = np.where(grp >= 0, grp, base + self.disp_grp)
        self.base = base
        self.out_grp = grp
        self.br.set_groups(g2, base + len(self.disp_names))
        self.br_counts_hist = []

    # ------------------------------------------------------------------ brain I/O
    def _brain(self, rates):
        ev = poisson_events(self.rng, rates, STEPS, 0.1)
        si = np.tile(self.cols, (self.B, 1))
        return self.br.run(STEPS, si, ev)[:, 0, :]  # (B, G) spikes in 10 ms

    def train(self, eta=0.6):
        """Paired odor A + PAM (1 s), then odor B alone (1 s), with 2 s gaps."""
        cx = self.cx
        kc_sp = np.zeros((self.B, cx.N))
        rates = np.zeros((self.B, self.cols.size))
        # record KC and PAM spikes during pairing through a temporary grouping
        g_save = (self.br.grp.copy(), self.br.n_grp)
        self.br.set_groups(np.arange(cx.N), cx.N)
        rates[:, self.sl["A"]] = 150.0
        rates[:, self.sl["pam"]] = 100.0
        for _ in range(100):
            kc_sp += self._brain(rates)
        rates[:] = 0
        for _ in range(200):
            self._brain(rates)
        rates[:, self.sl["B"]] = 150.0
        for _ in range(100):
            self._brain(rates)
        rates[:] = 0
        for _ in range(200):
            self._brain(rates)
        self.br.grp, self.br.n_grp = g_save
        dan_types = sorted(set(cx.t.to_numpy()[cx.pam]))
        dan_rate = [{d: float(kc_sp[b, (cx.t == d).to_numpy()].mean()) for d in dan_types}
                    for b in range(self.B)]
        learn(cx, self.br, kc_sp, dan_rate, eta=eta)
        return dan_rate

    # ------------------------------------------------------------------ arena
    def run(self, seconds, threats=(), frame_ms=50.0, start=None):
        a, bd, B = self.a, self.body, self.B
        rng = self.rng
        n_body = int(round(seconds * 1000 / DT_BODY))
        fr_every = int(round(frame_ms / DT_BODY))
        if start is None:
            r = a.R * 0.8 * np.sqrt(rng.random(B))
            ang = rng.random(B) * 2 * np.pi
            pos = np.c_[r * np.cos(ang), r * np.sin(ang)]
        else:
            pos = np.array(start, float)
        th = rng.random(B) * 2 * np.pi
        hunger = np.ones(B)
        air_t = np.zeros(B)
        air_v = np.zeros((B, 2))
        last_jump = np.full(B, -9.0)
        hist = []           # per body step output counts
        frames = []
        stats = {k: np.zeros(B) for k in ("feed_s", "jumps", "threat_hits", "near_A_s", "near_B_s")}
        threat_seen = [set() for _ in range(B)]
        for k in range(n_body):
            tsec = k * DT_BODY / 1000.0
            rates = np.zeros((B, self.cols.size))
            on_food = np.zeros(B, bool)
            for fx, fy, fr in a.food:
                on_food |= np.hypot(pos[:, 0] - fx, pos[:, 1] - fy) < fr
            grounded = air_t <= 0
            rates[:, self.sl["grn"]] = (150.0 * hunger * (on_food & grounded))[:, None]
            conc = {}
            for o, (sx, sy) in a.odor_src.items():
                d2 = (pos[:, 0] - sx) ** 2 + (pos[:, 1] - sy) ** 2
                conc[o] = np.exp(-d2 / (2 * a.sigma ** 2))
                rates[:, self.sl[o]] = (150.0 * conc[o])[:, None]
            # looming shadows
            shadow = []
            for i, (t0, sx, sy) in enumerate(threats):
                if t0 <= tsec < t0 + a.shadow_dur:
                    rad = 2.0 + (a.shadow_rmax - 2.0) * (tsec - t0) / a.shadow_dur
                    shadow.append((sx, sy, rad))
                    dx, dy = sx - pos[:, 0], sy - pos[:, 1]
                    inside = np.hypot(dx, dy) < rad + 6.0
                    rel = np.arctan2(dy, dx) - th
                    left = np.sin(rel) > 0
                    rates[:, self.sl["L"]] += (150.0 * (inside & left & grounded))[:, None]
                    rates[:, self.sl["R"]] += (150.0 * (inside & ~left & grounded))[:, None]
                    for b in np.flatnonzero(inside):
                        threat_seen[b].add(i)
            c = self._brain(np.minimum(rates, 300.0))
            hist.append(c[:, :self.base].astype(np.int16))
            disp = c[:, self.base:]
            # readouts
            h = np.stack(hist[-30:])                       # (<=30, B, base)
            mn9_hz = h[-10:, :, 0].sum(0) / (min(len(hist), 10) * DT_BODY / 1000) / 2.0
            gf_sp = h[-3:, :, 1].sum(0)
            mb_hz = h[-30:, :, 2:].sum(0) / (min(len(hist), 30) * DT_BODY / 1000)
            val = (mb_hz * self.cx.mbon_sign[None, :]).sum(1)
            # behaviour
            jump = grounded & (gf_sp >= bd.gf_spikes) & (tsec - last_jump > 1.0)
            for b in np.flatnonzero(jump):
                if shadow:
                    sx, sy, _ = shadow[0]
                    away = np.arctan2(pos[b, 1] - sy, pos[b, 0] - sx)
                else:
                    away = th[b]
                away += rng.normal(0, 0.5)
                air_t[b] = bd.air_s
                air_v[b] = bd.jump_mm / bd.air_s * np.array([np.cos(away), np.sin(away)])
                th[b] = away
                last_jump[b] = tsec
                stats["jumps"][b] += 1
            feeding = grounded & on_food & (mn9_hz > bd.feed_hz) & ~jump
            walking = grounded & ~feeding & ~jump
            dt = DT_BODY / 1000.0
            # odor steering from MB valence
            gx = sum(conc[o] * (a.odor_src[o][0] - pos[:, 0]) for o in conc)
            gy = sum(conc[o] * (a.odor_src[o][1] - pos[:, 1]) for o in conc)
            has_grad = np.hypot(gx, gy) > 1e-3
            phi = np.arctan2(gy, gx)
            steer = bd.odor_gain * np.tanh(val / bd.v_scale) * np.sin(phi - th) * has_grad
            th = th + walking * (steer * dt + rng.normal(0, bd.turn_sd * np.sqrt(dt), B))
            step = walking * bd.walk_speed * dt
            pos = pos + np.c_[np.cos(th), np.sin(th)] * step[:, None]
            airborne = air_t > 0
            pos[airborne] += air_v[airborne] * dt
            air_t = np.maximum(air_t - dt, 0)
            # walls
            rr = np.hypot(pos[:, 0], pos[:, 1])
            out = rr > a.R - 1.5
            if out.any():
                pos[out] *= ((a.R - 1.5) / rr[out])[:, None]
                th[out] = np.arctan2(-pos[out, 1], -pos[out, 0]) + rng.normal(0, 0.6, out.sum())
                air_t[out] = 0
            hunger = np.clip(hunger - feeding * bd.hunger_feed_rate * dt + (~feeding) * bd.hunger_recover * dt, 0, 1)
            stats["feed_s"] += feeding * dt
            stats["near_A_s"] += (conc["A"] > np.exp(-0.5)) * dt
            stats["near_B_s"] += (conc["B"] > np.exp(-0.5)) * dt
            if k % fr_every == 0:
                frames.append({
                    "t": round(tsec, 3),
                    "x": np.round(pos[:, 0], 2).tolist(), "y": np.round(pos[:, 1], 2).tolist(),
                    "th": np.round(th, 3).tolist(),
                    "s": [("air" if air_t[b] > 0 else "feed" if feeding[b] else "walk") for b in range(B)],
                    "mn9": np.round(mn9_hz, 1).tolist(), "val": np.round(val, 1).tolist(),
                    "act": (disp / np.maximum(self.n_disp, 1)[None, :] * 100).round(2).tolist(),
                    "shadow": [[round(s[0], 1), round(s[1], 1), round(s[2], 1)] for s in shadow],
                })
        stats["threats_seen"] = np.array([len(s) for s in threat_seen])
        return frames, {k: v.tolist() for k, v in stats.items()}
