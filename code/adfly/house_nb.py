"""Fused fly-house kernel: arena, senses, whole-CNS LIF brain and body in one numba loop.

Flies do not interact, so each fly runs start to finish on its own thread
(prange over flies) with no per-step synchronisation. The brain update is
identical to lif._run (Shiu et al. LIF with Brian2 refractory gating and the
adaptation extension); behaviour rules are identical to adfly.house.House.
Random numbers come from a per-fly xorshift64* stream, so a run is
reproducible for a given seed and independent of thread scheduling.
"""
import math

import numpy as np
from numba import njit, prange

from .lif import poisson_events
from .model import ODORS, Connectome, build_brains, learn

K_GRN, K_A, K_B, K_LL, K_LR = 0, 1, 2, 3, 4


@njit(inline="always")
def _u01(st):
    x = st[0]
    x ^= x >> np.uint64(12)
    x ^= x << np.uint64(25)
    x ^= x >> np.uint64(27)
    st[0] = x
    return ((x * np.uint64(2685821657736338717)) >> np.uint64(11)) * (1.0 / 9007199254740992.0)


@njit(inline="always")
def _gauss(st):
    u1 = _u01(st)
    if u1 < 1e-300:
        u1 = 1e-300
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * _u01(st))


@njit(parallel=True, cache=True)
def _house(v, g, a, ref, buf, row_ptr, dst, wsets, wset_of, vth, rfc, da,
           v0, v_rst, em, es, cg, ea, ca, delay, stim_w, gate,
           in_idx, in_kind, out_id, mb_sign, disp_id, disp_n,
           arena, food, odor, threats, body, v0_tab, v0_lv,
           k0, n_body, spb, dt_ms, frame_every,
           st, ring, rng, seen,
           fx, fy, fth, fs, fmn9, fval, fact, stats):
    B, N = v.shape
    D = buf.shape[1]
    n_mb = mb_sign.shape[0]
    n_disp = disp_n.shape[0]
    RW = ring.shape[1]
    R, sigma, sh_dur, sh_rmax = arena[0], arena[1], arena[2], arena[3]
    walk, turn_sd, odor_gain, v_scale = body[0], body[1], body[2], body[3]
    feed_hz, gf_need, jump_mm, air_s = body[4], body[5], body[6], body[7]
    h_feed, h_rec, rmax = body[8], body[9], body[10]
    dtb = spb * dt_ms / 1000.0
    for b in prange(B):
        W = wsets[wset_of[b]]
        rs = rng[b:b + 1]
        spk = np.empty(N, np.int32)
        mbc = np.zeros(n_mb, np.int32)
        dc = np.zeros(n_disp, np.int32)
        rate = np.zeros(5)
        x, y, th = st[b, 0], st[b, 1], st[b, 2]
        hunger, air_t, avx, avy, last_jump = st[b, 3], st[b, 4], st[b, 5], st[b, 6], st[b, 7]
        for kk in range(n_body):
            k = k0 + kk
            tsec = k * dtb
            grounded = air_t <= 0.0
            # ---------------- senses
            on_food = False
            for f in range(food.shape[0]):
                if math.hypot(x - food[f, 0], y - food[f, 1]) < food[f, 2]:
                    on_food = True
            rate[K_GRN] = 150.0 * hunger if (on_food and grounded) else 0.0
            cA = math.exp(-((x - odor[0, 0]) ** 2 + (y - odor[0, 1]) ** 2) / (2 * sigma * sigma))
            cB = math.exp(-((x - odor[1, 0]) ** 2 + (y - odor[1, 1]) ** 2) / (2 * sigma * sigma))
            rate[K_A] = 150.0 * cA
            rate[K_B] = 150.0 * cB
            rate[K_LL] = 0.0
            rate[K_LR] = 0.0
            shadow_on = False
            shx, shy = 0.0, 0.0
            for ti in range(threats.shape[0]):
                t0 = threats[ti, 0]
                if t0 <= tsec < t0 + sh_dur:
                    rad = 2.0 + (sh_rmax - 2.0) * (tsec - t0) / sh_dur
                    dx, dy = threats[ti, 1] - x, threats[ti, 2] - y
                    if not shadow_on:
                        shadow_on = True
                        shx, shy = threats[ti, 1], threats[ti, 2]
                    if math.hypot(dx, dy) < rad + 6.0:
                        seen[b, ti] = True
                        if grounded:
                            if math.sin(math.atan2(dy, dx) - th) > 0:
                                rate[K_LL] += 150.0
                            else:
                                rate[K_LR] += 150.0
            for q in range(5):
                if rate[q] > rmax:
                    rate[q] = rmax
            # ---------------- brain: spb steps of dt_ms
            mn9c = 0
            gfc = 0
            for j in range(n_mb):
                mbc[j] = 0
            for j in range(n_disp):
                dc[j] = 0
            for s in range(spb):
                t = k * spb + s
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
                            spk[nspk] = i
                            nspk += 1
                    if ai != 0.0:
                        ai = ai * ea
                        if -1e-20 < ai < 1e-20:
                            ai = 0.0
                        a[b, i] = ai
                    xx = buf[b, slot, i]
                    if xx != 0.0:
                        if not (gate and r > 0):
                            g[b, i] += xx
                        buf[b, slot, i] = 0.0
                for m in range(in_idx.shape[0]):
                    p = rate[in_kind[m]] * dt_ms * 1e-3
                    if p > 0.0 and _u01(rs) < p:
                        v[b, in_idx[m]] += stim_w
                tslot = (t + delay) % D
                for q in range(nspk):
                    i = spk[q]
                    for e in range(row_ptr[i], row_ptr[i + 1]):
                        buf[b, tslot, dst[e]] += W[e]
                    v[b, i] = v_rst
                    g[b, i] = 0.0
                    a[b, i] += da[b, i]
                    ref[b, i] = rfc[i]
                    o = out_id[i]
                    if o == 0:
                        mn9c += 1
                    elif o == 1:
                        gfc += 1
                    elif o >= 2:
                        mbc[o - 2] += 1
                    dc[disp_id[i]] += 1
            # ---------------- readouts over ring windows of body steps
            w = k % RW
            ring[b, w, 0] = mn9c
            ring[b, w, 1] = gfc
            for j in range(n_mb):
                ring[b, w, 2 + j] = mbc[j]
            n10 = min(k + 1, 10)
            n30 = min(k + 1, RW)
            mn9_sum = 0
            for q in range(n10):
                mn9_sum += ring[b, (k - q) % RW, 0]
            mn9_hz = mn9_sum / (n10 * dtb) / 2.0
            gf_sp = ring[b, w, 1] + (ring[b, (k - 1) % RW, 1] if k >= 1 else 0) + (ring[b, (k - 2) % RW, 1] if k >= 2 else 0)
            val = 0.0
            for j in range(n_mb):
                c = 0
                for q in range(n30):
                    c += ring[b, (k - q) % RW, 2 + j]
                val += mb_sign[j] * c / (n30 * dtb)
            # ---------------- behaviour
            jump = False
            if grounded and gf_sp >= gf_need and tsec - last_jump > 1.0:
                jump = True
                if shadow_on:
                    away = math.atan2(y - shy, x - shx)
                else:
                    away = th
                away += 0.5 * _gauss(rs)
                air_t = air_s
                avx = jump_mm / air_s * math.cos(away)
                avy = jump_mm / air_s * math.sin(away)
                th = away
                last_jump = tsec
                stats[b, 1] += 1.0
            feeding = grounded and on_food and (mn9_hz > feed_hz) and not jump
            walking = grounded and (not feeding) and (not jump)
            # learned valence: this fly's MB valence minus its own naive response
            # to the same odor concentrations (linear interpolation over levels)
            base = 0.0
            for o in range(2):
                c = cA if o == 0 else cB
                nl = v0_lv.shape[0]
                bo = v0_tab[b, o, nl - 1]
                for q in range(nl - 1):
                    if v0_lv[q] <= c < v0_lv[q + 1]:
                        u = (c - v0_lv[q]) / (v0_lv[q + 1] - v0_lv[q])
                        bo = v0_tab[b, o, q] * (1.0 - u) + v0_tab[b, o, q + 1] * u
                        break
                base += bo
            lval = val - base
            if walking:
                gx = cA * (odor[0, 0] - x) + cB * (odor[1, 0] - x)
                gy = cA * (odor[0, 1] - y) + cB * (odor[1, 1] - y)
                steer = 0.0
                if math.hypot(gx, gy) > 1e-3:
                    steer = odor_gain * math.tanh(lval / v_scale) * math.sin(math.atan2(gy, gx) - th)
                th += steer * dtb + turn_sd * math.sqrt(dtb) * _gauss(rs)
                x += walk * dtb * math.cos(th)
                y += walk * dtb * math.sin(th)
            if air_t > 0.0:
                x += avx * dtb
                y += avy * dtb
                air_t = max(air_t - dtb, 0.0)
            rr = math.hypot(x, y)
            if rr > R - 1.5:
                x *= (R - 1.5) / rr
                y *= (R - 1.5) / rr
                th = math.atan2(-y, -x) + 0.6 * _gauss(rs)
                air_t = 0.0
            if feeding:
                hunger = max(hunger - h_feed * dtb, 0.0)
                stats[b, 0] += dtb
            else:
                hunger = min(hunger + h_rec * dtb, 1.0)
            if cA > math.exp(-0.5):
                stats[b, 2] += dtb
            if cB > math.exp(-0.5):
                stats[b, 3] += dtb
            # ---------------- frames
            if k % frame_every == 0:
                fi = k // frame_every
                if fi < fx.shape[1]:
                    fx[b, fi] = x
                    fy[b, fi] = y
                    fth[b, fi] = th
                    fs[b, fi] = 2 if air_t > 0.0 else (1 if feeding else 0)
                    fmn9[b, fi] = mn9_hz
                    fval[b, fi] = lval
                    for j in range(n_disp):
                        fact[b, fi, j] = dc[j] / max(disp_n[j], 1) / dtb
        st[b, 0], st[b, 1], st[b, 2] = x, y, th
        st[b, 3], st[b, 4], st[b, 5], st[b, 6], st[b, 7] = hunger, air_t, avx, avy, last_jump


class HouseNB:
    """Same arena and rules as adfly.house.House, executed by the fused kernel."""

    def __init__(self, cx: Connectome, flies, arena, body, seed=0):
        self.cx, self.flies, self.a, self.body = cx, flies, arena, body
        self.B = len(flies)
        self.seed = seed
        self.br = build_brains(cx, flies)
        t = cx.t
        grn = cx.idx(["LB3c", "LB3b"])
        pnA, pnB = cx.upn(ODORS["A"]), cx.upn(ODORS["B"])
        lL, lR = cx.idx(["LPLC2", "LC4"], ("L",)), cx.idx(["LPLC2", "LC4"], ("R",))
        self.pnA, self.pnB, self.pam = pnA, pnB, cx.pam
        self.in_idx = np.concatenate([grn, pnA, pnB, lL, lR]).astype(np.int32)
        self.in_kind = np.concatenate([np.full(grn.size, K_GRN), np.full(pnA.size, K_A), np.full(pnB.size, K_B),
                                       np.full(lL.size, K_LL), np.full(lR.size, K_LR)]).astype(np.int32)
        self.br.no_refractory(np.concatenate([self.in_idx, self.pam]))
        self.out_id = np.full(cx.N, -1, np.int32)
        self.out_id[cx.idx(["MN9"])] = 0
        self.out_id[cx.idx(["DNp01"])] = 1
        self.out_id[cx.mbon] = 2 + np.arange(cx.mbon.size)
        self.disp_names = ["KC", "PN", "MBON", "DAN", "LPLC2/LC4", "GF", "MN9", "GRN", "VNC", "other"]
        dmap = {"KC": cx.is_kc, "PN": (cx.cls == "ALPN").to_numpy(),
                "DAN": t.str.match(r"^(PAM|PPL1)").to_numpy(),
                "LPLC2/LC4": t.isin(["LPLC2", "LC4"]).to_numpy(),
                "GRN": t.isin(["LB3c", "LB3b"]).to_numpy(),
                "VNC": cx.neu.superclass.str.startswith("vnc").to_numpy()}
        self.disp_id = np.full(cx.N, len(self.disp_names) - 1, np.int32)
        for k, m in dmap.items():
            self.disp_id[m] = self.disp_names.index(k)
        self.disp_id[cx.mbon] = self.disp_names.index("MBON")
        self.disp_id[cx.idx(["DNp01"])] = self.disp_names.index("GF")
        self.disp_id[cx.idx(["MN9"])] = self.disp_names.index("MN9")
        self.disp_n = np.bincount(self.disp_id, minlength=len(self.disp_names)).astype(np.int32)

    def probe_naive(self, levels=(0.25, 0.5, 1.0), ms=1000.0, gap_ms=500.0, rng_seed=3):
        """Each fly's naive MB valence for odor A and B at several concentrations.

        Run before training. The house subtracts this baseline, so odor steering
        follows only what the fly learned (naive flies choose evenly in T-maze
        assays, while raw MBON sums in this model carry an innate bias).
        """
        br, cx, B = self.br, self.cx, self.B
        rng = np.random.default_rng(rng_seed)
        grp = np.full(cx.N, -1, np.int32)
        grp[cx.mbon] = np.arange(cx.mbon.size)
        br.set_groups(grp, cx.mbon.size)
        lv = np.array((0.0,) + tuple(levels))
        tab = np.zeros((B, 2, lv.size))
        n_on, n_gap = int(round(ms / 0.1)), int(round(gap_ms / 0.1))
        for o, pn in enumerate((self.pnA, self.pnB)):
            si = np.tile(pn.astype(np.int32), (B, 1))
            for q, c in enumerate(levels, start=1):
                cnt = br.run(n_on, si, poisson_events(rng, np.full(si.shape, 150.0 * c), n_on, 0.1))[:, 0, :]
                tab[:, o, q] = (cnt / (ms / 1000.0) * cx.mbon_sign[None, :]).sum(1)
                br.run(n_gap, si, np.zeros((B, n_gap, pn.size), bool))
        self.v0_tab, self.v0_lv = tab, lv
        return tab

    def train(self, eta=0.6, rng_seed=1):
        """Odor A + PAM (1 s), 2 s gap, odor B (1 s), 2 s gap; then KC->MBON depression."""
        br, cx, B = self.br, self.cx, self.B
        rng = np.random.default_rng(rng_seed)
        br.set_groups(np.arange(cx.N), cx.N)
        none = np.zeros((B, 0), np.int32)

        def seg(idx, rates, ms):
            n = int(round(ms / 0.1))
            if idx.size == 0:
                return br.run(n, none, np.zeros((B, n, 0), bool))[:, 0, :]
            si = np.tile(idx.astype(np.int32), (B, 1))
            return br.run(n, si, poisson_events(rng, np.tile(rates, (B, 1)), n, 0.1))[:, 0, :]

        pair = np.concatenate([self.pnA, self.pam])
        kc_sp = seg(pair, np.concatenate([np.full(self.pnA.size, 150.0), np.full(self.pam.size, 100.0)]), 1000)
        seg(np.array([], np.int32), None, 2000)
        seg(self.pnB, np.full(self.pnB.size, 150.0), 1000)
        seg(np.array([], np.int32), None, 2000)
        dan_types = sorted(set(cx.t.to_numpy()[cx.pam]))
        dan_rate = [{d: float(kc_sp[b, (cx.t == d).to_numpy()].mean()) for d in dan_types} for b in range(B)]
        learn(cx, br, kc_sp, dan_rate, eta=eta)
        return dan_rate

    def run(self, seconds, threats, frame_ms=50.0, chunk_s=5.0, progress=None):
        br, a, bd, B = self.br, self.a, self.body, self.B
        spb = 100
        dt_ms = br.p.dt
        n_body = int(round(seconds * 1000 / (spb * dt_ms)))
        frame_every = int(round(frame_ms / (spb * dt_ms)))
        n_frames = (n_body + frame_every - 1) // frame_every
        init = np.random.default_rng(self.seed)
        r0 = a.R * 0.8 * np.sqrt(init.random(B))
        ang = init.random(B) * 2 * np.pi
        st = np.zeros((B, 8))
        st[:, 0], st[:, 1] = r0 * np.cos(ang), r0 * np.sin(ang)
        st[:, 2] = init.random(B) * 2 * np.pi
        st[:, 3] = 1.0
        st[:, 7] = -9.0
        ring = np.zeros((B, 30, 2 + self.cx.mbon.size), np.int32)
        rng = (np.arange(B, dtype=np.uint64) * np.uint64(0x9E3779B97F4A7C15) + np.uint64(self.seed * 7919 + 1)) | np.uint64(1)
        seen = np.zeros((B, max(len(threats), 1)), bool)
        fx = np.zeros((B, n_frames), np.float32); fy = np.zeros_like(fx); fth = np.zeros_like(fx)
        fs = np.zeros((B, n_frames), np.int8); fmn9 = np.zeros_like(fx); fval = np.zeros_like(fx)
        fact = np.zeros((B, n_frames, len(self.disp_names)), np.float32)
        stats = np.zeros((B, 4))
        arena = np.array([a.R, a.sigma, a.shadow_dur, a.shadow_rmax])
        food = np.array(a.food, float)
        odor = np.array([a.odor_src["A"], a.odor_src["B"]], float)
        thr = np.array(threats, float).reshape(-1, 3)
        body = np.array([bd.walk_speed, bd.turn_sd, bd.odor_gain, bd.v_scale, bd.feed_hz, bd.gf_spikes,
                         bd.jump_mm, bd.air_s, bd.hunger_feed_rate, bd.hunger_recover, 300.0])
        br.reset_state()
        chunk = int(round(chunk_s * 1000 / (spb * dt_ms)))
        for k0 in range(0, n_body, chunk):
            nb = min(chunk, n_body - k0)
            _house(br.v, br.g, br.a, br.ref, br.buf, br.row_ptr, br.dst, br.wsets, br.wset_of, br.vth,
                   br.rfc, br.da, np.float32(br.p.v_0), np.float32(br.p.v_rst), np.float32(br.em),
                   np.float32(br.es), np.float32(br.cg), np.float32(br.ea), np.float32(br.ca),
                   br.p.delay_steps, np.float32(br.p.w_syn * br.p.f_poi), br.gate,
                   self.in_idx, self.in_kind, self.out_id, self.cx.mbon_sign.astype(np.float64),
                   self.disp_id, self.disp_n, arena, food, odor, thr, body, self.v0_tab, self.v0_lv,
                   k0, nb, spb, dt_ms, frame_every, st, ring, rng, seen,
                   fx, fy, fth, fs, fmn9, fval, fact, stats)
            if progress:
                progress((k0 + nb) * spb * dt_ms / 1000.0)
        names = {0: "walk", 1: "feed", 2: "air"}
        frames = []
        for fi in range(n_frames):
            tsec = fi * frame_ms / 1000.0
            sh = []
            for (t0, sx, sy) in threats:
                if t0 <= tsec < t0 + a.shadow_dur:
                    sh.append([sx, sy, round(2.0 + (a.shadow_rmax - 2.0) * (tsec - t0) / a.shadow_dur, 1)])
            frames.append({"t": round(tsec, 3),
                           "x": np.round(fx[:, fi], 2).tolist(), "y": np.round(fy[:, fi], 2).tolist(),
                           "th": np.round(fth[:, fi], 3).tolist(),
                           "s": [names[int(q)] for q in fs[:, fi]],
                           "mn9": np.round(fmn9[:, fi], 1).tolist(), "val": np.round(fval[:, fi], 1).tolist(),
                           "act": np.round(fact[:, fi, :], 2).tolist(), "shadow": sh})
        out = {"feed_s": stats[:, 0].tolist(), "jumps": stats[:, 1].tolist(),
               "near_A_s": stats[:, 2].tolist(), "near_B_s": stats[:, 3].tolist(),
               "threats_seen": seen.sum(1).tolist()}
        return frames, out
