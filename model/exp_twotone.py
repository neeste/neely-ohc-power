"""
exp_twotone.py

The observables as the experiments actually measured them, on the same line.

Everything to date holds the probe as a small signal on the suppressor's
line: suppression is read VERTICALLY (dB of read-out reduction at fixed
probe) at a criterion near 1 dB. The sources behind observations 2 and 3 did
neither.

  - Fahey-Allen (1985): the neural threshold is the suppressor level that
    removes the driven response to a probe 6 to 10 dB above the fiber's own
    threshold. That is an EQUIVALENT-ATTENUATION criterion of roughly 8 dB,
    not 1 dB.
  - Delgutte (1990): the 2 dB/dB growth is HORIZONTAL, the dB of probe level
    needed to hold a criterion response per dB of suppressor, measured with
    the probe driven 30 to 80 dB SPL across a compressive input-output
    function. In BM dB the same data are near 1 dB/dB.
  - Cooper (1996): the BM threshold is a 10 percent (about 1 dB) VERTICAL
    reduction of a probe at moderate level, tens of dB above threshold.
  - Ruggero (1992): suppression decreases with probe intensity, so the
    moderate-level BM probe and the near-threshold neural probe are not
    interchangeable read-outs.

None of that is expressible while the probe stays out of the saturator. This
file puts it in: the probe and the suppressor drive the same nonlinearity,

    u^2 = u_s^2 + u_p^2      (power sum, both at their own frequencies)
    s   = u^2 / (1 + u^2)

solved self-consistently as before, and the three observables are then
measured by their own protocols. The hypothesis on trial: observation 3's
threshold gap is largely a PROBE-LEVEL effect (the moderate-level BM probe
has already saturated the gain the suppressor would otherwise remove), and
observation 2's 2 dB/dB is the horizontal measure seen through the probe's
compressive input-output function. Both would then be properties the hybrid
line already has, mismeasured by the old scorer's protocol.

THE OPERATING POINT NOW MATTERS. The prior single-tone machinery could say
"u = 1 at threshold is a calibration that moves observation 1 dB for dB and
leaves 2 and 3 unchanged". With the probe inside the saturator that is no
longer true: where the neural threshold sits relative to the saturation knee
(u_thr) sets how compressed the probe I/O is at each SL, so u_thr is scanned,
and observations 2 and 3 become predictions as functions of it while
observation 1 remains the calibration that pins absolute SPL.

Run: python3 exp_twotone.py [--quick]
"""

from __future__ import annotations

import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tl_model as T                      # noqa: E402
import exp_accumulation as E              # noqa: E402

N_RUN = 1024
P_ORD = 2.0        # transducer order: s = (u^2)^(p/2) / (1 + (u^2)^(p/2)).
                   # p = 2 is the project's s = u^2/(1+u^2). The kinematics
                   # give d ln s/dL = (p/2) 0.2303 (1-s), so p multiplies
                   # every growth slope by p/2. Observation 2 is the
                   # constraint that sets it (build plan, step 5).
GAINS = (10.0, 20.0, 39.0)
U_THRS = (0.03, 0.1, 0.3, 1.0)
SL_NEURAL = 8.0                 # FA-85: probe 6 to 10 dB above threshold
SL_COOPER = (30.0, 40.0, 50.0)  # Cooper's probe, dB above the neural anchor
EA_CRIT = SL_NEURAL             # response pushed back to threshold
BM_CRIT = 1.0                   # Cooper's 10 percent
OCT_BELOW = 0.5
GROWTH_SPAN = (3.0, 21.0, 3.0)  # above neural threshold: lo, hi, step
TARGET_DT = (18.0, 32.0)
TARGET_SL = (1.4, 2.6)


def fmt(v, w=8, p=2):
    return f"{'-':>{w}}" if (v is None or (isinstance(v, float)
                                           and math.isnan(v))) else \
        f"{v:{w}.{p}f}"


# ---------------------------------------------------------------------------
# the two-tone self-consistent solve
# ---------------------------------------------------------------------------

class TwoTone:
    """
    One operating point: world, pinned local gain, saturation anchor u_thr.
    Suppression is pure gain control (gamma -> gamma (1 - s)), the
    physiological default and the route the old observables failed on.
    """

    def __init__(self, coch, gain_target, u_thr, N=N_RUN,
                 beta=0.0, depth=0.0):
        self.coch, self.N, self.u_thr = coch, N, float(u_thr)
        self.beta, self.depth = float(beta), float(depth)
        self.f_p = float(coch.cf(E.X_PROBE))
        self.f_s = self.f_p * 2.0 ** (-OCT_BELOW)
        self.g0, self.got = T.gamma_for_gain(coch, gain_target, self.f_p,
                                             "local", N=N, x_ref=E.X_READ)
        self.gain_target = gain_target
        self.zero = np.zeros(N + 1)
        pr = E.line(coch, self.f_p, self.g0, self.zero, self.beta,
                    self.depth, N)
        self.j = int(round(E.X_READ * N))
        self.C = float(np.max(np.abs(pr["shear"])))   # unit-amp probe norm
        # probe threshold amplitude: unit shear norm times u_thr
        self.A0 = self.u_thr
        self.P0 = 20.0 * math.log10(self.A0)

    def solve(self, L_s, P_p, s_start, itmax=150):
        """Self-consistent s(x) with suppressor at L_s and probe at P_p,
        both dB re the unit-amp norm (P_p = P0 + SL)."""
        As = 10.0 ** (L_s / 20.0) if L_s is not None else 0.0
        Ap = 10.0 ** (P_p / 20.0) if P_p is not None else 0.0

        def F(sv):
            u2 = 0.0
            if As > 0.0:
                rs = E.line(self.coch, self.f_s, self.g0, sv, self.beta,
                            self.depth, self.N)
                us = As * np.abs(rs["shear"]) / self.C
                u2 = u2 + us * us
            if Ap > 0.0:
                rp = E.line(self.coch, self.f_p, self.g0, sv, self.beta,
                            self.depth, self.N)
                up = Ap * np.abs(rp["shear"]) / self.C
                u2 = u2 + up * up
            v = u2 ** (P_ORD / 2.0)
            return v / (1.0 + v)

        return E.anderson(F, s_start, itmax=itmax)

    def readout(self, s, P_p):
        """(cilia, BM) probe response in dB re 1, on the line s leaves."""
        r = E.line(self.coch, self.f_p, self.g0, s, self.beta,
                   self.depth, self.N)
        Ap = 10.0 ** (P_p / 20.0)
        return (20.0 * math.log10(Ap * abs(r["shear"][self.j])),
                20.0 * math.log10(Ap * abs(r["X_b"][self.j])))

    def response(self, L_s, P_p, s_start=None):
        s0 = self.zero.copy() if s_start is None else s_start
        s, d, it = self.solve(L_s, P_p, s0)
        c, b = self.readout(s, P_p)
        return c, b, s, d


# ---------------------------------------------------------------------------
# the three protocols
# ---------------------------------------------------------------------------

def probe_io(tt, sl_lo=-10.0, sl_hi=60.0, step=2.0):
    """Probe-alone input-output at the read place, no suppressor."""
    out = []
    s = tt.zero.copy()
    for sl in np.arange(sl_lo, sl_hi + 1e-9, step):
        c, b, s, _ = tt.response(None, tt.P0 + sl, s)
        out.append((float(sl), c, b))
    return np.asarray(out)


def io_slope(io, sl, half=2.0):
    """d(cilia out)/d(probe in) around SL, from the I/O table."""
    x, y = io[:, 0], io[:, 1]
    return float((np.interp(sl + half, x, y)
                  - np.interp(sl - half, x, y)) / (2.0 * half))


def bisect_supp(tt, P_p, target_c, lo=-60.0, hi=100.0, tol=0.02):
    """Suppressor level at which the cilia probe response falls to
    target_c. Returns nan if it never does."""
    s = tt.zero.copy()
    c_hi, _, s_hi, _ = tt.response(hi, P_p, s)
    if c_hi > target_c:
        return float("nan"), None
    c_lo, _, s_lo, _ = tt.response(lo, P_p, s)
    if c_lo < target_c:
        return float("nan"), None
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        c, _, s_m, _ = tt.response(mid, P_p, s_lo)
        if c > target_c:
            lo, s_lo = mid, s_m
        else:
            hi = mid
    return 0.5 * (lo + hi), s_lo


def bisect_bm(tt, P_p, drop_db, lo=-60.0, hi=100.0, tol=0.02):
    """Suppressor level at which the BM probe response falls drop_db below
    its unsuppressed value at the same probe level (Cooper's vertical
    criterion, fixed moderate-level probe)."""
    _, b0, s0, _ = tt.response(None, P_p)
    target = b0 - drop_db
    s = s0
    _, b_hi, s_hi, _ = tt.response(hi, P_p, s)
    if b_hi > target:
        return float("nan")
    _, b_lo, s_lo, _ = tt.response(lo, P_p, s)
    if b_lo < target:
        return float("nan")
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        _, b, s_m, _ = tt.response(mid, P_p, s_lo)
        if b > target:
            lo, s_lo = mid, s_m
        else:
            hi = mid
    return 0.5 * (lo + hi)


def bisect_probe(tt, L_s, target_c, lo_sl=-20.0, hi_sl=90.0, tol=0.02):
    """Delgutte: the probe SL restoring the criterion cilia response with
    the suppressor at L_s."""
    s = tt.zero.copy()
    c_hi, _, s_hi, _ = tt.response(L_s, tt.P0 + hi_sl, s)
    if c_hi < target_c:
        return float("nan")
    c_lo, _, s_lo, _ = tt.response(L_s, tt.P0 + lo_sl, s)
    if c_lo > target_c:
        return float("nan")
    lo, hi = lo_sl, hi_sl
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        c, _, s_m, _ = tt.response(L_s, tt.P0 + mid, s_lo)
        if c < target_c:
            lo, s_lo = mid, s_m
        else:
            hi = mid
    return 0.5 * (lo + hi)


def measure(tt):
    """All three observables by their own protocols."""
    io = probe_io(tt)
    # criterion responses
    c_thr = float(np.interp(0.0, io[:, 0], io[:, 1]))     # probe at SL 0
    c_n = float(np.interp(SL_NEURAL, io[:, 0], io[:, 1]))  # probe at 8 SL
    # FA-85 neural threshold: two-tone response pushed back to threshold
    L_n, _ = bisect_supp(tt, tt.P0 + SL_NEURAL, c_thr)
    # Delgutte horizontal growth above that threshold
    slope = float("nan")
    pts = []
    if not math.isnan(L_n):
        lo, hi, st = GROWTH_SPAN
        for d in np.arange(lo, hi + 1e-9, st):
            sl = bisect_probe(tt, L_n + d, c_n)
            if not math.isnan(sl):
                pts.append((float(d), sl))
        if len(pts) >= 3:
            x = np.asarray([p[0] for p in pts])
            y = np.asarray([p[1] for p in pts])
            A = np.vstack([x, np.ones_like(x)]).T
            slope = float(np.linalg.lstsq(A, y, rcond=None)[0][0])
    # Cooper BM thresholds at moderate probe levels
    bm = {sl: bisect_bm(tt, tt.P0 + sl, BM_CRIT) for sl in SL_COOPER}
    return dict(io=io, c_thr=c_thr, c_n=c_n, L_n=L_n, slope=slope,
                growth=pts, bm=bm,
                comp={sl: io_slope(io, sl) for sl in SL_COOPER})


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------

def verify(coch):
    print("=" * 78)
    print("VERIFY")
    print("=" * 78)
    ok = True
    tt = TwoTone(coch, 20.0, 0.1)

    # V1: probe amplitude -> 0 recovers the single-tone suppressor solve
    s1, d1, _ = tt.solve(0.0, None, tt.zero.copy())
    cfg = E.Config(coch, 20.0, OCT_BELOW, 0.0, 0.0, N=N_RUN)
    amp_eq = 20.0 * math.log10(cfg.C / tt.C)  # cfg uses u=amp|shear|/C too
    s2, d2, _ = cfg.solve(0.0 + amp_eq - amp_eq, tt.zero.copy())
    # same normalization C differs; compare at matched physical drive:
    s2b, _, _ = cfg.solve(20.0 * math.log10(tt.C / cfg.C), tt.zero.copy())
    err = float(np.max(np.abs(s1 - s2b)))
    print(f"\n[V1] suppressor-only, probe silent: two-tone solve against "
          f"exp_accumulation's\n     single-tone solve at matched drive: "
          f"max |s difference| = {err:.3e}")
    ok &= err < 1e-8

    # V2: both tones silent -> s = 0
    s0, _, _ = tt.solve(None, None, 0.3 * np.ones(N_RUN + 1))
    print(f"[V2] no drive: s converges to {float(np.max(s0)):.3e}")
    ok &= float(np.max(s0)) < 1e-12

    # V3: probe I/O is linear far below the anchor, compressive above it
    io = probe_io(tt)
    lo = io_slope(io, -6.0)
    hi = io_slope(io, 40.0)
    print(f"[V3] probe I/O slope at SL -6: {lo:.3f} dB/dB (want near 1); "
          f"at SL 40: {hi:.3f} (want < 1)")
    ok &= abs(lo - 1.0) < 0.1 and hi < 0.9

    # V4: equivalent attenuation equals vertical suppression for a small probe
    tt_small = TwoTone(coch, 20.0, 0.001)   # anchor deep in the linear range
    ios = probe_io(tt_small, -10, 20, 2.0)
    c_thr = float(np.interp(0.0, ios[:, 0], ios[:, 1]))
    c_n = float(np.interp(8.0, ios[:, 0], ios[:, 1]))
    L_ea, _ = bisect_supp(tt_small, tt_small.P0 + 8.0, c_thr)
    # vertical: suppression of the same small probe reaches 8 dB
    s = tt_small.zero.copy()
    L_v = float("nan")
    lo_, hi_ = -60.0, 100.0
    c0, _, _, _ = tt_small.response(None, tt_small.P0 + 8.0)
    while hi_ - lo_ > 0.02:
        mid = 0.5 * (lo_ + hi_)
        c, _, s, _ = tt_small.response(mid, tt_small.P0 + 8.0, s)
        if c0 - c < 8.0:
            lo_ = mid
        else:
            hi_ = mid
    L_v = 0.5 * (lo_ + hi_)
    print(f"[V4] linear probe: equivalent-attenuation threshold "
          f"{L_ea:.2f} against vertical\n     8 dB threshold {L_v:.2f}, "
          f"difference {abs(L_ea - L_v):.3f} dB (want small)")
    ok &= abs(L_ea - L_v) < 0.5

    print(f"\n     VERIFY {'passed' if ok else 'FAILED'}")
    return ok


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv):
    quick = "--quick" in argv
    t0 = time.time()
    coch = E.COCH
    f_p = float(coch.cf(E.X_PROBE))
    r = T.response(coch, f_p, N=4096, gamma=T.CEILING[E.NAME])
    E.X_READ = T._peak(r["x"], r["X_b"])[0]

    print("=" * 78)
    print("TWO TONES IN ONE SATURATOR: THE OBSERVABLES AS MEASURED")
    print("=" * 78)
    print(f"  world {E.NAME}, N = {N_RUN}, probe at x = {E.X_PROBE} "
          f"(f = {f_p:.5f}), read at x = {E.X_READ:.4f}")
    print(f"  suppressor {OCT_BELOW} oct below; neural probe at "
          f"+{SL_NEURAL:.0f} dB SL (FA-85); BM probe at "
          f"+{SL_COOPER} dB SL (Cooper); BM criterion {BM_CRIT} dB vertical;"
          f"\n  neural criterion: response returned to threshold "
          f"(equivalent attenuation {EA_CRIT:.0f} dB)")

    ok = verify(coch)

    gains = (20.0, 39.0) if quick else GAINS
    uthrs = (0.1, 1.0) if quick else U_THRS

    print("\n" + "=" * 78)
    print("THE SCAN: OPERATING POINT AGAINST THE MEASURED PROTOCOLS")
    print("=" * 78)
    print(f"\n  {'gain':>5} {'u_thr':>6} {'L_n':>8} {'growth':>7} "
          f"{'comp@40':>8} "
          + " ".join(f"{'L_bm@' + str(int(s)):>9}" for s in SL_COOPER)
          + " " + " ".join(f"{'dT@' + str(int(s)):>8}" for s in SL_COOPER)
          + f" {'box':>4}")
    results = []
    for gain in gains:
        for u in uthrs:
            tt = TwoTone(coch, gain, u)
            m = measure(tt)
            dts = {sl: (m["bm"][sl] - m["L_n"]
                        if not (math.isnan(m["bm"][sl])
                                or math.isnan(m["L_n"])) else float("nan"))
                   for sl in SL_COOPER}
            box = any(TARGET_DT[0] <= dts[sl] <= TARGET_DT[1]
                      for sl in SL_COOPER
                      if not math.isnan(dts[sl])) and \
                (TARGET_SL[0] <= m["slope"] <= TARGET_SL[1])
            results.append(dict(gain=gain, u=u, m=m, dts=dts, box=box))
            print(f"  {gain:5.0f} {u:6.2f} {fmt(m['L_n'])} "
                  f"{fmt(m['slope'], 7)} {fmt(m['comp'][40.0], 8)} "
                  + " ".join(fmt(m["bm"][s], 9) for s in SL_COOPER)
                  + " " + " ".join(fmt(dts[s], 8) for s in SL_COOPER)
                  + f" {'YES' if box else 'no':>4}")
    print("\n  L_n and L_bm are suppressor levels in dB re the unit-amp "
          "norm, so only\n  DIFFERENCES mean anything; dT@SL = L_bm at that "
          "probe SL minus L_n.\n  growth is Delgutte's horizontal slope "
          "(dB probe per dB suppressor);\n  comp@40 is the probe I/O slope "
          "at 40 dB SL, the compression that\n  multiplies a vertical slope "
          "into a horizontal one.")

    hits = [r for r in results if r["box"]]
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    print(f"  operating points meeting Delta-T in {TARGET_DT} at any Cooper "
          f"SL AND\n  horizontal growth in {TARGET_SL}: {len(hits)} of "
          f"{len(results)}")
    if hits:
        for h in hits:
            print(f"    gain {h['gain']:.0f}, u_thr {h['u']}: growth "
                  f"{h['m']['slope']:.2f}, dT "
                  + ", ".join(f"{sl:.0f}SL {h['dts'][sl]:.1f}"
                              for sl in SL_COOPER
                              if not math.isnan(h['dts'][sl])))
        print("  Next: suppressor-frequency locus (obs 1 flatness), "
              "stability under drive,\n  and the full six-observation "
              "score sheet at the best point.")
    else:
        print("  The measured protocols do not rescue observations 2 and 3 "
              "here. Check the\n  growth and comp columns: if growth "
              "tracks 1/comp as the algebra says it\n  must, the failure "
              "is in Delta-T alone and the probe-level effect is too\n"
              "  weak in this world.")
    print(f"\n  total {time.time() - t0:.1f} s")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
