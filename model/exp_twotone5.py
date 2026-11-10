"""
exp_twotone5.py

The transducer-slope gate: the last structural lever for observation 2.

WHY THE PREVIOUS LEVERS HAD TO FAIL. The horizontal measure (probe level to
criterion response) is invariant to any static monotone function of the
probe's own response amplitude: if the criterion is g(response) = const with
g monotone, the restoring probe level is the same for every g. So no
rate-level function, IHC or synaptic, can raise the growth slope by itself.
The measure depends only on the map (probe level, suppressor level) ->
probe-component drive, and in that map the model's growth is bounded by

    growth = (vertical suppression rate) / (probe I/O slope, suppressed)

whose numerator is capped by the removable gain and whose denominator rises
toward 1 as the suppressor removes it (compression and suppression draw on
one reservoir). Measured cap: 1.3 over 20 dB, 1.0 over 30 dB, across
saturator orders 1 to 4, route mixtures, both worlds, 0.5 to 2.5 oct.

WHAT IS NOT INVARIANT. Two-tone interaction inside the transducer. The
neural drive is not the bundle displacement; it is the receptor-current
modulation, transducer slope times displacement. A low-side suppressor
produces a large bundle motion at the probe place (the tails are big) and
biases the transducer onto its saturated tail, so the small-signal slope the
probe sees falls with suppressor level WITHOUT BOUND as s -> 1. That is the
IHC-level component of low-side suppression measured by Cheatham and Dallos
in receptor potentials, and it acts on the NEURAL read-out only: Cooper's
interferometer reads BM displacement and never sees it. So it raises the
neural growth and Delta-T together, which is exactly the shape of the two
residuals.

THE MODEL. One new element, the gate:

    r_n(j) = G_t(j) * |shear_p(j)|,   G_t = (1 + (u_s(j)/q)^2)^(-1)

u_s(j) the suppressor's cilia drive at the read place in the same units as
the OHC saturator (u = 1 at its knee), q the IHC/OHC knee ratio: q < 1 means
the IHC transducer saturates below the OHC knee, q -> inf recovers
exp_twotone exactly (V1). q is the one new parameter and observation 2 sets
it, as the build plan assigned. The BM protocol is untouched. The gate uses
the SUPPRESSOR component only (the probe's own contribution to its transducer
operating point is the static-monotone part that cancels; including it would
double-count the compression already in shear_p).

Scan: q, u_thr, beta/depth at gain 39, world B. For hits: locus, rule 1 at
levels, and the growth-curve shape.

Run: python3 exp_twotone5.py [--quick]
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
import exp_twotone as TT                  # noqa: E402
from exp_twotone2 import setup_world, fmt   # noqa: E402

GAIN = 39.0
QS = (0.1, 0.3, 1.0, 3.0)
U_THRS = (0.1, 0.3)
ROUTES = ((0.0, 0.0), (0.25, 4.0))
SLS = (40.0, 50.0, 60.0)
TARGET_DT = (18.0, 32.0)
TARGET_SL = (1.4, 2.6)
LOCUS_OCTS = (0.5, 0.75, 1.0, 1.5, 2.0, 2.5)


class Gated(TT.TwoTone):
    """TwoTone with the transducer-slope gate on the neural read-out."""

    def __init__(self, coch, gain, u_thr, q, beta=0.0, depth=0.0,
                 N=TT.N_RUN):
        super().__init__(coch, gain, u_thr, N=N, beta=beta, depth=depth)
        self.q = float(q)
        self._Ls = None

    def solve(self, L_s, P_p, s_start, itmax=150):
        self._Ls = L_s          # remembered for the gate in readout()
        return super().solve(L_s, P_p, s_start, itmax)

    def readout(self, s, P_p):
        r = E.line(self.coch, self.f_p, self.g0, s, self.beta,
                   self.depth, self.N)
        Ap = 10.0 ** (P_p / 20.0)
        c_lin = Ap * abs(r["shear"][self.j])
        b = Ap * abs(r["X_b"][self.j])
        gate = 1.0
        if self._Ls is not None:
            As = 10.0 ** (self._Ls / 20.0)
            rs = E.line(self.coch, self.f_s, self.g0, s, self.beta,
                        self.depth, self.N)
            us = As * abs(rs["shear"][self.j]) / self.C
            gate = 1.0 / (1.0 + (us / self.q) ** 2)
        return (20.0 * math.log10(gate * c_lin), 20.0 * math.log10(b))


def run_point(coch, u_thr, q, beta, depth):
    TT.P_ORD = 2.0
    TT.OCT_BELOW = 0.5
    TT.SL_COOPER = SLS
    TT.GROWTH_SPAN = (3.0, 33.0, 3.0)
    tt = Gated(coch, GAIN, u_thr, q, beta=beta, depth=depth)
    m = TT.measure(tt)
    dts = {}
    for sl in SLS:
        b = m["bm"][sl]
        if math.isnan(m["L_n"]):
            dts[sl] = float("nan")
        elif math.isnan(b):
            dts[sl] = float("inf")
        else:
            dts[sl] = b - m["L_n"]
    TT.GROWTH_SPAN = (3.0, 21.0, 3.0)
    g21 = TT.measure(tt)["slope"]
    return tt, m, dts, g21


def in_box(g33, g21, dts):
    ok_g = any(TARGET_SL[0] <= g <= TARGET_SL[1]
               for g in (g33, g21) if g is not None and not math.isnan(g))
    ok_d = any((not math.isnan(d)) and (not math.isinf(d))
               and TARGET_DT[0] <= d <= TARGET_DT[1] for d in dts.values())
    return ok_g and ok_d


def verify(coch):
    print("=" * 78)
    print("VERIFY")
    print("=" * 78)
    ok = True
    # V1: q -> inf recovers exp_twotone bit for bit
    TT.P_ORD = 2.0
    TT.OCT_BELOW = 0.5
    g1 = Gated(coch, GAIN, 0.1, 1e12)
    t1 = TT.TwoTone(coch, GAIN, 0.1)
    c_g, b_g, s_g, _ = g1.response(0.0, g1.P0 + 8.0)
    c_t, b_t, s_t, _ = t1.response(0.0, t1.P0 + 8.0)
    d = max(abs(c_g - c_t), abs(b_g - b_t), float(np.max(np.abs(s_g - s_t))))
    print(f"\n[V1] q = 1e12: gated model against exp_twotone, max "
          f"|difference| = {d:.3e}")
    ok &= d < 1e-9
    # V2: the gate never touches the BM read-out
    g2 = Gated(coch, GAIN, 0.1, 0.1)
    c2, b2, s2, _ = g2.response(0.0, g2.P0 + 8.0)
    print(f"[V2] q = 0.1: BM read-out unchanged: "
          f"|{b2:.6f} - {b_t:.6f}| = {abs(b2 - b_t):.3e}; neural moved "
          f"{c_t - c2:+.2f} dB")
    ok &= abs(b2 - b_t) < 1e-9 and (c_t - c2) > 1.0
    # V3: no suppressor -> gate = 1 exactly
    c3, b3, _, _ = g2.response(None, g2.P0 + 8.0)
    c3t, b3t, _, _ = t1.response(None, t1.P0 + 8.0)
    print(f"[V3] suppressor off: gate inert, difference "
          f"{abs(c3 - c3t):.3e}")
    ok &= abs(c3 - c3t) < 1e-12
    print(f"\n     VERIFY {'passed' if ok else 'FAILED'}")
    return ok


def main(argv):
    quick = "--quick" in argv
    t0 = time.time()
    coch = setup_world("B")

    qs = (0.3, 1.0) if quick else QS
    uthrs = (0.1,) if quick else U_THRS
    routes = ((0.0, 0.0),) if quick else ROUTES

    print("=" * 78)
    print("THE TRANSDUCER-SLOPE GATE ON THE NEURAL READ-OUT")
    print("=" * 78)
    print(f"  world B, gain {GAIN:.0f}, suppressor 0.5 oct below, read at "
          f"x = {E.X_READ:.4f}\n  gate G_t = 1/(1 + (u_s/q)^2) on the "
          f"neural read-out only; q -> inf is the old model")

    ok = verify(coch)

    print(f"\n  {'q':>6} {'u_thr':>6} {'beta':>5} {'depth':>6} {'L_n':>8} "
          f"{'gr33':>6} {'gr21':>6} "
          + " ".join(f"{'dT@' + str(int(s)):>8}" for s in SLS)
          + f" {'box':>4}")
    results = []
    for q in qs:
        for u in uthrs:
            for beta, depth in routes:
                try:
                    tt, m, dts, g21 = run_point(coch, u, q, beta, depth)
                except Exception as exc:              # noqa: BLE001
                    print(f"  {q:6.2f} {u:6.2f} {beta:5.2f} {depth:6.2f} "
                          f"failed: {exc}")
                    continue
                box = in_box(m["slope"], g21, dts)
                results.append(dict(q=q, u=u, beta=beta, depth=depth,
                                    m=m, g21=g21, dts=dts, box=box))
                print(f"  {q:6.2f} {u:6.2f} {beta:5.2f} {depth:6.2f} "
                      f"{fmt(m['L_n'])} {fmt(m['slope'], 6)} "
                      f"{fmt(g21, 6)} "
                      + " ".join(("     inf" if math.isinf(dts[s])
                                  else fmt(dts[s], 8)) for s in SLS)
                      + f" {'YES' if box else 'no':>4}")

    hits = [r for r in results if r["box"]]
    print("\n" + "=" * 78)
    print(f"BOX COUNT: {len(hits)} of {len(results)}")
    print("=" * 78)

    pool = hits if hits else sorted(
        results,
        key=lambda r: -max([v for v in (r["m"]["slope"], r["g21"])
                            if v is not None and not math.isnan(v)],
                           default=0.0))[:2]
    for r in pool[:4]:
        q, u, beta, depth = r["q"], r["u"], r["beta"], r["depth"]
        print(f"\n  DEEP CHECK at q {q}, u_thr {u}, beta {beta}, "
              f"depth {depth}, box={r['box']}")
        pts = []
        for lo in LOCUS_OCTS:
            TT.OCT_BELOW = lo
            tt = Gated(coch, GAIN, u, q, beta=beta, depth=depth)
            io = TT.probe_io(tt, -10, 14, 2.0)
            c_thr = float(np.interp(0.0, io[:, 0], io[:, 1]))
            L_n, s_n = TT.bisect_supp(tt, tt.P0 + TT.SL_NEURAL, c_thr)
            pts.append((lo, L_n, s_n))
            print(f"    {lo:5.2f} oct below: L_n {fmt(L_n)}")
        good = [(lo, L) for lo, L, _ in pts if not math.isnan(L)]
        if len(good) >= 3:
            x = np.asarray([math.log10(2.0 ** (-lo)) for lo, _ in good])
            y = np.asarray([L for _, L in good])
            A = np.vstack([x, np.ones_like(x)]).T
            sl = float(np.linalg.lstsq(A, y, rcond=None)[0][0])
            print(f"    locus slope {sl:+.2f} dB/decade (target 0 +/- 10)")
        TT.OCT_BELOW = 0.5
        s_n = pts[0][2]
        if s_n is not None:
            cfg = E.Config(coch, GAIN, 0.5, beta, depth, N=TT.N_RUN)
            worst = E.section_margin(cfg, s_n, 128)
            g, clean = E.growing_modes_mod(cfg, s_n, N=1024)
            print(f"    rule 1: worst section margin {worst:+.3e}; "
                  f"whole-line growing modes {g} (clean={clean})")

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    if hits:
        print("  observations 2 and 3 are met together. The gate is the "
              "IHC-level\n  component of low-side suppression (Cheatham-"
              "Dallos), it has one\n  parameter, and Cooper's read-out never "
              "sees it. Phase 4 next.")
    else:
        print("  the gate did not close the gap; with the invariance "
              "argument this\n  exhausts the structural levers for "
              "observation 2 in this model class.")
    print(f"\n  total {time.time() - t0:.1f} s")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
