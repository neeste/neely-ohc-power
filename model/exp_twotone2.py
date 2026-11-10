"""
exp_twotone2.py

Refinement of exp_twotone.py, which established that the measured protocols
move both failing observables most of the way to their targets:

    growth (horizontal)   0.34 -> 1.30 dB/dB     target 1.4 to 2.6
    Delta-T at 40 dB SL   ~0   -> 13.8 dB        target 18 to 32
    Delta-T at 50 dB SL   BM never reaches criterion (unbounded)

both at the highest gain scanned. The remaining gaps are a factor 1.1 in
growth and a few dB in Delta-T, and every lever that was frozen in the first
pass plausibly bears on them:

  - the suppressor sat at 0.5 oct below the probe; Delgutte's 2 dB/dB is at
    f_s/CF near 0.44, about 1.2 oct below, and FA-85's locus spans far more
  - the gain stopped at 39 dB; the certified ceiling is a little higher
  - only world B was scanned
  - the growth window stopped 21 dB above threshold; Delgutte's slope is
    read over about 30 dB
  - Cooper's probe SL was capped at 50

This file scans those, reports the box count, and for the best points runs
the observation 1 locus (neural threshold against suppressor frequency,
which must be flat) and rule 1 (per-section pole margins on the two-tone
suppressed line at the drives that define the thresholds).

Run: python3 exp_twotone2.py [--quick]
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

WORLD_NAMES = ("B", "A")
GAINS = (30.0, 39.0)
U_THRS = (0.1, 0.3, 1.0)
OCTS = (0.5, 1.0, 1.5)
P_ORDS = (2.0, 3.0, 4.0)
SLS = (30.0, 40.0, 50.0, 60.0)
TARGET_DT = (18.0, 32.0)
TARGET_SL = (1.4, 2.6)
LOCUS_OCTS = (0.5, 0.75, 1.0, 1.5, 2.0, 2.5)


def fmt(v, w=8, p=2):
    return f"{'-':>{w}}" if (v is None or (isinstance(v, float)
                                           and math.isnan(v))) else \
        f"{v:{w}.{p}f}"


def setup_world(name):
    coch = T.WORLDS[name]
    f_p = float(coch.cf(E.X_PROBE))
    r = T.response(coch, f_p, N=4096, gamma=T.CEILING[name])
    E.X_READ = T._peak(r["x"], r["X_b"])[0]
    return coch


def run_point(coch, gain, u_thr, oct_below, sls, growth_hi=33.0, p=2.0):
    TT.P_ORD = p
    TT.OCT_BELOW = oct_below
    TT.SL_COOPER = sls
    TT.GROWTH_SPAN = (3.0, growth_hi, 3.0)
    tt = TT.TwoTone(coch, gain, u_thr)
    m = TT.measure(tt)
    dts = {}
    for sl in sls:
        b = m["bm"][sl]
        if math.isnan(m["L_n"]):
            dts[sl] = float("nan")
        elif math.isnan(b):
            dts[sl] = float("inf")     # BM never reaches criterion
        else:
            dts[sl] = b - m["L_n"]
    return tt, m, dts


def in_box(m, dts):
    if not (TARGET_SL[0] <= m["slope"] <= TARGET_SL[1]):
        return False
    return any((not math.isnan(d)) and (not math.isinf(d))
               and TARGET_DT[0] <= d <= TARGET_DT[1] for d in dts.values())


def main(argv):
    quick = "--quick" in argv
    t0 = time.time()

    worlds = ("B",) if quick else WORLD_NAMES
    gains = (39.0,) if quick else GAINS
    uthrs = (0.1, 1.0) if quick else U_THRS
    octs = (0.5, 1.0) if quick else OCTS
    pords = (2.0, 4.0) if quick else P_ORDS

    print("=" * 78)
    print("TWO-TONE PROTOCOLS, REFINED")
    print("=" * 78)
    print(f"  target box: horizontal growth in {TARGET_SL}, Delta-T in "
          f"{TARGET_DT} at any Cooper SL\n  (inf = BM never reaches its "
          f"criterion; counts as ABOVE the box, not in it)")

    results = []
    for w in worlds:
        coch = setup_world(w)
        print(f"\n  --- world {w}, read at x = {E.X_READ:.4f} ---")
        print(f"  {'gain':>5} {'u_thr':>6} {'oct':>5} {'p':>3} {'L_n':>8} "
              f"{'growth':>7} "
              + " ".join(f"{'dT@' + str(int(s)):>8}" for s in SLS)
              + f" {'box':>4}")
        for gain in gains:
            for u in uthrs:
                for oc in octs:
                  for p in pords:
                    try:
                        tt, m, dts = run_point(coch, gain, u, oc, SLS, p=p)
                    except Exception as exc:          # noqa: BLE001
                        print(f"  {gain:5.0f} {u:6.2f} {oc:5.2f} {p:3.0f} "
                              f"failed: {exc}")
                        continue
                    box = in_box(m, dts)
                    results.append(dict(w=w, coch=coch, gain=gain, u=u,
                                        oc=oc, p=p, m=m, dts=dts, box=box))
                    print(f"  {gain:5.0f} {u:6.2f} {oc:5.2f} {p:3.0f} "
                          f"{fmt(m['L_n'])} {fmt(m['slope'], 7)} "
                          + " ".join(("     inf" if math.isinf(dts[s])
                                      else fmt(dts[s], 8))
                                     for s in SLS)
                          + f" {'YES' if box else 'no':>4}")

    hits = [r for r in results if r["box"]]
    near = sorted((r for r in results if not r["box"]),
                  key=lambda r: -(r["m"]["slope"] or 0))[:5]
    print("\n" + "=" * 78)
    print(f"BOX COUNT: {len(hits)} of {len(results)}")
    print("=" * 78)

    # ----- the locus and rule 1 on the best points ------------------------
    best = hits if hits else near[:2]
    seen = set()
    for r in best:
        w, gain, u, oc = r["w"], r["gain"], r["u"], r["oc"]
        p = r.get("p", 2.0)
        key = (w, gain, u, p)
        if key in seen:
            continue
        seen.add(key)
        TT.P_ORD = p
        coch = r["coch"]
        setup_world(w)
        print(f"\n  OBSERVATION 1 LOCUS at world {w}, gain {gain:.0f}, "
              f"u_thr {u}, p {p:.0f}, box={r['box']}")
        print(f"  {'oct below':>10} {'L_n':>8} {'re -0.5 oct':>12}")
        base = float("nan")
        pts = []
        for lo in LOCUS_OCTS:
            TT.OCT_BELOW = lo
            tt = TT.TwoTone(coch, gain, u)
            io = TT.probe_io(tt, -10, 14, 2.0)
            c_thr = float(np.interp(0.0, io[:, 0], io[:, 1]))
            L_n, _ = TT.bisect_supp(tt, tt.P0 + TT.SL_NEURAL, c_thr)
            if lo == 0.5:
                base = L_n
            pts.append((lo, L_n))
            print(f"  {lo:10.2f} {fmt(L_n)} {fmt(L_n - base, 12)}")
        good = [(lo, L) for lo, L in pts if not math.isnan(L)]
        if len(good) >= 3:
            x = np.asarray([math.log10(2.0 ** (-lo)) for lo, _ in good])
            y = np.asarray([L for _, L in good])
            A = np.vstack([x, np.ones_like(x)]).T
            sl = float(np.linalg.lstsq(A, y, rcond=None)[0][0])
            print(f"  locus slope {sl:+.2f} dB/decade of suppressor "
                  f"frequency (target 0 +/- 10)")

        # rule 1: section margins at the defining drives
        TT.OCT_BELOW = oc
        tt = TT.TwoTone(coch, gain, u)
        io = TT.probe_io(tt, -10, 14, 2.0)
        c_thr = float(np.interp(0.0, io[:, 0], io[:, 1]))
        L_n, s_n = TT.bisect_supp(tt, tt.P0 + TT.SL_NEURAL, c_thr)
        worst = float("-inf")
        if s_n is not None:
            cfg = E.Config(coch, gain, oc, 0.0, 0.0, N=TT.N_RUN)
            worst = E.section_margin(cfg, s_n, 128)
        print(f"  rule 1 at the neural threshold: worst section margin "
              f"{worst:+.3e} (negative is stable)")

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    if hits:
        print(f"  {len(hits)} operating point(s) meet observations 2 and 3 "
              f"under the measured\n  protocols. Next: full six-observation "
              f"score sheet at the best point\n  (exp_obs46 machinery for 4, "
          "5, 6 plus these protocols for 1, 2, 3).")
    else:
        print("  Still zero in the box. The near-misses above say which "
              "lever is short:\n  if growth saturates below 1.4 at every "
              "frequency and gain, the residue is\n  the saturator shape; "
              "if Delta-T tops out finite, it is the probe-level\n  effect. "
              "Either way the next scan is named by the table.")
    print(f"\n  total {time.time() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
