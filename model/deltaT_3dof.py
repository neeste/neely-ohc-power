"""
deltaT_3dof.py
The same question with the reticular lamina separated from the basilar membrane.

WHY
---
In the 2-DOF section of deltaT_sweep.py the reticular lamina is lumped into
the organ-of-Corti mass, so RL and BM cannot move differently. Two results
followed:

    1  only parameters reaching row 2 can produce Delta-T, so gain control
       and BM-stiffness migration give exactly zero
    2  Delta-T and the growth slope trade off, and observations 2 and 3
       cannot both be met

Result 1 is the reason to expect the third degree of freedom to matter. The
OHC sits between the BM and the RL. With them lumped it has nothing to push
against and had to be modeled as reacting on the bony structure, which is
what confined gamma to row 1. Separating them puts the OHC force between two
free masses, so gamma reaches the internal coordinates and can change H_o.
The measurements say RL and BM do move differently.

THE SECTION
-----------
Three masses, driven by pressure on the BM:

    M_b  basilar membrane and lower organ of Corti, held to bone by K_b, R_b
    M_r  reticular lamina, coupled to M_b through the pillar cells K_p, R_p
    M_t  tectorial membrane, held to bone by K_t, R_t

    hair bundle K_c, R_c couples M_r to M_t
    cilia shear is G (x_r - x_t)

K_a is a separate OHC force scale rather than K_b. Tying it to K_b would make
the bm route modulate the amplifier as a side effect, which showed up as a
spurious Delta-T of 25 dB on a route that cannot produce one.
    OHC force acts between M_b and M_r, proportional to bundle displacement:

        F_ohc = gamma K_a G (x_r - x_t)     applied to M_r, reacted on M_b

Three read-outs, which are the three the paper's proposed experiment asks for:

    BM     x_b          what the interferometer sees
    RL     x_r          what OCT sees
    cilia  G(x_r - x_t) what the hair cell senses

Delta-T is the BM threshold minus the cilia threshold, as before.

Checked against the 2-DOF model in the stiff-pillar limit: as K_p grows, the
RL locks to the BM and the section must reproduce deltaT_sweep.py. verify()
does that comparison numerically.

    python3 deltaT_3dof.py            # verify, then scan, then report
    python3 deltaT_3dof.py --verify   # checks only
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, replace

import numpy as np

import deltaT_sweep as two

CRITERION_DB = two.CRITERION_DB
DELTA_T_LO, DELTA_T_HI = two.DELTA_T_LO, two.DELTA_T_HI
ROUTES = ("bundle", "tm", "bm", "pillar")


@dataclass(frozen=True)
class Section3:
    M_b: float = 1.0
    K_b: float = 1.0
    R_b: float = 0.50
    M_r: float = 0.20        # reticular lamina
    R_r: float = 0.02
    K_p: float = 5.0         # pillar cells, BM to RL; large means locked
    R_p: float = 0.02
    M_t: float = 0.10
    K_t: float = 0.05
    R_t: float = 0.02
    K_c: float = 0.50
    R_c: float = 0.02
    K_a: float = 1.0         # OHC force scale, deliberately NOT tied to K_b
    G: float = 1.0
    gamma: float = 0.0

    def with_ratio(self, kt_over_kc: float) -> "Section3":
        return replace(self, K_c=self.K_t / kt_over_kc)


def response(sec: Section3, w):
    """Returns (X_b, X_r, shear, H_o) for unit force on the BM."""
    w = np.atleast_1d(np.asarray(w, dtype=float))
    jw = 1j * w
    pil = jw * sec.R_p + sec.K_p               # pillar coupling
    bun = jw * sec.R_c + sec.K_c               # hair bundle coupling
    act = sec.gamma * sec.K_a * sec.G          # OHC, between M_b and M_r

    n = w.size
    A = np.zeros((n, 3, 3), dtype=complex)
    A[:, 0, 0] = -w**2 * sec.M_b + jw * (sec.R_b + sec.R_p) + sec.K_b + sec.K_p
    A[:, 0, 1] = -pil + act
    A[:, 0, 2] = -act
    A[:, 1, 0] = -pil
    A[:, 1, 1] = (-w**2 * sec.M_r + jw * (sec.R_r + sec.R_p + sec.R_c)
                  + sec.K_p + sec.K_c - act)
    A[:, 1, 2] = -bun + act
    A[:, 2, 1] = -bun
    A[:, 2, 2] = -w**2 * sec.M_t + jw * (sec.R_t + sec.R_c) + sec.K_t + sec.K_c

    rhs = np.zeros((n, 3, 1), dtype=complex)   # numpy 2 wants a column
    rhs[:, 0, 0] = 1.0
    X = np.linalg.solve(A, rhs)[:, :, 0]
    X_b, X_r, X_t = X[:, 0], X[:, 1], X[:, 2]
    shear = sec.G * (X_r - X_t)
    with np.errstate(divide="ignore", invalid="ignore"):
        H_o = shear / X_b
    if X_b.size == 1:
        return X_b[0], X_r[0], shear[0], H_o[0]
    return X_b, X_r, shear, H_o


def suppressed(sec: Section3, u: float, beta: float,
               route: str = "bundle", depth: float = 1.0) -> Section3:
    s = two.saturate(u)
    sec = replace(sec, gamma=sec.gamma * (1.0 - (1.0 - beta) * s))
    f = beta * depth * s
    if route == "bundle":
        return replace(sec, K_c=sec.K_c * (1.0 + f))
    if route == "tm":
        return replace(sec, K_t=sec.K_t * (1.0 - f))
    if route == "bm":
        return replace(sec, K_b=sec.K_b * (1.0 - f))
    if route == "pillar":
        return replace(sec, K_p=sec.K_p * (1.0 - f))
    raise ValueError(route)


def suppression_db(sec, u, beta, w_probe, route="bundle", depth=1.0):
    """Suppression in dB for the BM, RL and cilia read-outs."""
    b0, r0, s0, _ = response(sec, w_probe)
    b1, r1, s1, _ = response(suppressed(sec, u, beta, route, depth), w_probe)
    return (20 * math.log10(abs(b0) / abs(b1)),
            20 * math.log10(abs(r0) / abs(r1)),
            20 * math.log10(abs(s0) / abs(s1)))


def _threshold_db(sec, beta, w_probe, which, route="bundle", depth=1.0,
                  u_lo=1e-4, u_hi=1e4, tol=1e-4):
    def f(u):
        return suppression_db(sec, u, beta, w_probe, route, depth)[which] \
            - CRITERION_DB
    if f(u_hi) < 0 or f(u_lo) > 0:
        return None
    lo, hi = math.log10(u_lo), math.log10(u_hi)
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if f(10.0**mid) < 0:
            lo = mid
        else:
            hi = mid
    return 20.0 * 0.5 * (lo + hi)


def thresholds(sec, beta, w_probe, route="bundle", depth=1.0):
    """(BM, RL, cilia) suppression thresholds in dB re unit drive."""
    return tuple(_threshold_db(sec, beta, w_probe, i, route, depth)
                 for i in range(3))


def delta_T(sec, beta, w_probe, route="bundle", depth=1.0, want_rl=False):
    """BM minus cilia threshold. The RL threshold costs a third more work and
    is only chased when asked for."""
    t_bm = _threshold_db(sec, beta, w_probe, 0, route, depth)
    t_cil = _threshold_db(sec, beta, w_probe, 2, route, depth)
    t_rl = _threshold_db(sec, beta, w_probe, 1, route, depth) if want_rl else None
    if t_bm is None or t_cil is None:
        return None, (t_bm, t_rl, t_cil)
    return t_bm - t_cil, (t_bm, t_rl, t_cil)


def growth_slope(sec, beta, w_probe, t_cil_db, route="bundle", depth=1.0):
    us = [10.0**((t_cil_db + d) / 20.0) for d in (2.0, 12.0)]
    ys = [suppression_db(sec, u, beta, w_probe, route, depth)[2] for u in us]
    return (ys[1] - ys[0]) / 10.0


def matrices(sec: Section3):
    """Mass, damping and stiffness matrices. The OHC term is frequency
    independent, so it lives entirely in K, which is what makes the quadratic
    eigenvalue problem below exact rather than approximate."""
    act = sec.gamma * sec.K_a * sec.G
    M = np.diag([sec.M_b, sec.M_r, sec.M_t]).astype(float)
    C = np.array([
        [sec.R_b + sec.R_p, -sec.R_p, 0.0],
        [-sec.R_p, sec.R_r + sec.R_p + sec.R_c, -sec.R_c],
        [0.0, -sec.R_c, sec.R_t + sec.R_c]])
    K = np.array([
        [sec.K_b + sec.K_p, -sec.K_p + act, -act],
        [-sec.K_p, sec.K_p + sec.K_c - act, -sec.K_c + act],
        [0.0, -sec.K_c, sec.K_t + sec.K_c]])
    return M, C, K


def stability_margin(sec: Section3) -> float:
    """Largest real part of the poles. Negative means stable. Solves the
    quadratic eigenvalue problem (lambda^2 M + lambda C + K) x = 0 by
    companion linearization."""
    M, C, K = matrices(sec)
    Mi = np.linalg.inv(M)
    Z = np.zeros((3, 3))
    A = np.block([[Z, np.eye(3)], [-Mi @ K, -Mi @ C]])
    return float(np.max(np.linalg.eigvals(A).real))


def gain_db(sec: Section3) -> float:
    """Peak BM gain over the same section with gamma = 0."""
    passive = replace(sec, gamma=0.0)
    Xp, _, _, _ = response(passive, peak_w(passive))
    Xa, _, _, _ = response(sec, peak_w(sec))
    return 20.0 * math.log10(abs(Xa) / abs(Xp))


def gamma_critical(sec: Section3, hi: float = 200.0, tol: float = 1e-6) -> float:
    """Largest gamma that keeps every pole in the left half plane."""
    if stability_margin(replace(sec, gamma=0.0)) >= 0:
        return 0.0
    lo = 0.0
    if stability_margin(replace(sec, gamma=hi)) < 0:
        return hi
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if stability_margin(replace(sec, gamma=mid)) < 0:
            lo = mid
        else:
            hi = mid
    return lo


def gamma_for_gain(sec: Section3, target_db: float, margin: float = 1e-4):
    """Gamma giving target_db of peak BM gain, or None if unreachable while
    stable. Bisects below the instability boundary."""
    g_c = gamma_critical(sec)
    if g_c <= 0:
        return None
    hi = g_c * (1.0 - margin)
    if gain_db(replace(sec, gamma=hi)) < target_db:
        return None
    lo = 0.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if gain_db(replace(sec, gamma=mid)) < target_db:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def peak_w(sec: Section3) -> float:
    """Frequency of the BM response peak. Refined locally, because a section
    near instability has a very narrow peak that a uniform grid can miss."""
    w = np.linspace(0.4, 1.6, 1201)
    X_b, _, _, _ = response(sec, w)
    i = int(np.argmax(np.abs(X_b)))
    lo, hi = w[max(i - 1, 0)], w[min(i + 1, w.size - 1)]
    for _ in range(4):
        w = np.linspace(lo, hi, 41)
        X_b, _, _, _ = response(sec, w)
        i = int(np.argmax(np.abs(X_b)))
        lo, hi = w[max(i - 1, 0)], w[min(i + 1, w.size - 1)]
    return float(0.5 * (lo + hi))


# ---------------------------------------------------------------------------

def verify() -> bool:
    ok = True

    # 1. Stiff-pillar limit must reproduce the 2-DOF section.
    print("  stiff-pillar limit, |H_o| at the probe against the 2-DOF model:")
    ref = two.Section()
    _, _, H2 = two.response(ref, 1.0)
    H2 = abs(complex(H2))
    for K_p in (5.0, 50.0, 500.0, 5000.0):
        s3 = Section3(K_p=K_p, M_r=1e-6, R_r=0.0, R_p=0.0)
        _, _, _, H3 = response(s3, 1.0)
        H3 = abs(complex(H3))
        print(f"    K_p={K_p:7.0f}  3-DOF {H3:.5f}   2-DOF {H2:.5f}   "
              f"diff {abs(H3-H2):.2e}")
    if abs(H3 - H2) > 1e-3:
        print("  FAIL does not converge to the 2-DOF section")
        ok = False
    else:
        print("  converges, so the extra degree of freedom is the only change")

    # 2. The section must amplify somewhere. Use a compliant pillar, since a
    #    stiff one locks the RL to the BM and recovers the 2-DOF behavior.
    best = None
    for K_p in (0.2, 0.5, 1.0, 2.0):
        for r in (0.2, 0.5, 1.0, 3.0):
            base = Section3(K_p=K_p).with_ratio(r)
            Xp, _, _, _ = response(base, peak_w(base))
            act = replace(base, gamma=0.7)
            Xa, _, _, _ = response(act, peak_w(act))
            g = 20 * math.log10(abs(Xa) / abs(Xp))
            if best is None or g > best[2]:
                best = (K_p, r, g)
    print(f"  amplifies best at K_p={best[0]}, K_t/K_c={best[1]}, "
          f"{best[2]:+.1f} dB at gamma=0.7")
    if best[2] <= 0:
        print("  FAIL never amplifies")
        ok = False

    # 3. K_b is confined to row 0, and X_r/X_b and X_t/X_b are cofactors that
    #    exclude row 0, so H_o cannot depend on it. Migration still gives no
    #    gap, third degree of freedom or not. Tested with beta = 1 so that no
    #    gain-control component is mixed in.
    worst, n = 0.0, 0
    for K_p in (0.2, 0.5, 1.0, 2.0):
        for r in np.logspace(-1, 0.8, 8):
            for dep in (-2, -0.5, 0.5, 1.0):
                sec = replace(Section3(K_p=K_p).with_ratio(r), gamma=0.7)
                d, _ = delta_T(sec, 1.0, peak_w(sec), "bm", dep)
                if d is not None:
                    worst = max(worst, abs(d))
                    n += 1
    print(f"  pure K_b modulation over {n} points: |Delta-T| <= {worst:.4f} dB")
    if worst > 1e-6:
        print("  FAIL migration should still give exactly zero")
        ok = False

    # 4. What the third mass changes: gain control alone can now open a gap.
    vals = []
    for K_p in (0.2, 0.5, 1.0, 2.0):
        for r in np.logspace(-1, 0.8, 10):
            sec = replace(Section3(K_p=K_p).with_ratio(r), gamma=0.7)
            d, _ = delta_T(sec, 0.0, peak_w(sec))
            if d is not None:
                vals.append(d)
    if vals:
        print(f"  gain control alone over {len(vals)} points: Delta-T from "
              f"{min(vals):+.2f} to {max(vals):+.2f} dB")
        print("    nonzero, where the 2-DOF section gave exactly zero: the OHC "
              "now\n    pushes between two free masses, so gamma reaches H_o")
    else:
        print("  FAIL gain control gave no evaluable points")
        ok = False

    return ok


GAMMAS = (0.0, 0.3, 0.5, 0.7, 0.85)
RATIOS = tuple(np.logspace(-1.1, 0.8, 14))
DEPTHS = (-5, -2, -1.0, -0.5, 0.5, 1.0)
BETAS = (0.0, 0.5, 0.75, 1.0)
PILLARS = (0.2, 0.5, 1.0, 2.0)


def scan(route: str):
    out = []
    for K_p in PILLARS:
        for g in GAMMAS:
            for r in RATIOS:
                sec = replace(Section3(K_p=K_p).with_ratio(r), gamma=g)
                wp = peak_w(sec)
                for dep in DEPTHS:
                    for b in BETAS:
                        d, (t_bm, _, t_cil) = delta_T(sec, b, wp, route, dep)
                        if d is None:
                            continue
                        sl = growth_slope(sec, b, wp, t_cil, route, dep)
                        rl_gap = None
                        if DELTA_T_LO <= d <= DELTA_T_HI and 1.4 <= sl <= 2.6:
                            _, (_, t_rl, _) = delta_T(sec, b, wp, route, dep,
                                                      want_rl=True)
                            rl_gap = None if t_rl is None else t_rl - t_cil
                        out.append((d, sl, rl_gap, r, dep, b, g, K_p, route))
    return out


def report():
    print("\nRoute comparison")
    print(f"  {'route':>7} {'points':>7} {'max Delta-T':>12} {'max slope':>10}")
    allrows = {}
    for route in ROUTES:
        rows = scan(route)
        allrows[route] = rows
        pos = [x for x in rows if x[0] > 0]
        if pos:
            print(f"  {route:>7} {len(rows):7d} {max(x[0] for x in pos):11.1f} dB"
                  f" {max(x[1] for x in pos):9.2f}")
        else:
            print(f"  {route:>7} {len(rows):7d} {'none > 0':>12} {'':>10}")

    every = [x for rs in allrows.values() for x in rs if x[0] > 0]
    print(f"\nAll routes together, {len(every)} points with Delta-T > 0.")
    both = [x for x in every
            if DELTA_T_LO <= x[0] <= DELTA_T_HI and 1.4 <= x[1] <= 2.6]
    print(f"  satisfying observation 2 and observation 3 together: {len(both)}")
    if both:
        print(f"\n  {'route':>7} {'Delta-T':>8} {'slope':>6} {'RL-cilia':>9} "
              f"{'K_t/K_c':>8} {'depth':>6} {'beta':>5} {'gamma':>6} {'K_p':>5}")
        for x in sorted(both, key=lambda t: -t[0])[:10]:
            rl = "   n/a" if x[2] is None else f"{x[2]:9.1f}"
            print(f"  {x[8]:>7} {x[0]:8.1f} {x[1]:6.2f} {rl} {x[3]:8.3f} "
                  f"{x[4]:6.1f} {x[5]:5.2f} {x[6]:6.2f} {x[7]:5.1f}")
    else:
        arr = np.array([(x[0], x[1]) for x in every])
        hi_d = arr[arr[:, 0] >= DELTA_T_LO]
        hi_s = arr[arr[:, 1] >= 1.4]
        print(f"  best slope with Delta-T >= {DELTA_T_LO:.0f}: "
              f"{hi_d[:, 1].max() if len(hi_d) else float('nan'):.2f}")
        print(f"  best Delta-T with slope >= 1.4: "
              f"{hi_s[:, 0].max() if len(hi_s) else float('nan'):.1f} dB")


# ---------------------------------------------------------------------------
# Local refinement around the feasible region
# ---------------------------------------------------------------------------

REFINE_KP = tuple(np.logspace(-0.7, 1.2, 20))
REFINE_RATIO = tuple(np.logspace(-0.6, 0.7, 18))
REFINE_GAMMA = (0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9)
REFINE_DEPTH = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0)
REFINE_BETA = (0.8, 0.9, 1.0)


def stable_under_drive(sec: Section3, beta: float, route: str, depth: float,
                       u_lo_db: float = -60.0, u_hi_db: float = 80.0,
                       step_db: float = 1.0) -> bool:
    """
    Stable at rest AND at every drive up to u_hi_db.

    Checking only the section at rest is not enough. The stiffness matrix is
    non-reciprocal, so this is a circulatory system, and a circulatory system
    can be destabilized by adding damping (Ziegler). Every point that passed
    the at-rest test in an earlier version of this file crossed into
    oscillation at a drive 23 to 46 dB BELOW its own BM threshold, which means
    the threshold defining its Delta-T was read off a section already ringing.
    """
    if stability_margin(sec) >= 0:
        return False
    for u_db in np.arange(u_lo_db, u_hi_db + step_db, step_db):
        if stability_margin(suppressed(sec, 10.0**(u_db / 20.0), beta,
                                       route, depth)) >= 0:
            return False
    return True


def refine(route: str = "pillar", stable_only: bool = True):
    """
    Fine scan of the region where observations 2 and 3 are both met.

    stable_only rejects sections with a pole in the right half plane. It
    defaults to True because an earlier version of this scan lacked the check
    and reported 80 feasible points of which 67 were spontaneously
    oscillating. A section that rings on its own is not a model of anything.
    """
    feasible, curve = [], []
    for K_p in REFINE_KP:
        for r in REFINE_RATIO:
            for g in REFINE_GAMMA:
                sec = replace(Section3(K_p=K_p).with_ratio(r), gamma=g)
                if stable_only and stability_margin(sec) >= 0:
                    continue
                wp = peak_w(sec)
                for dep in REFINE_DEPTH:
                    for b in REFINE_BETA:
                        d, (_, _, t_cil) = delta_T(sec, b, wp, route, dep)
                        if d is None or d <= 0:
                            continue
                        sl = growth_slope(sec, b, wp, t_cil, route, dep)
                        curve.append((d, sl))
                        if not (DELTA_T_LO <= d <= DELTA_T_HI
                                and 1.4 <= sl <= 2.6):
                            continue
                        if stable_only and not stable_under_drive(
                                sec, b, route, dep):
                            continue
                        _, (_, t_rl, _) = delta_T(sec, b, wp, route, dep,
                                                  want_rl=True)
                        feasible.append((d, sl,
                                         None if t_rl is None else t_rl - t_cil,
                                         K_p, r, dep, b, g))
    return feasible, np.array(curve)


def refine_report(route: str = "pillar"):
    feasible, curve = refine(route)
    print(f"\nRefinement on the {route} route, {len(curve)} points with "
          f"Delta-T > 0")
    print(f"  {'Delta-T (dB)':>16} {'n':>6} {'max slope':>10}")
    edges = [0, 2, 4, 6, 8, 10, 14, 18, 22, 26, 32]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (curve[:, 0] >= lo) & (curve[:, 0] < hi)
        if m.sum():
            print(f"  {lo:6.0f} to {hi:<6.0f} {m.sum():6d} "
                  f"{curve[m, 1].max():10.2f}")
    hd = curve[curve[:, 0] >= DELTA_T_LO]
    hs = curve[curve[:, 1] >= 1.4]
    print(f"\n  best slope with Delta-T >= {DELTA_T_LO:.0f}: "
          f"{hd[:, 1].max() if len(hd) else float('nan'):.2f}")
    print(f"  best Delta-T with slope >= 1.4: "
          f"{hs[:, 0].max() if len(hs) else float('nan'):.1f} dB")
    print(f"\n  points meeting observations 2 and 3 together: {len(feasible)}")
    if not feasible:
        return
    a = np.array([[h[0], h[1], h[3], h[4], h[5], h[6], h[7]] for h in feasible])
    names = ["Delta-T", "slope", "K_p", "K_t/K_c", "depth", "beta", "gamma"]
    print(f"\n  {'parameter':>10} {'min':>9} {'max':>9} {'median':>9}")
    for i, n in enumerate(names):
        print(f"  {n:>10} {a[:, i].min():9.3f} {a[:, i].max():9.3f} "
              f"{np.median(a[:, i]):9.3f}")
    rl = [h[2] for h in feasible if h[2] is not None]
    if rl:
        print(f"  {'RL-cilia':>10} {min(rl):9.3f} {max(rl):9.3f} "
              f"{np.median(rl):9.3f}")
        print("\n  The RL threshold equals the cilia threshold at every "
              "feasible point,\n  with the BM well above both. That is the "
              "second row of the outcome\n  table in Section 5.2 of the "
              "manuscript.")


def refine_at_gain(target_db: float = 40.0, route: str = "pillar"):
    """Same refinement, but gamma is solved at each point to deliver
    target_db of peak BM gain rather than taken from a grid."""
    feasible, curve, reachable, total = [], [], 0, 0
    for K_p in REFINE_KP:
        for r in REFINE_RATIO:
            base = Section3(K_p=K_p).with_ratio(r)
            total += 1
            g = gamma_for_gain(base, target_db)
            if g is None:
                continue
            reachable += 1
            sec = replace(base, gamma=g)
            if stability_margin(sec) >= 0:      # belt and braces
                continue
            wp = peak_w(sec)
            for dep in REFINE_DEPTH:
                for b in REFINE_BETA:
                    d, (_, _, t_cil) = delta_T(sec, b, wp, route, dep)
                    if d is None or d <= 0:
                        continue
                    sl = growth_slope(sec, b, wp, t_cil, route, dep)
                    curve.append((d, sl))
                    if not (DELTA_T_LO <= d <= DELTA_T_HI and 1.4 <= sl <= 2.6):
                        continue
                    _, (_, t_rl, _) = delta_T(sec, b, wp, route, dep,
                                              want_rl=True)
                    feasible.append((d, sl,
                                     None if t_rl is None else t_rl - t_cil,
                                     K_p, r, dep, b, g))
    return feasible, np.array(curve), reachable, total


def gain_report(target_db: float = 40.0, route: str = "pillar"):
    feasible, curve, reachable, total = refine_at_gain(target_db, route)
    print(f"\nRefinement with gamma solved for {target_db:.0f} dB of gain, "
          f"{route} route")
    print(f"  {reachable} of {total} (K_p, K_t/K_c) pairs can reach "
          f"{target_db:.0f} dB while stable")
    if not len(curve):
        print("  no evaluable points")
        return
    print(f"  {len(curve)} points with Delta-T > 0")
    hd = curve[curve[:, 0] >= DELTA_T_LO]
    hs = curve[curve[:, 1] >= 1.4]
    print(f"  best slope with Delta-T >= {DELTA_T_LO:.0f}: "
          f"{hd[:, 1].max() if len(hd) else float('nan'):.2f}")
    print(f"  best Delta-T with slope >= 1.4: "
          f"{hs[:, 0].max() if len(hs) else float('nan'):.1f} dB")
    print(f"  points meeting observations 2 and 3 together: {len(feasible)}")
    if not feasible:
        return
    a = np.array([[h[0], h[1], h[3], h[4], h[5], h[6], h[7]] for h in feasible])
    names = ["Delta-T", "slope", "K_p", "K_t/K_c", "depth", "beta", "gamma"]
    print(f"\n  {'parameter':>10} {'min':>9} {'max':>9} {'median':>9}")
    for i, n in enumerate(names):
        print(f"  {n:>10} {a[:, i].min():9.3f} {a[:, i].max():9.3f} "
              f"{np.median(a[:, i]):9.3f}")
    rl = [h[2] for h in feasible if h[2] is not None]
    if rl:
        print(f"  {'RL-cilia':>10} {min(rl):9.3f} {max(rl):9.3f} "
              f"{np.median(rl):9.3f}")


def main(argv):
    print("Verification")
    ok = verify()
    if "--verify" in argv:
        return 0 if ok else 1
    if "--refine" in argv:
        refine_report()
        return 0 if ok else 1
    if "--gain" in argv:
        i = argv.index("--gain")
        tgt = float(argv[i + 1]) if len(argv) > i + 1 else 40.0
        gain_report(tgt)
        return 0 if ok else 1
    report()
    refine_report()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
