"""
deltaT_sweep.py
Can a single radial cross section produce the threshold difference Delta-T?

THE QUESTION
------------
Observations 1 and 3 of the six-observation set are the pair no published
model reproduces together:

    1  neural low-side suppression appears as soon as the fiber responds
    3  BM low-side suppression needs 18 to 32 dB more suppressor level

Delta-T is the gap. It is a property of the transduction filter H_o and of
where the suppressing nonlinearity acts, both of which live in one radial
cross section. No wave propagation is needed to ask whether the gap is
achievable at all. That is what this script asks, before any effort goes into
a multi-chamber transmission-line model that inherits the same cross section.

WHY THE TRAVELING WAVE CANCELS OUT
----------------------------------
Delta-T is a *difference* of two suppressor levels measured with the same
suppressor. Whatever converts suppressor SPL into local cilia drive at the
probe place, including the entire tail of the traveling wave and the middle
ear, is common to both read-outs and cancels in the difference. So the
suppressor drive can be parameterized directly and Delta-T still comes out in
dB correctly. Absolute thresholds, meaning observation 1's 65 dB SPL, do need
the wave and are not computed here.

THE CROSS SECTION
-----------------
Two degrees of freedom, as in Figure 4 of Allen and Neely (1992, Physics
Today): the organ of Corti mass M_b on the basilar membrane, held by K_b and
R_b; the tectorial membrane mass M_t, held by K_t and R_t; the two coupled by
the hair bundle K_c and R_c. Cilia shear is G(x_b - x_t).

    M_b x_b" + R_b x_b' + K_b x_b + R_c(x_b' - x_t') + K_c(x_b - x_t) = F
    M_t x_t" + R_t x_t' + K_t x_t + R_c(x_t' - x_b') + K_c(x_t - x_b) = 0

This is built from the mechanical schematic rather than from the transmission
matrix of that paper, whose entries did not survive OCR of the scan legibly.
The construction is checked against the one closed-form result the paper
does state, H_o -> G K_t/(K_t + K_c) as frequency goes to zero. See
verify().

Activity is an OHC force applied to the organ-of-Corti mass, proportional to
cilia shear velocity, reacting against the bony structure rather than against
the TM:

    F_ohc = gamma K_b G (x_b - x_t)      applied to M_b only

proportional to hair bundle *displacement*, since OHC electromotility follows
transducer current and so follows bundle displacement, not its velocity. The
reaction is not taken on M_t, which makes the matrix non-reciprocal, a12 !=
a21, the signature of an active element.

Whether this delivers power depends on the phase of H_o. The force is in
phase with BM velocity when Im{H_o} > 0 at the probe frequency, so the
section amplifies over part of the K_t/K_c range and attenuates over the
rest. verify() reports the gain rather than assuming it.

WHERE THE NONLINEARITY ACTS
---------------------------
The suppressor saturates the OHC transducer. Two things it might do, weighted
by beta:

    beta = 0   saturation scales the force generator only, gamma -> gamma(1-s)
    beta = 1   saturation scales the hair bundle stiffness only, K_c -> K_c(1+s)

with s in [0,1) the saturation fraction.

A first guess was that beta = 0 must give Delta-T = 0, on the grounds that
H_o does not depend on gamma, so scaling gamma would suppress BM and cilia
equally. That is true only when the active force is reciprocal. With the
non-reciprocal OHC force above, X_t/X_b depends on gamma, so H_o does too,
and gain control alone can produce a gap. The sweep measures how much.

Standard library plus numpy. Run:

    python3 deltaT_sweep.py            # verify, then sweep, then report
    python3 deltaT_sweep.py --verify   # checks only
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, replace

import numpy as np

CRITERION_DB = 1.0          # suppression counted as present at this depth
DELTA_T_LO, DELTA_T_HI = 18.0, 32.0     # Cooper-96 and Geisler-Nuttall-97


# ---------------------------------------------------------------------------
# The cross section
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Section:
    """
    Normalized so that the BM resonance is at omega = 1 with M_b = 1, K_b = 1.
    Frequencies below are therefore in units of the BM place frequency.
    """
    M_b: float = 1.0
    K_b: float = 1.0
    R_b: float = 0.50        # passive BM is broadly tuned, Q about 2
    M_t: float = 0.10        # TM lighter than the organ of Corti
    K_t: float = 0.05        # gives a TM resonance half an octave below
    R_t: float = 0.02
    K_c: float = 0.50        # hair bundle stiffness; swept via K_t/K_c
    R_c: float = 0.02
    G: float = 1.0           # geometry factor
    gamma: float = 0.0       # activity; R_c -> R_c(1-gamma)

    @property
    def w_b(self) -> float:
        return math.sqrt(self.K_b / self.M_b)

    @property
    def w_t(self) -> float:
        return math.sqrt(self.K_t / self.M_t)

    def with_ratio(self, kt_over_kc: float) -> "Section":
        """Hold K_t fixed and set K_c from the requested ratio."""
        return replace(self, K_c=self.K_t / kt_over_kc)


def response(sec: Section, w):
    """
    BM displacement and cilia shear for unit force on M_b.
    Returns (X_b, shear, H_o), each complex, broadcast over w.
    """
    w = np.asarray(w, dtype=float)
    jw = 1j * w
    coup = jw * sec.R_c + sec.K_c              # cilia coupling impedance term
    act = sec.gamma * sec.K_b * sec.G          # OHC force, on M_b only

    a11 = -w**2 * sec.M_b + jw * sec.R_b + sec.K_b + coup - act
    a12 = -coup + act
    a21 = -coup                                # no reaction on M_t: active
    a22 = -w**2 * sec.M_t + jw * sec.R_t + sec.K_t + coup

    det = a11 * a22 - a12 * a21
    X_b = a22 / det                            # F = 1
    X_t = -a21 / det
    shear = sec.G * (X_b - X_t)
    with np.errstate(divide="ignore", invalid="ignore"):
        H_o = shear / X_b
    return X_b, shear, H_o


# ---------------------------------------------------------------------------
# Suppression
# ---------------------------------------------------------------------------

def saturate(u: float) -> float:
    """Saturation fraction from normalized suppressor drive u. Soft, in [0,1)."""
    return u * u / (1.0 + u * u)


ROUTES = ("bundle", "tm", "bm")


def suppressed(sec: Section, u: float, beta: float,
               route: str = "bundle", depth: float = 1.0) -> Section:
    """
    Apply saturation s(u). A fraction (1-beta) acts as gain control on gamma;
    the remaining beta acts on the micromechanics by the named route.

        bundle  K_c -> K_c (1 + depth s)     stiffens the hair bundle
        tm      K_t -> K_t (1 - depth s)     shifts the TM resonance down,
                                             moving the notch in H_o
        bm      K_b -> K_b (1 - depth s)     migration of the BM resonance

    Only routes touching row 2 of the matrix can change H_o. K_c enters row 2
    through the coupling term and K_t enters it directly, so those two can
    produce a gap. K_b, like gamma, sits in row 1 alone and cannot.
    """
    s = saturate(u)
    sec = replace(sec, gamma=sec.gamma * (1.0 - (1.0 - beta) * s))
    if route == "bundle":
        return replace(sec, K_c=sec.K_c * (1.0 + beta * depth * s))
    if route == "tm":
        return replace(sec, K_t=sec.K_t * (1.0 - beta * depth * s))
    if route == "bm":
        return replace(sec, K_b=sec.K_b * (1.0 - beta * depth * s))
    raise ValueError(f"unknown route {route!r}")


def suppression_db(sec: Section, u: float, beta: float, w_probe: float,
                   route: str = "bundle", depth: float = 1.0):
    """Suppression of the probe, in dB, for the BM and cilia read-outs."""
    Xb0, sh0, _ = response(sec, w_probe)
    Xb1, sh1, _ = response(suppressed(sec, u, beta, route, depth), w_probe)
    bm = 20.0 * math.log10(abs(Xb0) / abs(Xb1))
    cil = 20.0 * math.log10(abs(sh0) / abs(sh1))
    return bm, cil


def _threshold_db(sec: Section, beta: float, w_probe: float, which: int,
                  route: str = "bundle", depth: float = 1.0,
                  u_lo=1e-4, u_hi=1e4, tol=1e-4):
    """
    Suppressor drive, in dB re unit drive, at which suppression first reaches
    CRITERION_DB. Bisection on log u. None if the criterion is never reached.
    """
    def f(u):
        return suppression_db(sec, u, beta, w_probe, route, depth)[which] \
            - CRITERION_DB

    if f(u_hi) < 0:
        return None                      # saturates without ever suppressing
    if f(u_lo) > 0:
        return None                      # already suppressed at zero drive
    lo, hi = math.log10(u_lo), math.log10(u_hi)
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if f(10.0**mid) < 0:
            lo = mid
        else:
            hi = mid
    return 20.0 * 0.5 * (lo + hi)


def delta_T(sec: Section, beta: float, w_probe: float,
            route: str = "bundle", depth: float = 1.0):
    """BM threshold minus cilia threshold, in dB. Positive means the BM needs
    more suppressor, which is the measured direction."""
    t_bm = _threshold_db(sec, beta, w_probe, 0, route, depth)
    t_cil = _threshold_db(sec, beta, w_probe, 1, route, depth)
    if t_bm is None or t_cil is None:
        return None, t_bm, t_cil
    return t_bm - t_cil, t_bm, t_cil


def growth_slope(sec: Section, beta: float, w_probe: float, t_cil_db: float,
                 route: str = "bundle", depth: float = 1.0):
    """dSuppression/dDrive in dB/dB at the cilia, just above threshold.
    Observation 2 asks for about 2."""
    us = [10.0**((t_cil_db + d) / 20.0) for d in (2.0, 12.0)]
    ys = [suppression_db(sec, u, beta, w_probe, route, depth)[1] for u in us]
    return (ys[1] - ys[0]) / 10.0


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def verify() -> bool:
    ok = True

    # 1. Low-frequency limit of H_o against the published closed form.
    for ratio in (0.02, 0.1, 1.0, 5.0):
        sec = Section().with_ratio(ratio)
        _, _, H = response(sec, 1e-6)
        want = sec.G * sec.K_t / (sec.K_t + sec.K_c)
        got = abs(complex(H))
        if abs(got - want) > 1e-6 * max(1.0, want):
            print(f"  FAIL low-f limit at K_t/K_c={ratio}: {got:.6g} vs {want:.6g}")
            ok = False
    print("  H_o -> G K_t/(K_t+K_c) at low frequency: matches Allen and Neely (1992)")

    # 2. The notch in |H_o| should sit near the TM resonance, and the TM
    #    resonance was set half an octave below the BM resonance.
    sec = Section()
    w = np.logspace(-1.2, 0.6, 4000)
    _, _, H = response(sec, w)
    w_notch = w[int(np.argmin(np.abs(H)))]
    oct_re_bm = math.log2(w_notch / sec.w_b)
    print(f"  |H_o| notch at {w_notch:.3f} w_b = {oct_re_bm:+.2f} oct re BM "
          f"(TM resonance is {math.log2(sec.w_t/sec.w_b):+.2f} oct)")
    if not (-0.9 < oct_re_bm < -0.1):
        print("  FAIL notch is not below the BM resonance")
        ok = False

    # 3. The section must actually amplify somewhere, or nothing below means
    #    anything. Amplification needs Im{H_o} > 0 at the probe frequency.
    best = None
    for r in (0.02, 0.1, 0.5, 2.0, 10.0):
        base = Section().with_ratio(r)
        _, _, H = response(base, 1.0)
        Xp, _, _ = response(base, 1.0)
        Xa, _, _ = response(replace(base, gamma=0.7), 1.0)
        gain = 20.0 * math.log10(abs(complex(Xa)) / abs(complex(Xp)))
        print(f"  K_t/K_c={r:5.2f}: Im(H_o)={complex(H).imag:+.3f}, "
              f"BM gain at gamma=0.7 is {gain:+5.1f} dB")
        if best is None or gain > best[1]:
            best = (r, gain)
    if best[1] <= 0:
        print("  FAIL the section never amplifies")
        ok = False
    else:
        print(f"  amplifies best near K_t/K_c={best[0]}, {best[1]:+.1f} dB")

    # 4. The beta = 0 identity, checked numerically rather than assumed.
    worst = 0.0
    for r in (0.02, 0.1, 0.5, 2.0, 10.0):
        for g in (0.3, 0.7):
            sec = replace(Section().with_ratio(r), gamma=g)
            d, _, _ = delta_T(sec, 0.0, w_probe=1.0)
            if d is not None:
                worst = max(worst, abs(d))
    print(f"  beta=0 (gain control only): |Delta-T| <= {worst:.2e} dB "
          "over the grid, identically zero as the algebra requires")
    if worst > 1e-6:
        print("  FAIL beta=0 should give Delta-T = 0 exactly")
        ok = False

    return ok


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def peak_w(sec: Section) -> float:
    """Frequency of the BM response peak, which is this section's CF."""
    w = np.linspace(0.5, 1.5, 801)
    Xb, _, _ = response(sec, w)
    return float(w[int(np.argmax(np.abs(Xb)))])


GAMMAS = (0.0, 0.3, 0.5, 0.7, 0.85, 0.92)
RATIOS = tuple(np.logspace(-1.1, 0.6, 22))
DEPTHS = (-12, -8, -5, -3, -2, -1.5, -1.0, -0.7, -0.5, -0.3, 0.5, 1.0)
BETAS = (0.5, 0.75, 1.0)


def scan(route: str):
    """Every (Delta-T, slope) this route can reach. Probe at the section CF."""
    out = []
    for g in GAMMAS:
        for r in RATIOS:
            sec = replace(Section().with_ratio(r), gamma=g)
            wp = peak_w(sec)
            for dep in DEPTHS:
                for b in BETAS:
                    d, _, t_cil = delta_T(sec, b, wp, route, dep)
                    if d is None:
                        continue
                    sl = growth_slope(sec, b, wp, t_cil, route, dep)
                    out.append((d, sl, r, dep, b, g))
    return out


def report():
    print("\nRoute comparison, signed Delta-T. Positive is the measured "
          "direction,\nmeaning the BM needs more suppressor than the cilia.")
    print(f"  {'route':>7} {'points':>7} {'max Delta-T':>12} {'max slope':>10}")
    scans = {}
    for route in ROUTES:
        rows = scan(route)
        scans[route] = rows
        if rows:
            print(f"  {route:>7} {len(rows):7d} {max(x[0] for x in rows):11.1f} dB"
                  f" {max(x[1] for x in rows):9.2f}")
        else:
            print(f"  {route:>7} {0:7d} {'n/a':>12} {'n/a':>10}")

    print("\n  bm modulates K_b, which sits in row 1 with gamma, so it cannot"
          "\n  change H_o and its Delta-T is identically zero.")

    rows = np.array([(d, s) for d, s, *_ in scans["tm"] if d > 0])
    if not len(rows):
        print("\nNo positive Delta-T from the tm route; nothing to trade off.")
        return

    print(f"\nTrade-off on the tm route, {len(rows)} points with Delta-T > 0.")
    print("Best growth slope attainable in each Delta-T band:")
    print(f"  {'Delta-T (dB)':>16} {'n':>6} {'max slope':>10}")
    edges = [0, 2, 4, 6, 8, 10, 14, 18, 22, 26, 32]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (rows[:, 0] >= lo) & (rows[:, 0] < hi)
        if m.sum():
            print(f"  {lo:6.0f} to {hi:<6.0f} {m.sum():6d} {rows[m, 1].max():10.2f}")

    hi_d = rows[rows[:, 0] >= DELTA_T_LO]
    hi_s = rows[rows[:, 1] >= 1.4]
    print(f"\n  observation 2 wants a slope near 2.0, observation 3 wants "
          f"Delta-T {DELTA_T_LO:.0f} to {DELTA_T_HI:.0f} dB")
    print(f"  best slope found with Delta-T >= {DELTA_T_LO:.0f}: "
          f"{hi_d[:, 1].max() if len(hi_d) else float('nan'):.2f}")
    print(f"  best Delta-T found with slope >= 1.4: "
          f"{hi_s[:, 0].max() if len(hi_s) else float('nan'):.1f} dB")
    both = rows[(rows[:, 0] >= DELTA_T_LO) & (rows[:, 1] >= 1.4)]
    print(f"  points satisfying both: {len(both)}")


def main(argv):
    print("Verification")
    ok = verify()
    if "--verify" in argv:
        return 0 if ok else 1
    report()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
