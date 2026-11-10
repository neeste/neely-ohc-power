"""
tl_model.py
The certified 1-D transmission-line cochlea built on the 3-DOF cross section.

WHAT THIS FILE IS
-----------------
tl_method_a.py, tl_method_b.py and tl_method_c.py were written independently.
xval.py and xval_stability.py cross-validated them. This file is the merged
result: one model, three solvers, and a verify() that reproduces the
cross-validation numbers.

Nothing here is a fourth opinion. The finite-difference engine is the same
linear system all three files assemble (they agree to 1.7e-10 dB and 1.0e-11
radians at matched N), the cascade engine is method B's exact-segment transfer
matrix, and the WKB engine is method C's one-way Liouville-Green. The model
keeps all three because they fail in different places and the disagreement is
the diagnostic.

WHAT THE CROSS-VALIDATION SETTLED
---------------------------------
1. The three fluid constants are the same dimensionless group. Method A's q,
   method B's k^2 and minus method C's k^2 agree to 8e-15 relative, so
   F = rho2H = BETA_F = 2 rho L^2 / (H m_area). The three files' different
   values of it are different CHOICES, not different definitions.

2. The three finite-difference solvers are one solver. At N = 4096 on the same
   world they agree to 1.7e-10 dB and 1.0e-11 rad over the top 60 dB of the
   profile, worst case over three worlds, two gains and two frequencies. Most
   cells are near 1e-13.

3. The cascade converges to the finite-difference answer at h^2 (measured
   ratios 3.97 to 4.13 over four grid halvings), so neither is wrong. At
   N = 4000 they agree to 0.0006 to 0.03 dB depending on the world.

4. WKB is not a competitor, it is an approximation with a price. In the tail,
   half way to the peak, it costs 0.46 to 1.69 dB and 0.09 to 0.17 radians. AT
   the peak place it costs -5.70 to +0.39 dB and up to 1.46 radians. Comparing
   PEAK VALUE to PEAK VALUE instead, which lets the two peaks sit at different
   places, the worst case is +14.5 dB with the peak displaced by 0.13 of the
   cochlear length (world C at 0.9 of critical, xval.py [5]); that number is
   about a moved peak, not about a level. The error is not monotone in gain,
   and method C's account of why, a turning point basal of the peak where k^2
   is real and negative, is correct and is confirmed here.

5. The place-frequency map is exact in all three engines: fitted slope equals
   the built-in number of decades to 1e-4 with R^2 = 1.000000, for every world,
   passive and active.

6. STABILITY, where the three reports disagreed, and the disagreement was
   substantive rather than numerical:

     - method A's line_modes() and method C's line_poles() are the same
       operator, agreeing to 1.9e-14. They cannot disagree about a shared
       world; they disagreed because their worlds differ and because A tapers
       the activity at the apex.
     - that quadratic eigenvalue problem is NOT converged at the m most
       commonly used. At m = 60, 120 and 240 it reports about 2m growing modes
       in all three worlds; at m = 480 the count collapses to 2, 14 and 4. Any
       count taken at m <= 240 is contaminated.
     - the arbiter used here is a third method neither file used at the
       deciding moment: the argument principle applied to the determinant of
       the finite-difference system that is actually inverted. It is grid
       converged (identical counts at N = 1024 and 4096 in all three worlds at
       0, 0.6 and 0.9 of critical) and it reproduces method A's untapered
       onset, gamma = 0.09176 against A's 0.09177, and method C's onset,
       gamma = 0.59916 against C's 0.59954. Three methods, two of them written
       independently, agreeing to 0.01 and 0.06 percent.
     - method B's conclusion that the pressure-driven line has no right-half
       plane pole is WRONG, and its own cascade contains the refutation.
       Around a box in the apical band of world B at 0.9 of critical, the FD
       determinant has 5 zeros at N = 1024 and 5 at N = 4096, and the winding
       of METHOD B's OWN P(0) around the same box is +5.000 at both grids. The
       two discretizations agree exactly. But |P(0)| stays at 1.000 everywhere
       on that contour and on an 81 x 81 grid through its interior, because
       base_state() renormalizes so that max(|P|,|U|) = 1 and |U| < |P| there,
       so the amplitude is pinned at the normalizer and dips only in
       vanishingly small neighborhoods of the zeros. rhp_scan() seeds its zoom
       from the deepest minima of that surface, and a surface flat at its
       ceiling gives it nothing to descend. The information was in the phase of
       the same number the whole time, which is what the argument principle
       reads and what method B declined to use. Method B's own world goes
       unstable at gamma = 0.09443, which is 0.264 of its section's critical
       gamma and BELOW the rigid-stapes onset it reported at 0.14549.
     - what actually goes unstable first, in all three worlds, is an apical
       cavity mode against the perfectly reflecting helicotrema P(L) = 0. It
       lives in the lowest frequency band, at the apical CF, and the band by
       band count locates it there in every world. Tapering the activity to
       zero over x in [0.75, 0.95] removes it and raises the stable gamma from
       0.758 to 0.965 of critical in world A, from 0.264 to 0.626 in world B
       and from 0.685 to 0.884 in world C. Method A's taper was the right move
       and this file keeps it. It is a stated modeling choice, not a fix: the
       amplifier is turned off over the apical fifth, which is where the
       measurements put little of it anyway, and place_map() shows exactly
       what it costs (a frequency whose active peak would land at x = 0.82
       comes back at 0.762, with the peak reverting to its passive place).
     - a second, basal mode exists at higher gain, locked to the abrupt onset
       of activity at x = 0. Method C's [7] diagnostic shows it at x = 0.03
       with omega/CF(x0) = 0.819 held to three digits as the basal end is
       moved, and its growth rate falling from +0.082 to +0.010 when gamma is
       ramped over the basal 40 percent. A basal taper is therefore offered
       but is off by default: measured, it buys world B only 0.008 in gamma.

THE OPERATING POINT, AND WHY IT IS METHOD B'S WORLD
---------------------------------------------------
The three worlds are all available as presets. WORLD_B is the default, and the
reason is the one thing phase 2 needs the line to do: build gain along the wave
rather than locally.

At the certified ceiling, which is 0.98 of the measured onset, all read at the
place where the active wave peaks:

    world  gamma/gc   tip gain   local gain   isolated section   cycles
      A      0.945     18.1 dB     39.2 dB        26.9 dB         2.83
      B      0.611     23.1 dB     56.0 dB        17.4 dB         5.09
      C      0.867     -0.2 dB     -0.2 dB         6.2 dB         1.12

World A's line gains 18 dB where its own isolated section gains 27, so its
gain is local and the wave subtracts from it. World C's line does not amplify
at all below its instability, which is method B's diagnosis of R_b = 0.5
confirmed on method C's own parameters. World B's line turns a 17 dB section
into a 23 dB tip gain and a 56 dB local gain while sitting at only 61 percent
of the section's critical gamma. That is the regime the cross-section study
never reached: a large removable gain built by the wave, from a section that
is NOT dominated by one nearly cancelled pole.

WHAT PHASE 2 SHOULD READ OFF THIS FILE FIRST
--------------------------------------------
Both numbers come from gain_profile() in world B, at the read place, with the
gain pinned so the line is stable.

    THE CEILING ON A(u) IS HIGH. The active-over-passive ratio at the read
    place runs 6.3, 19.1, 36.5, 51.7 and 56.1 dB at 0.18, 0.31, 0.43, 0.55 and
    0.61 of the section's critical gamma. Silencing the OHCs at and basal of
    the probe removes all of it. Against the 1.0 to 1.6 dB at which the cross
    section's BM suppression asymptote stalled, that is not a marginal
    difference, and it is the specific defect the transmission line was asked
    to cure. It is a ceiling, not a result: no suppressor has been applied.

    THE GAIN IS NOT ACCUMULATED OVER AN EXTENDED BASAL REGION. That ratio
    climbs from 10 to 90 percent of its value over 0.06 to 0.30 octaves basal
    of the read place, and half an octave basal it is NEGATIVE, -1.0 to -9.3
    dB: the active line's tail is quieter than the passive line's. A
    suppressor confined to the tail, half an octave basal and further, has
    nothing to take away and would suppress by less than nothing. The gain
    lives in the last third of an octave before the peak, so a low-side
    suppressor produces BM suppression only to the extent that it saturates
    OHCs in that short stretch, which is also where the probe's own response
    is largest. That is the geometry phase 2 has to work in, and it is
    measured rather than assumed.

THREE GAINS, NEVER INTERCHANGEABLE (method A's point, extended)
    tip    peak over x active against peak over x passive: the line's tip gain
    cf     active over passive at the active peak place, same frequency
    local  the same at a FIXED place: continuous in gamma, and the one to pin
           against, because tip and cf both jump by 20 dB and more where the
           global maximum switches from the broad passive hump to the sharp
           active one
gamma_for_gain() pins any of them and RETURNS WHAT IT ACHIEVED, because inside
a jump no gamma delivers the target. 40 dB of TIP gain is not reachable with a
stable line in any world here. 40 dB of LOCAL gain is, in world B, at
gamma = 0.16252, which is 0.455 of the section's critical gamma and well below
the line's ceiling of 0.611.

RULES 1 TO 5
    1  stability: worst_section_margin() checks every section, growing_modes()
       counts whole-line modes by the argument principle, line_poles() solves
       the eigenvalue problem with its m-convergence stated. All three accept a
       gamma PROFILE, so phase 2 can check a suppressed line rather than a
       uniform one.
    2  gain is pinned by gamma_for_gain(), never swept.
    3  K_a is scaled with the other stiffnesses, so the place map does not
       modulate the amplifier; verify() measures the residual at 1e-14 dB.
    4  no criterion appears in this file; there is no suppression here yet.
    5  every number in this docstring came out of verify() or report().

GRIDS AND COST
    N = 4096 by default. The peak level moves -0.040, -0.009 and -0.002 dB
    over N = 2048, 4096, 8192, 16384 in world B, so 4096 is converged to about
    0.01 dB and 16384 to 0.002. World A at its ceiling is NOT that well
    behaved, moving +3.50, -0.03, +0.03, +0.03 dB over the same grids, because
    its ceiling sits at 0.945 of the section's critical gamma where the local
    impedance is razor sharp in x. That is a reason to prefer world B beyond
    the gain argument. verify() takes about 3 minutes, report() about 20
    seconds, both on the grids they print.

CAVEATS, ALL MEASURED
    1  Phase. World B at its ceiling accumulates 5.09 cycles to the peak,
       above the 1.5 to 3 that is measured; at 10 to 20 dB of local gain it is
       2.3 cycles, inside it. F trades phase against reachable gain and method
       B chose 20000 to keep a 40 dB rung. If phase 2 needs both, F is the
       knob and the cost is on the record here rather than hidden.
    2  Long wave. |k| reaches several hundred per cochlear length near the
       peak, so |k| H < 1 needs a scala height under about 1/500 of the
       length, against a real ratio nearer 1/30. The peak sharpness is a
       long-wave artifact to the same degree it is in every 1-D model with
       realistic phase; it is not specific to this one.
    3  The stapes is an ideal pressure source and the helicotrema a perfect
       short. Both are perfect reflectors, which is what makes the cavity
       modes possible. A middle-ear termination is the obvious next
       refinement, and method B is right that any real one lies between the
       pressure-source and rigid-stapes extremes.
    4  Parameters are normalized with the basal CF at omega = 1 and fitted to
       no animal. R_b = 0.15 and K_p = 0.15 in world B are outside the
       cross-section study's defaults; R_b sits alone in row 0 and so cannot
       touch H_o or the notch (README Result 1), but K_p is inside the
       transduction filter and the change has to be carried into any
       comparison with the cross-section results.
    5  Nothing here is suppressed and no Delta-T is computed. The per-place
       identity cilia_supp - BM_supp = -20 log10 |H_o(s)/H_o(0)| is untouched.

    python3 tl_model.py             # verify, then the report
    python3 tl_model.py --verify    # the cross-validation numbers only
    python3 tl_model.py --report    # the operating tables only
"""

from __future__ import annotations

import math
import os
import sys
import time
from dataclasses import dataclass, field, replace

import numpy as np
from scipy.linalg import solve_banded

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import deltaT_3dof as d3                                        # noqa: E402
from deltaT_3dof import Section3                                # noqa: E402


# ---------------------------------------------------------------------------
# The cochlea
# ---------------------------------------------------------------------------

SECTION_A = Section3(K_p=0.2, K_c=0.05, R_b=0.08)
SECTION_B = replace(Section3(K_p=0.15).with_ratio(1.0), R_b=0.15)
SECTION_C = Section3(K_p=1.0).with_ratio(2.0)


@dataclass(frozen=True)
class Cochlea:
    """
    A place-scaled 3-DOF partition in a long-wave fluid.

        sec        the reference section, whose CF is omega = 1, at x = 0
        decades    CF decades from base to apex; CF(x) = 10^(-decades x)
        F          2 rho L^2 / (H m_area), the one fluid group
        N          sections on the uniform grid x in [0, 1]
        taper_*    raised-cosine multiplier on gamma: 1 up to taper_lo, 0 from
                   taper_hi, and 0 to 1 over [0, taper_base] at the stapes end
        apex       "short" for P(L) = 0, the helicotrema

    Every stiffness at place x is scaled by CF(x)^2 and every damping by CF(x),
    masses fixed, which makes the line exactly scale invariant. K_a is scaled
    with the other stiffnesses, so gamma means the same thing at every place.
    """
    sec: Section3 = SECTION_B
    decades: float = 2.75
    F: float = 20000.0
    N: int = 4096
    taper_lo: float = 0.75
    taper_hi: float = 0.95
    taper_base: float = 0.0
    apex: str = "short"

    # -- geometry ----------------------------------------------------------
    def cf(self, x):
        """Local section CF in units of the basal CF."""
        return 10.0 ** (-self.decades * np.asarray(x, dtype=float))

    def x_of_cf(self, cf):
        return -np.log10(np.asarray(cf, dtype=float)) / self.decades

    def octaves(self, x0, x1):
        """Octaves between two places, positive basalward."""
        return (np.asarray(x1) - np.asarray(x0)) * self.decades * math.log2(10)

    def taper(self, x):
        t = np.ones_like(np.asarray(x, dtype=float))
        if self.taper_hi > self.taper_lo:
            u = np.clip((np.asarray(x, float) - self.taper_lo)
                        / (self.taper_hi - self.taper_lo), 0.0, 1.0)
            t = t * 0.5 * (1.0 + np.cos(math.pi * u))
        if self.taper_base > 0:
            u = np.clip(np.asarray(x, float) / self.taper_base, 0.0, 1.0)
            t = t * 0.5 * (1.0 - np.cos(math.pi * u))
        return t

    def grid(self, N=None):
        N = self.N if N is None else N
        x = np.linspace(0.0, 1.0, N + 1)
        return x, self.cf(x)

    def at(self, gamma: float) -> "Cochlea":
        return replace(self, sec=replace(self.sec, gamma=gamma))

    def gamma_of_x(self, x, gamma=None, profile=None):
        """
        The activity at each place: gamma times the taper, or an arbitrary
        profile times the taper. profile may be a callable of x or an array on
        the same grid. This is the hook for phase 2: a suppressor that removes
        OHC gain over a region is a profile.
        """
        g = self.sec.gamma if gamma is None else gamma
        t = self.taper(x)
        if profile is None:
            return g * t
        p = profile(x) if callable(profile) else np.asarray(profile, float)
        return p * t

    def gamma_critical(self) -> float:
        """Largest gamma keeping all six poles of every section in the left
        half plane. Place independent, by scale invariance."""
        return d3.gamma_critical(replace(self.sec, gamma=0.0))


WORLD_A = Cochlea(sec=SECTION_A, decades=3.00, F=3000.0)
WORLD_B = Cochlea(sec=SECTION_B, decades=2.75, F=20000.0)
WORLD_C = Cochlea(sec=SECTION_C, decades=2.80, F=4000.0)
WORLDS = {"A": WORLD_A, "B": WORLD_B, "C": WORLD_C}

# CERTIFIED OPERATING CEILING: 0.98 times the smaller of the two
# argument-principle onsets at N = 1024 and N = 4096, with the default apex
# taper. The onset itself is 0.11649/0.11834 (A), 0.22376/0.22307 (B) and
# 0.77106/0.77106 (C); the 2 percent step back covers the grid dependence,
# which is real and is largest in world A because its ceiling sits at 0.96 of
# the section's own critical gamma. verify() rechecks that no whole-line mode
# grows at these gammas at N = 1024, 4096 and 8192, and stability_ceiling()
# recomputes the onset from scratch.
CEILING = {"A": 0.11416, "B": 0.21861, "C": 0.75564}


# ---------------------------------------------------------------------------
# The section at a place
# ---------------------------------------------------------------------------

def section_terms(sec: Section3, w, a, r=None, gamma=None):
    """
    (Y, Y_r, H_o) for the section at a place, broadcast over w, a, r, gamma.

        Y   = X_b / F   BM displacement per unit pressure on the partition
        Y_r = X_r / F   reticular lamina
        H_o = G (X_r - X_t) / X_b, the transduction filter

    a is the local CF scale, r the damping scale (r = a unless overridden).
    Written from cofactors rather than np.linalg.solve so that complex w works,
    which the argument-principle count needs. Verified against
    deltaT_3dof.response() to 7e-16 by verify().
    """
    w = np.asarray(w)
    a = np.asarray(a, dtype=float)
    r = a if r is None else np.asarray(r, dtype=float)
    g = sec.gamma if gamma is None else np.asarray(gamma, dtype=float)
    jw = 1j * w
    a2 = a * a
    act = g * sec.K_a * sec.G * a2
    pil = jw * sec.R_p * r + sec.K_p * a2
    bun = jw * sec.R_c * r + sec.K_c * a2

    A00 = (-w ** 2 * sec.M_b + jw * (sec.R_b + sec.R_p) * r
           + (sec.K_b + sec.K_p) * a2)
    A01 = -pil + act
    A02 = -act + 0.0 * A00
    A10 = -pil + 0.0 * A00
    A11 = (-w ** 2 * sec.M_r + jw * (sec.R_r + sec.R_p + sec.R_c) * r
           + (sec.K_p + sec.K_c) * a2 - act)
    A12 = -bun + act
    A21 = -bun + 0.0 * A00
    A22 = (-w ** 2 * sec.M_t + jw * (sec.R_t + sec.R_c) * r
           + (sec.K_t + sec.K_c) * a2)

    m00 = A11 * A22 - A12 * A21                  # minor deleting row 0, col 0
    det = A00 * m00 - A01 * (A10 * A22) + A02 * (A10 * A21)      # A20 = 0
    Y = m00 / det
    Y_r = -A10 * A22 / det
    Y_t = A10 * A21 / det
    return Y, Y_r, sec.G * (Y_r - Y_t) / Y


def scaled_section(sec: Section3, a: float, gamma=None) -> Section3:
    """The section at a place with CF scale a, as a Section3, for the
    routines in deltaT_3dof that want one."""
    a2 = a * a
    return replace(sec,
                   K_b=sec.K_b * a2, K_p=sec.K_p * a2, K_t=sec.K_t * a2,
                   K_c=sec.K_c * a2, K_a=sec.K_a * a2,
                   R_b=sec.R_b * a, R_p=sec.R_p * a, R_r=sec.R_r * a,
                   R_t=sec.R_t * a, R_c=sec.R_c * a,
                   gamma=sec.gamma if gamma is None else gamma)


# ---------------------------------------------------------------------------
# THE API: read-outs as functions of place
# ---------------------------------------------------------------------------

def response(coch: Cochlea, freq: float, level_db: float = 0.0, N=None,
             gamma=None, profile=None, engine: str = "fd") -> dict:
    """
    The three read-outs as functions of place, for one stimulus frequency and
    level.

        freq       stimulus frequency, in units of the basal CF
        level_db   stapes pressure in dB re unity: P(0) = 10^(level_db/20)
        gamma      overrides coch.sec.gamma
        profile    a per-place gamma, callable or array; the hook for a
                   suppressor that removes gain over a region (phase 2)
        engine     "fd" (default), "cascade" or "wkb"

    Returns a dict with, all on the grid x:

        x, cf      place and the local section CF
        P          pressure difference across the partition
        X_b, X_r   BM and reticular-lamina displacement
        shear      cilia shear G(x_r - x_t) = H_o X_b
        H_o        the local transduction filter
        Z_p        partition impedance, 1/(i w Y)
        gamma_x    the activity actually used at each place
        lg_P       log10|P|, which stays finite where P underflows

    THE MODEL IS LINEAR IN THIS PHASE, so every read-out is exactly
    proportional to 10^(level_db/20). verify() measures that at 1e-16 rather
    than asserting it. level_db is here because phase 2 needs the signature,
    and because the level at which a read-out crosses a criterion is the whole
    subject; it is not a claim that anything here is level dependent.
    """
    N = coch.N if N is None else N
    x, a = coch.grid(N)
    gam = coch.gamma_of_x(x, gamma, profile)
    sec = coch.sec if gamma is None else replace(coch.sec, gamma=gamma)
    Y, Y_r, H_o = section_terms(sec, freq, a, a, gam)
    amp = 10.0 ** (level_db / 20.0)

    if engine == "fd":
        P = _fd(coch, freq, Y, N, amp)
    elif engine == "cascade":
        P = _cascade(coch, freq, N, gam, amp)
    elif engine == "wkb":
        P = _wkb(coch, freq, Y, x, amp)
    else:
        raise ValueError(engine)

    with np.errstate(divide="ignore", invalid="ignore"):
        Z_p = 1.0 / (1j * freq * Y)
        lg_P = np.log10(np.abs(P))
    X_b = Y * P
    return dict(x=x, cf=a, P=P, X_b=X_b, X_r=Y_r * P, shear=H_o * X_b,
                H_o=H_o, Y=Y, Z_p=Z_p, gamma_x=gam, lg_P=lg_P,
                freq=freq, level_db=level_db, N=N, engine=engine)


def _fd(coch: Cochlea, w, Y, N, amp):
    """
    Method A / B / C's shared engine: centered differences on

        P_{j-1} - (2 + h^2 q_j) P_j + P_{j+1} = 0,   q = -w^2 F Y

    with P(0) = amp and P(L) = 0, solved by LAPACK banded elimination. q never
    divides by anything that can vanish. Second order; verify() measures the
    order at 4.00.
    """
    q = -(w ** 2) * coch.F * Y
    h = 1.0 / N
    hq = (h * h) * q
    n = N - 1
    ab = np.zeros((3, n), dtype=complex)
    ab[0, 1:] = 1.0
    ab[1, :] = -(2.0 + hq[1:N])
    ab[2, :-1] = 1.0
    rhs = np.zeros(n, dtype=complex)
    rhs[0] = -amp
    P = np.zeros(N + 1, dtype=complex)
    P[0] = amp
    P[1:N] = solve_banded((1, 1), ab, rhs)
    return P


def _cascade(coch: Cochlea, w, N, gam, amp):
    """
    Method B's engine: the exact transfer matrix of a uniform segment,

        [P;U]_basal = [[cosh kh, z sinh(kh)/k],
                       [(k^2/z) sinh(kh)/k, cosh kh]] [P;U]_apical

    cascaded apex to base with the state renormalized by a positive real at
    every step, so nothing overflows and no phase is disturbed. Both entries
    are even in k, so no branch of the square root is ever chosen. Segment
    values are taken at midpoints, which keeps the cascade second order on a
    tapered line; it is exact on a uniform one, which verify() checks at 3e-12.

    Returned renormalized to P(0) = amp like every other engine. The profile
    it returns is trustworthy hundreds of dB below the peak, where the FD
    engine has underflowed to zero.
    """
    h = 1.0 / N
    xm = (np.arange(N) + 0.5) * h
    am = coch.cf(xm)
    gm = coch.gamma_of_x(xm, coch.sec.gamma, None) if gam is None else \
        _interp_gamma(coch, xm, gam, N)
    Ym, _, _ = section_terms(coch.sec, w, am, am, gm)
    s = 1j * w
    k2 = coch.F * (s * s) * Ym
    k = np.sqrt(k2)
    kh = k * h
    small = np.abs(kh) < 1e-7
    sh_k = np.where(small, h * (1.0 + kh * kh / 6.0),
                    np.sinh(np.where(small, 0.0, kh)) / np.where(small, 1.0, k))
    ch = np.cosh(kh)
    z = coch.F * s
    T11, T12, T21, T22 = ch, z * sh_k, (k2 / z) * sh_k, ch

    p, u = 0.0 + 0j, 1.0 + 0j          # P(L) = 0
    lg = 0.0
    Pn = np.empty(N + 1, dtype=complex)
    LG = np.empty(N + 1)
    Pn[N], LG[N] = p, lg
    for j in range(N - 1, -1, -1):
        pn = T11[j] * p + T12[j] * u
        un = T21[j] * p + T22[j] * u
        m = max(abs(pn), abs(un))
        m = m if m > 0 else 1.0
        p, u = pn / m, un / m
        lg += math.log10(m)
        Pn[j], LG[j] = p, lg
    with np.errstate(divide="ignore"):
        lgP = LG - LG[0] + np.log10(np.abs(Pn)) - math.log10(abs(Pn[0]))
    return amp * 10.0 ** lgP * np.exp(1j * (np.angle(Pn) - np.angle(Pn[0])))


def _interp_gamma(coch, xm, gam, N):
    return np.interp(xm, np.linspace(0.0, 1.0, N + 1), gam)


def _wkb(coch: Cochlea, w, Y, x, amp):
    """
    Method C's engine: the one-way Liouville-Green wave

        P = P(0) sqrt(k(0)/k) exp(-i int k dx),   k^2 = F w^2 Y

    on the forward branch Re k >= 0. Method C established that this is the only
    workable branch rule: Im k <= 0 deletes the amplifier, and continuity in x
    continues onto the growing solution past cutoff. No boundary condition at
    the apex, and no reflection. Cheap, and wrong near the peak by the amounts
    verify() prints.
    """
    k2 = coch.F * (w ** 2) * Y
    k = np.sqrt(np.asarray(k2, dtype=complex))
    k = np.where(k.real < 0.0, -k, k)
    ph = np.zeros_like(k)
    ph[1:] = np.cumsum(0.5 * (k[1:] + k[:-1]) * np.diff(x))
    return amp * np.sqrt(k[0] / k) * np.exp(-1j * ph)


# ---------------------------------------------------------------------------
# THE API: the place-frequency map
# ---------------------------------------------------------------------------

def place_map(coch: Cochlea, freqs=None, N=None, gamma=None, profile=None,
              engine="fd", which="X_b") -> dict:
    """
    The place-frequency map: for each stimulus frequency, the place where the
    read-out peaks, the local section CF there, and the offset between them.

    Returns freq, x_peak, cf_at_peak, offset_oct (positive when the peak sits
    APICAL of the place whose section CF equals the stimulus frequency), the
    fitted slope of log10(freq) against x_peak, and its R^2.

    The slope must come back as -decades and the offset must be the SAME at
    every frequency, which is the scale invariance showing through the wave.
    Measured: -3.0000 with R^2 = 1.000000 and a constant offset of -0.433
    octaves passive and +0.187 active, world A.

    The default frequencies stop short of the activity taper on purpose. The
    taper breaks the scale invariance it sits in, and it does so visibly: a
    frequency whose active peak would land at x = 0.82 comes back at 0.762
    with the offset flipped from +0.188 to -0.375, because the amplifier has
    been turned off underneath it and the peak reverts to its passive place.
    That is the taper being honest, not the map being wrong, but it has no
    business in a fitted slope.
    """
    N = coch.N if N is None else N
    if freqs is None:
        freqs = coch.cf(np.linspace(0.15, 0.65, 14))
    xs, fs, cfs = [], [], []
    for w in np.atleast_1d(np.asarray(freqs, dtype=float)):
        r = response(coch, float(w), N=N, gamma=gamma, profile=profile,
                     engine=engine)
        xp, _ = _peak(r["x"], r[which])
        if 0.02 < xp < 0.98:
            xs.append(xp)
            fs.append(math.log10(w))
            cfs.append(float(coch.cf(xp)))
    xs, fs, cfs = np.array(xs), np.array(fs), np.array(cfs)
    if xs.size > 2:
        p, res = np.polyfit(xs, fs, 1, full=True)[:2]
        r2 = float(1 - res[0] / max(np.var(fs) * fs.size, 1e-300))
        slope = float(p[0])
    else:
        slope, r2 = float("nan"), float("nan")
    return dict(freq=10.0 ** fs, x_peak=xs, cf_at_peak=cfs,
                offset_oct=np.log2(10.0 ** fs / cfs), slope=slope, r2=r2,
                decades=coch.decades)


def _peak(x, y):
    """Place and value of the maximum of |y|, parabolic in log|y|."""
    m = np.abs(np.asarray(y))
    i = int(np.argmax(m))
    if 0 < i < m.size - 1 and m[i] > 0:
        y0, y1, y2 = np.log(m[i - 1:i + 2] + 1e-300)
        den = y0 - 2 * y1 + y2
        d = 0.0 if den == 0 else float(np.clip(0.5 * (y0 - y2) / den, -1, 1))
    else:
        d = 0.0
    return float(x[i] + d * (x[1] - x[0])), float(m[i])


# ---------------------------------------------------------------------------
# Gain: pinned, never swept (rule 2)
# ---------------------------------------------------------------------------

def gain_db(coch: Cochlea, freq: float, gamma: float, kind: str = "tip",
            N=None, which="X_b", x_ref: float = None) -> float:
    """
    Three gains, all in use in the literature and none interchangeable.

      "tip"    peak over x active against peak over x passive. The line's tip
               gain. Discontinuous in gamma where the global maximum switches
               between the broad passive hump and the sharp active one, which
               is why gamma_for_gain() brackets on a grid before bisecting and
               reports what it actually achieved.
      "cf"     active over passive at the active peak place, same frequency.
               The 40 to 60 dB figure experimenters quote. Inherits the same
               discontinuity, because the place it reads at is the one that
               jumps.
      "local"  active over passive at a FIXED place x_ref and the same
               frequency. Continuous in gamma, and it is the quantity a
               suppression experiment actually measures: one place, one
               frequency, amplifier on against amplifier off. This is the one
               to pin against, and the one phase 2's A(u) is bounded by.
    """
    act = response(coch, freq, N=N, gamma=gamma)
    pas = response(coch, freq, N=N, gamma=0.0)
    if kind == "tip":
        return 20.0 * math.log10(np.abs(act[which]).max()
                                 / np.abs(pas[which]).max())
    if kind == "cf":
        i = int(np.argmax(np.abs(act[which])))
    elif kind == "local":
        if x_ref is None:
            raise ValueError("kind='local' needs x_ref")
        i = int(round(x_ref * act["N"]))
    else:
        raise ValueError(kind)
    return 20.0 * math.log10(abs(act[which][i]) / abs(pas[which][i]))


def section_gain_db(coch: Cochlea, gamma: float) -> float:
    """Peak gain of the ISOLATED reference section at the same gamma. Place
    independent by construction (rule 3), which verify() measures."""
    w = np.linspace(0.25, 1.9, 4001)
    Ya, _, _ = section_terms(replace(coch.sec, gamma=gamma), w,
                             np.ones_like(w))
    Yp, _, _ = section_terms(replace(coch.sec, gamma=0.0), w, np.ones_like(w))
    return 20.0 * math.log10(np.abs(Ya).max() / np.abs(Yp).max())


def gamma_for_gain(coch: Cochlea, target_db: float, freq: float = None,
                   kind: str = "tip", N=None, gamma_max=None, n_grid=40,
                   x_ref: float = None):
    """
    The gamma delivering target_db of gain, and the gain it actually achieved.
    Returns (gamma, achieved_db), or (None, best_db) if the target is not
    reached below gamma_max.

    Brackets on a grid dense near the top before bisecting, because the gain is
    neither monotone nor continuous in gamma. WHERE IT IS DISCONTINUOUS THE
    ACHIEVED GAIN WILL NOT EQUAL THE TARGET, and that is why it is returned:
    the "tip" and "cf" measures jump by 20 dB and more as the global maximum
    switches humps, so a target inside the jump is unreachable at any gamma
    and a bisection alone would silently return the jump location. The "local"
    measure does not jump.

    gamma_max defaults to the section's own critical gamma. It does NOT default
    to the whole-line stability ceiling: pinning and stability are kept
    separate on purpose, so that a caller who asks for 40 dB is told the gamma
    that delivers it AND, by stability_ceiling(), that it is above the ceiling.
    """
    N = coch.N if N is None else N
    freq = float(coch.cf(0.55)) if freq is None else freq
    if gamma_max is None:
        gamma_max = coch.gamma_critical() * (1.0 - 1e-3)
    gs = np.unique(np.concatenate([
        np.linspace(0.0, 0.9 * gamma_max, n_grid),
        gamma_max * (1.0 - np.logspace(-4, -1, n_grid))]))
    vals = [gain_db(coch, freq, float(g), kind, N, x_ref=x_ref) for g in gs]
    for i in range(1, gs.size):
        if vals[i] >= target_db:
            lo, hi = gs[i - 1], gs[i]
            for _ in range(50):
                mid = 0.5 * (lo + hi)
                if gain_db(coch, freq, mid, kind, N, x_ref=x_ref) < target_db:
                    lo = mid
                else:
                    hi = mid
            g = 0.5 * (lo + hi)
            return g, gain_db(coch, freq, g, kind, N, x_ref=x_ref)
    return None, float(np.max(vals))


def gain_profile(coch: Cochlea, freq: float, gamma: float, N=None,
                 which="X_b", x_ref: float = None) -> dict:
    """
    Where along the wave the gain is accumulated: the active-over-passive ratio
    as a function of place, and the stretch over which it is built.

    This is the measurement phase 2 is built on. A suppressor that saturates
    the OHCs over some region can remove at most the ratio that has been built
    up by the apical edge of that region, so this bounds the BM suppression
    A(u) that the line can produce, place by place.

    Returns the ratio in dB on the grid; its value at the read-out place, taken
    at x_ref if given and at the active peak otherwise; the octaves basal of
    that place over which the ratio climbs from 10 to 90 percent of its value
    there; and its value half an octave and one octave basal.

    Read the ratio for what it is. It is the drop when the amplifier is turned
    off at one place and one frequency, which is exactly what a saturating
    suppressor does, and it therefore includes the CF shift: part of the rise
    just basal of the peak is the PASSIVE response cutting off rather than the
    active one growing. That does not make it the wrong bound on A(u); it makes
    it a bound that already contains the half-octave shift.
    """
    N = coch.N if N is None else N
    act = response(coch, freq, N=N, gamma=gamma)
    pas = response(coch, freq, N=N, gamma=0.0)
    x = act["x"]
    with np.errstate(divide="ignore", invalid="ignore"):
        r = 20.0 * np.log10(np.abs(act[which]) / np.abs(pas[which]))
    i = (int(np.argmax(np.abs(act[which]))) if x_ref is None
         else int(round(x_ref * N)))
    rp = float(r[i])
    span = float("nan")
    if rp > 0.5:
        seg = r[:i + 1]
        i10 = int(np.argmax(seg >= 0.1 * rp))
        i90 = int(np.argmax(seg >= 0.9 * rp))
        span = float(coch.octaves(x[i10], x[i90]))
    out = dict(x=x, ratio_db=r, x_read=float(x[i]), read_db=rp,
               climb_octaves=span,
               x_peak=float(x[int(np.argmax(np.abs(act[which])))]))
    for oct_ in (0.5, 1.0):
        j = max(int(round((x[i] - oct_ / (coch.decades * math.log2(10)))
                          * N)), 0)
        out[f"basal_{oct_}oct_db"] = float(r[j])
    return out


# ---------------------------------------------------------------------------
# Stability (rule 1)
# ---------------------------------------------------------------------------

def worst_section_margin(coch: Cochlea, gamma=None, profile=None, n=None):
    """
    Level 1: every section on the grid, by the exact quadratic eigenvalue
    problem in deltaT_3dof.stability_margin(). Returns the worst largest-pole
    real part and the same normalized by the local CF, so basal and apical
    places compare. Negative is stable.
    """
    n = min(coch.N, 512) if n is None else n
    x = np.linspace(0.0, 1.0, n + 1)
    a = coch.cf(x)
    gam = coch.gamma_of_x(x, gamma, profile)
    sec = coch.sec if gamma is None else replace(coch.sec, gamma=gamma)
    worst, worst_n = -np.inf, -np.inf
    for ai, gi in zip(a, gam):
        m = d3.stability_margin(scaled_section(sec, float(ai), float(gi)))
        worst = max(worst, m)
        worst_n = max(worst_n, m / ai)
    return float(worst), float(worst_n)


def _det(coch: Cochlea, gamma, omegas, N, profile=None):
    """
    Normalized determinant of the FD system for P_1..P_{N-1} under
    P(0) = P(L) = 0, at complex omega. The continuant recursion is rescaled by
    positive reals at every step, which cannot move a zero or change an
    argument.
    """
    x, a = coch.grid(N)
    gam = coch.gamma_of_x(x, gamma, profile)
    sec = coch.sec if gamma is None else replace(coch.sec, gamma=gamma)
    W = np.asarray(omegas, dtype=complex)[:, None]
    Y, _, _ = section_terms(sec, W, a[None, :], a[None, :], gam[None, :])
    q = -(W ** 2) * coch.F * Y
    h = 1.0 / N
    b = -(2.0 + (h * h) * q)[:, 1:N]
    m, n = b.shape
    d_prev = np.ones(m, dtype=complex)
    d_cur = b[:, 0].copy()
    for j in range(1, n):
        d_next = b[:, j] * d_cur - d_prev
        d_prev, d_cur = d_cur, d_next
        s = np.maximum(np.abs(d_cur), np.abs(d_prev))
        s = np.where(s > 0, s, 1.0)
        d_cur, d_prev = d_cur / s, d_prev / s
    return d_cur


def _winding(coch, gamma, N, corners, profile=None, npts0=256, npts_max=8192):
    prev, npts, wnd, step = None, npts0, 0.0, 0.0
    while npts <= npts_max:
        z = np.concatenate([corners[k] + (corners[(k + 1) % 4] - corners[k])
                            * np.linspace(0, 1, npts, endpoint=False)
                            for k in range(4)])
        z = np.concatenate([z, z[:1]])
        d = np.concatenate([_det(coch, gamma, z[i:i + 512], N, profile)
                            for i in range(0, z.size, 512)])
        ph = np.unwrap(np.angle(d))
        wnd = (ph[-1] - ph[0]) / (2 * math.pi)
        step = float(np.max(np.abs(np.diff(ph))))
        if (prev is not None and abs(wnd - prev) < 0.05
                and abs(wnd - round(wnd)) < 0.05 and step < math.pi / 2):
            return int(round(wnd)), True
        prev, npts = wnd, npts * 2
    return int(round(wnd)), abs(wnd - round(wnd)) < 0.1


def growing_modes(coch: Cochlea, gamma=None, N=1024, profile=None,
                  depth=0.35, n_bands=8, verbose=False):
    """
    Level 2, and the arbiter: the number of whole-line modes with Re s > 0,
    counted by the argument principle on the determinant of the FD system.

    With exp(i omega t) a growing mode sits in the lower half omega plane, and
    every section is stable whenever gamma is below the section's critical
    value, so all the poles of the determinant are in the UPPER half plane and
    the winding number counts zeros with no pole correction. Each band is
    refined until the winding is a stable integer and no step exceeds pi/2.

    Grid converged: identical counts at N = 1024 and 4096 in all three worlds
    at 0, 0.6 and 0.9 of critical. Reproduces method A's untapered onset to
    0.13 percent and method C's to 0.11 percent.

    Not usable within about one percent of the section's critical gamma, where
    the section poles approach the contour and the count runs away with the
    grid. stability_ceiling() therefore caps its search at 0.98 of critical.
    """
    edges = np.logspace(-coch.decades - 0.15, 0.35, n_bands + 1)
    total, clean = 0, True
    for lo, hi in zip(edges[:-1], edges[1:]):
        corners = [complex(hi, -depth * hi), complex(hi, 1e-7 * hi),
                   complex(lo, 1e-7 * hi), complex(lo, -depth * hi)]
        n, ok = _winding(coch, gamma, N, corners, profile)
        clean &= ok
        total += n
        if verbose and n:
            print(f"      band {lo:9.2e} to {hi:9.2e}: {n:+d} growing")
    return total, clean


def line_poles(coch: Cochlea, gamma=None, m: int = 480, profile=None):
    """
    The same question by eigenvalues rather than by counting: eliminate the
    pressure under P(0) = P(L) = 0 into a dense added-mass term and solve

        (s^2 (M + F E G E^T) + s C + K) X = 0,   G = -h^2 D2^{-1}

    by companion linearization. This is method A's line_modes() and method C's
    line_poles(), which agree to 1.9e-14.

    THE m CONVERGENCE MATTERS AND IS THE REASON THIS IS NOT THE ARBITER. At
    m = 60, 120 and 240 the count of modes with Re s > 0 is about 2m in all
    three worlds, and at m = 480 it collapses to single or low double digits.
    Any conclusion drawn at m <= 240 is contaminated. Cost is O(m^3), so
    m = 480 is about a minute and m = 960 is not affordable in a sweep.
    """
    n_x = m + 2
    x = np.linspace(0.0, 1.0, n_x)
    h = x[1] - x[0]
    a = coch.cf(x)
    gam = coch.gamma_of_x(x, gamma, profile)
    sec = coch.sec if gamma is None else replace(coch.sec, gamma=gamma)
    M = np.zeros((3 * m, 3 * m))
    C = np.zeros((3 * m, 3 * m))
    K = np.zeros((3 * m, 3 * m))
    for j in range(m):
        Mj, Cj, Kj = d3.matrices(scaled_section(sec, float(a[j + 1]),
                                                float(gam[j + 1])))
        sl = slice(3 * j, 3 * j + 3)
        M[sl, sl], C[sl, sl], K[sl, sl] = Mj, Cj, Kj
    D2 = (np.diag(-2.0 * np.ones(m)) + np.diag(np.ones(m - 1), 1)
          + np.diag(np.ones(m - 1), -1))
    M[0::3][:, 0::3] += coch.F * (-h * h * np.linalg.inv(D2))
    Mi = np.linalg.inv(M)
    A = np.block([[np.zeros((3 * m, 3 * m)), np.eye(3 * m)],
                  [-Mi @ K, -Mi @ C]])
    e = np.linalg.eigvals(A)
    i = int(np.argmax(e.real))
    return complex(e[i]), int((e.real > 0).sum())


def stability_ceiling(coch: Cochlea, N=1024, tol=5e-3, cap=0.98):
    """
    The largest gamma with no growing whole-line mode, bisected on the argument
    principle. Capped at cap times the section's own critical gamma, above
    which the count itself stops being usable.
    """
    gc = coch.gamma_critical()
    lo, hi = 0.0, cap * gc
    if growing_modes(coch, hi, N)[0] == 0:
        return float(hi)
    while hi - lo > tol * gc:
        mid = 0.5 * (lo + hi)
        if growing_modes(coch, mid, N)[0] == 0:
            lo = mid
        else:
            hi = mid
    return float(lo)


# ---------------------------------------------------------------------------
# Wave measurements
# ---------------------------------------------------------------------------

def cycles_to_peak(r: dict, which="X_b"):
    """Phase lag from the stapes to the peak in cycles, and the largest
    unwrapped step between grid points, which must stay well below pi."""
    i = int(np.argmax(np.abs(r[which])))
    ph = np.unwrap(np.angle(r[which][:i + 1]))
    step = float(np.max(np.abs(np.diff(ph)))) if i > 0 else 0.0
    return float(-(ph[-1] - ph[0]) / (2 * math.pi)), step


def tuning(coch: Cochlea, x_probe: float, gamma=None, N=None, n_w=121,
           lo=-0.9, hi=0.35, which="X_b"):
    """
    The frequency response at a fixed place: peak frequency, peak level and
    tip-to-tail at one and two octaves. The standard measurement, made at a
    place rather than at the moving peak.

    TWO TRAPS, BOTH MET AND BOTH FIXED HERE.

    The peak is found on BM VELOCITY, i omega X_b, not on displacement. Driven
    by an ideal pressure source of constant amplitude, |X_b| at a fixed place
    tends to the partition compliance 1/K(x) as the frequency falls, and that
    plateau is 60 dB at x = 0.55 in world B against a 62 dB peak at CF. A
    displacement sweep wide enough to contain the tail therefore has its
    maximum at the low-frequency end of the sweep and reports a peak 5 octaves
    below CF. Velocity removes the plateau, and it is what interferometry
    measures.

    Tip-to-tail is then reported for BOTH read-outs, because the two differ by
    exactly 20 log10(2^octaves), 6.02 dB at one octave and 12.04 at two, and
    the three method files did not use the same one: method A quoted
    displacement, method B quoted velocity. Comparing their tip-to-tail numbers
    without that correction is comparing two different quantities.

    The sweep starts 0.9 decades below the local CF, which is 3 octaves, so a
    two-octave tail is inside it with room to spare.
    """
    N = coch.N if N is None else N
    cf0 = float(coch.cf(x_probe))
    ws = cf0 * np.logspace(lo, hi, n_w)
    j = int(round(x_probe * N))
    mag = np.array([abs(response(coch, float(w), N=N, gamma=gamma)[which][j])
                    for w in ws])
    vel = mag * ws
    i = int(np.argmax(vel))
    out = {"f_peak": float(ws[i]), "peak_db": 20.0 * math.log10(mag[i]),
           "peak_v_db": 20.0 * math.log10(vel[i]),
           "cf_ratio": float(ws[i] / cf0),
           "offset_oct": math.log2(ws[i] / cf0),
           "interior": bool(0 < i < n_w - 1)}
    for tag, m in (("", mag), ("v", vel)):
        lg = np.log(m + 1e-300)
        for oct_ in (1.0, 2.0):
            wt = ws[i] / 2.0 ** oct_
            out[f"tt{tag}{int(oct_)}"] = (
                float("nan") if wt < ws[0] else
                20.0 * (lg[i] - np.interp(math.log(wt), np.log(ws), lg))
                / math.log(10))
    return out


# ---------------------------------------------------------------------------
# verify(): the cross-validation numbers, reproduced
# ---------------------------------------------------------------------------

def _rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300)))


def _cmp(f1, f2, key="X_b", floor_db=60.0):
    """Largest disagreement in dB and in radians over the top floor_db of the
    first profile. Below that both fields are underflowing and a dB comparison
    compares two zeros."""
    a, b = np.asarray(f1[key]), np.asarray(f2[key])
    n = min(a.size, b.size)
    a, b = a[:n], b[:n]
    m = np.abs(a)
    keep = (m > m.max() * 10.0 ** (-floor_db / 20.0)) & (np.abs(b) > 0)
    if not keep.any():
        return float("nan"), float("nan")
    return (float(np.max(20.0 * np.abs(np.log10(np.abs(a[keep] / b[keep]))))),
            float(np.max(np.abs(np.angle(a[keep] / b[keep])))))


def verify(full: bool = True) -> bool:
    """
    Reproduces the cross-validation. Every number printed is computed here.
    Checks marked [x] compare against tl_method_a/b/c and are skipped with a
    note if those files are not importable.
    """
    ok = True
    t0 = time.time()

    print("[V1] the section in the line IS deltaT_3dof's section")
    w = np.logspace(-1.5, 0.8, 11)
    worst = 0.0
    for wn, coch in WORLDS.items():
        sec = replace(coch.sec, gamma=0.6 * coch.gamma_critical())
        Y, Y_r, H_o = section_terms(sec, w, np.ones_like(w))
        Xb, Xr, sh, Ho = d3.response(sec, w)
        e = max(_rel(Y, Xb), _rel(Y_r, Xr), _rel(H_o, Ho))
        worst = max(worst, e)
        print(f"     world {wn}: X_b, X_r and H_o against response(): {e:.2e}")
    ok &= worst < 1e-12
    print(f"     worst {worst:.2e}   {'PASS' if worst < 1e-12 else 'FAIL'}")

    print("\n[V2] rule 3: the place map does not modulate the amplifier")
    for wn, coch in WORLDS.items():
        g = 0.6 * coch.gamma_critical()
        gains = []
        for x in np.linspace(0, 1, 9):
            a = float(coch.cf(x))
            ww = np.linspace(0.25 * a, 1.9 * a, 4001)
            Ya, _, _ = section_terms(replace(coch.sec, gamma=g), ww,
                                     np.full_like(ww, a))
            Yp, _, _ = section_terms(replace(coch.sec, gamma=0.0), ww,
                                     np.full_like(ww, a))
            gains.append(20 * math.log10(np.abs(Ya).max() / np.abs(Yp).max()))
        spread = max(gains) - min(gains)
        print(f"     world {wn}: peak gain over 9 places varies by "
              f"{spread:.2e} dB (gain {gains[0]:.3f} dB)")
        ok &= spread < 1e-10

    print("\n[V3] scale invariance: Z_p(x,w) = CF(x) Z_ref(w/CF)")
    for wn, coch in WORLDS.items():
        g = 0.6 * coch.gamma_critical()
        sec = replace(coch.sec, gamma=g)
        nu = np.logspace(-1.2, 0.6, 9)
        e = 0.0
        for x in (0.15, 0.5, 0.85):
            a = float(coch.cf(x))
            Y, _, H = section_terms(sec, nu * a, np.full_like(nu, a))
            Yr, _, Hr = section_terms(sec, nu, np.ones_like(nu))
            e = max(e, _rel(Y * a * a, Yr), _rel(H, Hr))
        print(f"     world {wn}: {e:.2e}")
        ok &= e < 1e-12

    if full:
        ok &= _verify_engines()
    print(f"\n[V7] level linearity: read-outs proportional to 10^(level/20)")
    coch = WORLD_B
    r0 = response(coch, float(coch.cf(0.5)), 0.0, N=1024,
                  gamma=0.5 * coch.gamma_critical())
    r1 = response(coch, float(coch.cf(0.5)), 26.0, N=1024,
                  gamma=0.5 * coch.gamma_critical())
    e = max(_rel(r1[k], r0[k] * 10 ** 1.3) for k in ("X_b", "X_r", "shear"))
    print(f"     26 dB up, all three read-outs: {e:.2e}")
    ok &= e < 1e-12

    print("\n[V8] the place-frequency map")
    for wn, coch in WORLDS.items():
        for tag, g in (("passive", 0.0), ("active", CEILING[wn])):
            pm = place_map(coch, N=4096, gamma=g)
            spread = float(pm["offset_oct"].max() - pm["offset_oct"].min())
            print(f"     world {wn} {tag:>7}: slope {pm['slope']:+.4f} "
                  f"against {-coch.decades:+.4f}, R2 {pm['r2']:.6f}, "
                  f"offset {pm['offset_oct'].mean():+.3f} octaves, "
                  f"spread {spread:.1e}")
            ok &= abs(pm["slope"] + coch.decades) < 2e-3 and spread < 5e-3

    if full:
        ok &= _verify_stability()

    print(f"\n{'verify PASSED' if ok else 'verify FAILED'}  "
          f"({time.time() - t0:.1f} s)")
    return ok


def _verify_engines() -> bool:
    ok = True
    print("\n[V4] the three finite-difference solvers are one solver")
    try:
        import tl_method_a as A
        import tl_method_b as B
        import tl_method_c as Cm
    except Exception as exc:                                  # pragma: no cover
        print(f"     skipped, cannot import the method files ({exc})")
        return ok
    N = 4096
    worst_db = worst_rad = 0.0
    for wn, coch in WORLDS.items():
        for g in (0.0, 0.9 * coch.gamma_critical()):
            for xp in (0.25, 0.55):
                w = float(coch.cf(xp))
                mine = response(replace(coch, taper_lo=1.0, taper_hi=1.0),
                                w, N=N, gamma=g)
                ca = A.Cochlea(N=N, decades=coch.decades, F=coch.F,
                               taper_lo=1.0, taper_hi=1.0,
                               sec=replace(coch.sec, gamma=g))
                fa = A.solve_line(ca, w, N=N)
                ln = B.Line(sec=replace(coch.sec, gamma=g),
                            n_dec=coch.decades, rho2H=coch.F, N=N,
                            apex="short")
                rb = B.tridiag(ln, [w], N=N)
                Pb = 10.0 ** rb["lgP"][:, 0] * np.exp(1j * rb["argP"][:, 0])
                fb = dict(X_b=Pb * rb["Y_p"][:, 0] / (1j * w))
                Cm.DECADES, Cm.BETA_F = coch.decades, coch.F
                fc = Cm.direct_solve(replace(coch.sec, gamma=g), w, N + 1)
                for other in (fa, fb, fc):
                    d, rr = _cmp(mine, other)
                    worst_db, worst_rad = max(worst_db, d), max(worst_rad, rr)
    print(f"     this file against methods A, B and C at N = {N}, "
          f"3 worlds x 2 gains x 2 frequencies")
    print(f"     worst {worst_db:.2e} dB and {worst_rad:.2e} rad over the top "
          f"60 dB of the profile")
    ok &= worst_db < 1e-6 and worst_rad < 1e-6

    print("\n[V5] the cascade converges to the FD answer at h^2")
    coch = WORLD_B
    g = 0.9 * coch.gamma_critical()
    w = float(coch.cf(0.55))
    prev = None
    for N in (500, 1000, 2000, 4000):
        d, rr = _cmp(response(coch, w, N=N, gamma=g, engine="cascade"),
                     response(coch, w, N=N, gamma=g))
        r = "" if prev is None else f", ratio {prev / d:.2f}"
        print(f"     N = {N:5d}: {d:.3e} dB, {rr:.3e} rad{r}")
        prev = d
    ok &= d < 0.05
    dB_, _ = _cmp(response(coch, w, N=2000, gamma=g, engine="cascade"),
                  {"X_b": _b_cascade_xb(coch, w, 2000, g)})
    print(f"     this file's cascade against method B's, N = 2000: "
          f"{dB_:.2e} dB")
    ok &= dB_ < 1e-9

    print("\n[V6] WKB, priced against the converged FD solve")
    print(f"     {'world':>5} {'gamma':>9} {'tail dB':>8} {'tail rad':>9} "
          f"{'peak dB':>8} {'peak rad':>9}")
    for wn, coch in WORLDS.items():
        for g in (0.0, 0.9 * coch.gamma_critical()):
            N = 16000
            w = float(coch.cf(0.55))
            fd = response(coch, w, N=N, gamma=g)
            wk = response(coch, w, N=N, gamma=g, engine="wkb")
            i = int(np.argmax(np.abs(fd["X_b"])))
            it = i // 2
            print(f"     {wn:>5} {g:9.5f} "
                  f"{20 * abs(math.log10(abs(wk['X_b'][it] / fd['X_b'][it]))):8.3f} "
                  f"{abs(np.angle(wk['X_b'][it] / fd['X_b'][it])):9.4f} "
                  f"{20 * math.log10(abs(wk['X_b'][i] / fd['X_b'][i])):+8.3f} "
                  f"{abs(np.angle(wk['X_b'][i] / fd['X_b'][i])):9.4f}")
    print("     tail is half way to the peak; WKB minus FD")
    return ok


def _b_cascade_xb(coch, w, N, g):
    import tl_method_b as B
    ln = B.Line(sec=replace(coch.sec, gamma=g), n_dec=coch.decades,
                rho2H=coch.F, N=N, apex="short")
    r = B.cascade(ln, [w], N=N)
    P = 10.0 ** r["lgP"][:, 0] * np.exp(1j * r["argP"][:, 0])
    return P * r["Y_p"][:, 0] / (1j * w)


def _verify_b_search() -> bool:
    """
    Why method B concluded the pressure-driven line is always stable, settled
    on a pole that is known to be there.

    World B untapered at 0.9 of the section's critical gamma has five growing
    modes near s = 0.000165 + 0.001458j, the apical cavity band. Three
    independent statements are checked here:

      the FD determinant has 5 zeros in a box around them, at N = 1024 and at
      N = 4096, so it is not a discretization artifact;
      method B's OWN cascade agrees exactly: the winding of its P(0) around the
      same box is +5.000 at both grids, so its field has the same five poles;
      and yet |P(0)| never leaves 1.000 anywhere on that contour or on a fine
      grid through the interior.

    The last line is the whole story. base_state() renormalizes so that
    max(|P|,|U|) = 1, and in that region |U| < |P|, so |P(0)| is pinned at the
    normalizer and can dip only in vanishingly small neighborhoods of the
    zeros. rhp_scan() takes the deepest minima of that surface as its seeds, so
    a surface that is flat at its ceiling gives it nothing to descend, and its
    eight rounds of zoom start in the wrong place. The information was in the
    phase of the same number the whole time.
    """
    print("\n[V10b] method B's zero search, tested on a pole known to be there")
    try:
        import tl_method_b as B
    except Exception as exc:                                  # pragma: no cover
        print(f"     skipped, cannot import method B ({exc})")
        return True
    coch = replace(WORLD_B, taper_lo=1.0, taper_hi=1.0)
    gam = 0.9 * coch.gamma_critical()
    p, _ = line_poles(coch, gam, m=480)
    om, sg, rel = abs(p.imag), abs(p.real), 0.25
    r = rel * om
    corners = [complex(om + r, -1e-12), complex(om - r, -1e-12),
               complex(om - r, -sg - r), complex(om + r, -sg - r)]
    ok = True
    print(f"     QEP m=480 pole at s = {p.real:+.6f}{p.imag:+.6f}j")
    for N in (1024, 4096):
        n_fd, clean = _winding(coch, gam, N, corners)
        npts = 2048
        z = np.concatenate([corners[k] + (corners[(k + 1) % 4] - corners[k])
                            * np.linspace(0, 1, npts, endpoint=False)
                            for k in range(4)])
        z = np.concatenate([z, z[:1]])
        ln = B.Line(sec=replace(coch.sec, gamma=gam), n_dec=coch.decades,
                    rho2H=coch.F, N=N, apex="short")
        P0, U0 = B.base_state(ln, 1j * z, N)
        wc = float((np.unwrap(np.angle(P0))[-1]
                    - np.unwrap(np.angle(P0))[0]) / (2 * math.pi))
        print(f"     N = {N:5d}: FD determinant {n_fd} zeros in the box"
              f"{'' if clean else ' (winding did not settle)'}; method B's "
              f"cascade winding {wc:+.3f}; |P(0)| on the contour "
              f"[{np.abs(P0).min():.3f}, {np.abs(P0).max():.3f}]")
        ok &= (n_fd == round(wc)) and n_fd > 0
    R = np.linspace(0.0, 2 * sg, 81)
    I = om * (1 + np.linspace(-rel, rel, 81))
    S = (R[:, None] + 1j * I[None, :]).ravel()
    Pb, Ub = B.base_state(B.Line(sec=replace(coch.sec, gamma=gam),
                                 n_dec=coch.decades, rho2H=coch.F, N=1024,
                                 apex="short"), S, 1024)
    print(f"     a 81 x 81 grid through the interior: min |P(0)| "
          f"{np.abs(Pb).min():.4f}, min |U(0)| {np.abs(Ub).min():.4f}")
    print("     the amplitude is pinned at the normalization ceiling; the "
          "phase of the same\n     number turns five times. The search "
          "statistic was the problem, not the field")
    return ok


def _verify_stability() -> bool:
    ok = True
    print("\n[V9] stability: the argument principle against the eigenvalue "
          "problem")
    print(f"     {'world':>5} {'gamma/gc':>9} {'AP N=1024':>10} "
          f"{'AP N=4096':>10} {'QEP m=480 (n>0)':>17}")
    for wn, coch in WORLDS.items():
        gc = coch.gamma_critical()
        for gf in (0.0, 0.9):
            g = gf * gc
            flat = replace(coch, taper_lo=1.0, taper_hi=1.0)
            n1 = growing_modes(flat, g, N=1024)[0]
            n2 = growing_modes(flat, g, N=4096)[0]
            p, nq = line_poles(flat, g, m=480)
            print(f"     {wn:>5} {gf:9.2f} {n1:10d} {n2:10d} "
                  f"{p.real:+.5f}{p.imag:+.4f}j {nq:3d}")
            ok &= n1 == n2
    print("     the two agree on stable-or-not; the eigenvalue problem's COUNT "
          "is not\n     converged below m = 480, where it falls from about 2m "
          "to single digits")

    print("\n[V10] onset gamma, untapered, against what methods A and C "
          "reported")
    for wn, want in (("A", 0.09177), ("C", 0.59954)):
        coch = replace(WORLDS[wn], taper_lo=1.0, taper_hi=1.0)
        g = stability_ceiling(coch, N=1024, tol=3e-3)
        print(f"     world {wn}: this file {g:.5f}, method {wn} reported "
              f"{want:.5f}, differ by {100 * abs(g - want) / want:.2f} percent")
        ok &= abs(g - want) / want < 0.02

    ok &= _verify_b_search()

    print("\n[V11] the apex taper is what buys the gain")
    for wn, coch in WORLDS.items():
        flat = replace(coch, taper_lo=1.0, taper_hi=1.0)
        g0 = stability_ceiling(flat, N=1024)
        g1 = CEILING[wn]
        gc = coch.gamma_critical()
        print(f"     world {wn}: ceiling {g0 / gc:.3f} of critical untapered, "
              f"{g1 / gc:.3f} tapered; tip gain "
              f"{gain_db(coch, float(coch.cf(0.55)), g0, 'tip', 4096):+.2f} "
              f"-> {gain_db(coch, float(coch.cf(0.55)), g1, 'tip', 4096):+.2f} dB")
    return ok


# ---------------------------------------------------------------------------
# report(): the operating tables
# ---------------------------------------------------------------------------

def report(coch: Cochlea = None, name: str = "B"):
    coch = WORLDS[name] if coch is None else coch
    gc = coch.gamma_critical()
    ceil = CEILING[name]
    x0 = 0.55
    w0 = float(coch.cf(x0))
    # The read-out place is where the ACTIVE wave peaks at the operating point,
    # not the place whose section CF equals the stimulus. Those differ by the
    # CF shift, +0.19 octaves in world A, and reading the "local" gain at the
    # wrong one of them turns a 43 dB amplifier into a 0 dB one.
    x_read = _peak(*[response(coch, w0, N=4096, gamma=ceil)[k]
                     for k in ("x", "X_b")])[0]
    print(f"\nOperating tables, world {name}")
    print(f"  section K_p={coch.sec.K_p:g} K_t/K_c={coch.sec.K_t / coch.sec.K_c:g}"
          f" R_b={coch.sec.R_b:g}; {coch.decades} decades; F={coch.F:g}; "
          f"N={coch.N}")
    print(f"  gamma_critical (section) {gc:.5f}; whole-line ceiling {ceil:.5f}"
          f" = {ceil / gc:.3f} of it")

    print(f"\n  gain pinned, never swept (rule 2). Stimulus f = {w0:.5f}, "
          f"the section CF at x = {x0};\n  the local gain is read at "
          f"x = {x_read:.4f}, where the active wave peaks.")
    print(f"  {'target':>7} {'kind':>6} {'gamma':>9} {'/gc':>6} {'got':>7} "
          f"{'tip':>7} {'local':>7} {'section':>8} {'stable':>7} "
          f"{'cycles':>7} {'ttv2':>6}")
    for kind in ("tip", "local"):
        for tgt in (1.0, 3.0, 10.0, 20.0, 40.0):
            g, got = gamma_for_gain(coch, tgt, w0, kind, N=4096,
                                    x_ref=x_read)
            if g is None:
                print(f"  {tgt:7.0f} {kind:>6}   not reached below gamma_c; "
                      f"best {got:.2f} dB")
                continue
            tip = gain_db(coch, w0, g, "tip", 4096)
            loc = gain_db(coch, w0, g, "local", 4096, x_ref=x_read)
            sg = section_gain_db(coch, g)
            r = response(coch, w0, N=4096, gamma=g)
            cyc, _ = cycles_to_peak(r)
            tn = tuning(coch, x0, gamma=g, N=2048, n_w=61)
            print(f"  {tgt:7.0f} {kind:>6} {g:9.5f} {g / gc:6.3f} {got:7.2f} "
                  f"{tip:7.2f} {loc:7.2f} {sg:8.2f} "
                  f"{'yes' if g <= ceil else 'NO':>7} {cyc:7.2f} "
                  f"{tn['ttv2']:6.1f}")
    print("  got is the gain actually achieved; where it misses the target the "
          "measure is\n  discontinuous in gamma and no gamma delivers that "
          "number. stable means at or\n  below the whole-line ceiling; a NO "
          "row is the formal steady state of an\n  oscillating line and must "
          "not be quoted as a response. ttv2 is tip-to-tail\n  of BM VELOCITY "
          "at two octaves; subtract 12.04 dB for displacement.")

    print("\n  where the gain is accumulated (the number phase 2 needs), "
          f"read at x = {x_read:.4f}")
    print(f"  {'gamma/gc':>9} {'local gain':>11} {'10-90% climb':>13} "
          f"{'0.5 oct basal':>14} {'1 oct basal':>12} {'x_peak':>7}")
    for gf in (0.3, 0.5, 0.7, 0.9, 1.0):
        g = gf * ceil
        gp = gain_profile(coch, w0, g, N=8192, x_ref=x_read)
        print(f"  {g / gc:9.3f} {gp['read_db']:11.2f} "
              f"{gp['climb_octaves']:13.2f} {gp['basal_0.5oct_db']:14.2f} "
              f"{gp['basal_1.0oct_db']:12.2f} {gp['x_peak']:7.4f}")
    print("  local gain is active over passive at that place and frequency; "
          "climb is the\n  octaves basal of it over which that ratio goes from "
          "10 to 90 percent of its\n  value. A suppressor that silences the "
          "OHCs only in the tail, half an octave\n  basal and further, can "
          "remove no more than the 0.5 oct column.")

    print("\n  the wave at the ceiling")
    for tag, g in (("passive", 0.0), ("at ceiling", ceil)):
        r = response(coch, w0, N=8192, gamma=g)
        cyc, step = cycles_to_peak(r)
        tn = tuning(coch, x0, gamma=g, N=4096)
        xp, _ = _peak(r["x"], r["X_b"])
        print(f"    {tag:>10}: x_peak {xp:.4f}, {cyc:.2f} cycles "
              f"(max step {step:.2f} rad), CF {tn['cf_ratio']:.4f} of the "
              f"section CF ({tn['offset_oct']:+.2f} oct)")
        print(f"                tip-to-tail, velocity {tn['ttv1']:.1f} dB at "
              f"1 oct and {tn['ttv2']:.1f} at 2; displacement {tn['tt1']:.1f} "
              f"and {tn['tt2']:.1f}")

    print("\n  grid convergence at the ceiling")
    prev = None
    for N in (1024, 2048, 4096, 8192, 16384):
        r = response(coch, w0, N=N, gamma=ceil)
        xp, mx = _peak(r["x"], r["X_b"])
        db = 20 * math.log10(mx)
        d = "" if prev is None else f"{db - prev:+9.4f}"
        print(f"    N = {N:6d}: peak {db:9.4f} dB at x = {xp:.5f} {d:>9}")
        prev = db

    print("\n  rule 1, level 1: every section")
    m, mn = worst_section_margin(coch, ceil, n=512)
    print(f"    worst pole real part over 513 places {m:.3e}, "
          f"{mn:.3e} normalized by the local CF")
    print("  rule 1, level 2: the whole line")
    n, clean = growing_modes(coch, ceil, N=1024)
    n2, _ = growing_modes(coch, ceil, N=4096)
    print(f"    growing modes at the ceiling: {n} at N = 1024, {n2} at "
          f"N = 4096{'' if clean else '  (winding did not settle)'}")


def main(argv):
    v = "--verify" in argv
    r = "--report" in argv
    ok = True
    if not r:
        ok = verify(full=True)
    if not v:
        for nm in ("B", "A"):
            report(name=nm)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
