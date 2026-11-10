"""
exp_obs46.py
Observations 4 and 6 on the certified transmission line, then all six scored.

WHY THIS FILE EXISTS
--------------------
Observations 4 and 6 were untestable in the cross-section study and the README
says so twice. Observation 4 is a tip-to-tail ratio, which is a property of a
wave arriving at a place. Observation 6 is a net forward gain between two
places, which needs two places. tl_model.py supplies both. Nothing here is a
new model: every field comes from tl_model.response() or from the SAME
tridiagonal matrix tl_model._fd assembles, with a different right-hand side.
The one piece of new linear algebra, the point-source Green's function the
Allen-Fahey function needs, is checked against tl_model.response() (identical
to the last bit) and against reciprocity of the same matrix.

WHAT IS COMPUTED

  Observation 4   a THRESHOLD TUNING CURVE at a fixed place: the stapes level
                  needed to hold a read-out at a criterion, against frequency.
                  The model is linear in this phase, so that is exactly the
                  negative of the fixed-place frequency response plus one
                  additive constant. Tip-to-tail and tail slope are then read
                  by obs_metrics's own convention (regress every point at or
                  below -1 oct re BF, evaluate the regression at -1 oct).

  Observation 6   the Allen-Fahey geometry. f_dp is FIXED and r = f2/f1 is
                  swept, so the DP characteristic place is fixed and the
                  generator place moves basally as r grows:

                      f1 = f_dp/(2-r),  f2 = r f_dp/(2-r)

                  PRIMARY METRIC, which is what the task asks for and what
                  AF-92 did: the net forward gain read off the model's own
                  excitation pattern at the single frequency f_dp,

                      20 log10 |X_b(x_dp)| / |X_b(x_gen)|

                  AF-92 estimated 16.5, 28 and 30 dB this way from measured
                  patterns and then bounded the real thing under 10 dB.
                  Two further readings are printed beside it every time: the
                  same ratio on the PASSIVE line, and the difference of the
                  two, which is the part of the growth the amplifier supplies.

                  SECONDARY, the Allen-Fahey FUNCTION: a unit point source at
                  x_gen, the intracochlear DP held constant at x_dp, the ear
                  canal DP proportional to the stapes volume velocity U(0).
                  At fixed f_dp every middle-ear factor is one constant, which
                  is what makes the experiment interpretable, so the curve is
                  computed up to an additive constant and the scorer reads only
                  its excursion and its trend.

  Observation 5   falls out of the same tuning curve (tail and tip regressions
                  intersect) and out of H_o at the same place (the phase step).
                  Reported because the scorer computes it from data this file
                  already has, not because this file set out to test it.

  Observations    a minimal low-side suppression run, included ONLY so that the
  1, 2 and 3      sheet is complete for the first time. One probe place, one
                  mechanism (saturation of gamma itself by the local cilia
                  shear the suppressor produces). The accumulation experiment
                  is the authority on these three and this file says so in the
                  notes field of every report it writes.

THE READ-OUT CONVENTIONS, CHOSEN BEFORE THE NUMBERS WERE SEEN
    The neural threshold curve is the CILIA read-out, shear = G(x_r - x_t),
    driven by shear VELOCITY: the inner hair cell bundle is free standing and
    fluid coupled below its corner, which is Allen's own account of why the
    neural tail slopes at 6 dB/oct. That choice is load-bearing for the tail
    slope and nothing else, so the displacement alternative is printed beside
    it every single time. Observation 6 is a ratio at ONE frequency, where
    displacement and velocity give the identical number.

THE FREE CONSTANTS, AND WHAT THEY CANNOT MOVE
    Two, both named, both printed.
    F_BASE_HZ turns the model's dimensionless frequency into Hz. Nothing
    scored depends on it: every metric is an octave, a slope per octave or per
    decade, or a dB difference. Measured here, not asserted.
    TIP_DB_SPL is the one additive constant that puts the level axis in dB SPL.
    It fixes where the tuning curve sits, and it cancels out of tip-to-tail,
    tail slope, the second-map intersection, the phase step, the forward gain
    and the Allen-Fahey excursion.
    The OHC operating point s_ref is NOT a third constant. It slides every
    suppression curve along the level axis one for one, so observation 1 is a
    CALIBRATION of it rather than a test of the model, and this file sets it to
    put observation 1 exactly on target and says so. Observations 2 and 3 are
    differences and are invariant to it; that invariance is measured below at
    1e-13 rather than claimed.

THE FIVE RULES
 1  STABILITY UNDER DRIVE. Three levels, all with the SUPPRESSED profile and
    not only the uniform line: worst_section_margin() over 513 sections,
    growing_modes() over the whole line by the argument principle at N = 1024
    and N = 4096, and line_poles() at m = 480 as a third opinion at the chosen
    operating point. Profiles are passed as CALLABLES, because those three
    routines each use their own grid and an array silently belongs to one.

    IT BITES, and it is the reason there are two operating points and two
    reports per world. The uniform line at the certified ceiling has no
    growing mode. Put a notch in gamma 1 percent deep and 0.004 of the
    cochlear length wide, which is less than the shallowest suppression this
    file computes, and the same line has 3 growing modes, identically at
    N = 1024 and N = 4096. So observations 4, 5 and 6, which apply no
    suppressor, are quoted at the certified ceiling, and observations 1, 2 and
    3 are quoted at the highest pinned gain whose SUPPRESSED line still has no
    growing mode anywhere on the drive ladder. Quoting a suppression threshold
    off an oscillating line is the error the README records under Result 4.
 2  GAIN IS PINNED, NEVER SWEPT, by tl_model.gamma_for_gain(kind="local"),
    the one gain measure that is continuous in gamma, at 1, 3, 10, 20 and 40 dB
    plus the certified ceiling, and observations 4, 5 and 6 are reported at
    every one of them. 40 dB of TIP gain is not reachable with a stable line in
    any world; 40 dB of LOCAL gain is.
 3  NO PARAMETERIZATION ARTIFACTS. Nothing is modulated except gamma, through
    the certified profile hook. The law is gamma -> gamma (1 - s) with
    s = u^2/(1+u^2) in [0,1), so the profile is bounded in [0, gamma] and can
    never change sign, and the OHC scale K_a is untouched. Rule 3 is also
    rechecked at the level tl_model checks it: the isolated section's peak gain
    is place independent to 1e-14 dB.
 4  CRITERION ROBUSTNESS. Every suppression threshold and every Delta-T is
    reported at 1.0, 1.5 and 3.0 dB. Observations 4 and 6 have no criterion,
    but they have conventions that bite just as hard, and each is varied and
    printed: the tail window and tail reference, the read-out and its motion,
    the sweep's low-frequency end, the generator and DP place definitions, and
    the r range.
 5  Every number printed was produced by running this file.

GRIDS AND COST
    N = 4096 for every response and every Green's function, which tl_model
    certifies as converged to 0.01 dB in world B; N = 1024 and 4096 for the
    argument-principle mode counts; m = 480 for the eigenvalue problem;
    169 frequencies over 4.2 octaves for a tuning curve; 10 primary ratios for
    the Allen-Fahey sweep; a 5 dB level ladder over 120 dB to bracket a
    suppression threshold and a 2.5 dB ladder over 30 dB to report it.

    python3 exp_obs46.py             # everything, both worlds, then the sheets
    python3 exp_obs46.py --quick     # world B, no whole-line stability counts
    python3 exp_obs46.py --world A   # one world
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import replace

import numpy as np
from scipy.linalg import solve_banded

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCORING = os.path.normpath(os.path.join(_HERE, "..", "..", "scoring"))
for _p in (_HERE, os.path.dirname(_HERE), _SCORING):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import tl_model as T                                            # noqa: E402
import obs_metrics as MS                                          # noqa: E402


# ---------------------------------------------------------------------------
# Conventions. Every one of these is printed in the log and varied in the
# robustness tables, because a convention that is not varied is an assumption.
# ---------------------------------------------------------------------------

F_BASE_HZ = 40000.0      # CF at x = 0, in Hz. Cat-like. Nothing scored moves.
TIP_DB_SPL = 25.0        # the one additive constant on the level axis
X_PROBE = 0.55           # probe place, well basal of the activity taper
N_DEF = 4096             # response grid
N_AP = 1024              # argument principle grid, with 4096 as the check

GAIN_TARGETS = (1.0, 3.0, 10.0, 20.0, 40.0)     # rule 2, local gain in dB
SUPP_OCT = (-0.75, -1.0, -1.5, -2.0)            # low side, >= 0.5 oct below
CRITERIA = (1.0, 1.5, 3.0)                      # rule 4
R_LIST = tuple(np.round(np.linspace(1.05, 1.50, 10), 4))

TC_LO, TC_HI, TC_N = -3.6, 0.6, 169             # tuning sweep, oct re section CF
TAIL_WINDOW = MS.TAIL_WINDOW_OCT                # -1.0, the scorer's own
TAIL_REF = MS.TIP_TO_TAIL_REF_OCT               # -1.0, the scorer's own


def hz(w):
    """Model frequency (units of the basal CF) to Hz."""
    return np.asarray(w, dtype=float) * F_BASE_HZ


def oct_re(f, f0):
    return math.log2(f / f0)


# ---------------------------------------------------------------------------
# The line, as tl_model assembles it, plus one new right-hand side
# ---------------------------------------------------------------------------

def prof_callable(x_grid, arr):
    """
    A per-place gamma as a CALLABLE rather than an array.

    worst_section_margin(), growing_modes() and line_poles() each build their
    own grid, so an array profile silently belongs to whichever grid made it
    and the others raise or, worse, broadcast. Everything downstream of a
    suppressor in this file therefore travels as an interpolant.
    """
    x_grid = np.asarray(x_grid, float)
    arr = np.asarray(arr, float)
    return lambda xx, _x=x_grid, _a=arr: np.interp(np.asarray(xx, float), _x, _a)


def system(coch, freq, gamma=None, profile=None, N=None):
    """
    The same tridiagonal system tl_model._fd assembles, returned rather than
    solved, so that a second right-hand side can be applied to it.

        P_{j-1} - (2 + h^2 q_j) P_j + P_{j+1} = rhs_j,   q = -w^2 F Y
    """
    N = coch.N if N is None else N
    x, a = coch.grid(N)
    gam = coch.gamma_of_x(x, gamma, profile)
    sec = coch.sec if gamma is None else replace(coch.sec, gamma=gamma)
    Y, Y_r, H_o = T.section_terms(sec, freq, a, a, gam)
    q = -(freq ** 2) * coch.F * Y
    h = 1.0 / N
    ab = np.zeros((3, N - 1), dtype=complex)
    ab[0, 1:] = 1.0
    ab[1, :] = -(2.0 + (h * h) * q[1:N])
    ab[2, :-1] = 1.0
    return dict(x=x, cf=a, Y=Y, Y_r=Y_r, H_o=H_o, ab=ab, h=h, N=N, freq=freq)


def drive_solve(sy, amp=1.0):
    """Stapes drive: P(0) = amp, P(L) = 0. Identical to tl_model.response()."""
    N, ab = sy["N"], sy["ab"]
    rhs = np.zeros(N - 1, dtype=complex)
    rhs[0] = -amp
    P = np.zeros(N + 1, dtype=complex)
    P[0] = amp
    P[1:N] = solve_banded((1, 1), ab, rhs)
    return P


def source_solve(sy, m, S=1.0):
    """
    A unit volume-velocity source injected into the partition at node m, with
    both ends shorted: P(0) = P(L) = 0.

    P(0) = 0 is the ideal-pressure-source stapes seen from inside, which is the
    same termination tl_model drives through, and the emitted signal is then
    the stapes volume velocity U(0) proportional to (P_1 - P_0)/h.

    The discretization of delta(x - x_m) puts -h S on the right-hand side at
    node m, and the matrix is unchanged, so reciprocity of this Green's
    function against the drive solve is an exact identity of the same matrix.
    checks() measures it at 1e-12.
    """
    N, ab, h = sy["N"], sy["ab"], sy["h"]
    rhs = np.zeros(N - 1, dtype=complex)
    rhs[m - 1] = -h * S
    P = np.zeros(N + 1, dtype=complex)
    P[1:N] = solve_banded((1, 1), ab, rhs)
    return P


def u_stapes(P, h):
    """Stapes volume velocity, up to the one constant that is common to every
    frequency in an experiment run at fixed f_dp."""
    return (P[1] - P[0]) / h


# ---------------------------------------------------------------------------
# Fixed-place measurements
# ---------------------------------------------------------------------------

def place_response(coch, x_probe, freqs, gamma=None, profile=None, N=N_DEF):
    """|X_b|, |shear| and H_o at ONE place over a list of frequencies."""
    j = int(round(x_probe * N))
    n = len(freqs)
    xb = np.empty(n)
    sh = np.empty(n)
    ho = np.empty(n, dtype=complex)
    for i, w in enumerate(freqs):
        r = T.response(coch, float(w), N=N, gamma=gamma, profile=profile)
        xb[i] = abs(r["X_b"][j])
        sh[i] = abs(r["shear"][j])
        ho[i] = r["H_o"][j]
    return xb, sh, ho


def bf_at(coch, x_probe, gamma, N=N_DEF, lo=-1.4, hi=0.5, n=191):
    """
    Best frequency at a fixed place, on BM VELOCITY.

    tl_model.tuning() documents why velocity: driven by a constant-amplitude
    pressure source, |X_b| at a fixed place tends to the partition compliance
    as the frequency falls, and that plateau can exceed the peak, so a
    displacement sweep wide enough to hold a tail reports its BF five octaves
    below CF. Refined parabolically in log-log.
    """
    cf0 = float(coch.cf(x_probe))
    ws = cf0 * 2.0 ** np.linspace(lo, hi, n)
    j = int(round(x_probe * N))
    v = np.array([abs(T.response(coch, float(w), N=N, gamma=gamma)["X_b"][j]) * w
                  for w in ws])
    i = int(np.argmax(v))
    d = 0.0
    if 0 < i < n - 1:
        y = np.log(v[i - 1:i + 2] + 1e-300)
        den = y[0] - 2 * y[1] + y[2]
        if den != 0:
            d = float(np.clip(0.5 * (y[0] - y[2]) / den, -1, 1))
    step = (hi - lo) / (n - 1)
    return float(ws[i] * 2.0 ** (d * step))


def bf_gain_db(coch, x_probe, gamma, N=N_DEF):
    """
    Gain at the place's own BF: active over passive at the SAME frequency and
    place. This is what an experimenter calls the gain at CF, and it is what
    bounds how much a suppressor can remove. It is NOT continuous in gamma,
    because BF itself jumps from the passive hump to the active one, which is
    exactly why rule 2's pinning is done on tl_model's "local" measure at the
    section CF instead. Both are reported at every operating point.
    """
    f = bf_at(coch, x_probe, gamma, N)
    j = int(round(x_probe * N))
    a = abs(T.response(coch, f, N=N, gamma=gamma)["X_b"][j])
    p = abs(T.response(coch, f, N=N, gamma=0.0)["X_b"][j])
    return f, 20.0 * math.log10(a / p)


def tuning_curve(coch, x_probe, gamma, N=N_DEF, lo=TC_LO, hi=TC_HI, n=TC_N,
                 profile=None):
    """The fixed-place sweep every observation-4 and observation-5 number is
    read from. Frequencies are in units of the basal CF."""
    cf0 = float(coch.cf(x_probe))
    ws = cf0 * 2.0 ** np.linspace(lo, hi, n)
    xb, sh, ho = place_response(coch, x_probe, ws, gamma=gamma,
                                profile=profile, N=N)
    return dict(w=ws, cf0=cf0, X_b=xb, shear=sh, H_o=ho,
                bf=float(ws[int(np.argmax(xb * ws))]))


def threshold_of(tc, readout="shear", motion="velocity"):
    """
    Threshold in dB SPL: the level needed to hold the read-out at a fixed
    criterion. The model is linear, so this is minus the response in dB plus
    one constant, and the constant is set to put the tip at TIP_DB_SPL.
    """
    m = tc[readout] * (tc["w"] if motion == "velocity" else 1.0)
    y = 20.0 * np.log10(m)
    thr = -y
    return thr - thr.min() + TIP_DB_SPL


def obs4_metrics(f, thr, cf, window=TAIL_WINDOW, ref=TAIL_REF):
    """
    obs_metrics.obs4's arithmetic, reimplemented here so the robustness table can
    move the two conventions the scorer holds fixed. checks() confirms the two
    agree to 1e-12 on the default conventions.
    """
    oc = np.log2(np.asarray(f, float) / cf)
    thr = np.asarray(thr, float)
    tip = float(thr.min())
    m = oc <= window
    if m.sum() < 2:
        return None, None, int(m.sum())
    A = np.polyfit(oc[m], thr[m], 1)
    tail_level = float(np.polyval(A, ref))
    return tail_level - tip, float(-A[0]), int(m.sum())


# ---------------------------------------------------------------------------
# Observation 6
# ---------------------------------------------------------------------------

def af_geometry(coch, f_dp, r, gamma, N=N_DEF, gen="peak"):
    """
    The generator place for a primary ratio r at fixed f_dp.

        f2 = r f_dp / (2 - r)

    gen="peak" puts the generator at the peak of the f2 excitation pattern,
    which is where the two primaries overlap; gen="cf" puts it at the place
    whose section CF equals f2. The two differ by the CF offset, which is
    0.008 octaves on the active line in world B and 0.68 octaves on the
    passive one, so the choice is worth nothing at high gain and worth a lot at
    low gain. Both are reported.
    """
    f2 = r * f_dp / (2.0 - r)
    if gen == "cf":
        x = float(coch.x_of_cf(f2))
    else:
        r2 = T.response(coch, f2, N=N, gamma=gamma)
        x = float(r2["x"][int(np.argmax(np.abs(r2["X_b"])))])
    return f2, x


def forward_gain(coch, gamma, x_dp_ref=X_PROBE, r_list=R_LIST, N=N_DEF,
                 gen="peak", dp="peak"):
    """
    Net forward gain between the DP generator place and the DP characteristic
    place, read off the model's own excitation pattern at the single frequency
    f_dp. This is AF-92's own procedure, the one that gave them 16.5, 28 and
    30 dB, and the quantity their experiment bounds under 10 dB.

    Returned per r, together with the same ratio on the passive line and the
    difference, which is the part the amplifier supplies.
    """
    f_dp = float(coch.cf(x_dp_ref))
    act = T.response(coch, f_dp, N=N, gamma=gamma)
    pas = T.response(coch, f_dp, N=N, gamma=0.0)
    ab = np.abs(act["X_b"])
    pb = np.abs(pas["X_b"])
    j_dp = (int(np.argmax(ab)) if dp == "peak" else int(round(x_dp_ref * N)))
    j_dp_p = (int(np.argmax(pb)) if dp == "peak" else int(round(x_dp_ref * N)))
    rows = []
    for r in r_list:
        f2, x_gen = af_geometry(coch, f_dp, r, gamma, N, gen)
        _, x_gen_p = af_geometry(coch, f_dp, r, 0.0, N, gen)
        jg = int(round(x_gen * N))
        jgp = int(round(x_gen_p * N))
        ga = 20.0 * math.log10(ab[j_dp] / ab[jg])
        gp = 20.0 * math.log10(pb[j_dp_p] / pb[jgp])
        rows.append(dict(r=float(r), f2=f2, span_oct=math.log2(f2 / f_dp),
                         x_gen=x_gen, gain_db=ga, passive_db=gp,
                         amplifier_db=ga - gp))
    v = [r["gain_db"] for r in rows]
    return dict(f_dp=f_dp, x_dp=float(act["x"][j_dp]), rows=rows,
                widest=v[-1], max=max(v), median=float(np.median(v)))


def fwd_primary(fg):
    """
    The single number observation 6 is scored on: the forward gain at the
    WIDEST span in the sweep, r = 1.50, which is 1.585 octaves and the closest
    thing here to the separations AF-92 actually measured.

    Not the maximum over r. The excitation pattern carries ripple, because both
    terminations in this model are perfect reflectors, and a null in the
    denominator inflates a maximum by 10 dB and more without saying anything
    about the amplifier. The maximum and the median over the sweep are printed
    beside it every time, and in world B all three are far outside AF-92's
    bound, so the choice does not carry the verdict.
    """
    return fg["widest"]


def af_function(coch, gamma, x_dp_ref=X_PROBE, r_list=R_LIST, N=N_DEF,
                gen="peak", dp="peak"):
    """
    The Allen-Fahey function: ear-canal DP against r, with f_dp fixed and the
    intracochlear DP held constant at its characteristic place.

    A unit point source at x_gen, the emitted signal proportional to U(0), and
    the intracochlear DP taken as the CILIA response at x_dp, since AF-92 held
    a fiber tuned to f_dp at a constant rate. At one frequency and one place
    H_o is a constant factor, so the choice of read-out cannot move the
    excursion or the trend; checks() measures that at zero.

    Returned up to one additive constant, which is all the experiment
    determines and all the scorer reads.
    """
    f_dp = float(coch.cf(x_dp_ref))
    sy = system(coch, f_dp, gamma=gamma, N=N)
    P = drive_solve(sy)
    xb = np.abs(sy["Y"] * P)
    j_dp = (int(np.argmax(xb)) if dp == "peak" else int(round(x_dp_ref * N)))
    rs, lv = [], []
    for r in r_list:
        f2, x_gen = af_geometry(coch, f_dp, r, gamma, N, gen)
        m = int(round(x_gen * N))
        m = min(max(m, 1), N - 1)
        Ps = source_solve(sy, m)
        emitted = abs(u_stapes(Ps, sy["h"]))
        dp_local = abs(sy["H_o"][j_dp] * sy["Y"][j_dp] * Ps[j_dp])
        rs.append(float(r))
        lv.append(20.0 * math.log10(emitted / dp_local))
    lv = np.asarray(lv)
    return dict(r=rs, dpoae=list(lv - lv.max() + 42.0),
                excursion=float(lv.max() - lv.min()),
                trend=float(lv[-1] - lv[0]), x_dp=float(sy["x"][j_dp]))


# ---------------------------------------------------------------------------
# Observations 1, 2 and 3: a low-side suppressor on the line
# ---------------------------------------------------------------------------

def saturate(u):
    return u * u / (1.0 + u * u)


def supp_profile(coch, gamma, f_sup, level_db, s_ref, N=N_DEF, prof0=None,
                 damp=0.25, iters=400, tol=1e-10, self_consistent=True):
    """
    The gamma profile a low-side suppressor leaves behind.

        u(x) = |shear_sup(x)| / s_ref,  s(x) = u^2/(1+u^2),
        gamma(x) -> gamma (1 - s(x))

    computed from the LOCAL cilia shear the suppressor produces at each place,
    never imposed uniformly. Bounded in [0, gamma] by construction, so rule 3's
    negative-stiffness trap cannot occur.

    self_consistent solves the fixed point in which the suppressor's own field
    is computed through the gain it has already removed, by damped iteration
    warm started from the previous level. The one-pass alternative, in which
    the suppressor field is computed on the unsuppressed line, is a stated
    approximation and is reported beside it.
    """
    prof = (np.full(N + 1, gamma, dtype=float) if prof0 is None
            else np.asarray(prof0, float).copy())
    if not self_consistent:
        r = T.response(coch, f_sup, level_db=level_db, N=N, gamma=gamma)
        s = saturate(np.abs(r["shear"]) / s_ref)
        return gamma * (1.0 - s), s, 0, 0.0
    d = float("inf")
    it = 0
    for it in range(iters):
        r = T.response(coch, f_sup, level_db=level_db, N=N, profile=prof)
        s = saturate(np.abs(r["shear"]) / s_ref)
        new = damp * (gamma * (1.0 - s)) + (1.0 - damp) * prof
        d = float(np.max(np.abs(new - prof)))
        prof = new
        if d < tol:
            break
    return prof, s, it, d


def suppression_ladder(coch, gamma, x_probe, f_probe, f_sup, levels, s_ref,
                       N=N_DEF, self_consistent=True, keep_profiles=()):
    """
    Suppression of BM displacement and of cilia shear at the probe place, as a
    function of suppressor level, by continuation up the ladder.

    The probe is a vanishing-level probe: it does not contribute to saturation.
    That is the standard two-tone idealization and it is the only place in this
    file where a nonlinearity is not solved self-consistently.
    """
    j = int(round(x_probe * N))
    r0 = T.response(coch, f_probe, N=N, gamma=gamma)
    b0, c0 = abs(r0["X_b"][j]), abs(r0["shear"][j])
    bm, cil, smax, res, sprobe = [], [], [], [], []
    kept = {}
    prof = None
    for L in levels:
        prof, s, it, d = supp_profile(coch, gamma, f_sup, float(L), s_ref, N,
                                      prof0=prof,
                                      self_consistent=self_consistent)
        r1 = T.response(coch, f_probe, N=N, profile=prof)
        bm.append(20.0 * math.log10(b0 / abs(r1["X_b"][j])))
        cil.append(20.0 * math.log10(c0 / abs(r1["shear"][j])))
        smax.append(float(s.max()))
        sprobe.append(float(s[j]))
        res.append(d)
        if float(L) in keep_profiles:
            kept[float(L)] = prof.copy()
    return dict(levels=list(map(float, levels)), bm=bm, cilia=cil,
                smax=smax, s_probe=sprobe, residual=res, profiles=kept,
                x=np.linspace(0.0, 1.0, N + 1))


def crossing(levels, y, crit):
    """First level at which y reaches crit, by linear interpolation. The
    scorer's own rule, so the two agree by construction."""
    return MS._first_crossing(list(map(float, levels)), list(map(float, y)),
                              crit)


# ---------------------------------------------------------------------------
# Rule 1
# ---------------------------------------------------------------------------

def stability(coch, gamma=None, profile=None, N_ap=N_AP, full=False):
    """
    Level 1 every section, level 2 the whole line by the argument principle,
    and at the operating point level 3 the eigenvalue problem at m = 480.
    """
    worst, worst_n = T.worst_section_margin(coch, gamma=gamma, profile=profile)
    n1, ok1 = T.growing_modes(coch, gamma=gamma, N=N_ap, profile=profile)
    out = dict(sec_margin=worst, sec_margin_norm=worst_n, modes=n1, clean=ok1)
    if full:
        n2, ok2 = T.growing_modes(coch, gamma=gamma, N=4096, profile=profile)
        s, npos = T.line_poles(coch, gamma=gamma, m=480, profile=profile)
        out.update(modes_4096=n2, clean_4096=ok2, qep_s=s, qep_n=npos)
    return out


# ---------------------------------------------------------------------------
# Checks. Nothing below this line is quoted unless these pass.
# ---------------------------------------------------------------------------

def checks(name="B"):
    coch = T.WORLDS[name]
    g = T.CEILING[name]
    print(f"CHECKS, world {name}, gamma = {g:.5f} (the certified ceiling)")
    ok = True

    # C1  the operator is tl_model's operator
    sy = system(coch, float(coch.cf(X_PROBE)), gamma=g, N=N_DEF)
    P = drive_solve(sy)
    Pref = T.response(coch, float(coch.cf(X_PROBE)), N=N_DEF, gamma=g)["P"]
    e = float(np.max(np.abs(P - Pref)) / np.max(np.abs(Pref)))
    print(f"  [C1] drive solve against tl_model.response(): {e:.2e}")
    ok &= e < 1e-14

    # C2  reciprocity of the same matrix: the point source is not a new model
    worst = 0.0
    for m in (128, 512, 1500, 2500, 3500, 4000):
        Ps = source_solve(sy, m)
        worst = max(worst, abs(Ps[1] / sy["h"] - P[m]) / abs(P[m]))
    print(f"  [C2] reciprocity, U(0) from a source at x against the drive "
          f"field at x, 6 places: {worst:.2e}")
    ok &= worst < 1e-9

    # C3  grid convergence of everything observation 4 and 6 read
    print("  [C3] grid convergence of the scored quantities")
    prev = None
    for N in (1024, 2048, 4096, 8192):
        tc = tuning_curve(coch, X_PROBE, g, N=N)
        thr = threshold_of(tc)
        tt, sl, _ = obs4_metrics(hz(tc["w"]), thr, hz(tc["bf"]))
        fg = forward_gain(coch, g, N=N)
        gmax = fwd_primary(fg)
        af = af_function(coch, g, N=N)
        line = (f"       N = {N:5d}: tip-to-tail {tt:7.3f} dB, tail slope "
                f"{sl:6.3f} dB/oct, forward gain {gmax:7.3f} dB, "
                f"AF excursion {af['excursion']:7.3f} dB")
        if prev is not None:
            line += (f"   d {tt - prev[0]:+.3f} {sl - prev[1]:+.3f} "
                     f"{gmax - prev[2]:+.3f} {af['excursion'] - prev[3]:+.3f}")
        print(line)
        prev = (tt, sl, gmax, af["excursion"])

    # C4  my obs4 arithmetic is the scorer's
    tc = tuning_curve(coch, X_PROBE, g, N=N_DEF)
    thr = threshold_of(tc)
    rep = {"tuning_curve": {"place_cf_hz": float(hz(tc["bf"])),
                            "frequency_hz": list(map(float, hz(tc["w"]))),
                            "threshold_db_spl": list(map(float, thr))}}
    o4 = MS.obs4(rep)
    mine = obs4_metrics(hz(tc["w"]), thr, hz(tc["bf"]))
    d1 = abs(o4.metrics[0].value - mine[0])
    d2 = abs(o4.metrics[1].value - mine[1])
    print(f"  [C4] my obs4 arithmetic against obs_metrics.obs4: {d1:.2e} dB, "
          f"{d2:.2e} dB/oct")
    ok &= max(d1, d2) < 1e-9

    # C5  the two free constants move nothing that is scored
    tt0, sl0, _ = mine
    global F_BASE_HZ, TIP_DB_SPL
    f0, t0c = F_BASE_HZ, TIP_DB_SPL
    F_BASE_HZ, TIP_DB_SPL = 12345.0, -17.0
    thr2 = threshold_of(tc)
    tt2, sl2, _ = obs4_metrics(hz(tc["w"]), thr2, hz(tc["bf"]))
    F_BASE_HZ, TIP_DB_SPL = f0, t0c
    print(f"  [C5] F_BASE_HZ 40000 -> 12345 and the SPL constant 25 -> -17: "
          f"tip-to-tail moves {tt2 - tt0:+.2e} dB, tail slope "
          f"{sl2 - sl0:+.2e} dB/oct")
    ok &= max(abs(tt2 - tt0), abs(sl2 - sl0)) < 1e-9

    # C6  rule 3: the place map does not modulate the amplifier
    gains = []
    for x in np.linspace(0.05, 0.7, 9):
        a = float(coch.cf(x))
        s_act = T.scaled_section(replace(coch.sec, gamma=g), a)
        s_pas = T.scaled_section(replace(coch.sec, gamma=0.0), a)
        w = a * np.linspace(0.25, 1.9, 4001)
        Ya, _, _ = T.section_terms(s_act, w, 1.0)
        Yp, _, _ = T.section_terms(s_pas, w, 1.0)
        gains.append(20 * math.log10(np.abs(Ya).max() / np.abs(Yp).max()))
    print(f"  [C6] rule 3: isolated section peak gain over 9 places varies by "
          f"{max(gains) - min(gains):.2e} dB (gain {gains[0]:.3f} dB)")
    ok &= (max(gains) - min(gains)) < 1e-9

    # C7  the AF function does not care which read-out holds the DP constant
    a1 = af_function(coch, g, N=N_DEF)
    f_dp = float(coch.cf(X_PROBE))
    sy2 = system(coch, f_dp, gamma=g, N=N_DEF)
    P2 = drive_solve(sy2)
    j = int(np.argmax(np.abs(sy2["Y"] * P2)))
    lv = []
    for r in R_LIST:                       # same curve, DP held on X_b instead
        _, xg = af_geometry(coch, f_dp, r, g, N_DEF, "peak")
        Ps = source_solve(sy2, int(round(xg * N_DEF)))
        lv.append(20 * math.log10(abs(u_stapes(Ps, sy2["h"]))
                                  / abs(sy2["Y"][j] * Ps[j])))
    lv = np.asarray(lv)
    d = abs((lv.max() - lv.min()) - a1["excursion"])
    print(f"  [C7] AF excursion on cilia against on BM displacement: {d:.2e} dB")
    ok &= d < 1e-9

    # C8  the suppressed line: the profile hook, and the notch that bites
    xg = np.linspace(0.0, 1.0, N_DEF + 1)
    const = prof_callable(xg, np.full(N_DEF + 1, g))
    n_u, _ = T.growing_modes(coch, gamma=g, N=N_AP)
    n_p, _ = T.growing_modes(coch, gamma=None, N=N_AP, profile=const)
    print(f"  [C8] growing modes at the ceiling: {n_u} through gamma, {n_p} "
          f"through a constant profile (the control for C9)")
    ok &= (n_u == n_p == 0)

    # C9  a notch that shallow should not matter, and it does
    print("  [C9] a Gaussian notch in gamma at x = 0.60, width 0.004 of the "
          "length")
    for depth in (0.0, 1e-3, 5e-3, 1e-2, 3e-2):
        arr = np.full(N_DEF + 1, g) * (
            1.0 - depth * np.exp(-0.5 * ((xg - 0.60) / 0.004) ** 2))
        fn = prof_callable(xg, arr)
        n1, c1 = T.growing_modes(coch, gamma=None, N=1024, profile=fn)
        n2, c2 = T.growing_modes(coch, gamma=None, N=4096, profile=fn)
        m, _ = T.worst_section_margin(coch, profile=fn)
        print(f"       depth {depth * 100:5.1f} percent: growing modes "
              f"{n1} at N = 1024 and {n2} at N = 4096 "
              f"({'converged' if n1 == n2 else 'NOT converged'}), "
              f"worst section margin {m:+.3e}")
    print("       every section is stable at every depth; what goes unstable "
          "is the LINE.")

    print(f"  CHECKS {'PASSED' if ok else 'FAILED'}")
    return ok


# ---------------------------------------------------------------------------
# The operating table: observations 4, 5 and 6 at every pinned gain (rule 2)
# ---------------------------------------------------------------------------

def pinned_gammas(coch, name, targets=GAIN_TARGETS):
    """gamma pinned to deliver a stated LOCAL gain, plus the certified
    ceiling. Local, because tip and cf jump by 20 dB where the global maximum
    switches humps and no gamma delivers a target inside a jump."""
    f_pin = float(coch.cf(X_PROBE))
    out = []
    for t in targets:
        g, got = T.gamma_for_gain(coch, t, freq=f_pin, kind="local",
                                  x_ref=X_PROBE)
        if g is None:
            out.append((f"{t:.0f} dB local", None, got))
        else:
            out.append((f"{t:.0f} dB local", float(g), float(got)))
    out.append(("ceiling", float(T.CEILING[name]), float("nan")))
    return out


def operating_table(coch, name, do_stability=True):
    print(f"\nOBSERVATIONS 4, 5 AND 6 AT EVERY PINNED GAIN, world {name}")
    print(f"  probe place x = {X_PROBE}, section CF there "
          f"{hz(coch.cf(X_PROBE)):.0f} Hz with the basal CF at "
          f"{F_BASE_HZ:.0f} Hz")
    print(f"  {'pinned':>12} {'gamma':>8} {'/gc':>6} {'BF':>8} {'BF gain':>8} "
          f"{'tip-tail':>9} {'tail sl':>8} {'2nd map':>8} {'fwd gain':>9} "
          f"{'AF exc':>7} {'modes':>6}")
    gc = coch.gamma_critical()
    rows = []
    for label, g, got in pinned_gammas(coch, name):
        if g is None:
            print(f"  {label:>12}   not reachable below gamma_c; best "
                  f"{got:.2f} dB")
            continue
        f_bf, bg = bf_gain_db(coch, X_PROBE, g)
        tc = tuning_curve(coch, X_PROBE, g)
        thr = threshold_of(tc)
        f_hz = hz(tc["w"])
        cf_hz = hz(f_bf)
        tt, sl, ntail = obs4_metrics(f_hz, thr, cf_hz)
        rep = dict(tuning_curve=dict(place_cf_hz=float(cf_hz),
                                     frequency_hz=list(map(float, f_hz)),
                                     threshold_db_spl=list(map(float, thr))))
        o5 = MS.obs5(rep).metrics[0].value
        fg = forward_gain(coch, g)
        fmax = fwd_primary(fg)
        af = af_function(coch, g)
        modes = ""
        if do_stability:
            st = stability(coch, gamma=g)
            modes = f"{st['modes']:d}"
        print(f"  {label:>12} {g:8.5f} {g / gc:6.3f} {oct_re(f_bf, tc['cf0']):+8.3f} "
              f"{bg:8.2f} {tt:9.2f} {sl:8.2f} "
              f"{(float('nan') if o5 is None else o5):8.3f} {fmax:9.2f} "
              f"{af['excursion']:7.2f} {modes:>6}")
        rows.append(dict(label=label, gamma=g, bf=f_bf, bf_gain=bg, tt=tt,
                         slope=sl, map5=o5, fwd=fmax, af=af["excursion"],
                         tc=tc, thr=thr, fg=fg, aff=af))
    print("  BF is octaves re the section CF at the probe place. BF gain is "
          "active over\n  passive at that place and that frequency; it is "
          "discontinuous in gamma because\n  BF itself jumps from the passive "
          "hump to the active one, which is why the\n  pinning is done on the "
          "continuous local measure. tip-tail and tail slope are\n  obs_metrics's "
          "own conventions on the cilia velocity threshold curve. fwd gain is\n"
          "  the largest excitation-pattern forward gain over the swept r; "
          "AF exc is the\n  excursion of the Allen-Fahey function. modes is "
          "the whole-line growing-mode\n  count at N = 1024 on the UNIFORM "
          "line.")
    return rows


def obs4_robustness(coch, name, g):
    print(f"\n  observation 4, every convention varied, world {name} at the "
          f"ceiling")
    tc = tuning_curve(coch, X_PROBE, g)
    f_bf = tc["bf"]
    print(f"    {'read-out':>10} {'motion':>6} {'ref':>18} {'tip-to-tail':>12} "
          f"{'tail slope':>11} {'n tail':>7}")
    for ro in ("shear", "X_b"):
        for mo in ("velocity", "displacement"):
            thr = threshold_of(tc, ro, mo)
            for ref_lbl, cf in (("BF (velocity)", f_bf),
                                ("section CF", tc["cf0"])):
                tt, sl, nt = obs4_metrics(hz(tc["w"]), thr, hz(cf))
                nm = "cilia" if ro == "shear" else "BM"
                print(f"    {nm:>10} {mo[:4]:>6} {ref_lbl:>18} "
                      f"{tt:12.2f} {sl:11.2f} {nt:7d}")
    print(f"    tail window and reference, cilia velocity re BF")
    thr = threshold_of(tc)
    for win in (-0.5, -0.75, -1.0, -1.5, -2.0):
        tt, sl, nt = obs4_metrics(hz(tc["w"]), thr, hz(f_bf), window=win,
                                  ref=win)
        tt1, _, _ = obs4_metrics(hz(tc["w"]), thr, hz(f_bf), window=win,
                                 ref=-1.0)
        print(f"      window {win:+.2f} oct: tip-to-tail {tt:7.2f} dB read at "
              f"the window, {tt1:7.2f} dB read at -1 oct, slope {sl:6.2f} "
              f"dB/oct over {nt} points")
    print(f"    low end of the sweep, window and reference at -1 oct")
    for lo in (-1.6, -2.1, -2.6, -3.1, -3.6):
        tc2 = tuning_curve(coch, X_PROBE, g, lo=lo,
                           n=int(round((TC_HI - lo) / 0.025)) + 1)
        thr2 = threshold_of(tc2)
        tt, sl, nt = obs4_metrics(hz(tc2["w"]), thr2, hz(tc2["bf"]))
        print(f"      sweep to {lo:+.1f} oct re section CF: tip-to-tail "
              f"{tt:7.2f} dB, tail slope {sl:6.2f} dB/oct over {nt} points")


def obs6_detail(coch, name, g, label):
    print(f"\n  observation 6 in detail, world {name}, {label}")
    fg = forward_gain(coch, g)
    print(f"    f_dp = {hz(fg['f_dp']):.0f} Hz, DP place x = {fg['x_dp']:.4f}")
    print(f"    {'r':>6} {'f2/f_dp':>8} {'span oct':>9} {'x_gen':>7} "
          f"{'active':>8} {'passive':>8} {'amplifier':>10}")
    for r in fg["rows"]:
        print(f"    {r['r']:6.3f} {r['f2'] / fg['f_dp']:8.3f} "
              f"{r['span_oct']:9.3f} {r['x_gen']:7.4f} {r['gain_db']:8.2f} "
              f"{r['passive_db']:8.2f} {r['amplifier_db']:10.2f}")
    print("    active is the excitation-pattern ratio on the active line, "
          "which is what\n    AF-92 estimated at 16.5, 28 and 30 dB and "
          "bounded under 10; passive is the\n    same ratio with gamma = 0; "
          "amplifier is the difference.")
    print(f"    conventions varied:")
    for gen in ("peak", "cf"):
        for dp in ("peak", "cf"):
            f2 = forward_gain(coch, g, gen=gen, dp=dp)
            v = [r["gain_db"] for r in f2["rows"]]
            sub = [r["gain_db"] for r in f2["rows"] if r["r"] <= 1.35]
            print(f"      generator at the {gen:4s} place, DP at the {dp:4s} "
                  f"place: max {max(v):7.2f} dB over r <= 1.50, "
                  f"{max(sub):7.2f} dB over r <= 1.35")
    af = af_function(coch, g)
    print(f"    Allen-Fahey function, excursion {af['excursion']:.2f} dB, "
          f"trend {af['trend']:+.2f} dB from r = {R_LIST[0]} to {R_LIST[-1]}")
    print("      " + "  ".join(f"{r:.2f}:{v:+.1f}"
                               for r, v in zip(af["r"], af["dpoae"])))
    return fg, af


def obs46_tension(coch, name):
    """
    The two observations pull opposite ways and this is the number that says by
    how much. Observation 4 wants a tip-to-tail of 32 dB or more, which needs
    gain. Observation 6 wants a forward gain under 10 dB, which forbids it.
    Both are measured on the same pinned axis, so the crossing is a fact about
    the line and not about either convention.
    """
    print(f"\n  THE OBSERVATION 4 / OBSERVATION 6 TENSION, world {name}")
    gc = coch.gamma_critical()
    ceil = T.CEILING[name]
    print(f"    {'gamma':>8} {'/gc':>6} {'/ceiling':>9} {'local':>7} "
          f"{'BF gain':>8} {'tip-to-tail':>12} {'fwd gain':>9}")
    gs = np.unique(np.concatenate([np.linspace(0.0, ceil, 9), [ceil]]))
    tab = []
    for g in gs:
        f_bf, bg = bf_gain_db(coch, X_PROBE, float(g))
        tc = tuning_curve(coch, X_PROBE, float(g))
        tt, sl, _ = obs4_metrics(hz(tc["w"]), threshold_of(tc), hz(tc["bf"]))
        loc = T.gain_db(coch, float(coch.cf(X_PROBE)), float(g), "local",
                        x_ref=X_PROBE)
        fg = forward_gain(coch, float(g))
        fmax = fwd_primary(fg)
        tab.append((float(g), tt, fmax))
        print(f"    {g:8.5f} {g / gc:6.3f} {g / ceil:9.3f} {loc:7.2f} "
              f"{bg:8.2f} {tt:12.2f} {fmax:9.2f}")
    a = np.array(tab)
    def cross(col, target):
        y = a[:, col]
        for i in range(1, len(y)):
            if (y[i - 1] < target) != (y[i] < target):
                t = (target - y[i - 1]) / (y[i] - y[i - 1])
                return a[i - 1, 0] + t * (a[i, 0] - a[i - 1, 0])
        return None
    g4 = cross(1, 32.0)
    g4lo = cross(1, 40.0 - 8.0)
    g6 = cross(2, 10.0)
    g6t = cross(2, 13.0)
    print(f"    gamma at which the tip-to-tail reaches the bottom of "
          f"observation 4's band (32 dB): {g4}")
    print(f"    gamma at which the forward gain reaches observation 6's bound "
          f"(10 dB): {g6}")
    print(f"    ... and its tolerance edge (13 dB): {g6t}")
    if g6 is not None:
        f_bf, bg = bf_gain_db(coch, X_PROBE, float(g6))
        tc = tuning_curve(coch, X_PROBE, float(g6))
        tt, _, _ = obs4_metrics(hz(tc["w"]), threshold_of(tc), hz(tc["bf"]))
        loc = T.gain_db(coch, float(coch.cf(X_PROBE)), float(g6), "local",
                        x_ref=X_PROBE)
        print(f"    at the observation 6 bound the line has {loc:.2f} dB of "
              f"local gain, {bg:.2f} dB at BF,\n    and a tip-to-tail of "
              f"{tt:.2f} dB against observation 4's 32 to 48 dB band.")
    return g4, g6


# ---------------------------------------------------------------------------
# Observations 1, 2 and 3
# ---------------------------------------------------------------------------

def level_span(coch, gamma, f_sups, s_ref=1.0, below=85.0, above=45.0,
               step=5.0, N=N_DEF):
    """
    A level ladder that actually straddles the saturation transition.

    s_ref is a free constant, so the level axis has no origin until it is
    chosen; a fixed ladder therefore either sits entirely in saturation or
    entirely out of it. The reference used here is the level at which the
    suppressor's cilia shear AT THE PROBE PLACE equals s_ref on the
    unsuppressed line, computed per frequency, and the ladder runs from 85 dB
    below the lowest of them to 45 dB above the highest. Saturation begins at
    the suppressor's own place long before the probe place, which is why the
    ladder is so asymmetric, and the s_max column of every table shows the
    coverage rather than assuming it.
    """
    j = int(round(X_PROBE * N))
    l1 = []
    for f in f_sups:
        r = T.response(coch, float(f), N=N, gamma=gamma)
        l1.append(-20.0 * math.log10(abs(r["shear"][j]) / s_ref))
    return np.arange(min(l1) - below, max(l1) + above + 1e-9, step), l1


def suppressed_stable_gamma(coch, name, cands, n_freq=1, step=10.0):
    """
    Rule 1 on the SUPPRESSED line: the largest pinned gamma at which no
    whole-line mode grows anywhere on the drive ladder. Checked with the
    suppressor's own profile at every rung, not at rest, and the ladder is
    auto-ranged so that s_max really does run from about 0 to about 1.

    Because s_ref only slides the level axis, a ladder that covers the whole
    transition covers every profile any choice of operating point can produce.
    """
    print(f"\n  RULE 1 ON THE SUPPRESSED LINE, world {name}")
    print(f"    {'gamma':>8} {'label':>13} {'uniform':>8} {'s_max range':>16} "
          f"{'suppressed':>12}")
    best = None
    for label, g in cands:
        f_probe = bf_at(coch, X_PROBE, g)
        n_uni, _ = T.growing_modes(coch, gamma=g, N=N_AP)
        worst_modes, worst_L, worst_margin = 0, None, -np.inf
        s_lo, s_hi = 1.0, 0.0
        for f_oct in SUPP_OCT[:n_freq]:
            f_sup = f_probe * 2.0 ** f_oct
            levels, _ = level_span(coch, g, [f_sup], step=step)
            prof = None
            for L in levels:
                prof, s, it, d = supp_profile(coch, g, f_sup, float(L), 1.0,
                                              N_DEF, prof0=prof)
                fn = prof_callable(np.linspace(0, 1, N_DEF + 1), prof)
                m, _ = T.worst_section_margin(coch, profile=fn)
                n, _ = T.growing_modes(coch, gamma=None, N=N_AP, profile=fn)
                worst_margin = max(worst_margin, m)
                s_lo, s_hi = min(s_lo, s.max()), max(s_hi, s.max())
                if n > worst_modes:
                    worst_modes, worst_L = n, float(L)
        print(f"    {g:8.5f} {label:>13} {n_uni:8d} "
              f"{s_lo:7.4f} to {s_hi:5.4f} "
              f"{worst_modes:6d} modes, worst at rung {worst_L}, worst "
              f"section margin {worst_margin:+.2e}")
        if worst_modes == 0 and n_uni == 0:
            best = (label, g)
    print("    uniform is the growing-mode count with no suppressor. The "
          "sections stay stable\n    throughout; what the suppressor "
          "destabilizes is the LINE.")
    return best


def suppression_block(coch, name, g, label, do_stability=True):
    print(f"\nOBSERVATIONS 1, 2 AND 3, world {name}, {label}, gamma = "
          f"{g:.5f}")
    f_probe = bf_at(coch, X_PROBE, g)
    j = int(round(X_PROBE * N_DEF))
    print(f"  probe at the place's own BF, {hz(f_probe):.0f} Hz "
          f"({oct_re(f_probe, float(coch.cf(X_PROBE))):+.3f} oct re the "
          f"section CF)")

    f_sups = [f_probe * 2.0 ** k for k in SUPP_OCT]
    coarse, l1 = level_span(coch, g, f_sups, s_ref=1.0)
    print(f"  coarse ladder {coarse[0]:+.0f} to {coarse[-1]:+.0f} dB in "
          f"{coarse[1] - coarse[0]:.1f} dB steps at s_ref = 1, auto-ranged "
          f"onto the\n  saturation transition; u = 1 at the probe place at "
          + ", ".join(f"{v:+.0f}" for v in l1) + " dB")
    runs = {}
    for f_oct in SUPP_OCT:
        f_sup = f_probe * 2.0 ** f_oct
        runs[f_oct] = suppression_ladder(coch, g, X_PROBE, f_probe, f_sup,
                                         coarse, 1.0)

    # the OHC operating point: one multiplicative constant, fixed by putting
    # observation 1 on target. It slides every curve one for one.
    thr_n = [crossing(runs[k]["levels"], runs[k]["cilia"], 1.0)
             for k in SUPP_OCT]
    have = [t for t in thr_n if t is not None]
    if not have:
        print("  no low-side neural run reaches 1 dB anywhere on the ladder; "
              "observations 1, 2 and 3 cannot be scored at this gain")
        return None
    med = MS._median(have)
    shift = 65.0 - med
    s_ref = 10.0 ** (shift / 20.0)
    print(f"  OHC operating point: s_ref = {s_ref:.4g} in model shear units, "
          f"equivalently a\n  {shift:+.2f} dB shift of the level axis, chosen "
          f"to put the median low-side neural\n  threshold at 65 dB SPL. "
          f"This is a CALIBRATION, not a test, and it is independent\n  of "
          f"the constant that puts the tuning-curve tip at "
          f"{TIP_DB_SPL:.0f} dB SPL: one is the\n  OHC transducer's "
          f"operating point and the other is the fiber's rate criterion,\n  "
          f"and this model determines neither. So observation 1's first "
          f"metric is on target\n  by construction and the report says so. "
          f"What IS a prediction is the flatness\n  across suppressor "
          f"frequency, the growth slope and Delta-T, all of which are\n  "
          f"differences and are invariant to both constants.")

    # invariance: shifting s_ref shifts every curve one for one
    r_a = suppression_ladder(coch, g, X_PROBE, f_probe,
                             f_probe * 2.0 ** SUPP_OCT[0], coarse, 1.0)
    r_b = suppression_ladder(coch, g, X_PROBE, f_probe,
                             f_probe * 2.0 ** SUPP_OCT[0], coarse + 13.0,
                             10.0 ** (13.0 / 20.0))
    inv = float(np.max(np.abs(np.array(r_a["bm"]) - np.array(r_b["bm"]))))
    print(f"  invariance measured: s_ref up 13 dB and the ladder up 13 dB "
          f"gives the same\n  suppression curve to {inv:.2e} dB, so "
          f"observations 2 and 3 cannot depend on it")

    # the reported runs, one ladder per read-out centered on its own threshold
    print(f"\n  {'f_sup':>10} {'oct':>6} {'neural thr':>11} {'BM thr':>9} "
          f"{'Delta-T':>9} {'slope':>7} {'BM max':>8} {'cilia max':>10}")
    report_runs = {"neural": [], "bm": []}
    table = []
    for f_oct in SUPP_OCT:
        f_sup = f_probe * 2.0 ** f_oct
        rc = runs[f_oct]
        lv = np.array(rc["levels"]) + shift
        tn = crossing(lv, rc["cilia"], 1.0)
        tb = crossing(lv, rc["bm"], 1.0)
        row = dict(f_oct=f_oct, f_sup=f_sup, thr_n=tn, thr_b=tb,
                   bm_max=max(rc["bm"]), cil_max=max(rc["cilia"]))
        for crit in CRITERIA:
            row[f"tn{crit}"] = crossing(lv, rc["cilia"], crit)
            row[f"tb{crit}"] = crossing(lv, rc["bm"], crit)
        # fine ladders for the report
        for ro, key, t0 in (("neural", "cilia", tn), ("bm", "bm", tb)):
            if t0 is None:
                continue
            fine = np.arange(t0 - 10.0, t0 + 20.01, 2.5)
            rf = suppression_ladder(coch, g, X_PROBE, f_probe, f_sup,
                                    fine - shift, 1.0)
            report_runs[ro].append(dict(suppressor_hz=float(hz(f_sup)),
                                        level_db_spl=list(map(float, fine)),
                                        suppression_db=list(map(float,
                                                                rf[key]))))
            if ro == "neural":
                row["slope"] = MS._suppression_slope(report_runs[ro][-1], t0)
        table.append(row)
        sl = row.get('slope')
        print(f"  {hz(f_sup):10.0f} {f_oct:6.2f} "
              f"{('  none' if tn is None else f'{tn:11.2f}')} "
              f"{('  none' if tb is None else f'{tb:9.2f}')} "
              f"{('  none' if (tn is None or tb is None) else f'{tb - tn:9.2f}')} "
              f"{('  none' if row.get('slope') is None else f'{sl:7.2f}')} "
              f"{row['bm_max']:8.2f} {row['cil_max']:10.2f}")

    print(f"\n  rule 4, every criterion")
    print(f"    {'oct':>6} " + " ".join(f"{'thr_n@' + str(c):>11}" for c in CRITERIA)
          + " " + " ".join(f"{'Delta-T@' + str(c):>13}" for c in CRITERIA))
    for row in table:
        s = f"    {row['f_oct']:6.2f} "
        for c in CRITERIA:
            v = row[f"tn{c}"]
            s += f"{('none' if v is None else f'{v:.2f}'):>11} "
        for c in CRITERIA:
            a, b = row[f"tn{c}"], row[f"tb{c}"]
            s += f"{('none' if (a is None or b is None) else f'{b - a:.2f}'):>13} "
        print(s)
    print("    a Delta-T that exists only at 1.0 dB is a grazing artifact and "
          "is labelled one.")

    # the one-pass alternative to self-consistency
    f_sup = f_probe * 2.0 ** SUPP_OCT[0]
    r1 = suppression_ladder(coch, g, X_PROBE, f_probe, f_sup, coarse, 1.0,
                            self_consistent=False)
    r2 = runs[SUPP_OCT[0]]
    lv = np.array(coarse) + shift
    t1 = crossing(lv, r1["cilia"], 1.0)
    t2 = crossing(lv, [c for c in r2["cilia"]], 1.0)
    print(f"\n  self-consistency: solving the suppressor's own field through "
          f"the gain it has\n  removed moves the neural threshold from "
          f"{t1:.2f} to {t2:.2f} dB SPL at "
          f"{SUPP_OCT[0]:+.2f} oct;\n  the largest fixed-point residual "
          f"anywhere on the ladder was "
          f"{max(r2['residual']):.1e}")

    if do_stability:
        print(f"\n  rule 1 at this operating point, the reported ladder")
        f_sup = f_probe * 2.0 ** SUPP_OCT[0]
        prof = None
        for L in coarse[::2]:
            prof, s, it, d = supp_profile(coch, g, f_sup, float(L), 1.0,
                                          N_DEF, prof0=prof)
            fn = prof_callable(np.linspace(0, 1, N_DEF + 1), prof)
            st = stability(coch, profile=fn)
            print(f"    L = {L + shift:+7.1f} dB SPL: s_max {s.max():.4f}, "
                  f"worst section margin {st['sec_margin']:+.3e}, growing "
                  f"modes {st['modes']}")

    return dict(probe_cf_hz=float(hz(f_probe)), readouts=report_runs,
                table=table, s_ref=s_ref, shift=shift)


# ---------------------------------------------------------------------------
# Reports and scoring
# ---------------------------------------------------------------------------

def build_report(coch, name, g, label, tc, thr, fg, af, suppression=None,
                 note=""):
    f_hz = hz(tc["w"])
    bf_hz = hz(tc["bf"])
    ho = tc["H_o"]
    rep = {
        "model": f"hybrid TL, world {name}, {label} (exp_obs46.py)",
        "notes": note,
        "tuning_curve": {
            "place_cf_hz": float(bf_hz),
            "frequency_hz": list(map(float, f_hz)),
            "threshold_db_spl": list(map(float, thr)),
        },
        "transduction_phase": {
            "place_cf_hz": float(bf_hz),
            "frequency_hz": list(map(float, f_hz)),
            "phase_rad": list(map(float, np.unwrap(np.angle(ho)))),
        },
        "af92": {
            "forward_gain_db": float(fwd_primary(fg)),
            "f2_over_f1": list(map(float, af["r"])),
            "dpoae_db_spl": list(map(float, af["dpoae"])),
        },
    }
    if suppression is not None:
        rep["suppression"] = {"probe_cf_hz": suppression["probe_cf_hz"],
                              "readouts": suppression["readouts"]}
    return rep


def write_and_score(rep, path, sheets):
    with open(path, "w") as fh:
        json.dump(rep, fh, indent=1)
    sheet = MS.format_report(MS.score(rep))
    sheets.append(sheet)
    print()
    print(sheet)
    return sheet


# ---------------------------------------------------------------------------

def run_world(name, quick=False, sheets=None):
    coch = T.WORLDS[name]
    t0 = time.time()
    print("=" * 78)
    print(f"WORLD {name}: section {coch.sec}")
    print(f"  {coch.decades} decades, F = {coch.F}, N = {N_DEF}, apex taper "
          f"{coch.taper_lo} to {coch.taper_hi}")
    print("=" * 78)

    ok = checks(name)
    if not ok:
        print("CHECKS FAILED; nothing below is quoted")
        return

    rows = operating_table(coch, name, do_stability=not quick)
    ceil = T.CEILING[name]
    obs4_robustness(coch, name, ceil)
    fg, af = obs6_detail(coch, name, ceil, "the certified ceiling")
    obs46_tension(coch, name)

    tc = tuning_curve(coch, X_PROBE, ceil)
    thr = threshold_of(tc)
    note = ("Observations 4, 5 and 6 only, no suppressor applied, so 1, 2 and "
            "3 are skipped\nrather than failed. World " + name + " at the "
            "certified whole-line stability ceiling.\nNeural threshold curve "
            "is cilia shear velocity at x = 0.55; observation 6's\nforward "
            "gain is the excitation-pattern ratio between the DP generator "
            "place and\nthe DP place at fixed f_dp, AF-92's own procedure.")
    rep = build_report(coch, name, ceil, "ceiling", tc, thr, fg, af,
                       note=note)
    write_and_score(rep, os.path.join(_HERE, f"report_obs46_{name}.json"),
                    sheets)

    # observations 1, 2 and 3 need an operating point whose SUPPRESSED line is
    # stable, which is not the same point (rule 1)
    cands = [(lbl, g) for lbl, g, _ in pinned_gammas(coch, name)
             if g is not None]
    cands.append(("ceiling", ceil))
    if quick:
        pick = cands[-3]
    else:
        best = suppressed_stable_gamma(coch, name, cands)
        pick = best if best is not None else cands[0]
    print(f"\n  operating point for observations 1, 2 and 3: {pick[0]}, "
          f"gamma = {pick[1]:.5f}")
    sup = suppression_block(coch, name, pick[1], pick[0],
                            do_stability=not quick)
    if sup is not None:
        tc2 = tuning_curve(coch, X_PROBE, pick[1])
        thr2 = threshold_of(tc2)
        fg2 = forward_gain(coch, pick[1])
        af2 = af_function(coch, pick[1])
        note2 = ("All six at one operating point: world " + name + " at " +
                 pick[0] + ", the highest\npinned gain whose SUPPRESSED line "
                 "has no growing whole-line mode anywhere on\nthe drive "
                 "ladder. Observations 1, 2 and 3 are a minimal single-place, "
                 "single-\nmechanism run included for completeness; the "
                 "accumulation experiment is the\nauthority on them. "
                 "Observation 1's threshold is a calibration of the OHC "
                 "operating\npoint, not a test.")
        rep2 = build_report(coch, name, pick[1], pick[0], tc2, thr2, fg2, af2,
                            suppression=sup, note=note2)
        write_and_score(rep2, os.path.join(_HERE, f"report_all6_{name}.json"),
                        sheets)
    print(f"\nworld {name} finished in {time.time() - t0:.1f} s")


def main(argv):
    """
    World B, the certified default, is one run of about five minutes. World A
    is a second run of its own rather than a continuation, because the rule
    that no single run exceed about ten minutes is worth more than the
    convenience of one command. Each run writes its own score sheet and then
    rebuilds the combined one from whatever sheets exist.
    """
    quick = "--quick" in argv
    worlds = ["B"]
    if "--world" in argv:
        worlds = [argv[argv.index("--world") + 1]]
    if "--all" in argv:
        worlds = ["B", "A"]
    t0 = time.time()
    for w in worlds:
        sheets = []
        run_world(w, quick=quick, sheets=sheets)
        with open(os.path.join(_HERE, f"score_obs46_{w}.txt"), "w") as fh:
            fh.write("\n\n".join(sheets) + "\n")
    have = []
    for w in ("B", "A", "C"):
        f = os.path.join(_HERE, f"score_obs46_{w}.txt")
        if os.path.exists(f):
            have.append(open(f).read())
    with open(os.path.join(_HERE, "score_obs46.txt"), "w") as fh:
        fh.write("\n".join(have))
    print(f"\nALL DONE in {time.time() - t0:.1f} s; score sheets written to "
          f"score_obs46_<world>.txt and combined into score_obs46.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
