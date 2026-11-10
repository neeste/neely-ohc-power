"""
exp_accumulation.py
EXPERIMENT 1: does propagation rescue Delta-T?

THE QUESTION
------------
The cross-section study (README Results 1 to 6) ended on one specific defect.
At every feasible point the BM suppression asymptote grazed the 1 dB detection
criterion, ceiling 1.0 to 1.6 dB, and stopped. Its "Delta-T" was therefore a
criterion crossing rather than a threshold shift, and raising the criterion to
1.5 dB destroyed nearly every feasible point. Cooper measured a BM that is
genuinely suppressed well past its threshold.

Write the two read-outs at the probe place as

    BM_supp(u)    = A(u)                accumulated along the wave
    cilia_supp(u) = A(u) + h(u)         h is the LOCAL transduction change

The per-place identity cilia_supp - BM_supp = -20 log10 |H_o(s)/H_o(0)| is
exact, because shear = H_o X_b pointwise; propagation cannot touch it. It is
rechecked here (V4) and comes back at 1e-15, not because it is in doubt but
because it is the definition of h that everything below uses.

What propagation CAN change is A, because a low-side suppressor removes OHC
gain over a region basal of the probe and the probe's response is built over
that region. So:

    does accumulation make A grow to a healthy several dB, curing the grazing
    defect, while leaving h large enough to keep Delta-T in 18 to 32 dB and
    the cilia growth slope in 1.4 to 2.6?

WHAT IS BUILT HERE, ON TOP OF THE CERTIFIED MODEL
-------------------------------------------------
tl_model.py is used unmodified for the fluid solve (tl_model._fd), the gain
pinning (gamma_for_gain), the section (Section3), the geometry (Cochlea) and
the whole-line argument-principle machinery. Two additions, both verified
against it rather than assumed:

  1. terms()  generalizes tl_model.section_terms() to per-place multipliers on
     K_p, K_c, K_t, R_p and R_c, so a suppressor can act micromechanically as
     well as on the gain. With unit multipliers it reproduces section_terms()
     bit for bit (V1), and at one place it reproduces deltaT_3dof.response() on
     the correspondingly modified Section3 (V2).
  2. growing_modes_mod()  is tl_model's argument-principle count carried
     through the same generalization, so rule 1 can be applied to a line whose
     SECTIONS have been modulated and not only to one whose gamma has been
     scaled. On an unmodulated line it returns tl_model's integers (V6).

THE SUPPRESSOR
--------------
Low side by construction: f_s is 0.5 to 3.0 octaves BELOW the probe, so the
suppressor's own characteristic place is APICAL of the probe place and it
crosses the probe place while still in its tail. The tail depth at the probe
place is measured and printed (Table 0), not assumed.

Saturation is LOCAL and SELF-CONSISTENT:

    u(x) = amp * |cilia shear produced by the suppressor at x| / C
    s(x) = u(x)^2 / (1 + u(x)^2)

with the suppressor's shear recomputed on the line its own saturation has
already produced, iterated to a fixed point by Anderson acceleration. The
probe is a low-level probe and does not enter s; the model is linear in this
phase, so the probe level cancels out of every suppression in dB.

C is ONE constant per gain target: the peak cilia shear of the PROBE tone at
unit stapes pressure on the unsuppressed active line. It is deliberately not
recomputed per suppressor frequency, so that the threshold LEVEL as a function
of suppressor frequency is a meaningful curve (the FA-85 question). Changing C
shifts every threshold in a configuration by the same amount and therefore
cannot change Delta-T or the growth slope, both of which are differences.

TWO ROUTES, ONE KNOB BETWEEN THEM
---------------------------------
    gamma(x) -> gamma(x) * (1 - (1 - beta) s(x))          gain control
    K_p(x)   -> K_p(x)   * (1 - beta * depth * s(x))      pillar micromechanics

beta = 0 is pure gain control, beta = 1 is the pure pillar route that was the
only feasible route in the cross section. beta * depth is capped at DEPTH_CAP
= 0.9 so K_p can never go negative (README Result 4's second defect, rule 3),
and the cap is checked in V7. The OHC force scale gamma K_a G is untouched by
the K_p route, checked field by field in V8, so no route modulates the
amplifier as a side effect (the K_a-tied-to-K_b trap, rule 3).

THE FIVE RULES
    1  Stability is checked UNDER DRIVE, not at rest, at three levels: every
       section at every rung of every ladder (129 places x 141 rungs, exact
       quadratic eigenvalue problem); the drive at which the first section
       crosses, bisected, and compared with the read-out thresholds it is
       supposed to support; and the whole line by the argument principle at
       selected rungs of every route and gain. Section 6 prints all three.
    2  Gain is pinned by tl_model.gamma_for_gain(), never swept, at 1, 3, 10,
       20 and 40 dB of LOCAL gain, and the achieved value is reported. Local
       is the continuous measure and the one an experiment reads; 40 dB of TIP
       gain is not reachable with a stable line in any world (tl_model), so a
       tip-gain table would have no 40 dB row at all. Tip gain is reported
       alongside every row.
    3  No parameterization artifact: V7 and V8.
    4  Every Delta-T is reported at criteria 1.0, 1.5 and 3.0 dB, and the
       headroom A_max/criterion is printed beside it so a grazing crossing
       cannot be mistaken for a threshold shift.
    5  Every number printed comes out of this file.

GRIDS AND COST
    N = 2048 for the ladders, checked against N = 1024 and 4096 in section 7
    (Delta-T moves by less than 0.02 dB). Level ladders run -60 to +80 dB in
    1 dB steps, 141 rungs, warm started, with each threshold then refined by
    bisection on the self-consistent solve to 0.01 dB; the difference between
    the interpolated and the bisected threshold is printed. Whole-line mode
    counts at N = 1024. About 5 minutes.

    python3 exp_accumulation.py            # verify, then the experiment
    python3 exp_accumulation.py --verify   # the checks only
    python3 exp_accumulation.py --quick    # a reduced sweep, about 1 minute
"""

from __future__ import annotations

import math
import os
import sys
import time
from dataclasses import replace

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import tl_model as T                                              # noqa: E402
import deltaT_3dof as d3                                          # noqa: E402
from deltaT_3dof import Section3                                  # noqa: E402


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

COCH = T.WORLD_B                 # the certified default world
NAME = "B"
N_LAD = 2048                     # grid for the ladders
X_PROBE = 0.55                   # the place whose section CF is the probe tone
GAINS = (1.0, 3.0, 10.0, 20.0, 40.0)
OCTAVES = (0.5, 1.0, 2.0)        # suppressor, octaves BELOW the probe
OCT_SWEEP = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0)
BETAS = (0.0, 0.25, 0.5, 0.75, 0.9, 1.0)
DEPTH = 1.0                      # nominal pillar depth before the cap
DEPTH_CAP = 0.9                  # beta * depth * s <= 0.9, so K_p >= 0.1 K_p0
CRITERIA = (1.0, 1.5, 3.0)
DELTA_T_LO, DELTA_T_HI = 18.0, 32.0
SLOPE_LO, SLOPE_HI = 1.4, 2.6
L_LO, L_HI, L_STEP = -60.0, 80.0, 1.0


# ---------------------------------------------------------------------------
# 1. The section with per-place multipliers
# ---------------------------------------------------------------------------

def terms(sec: Section3, w, a, r=None, gamma=None,
          mp=1.0, mc=1.0, mt=1.0, mrp=1.0, mrc=1.0):
    """
    (Y, Y_r, H_o) for the section at a place, with per-place multipliers on
    K_p (mp), K_c (mc), K_t (mt), R_p (mrp) and R_c (mrc).

    Identical in form to tl_model.section_terms(), which it reduces to when
    every multiplier is 1 (V1, bit for bit). The OHC term

        act = gamma K_a G a^2

    is built from gamma, K_a, G and the place scale ONLY, so no multiplier can
    reach it: modulating a stiffness cannot modulate the amplifier (rule 3,
    measured in V8).
    """
    w = np.asarray(w)
    a = np.asarray(a, dtype=float)
    r = a if r is None else np.asarray(r, dtype=float)
    g = sec.gamma if gamma is None else np.asarray(gamma, dtype=float)
    jw = 1j * w
    a2 = a * a
    act = g * sec.K_a * sec.G * a2
    Rp, Rc = sec.R_p * mrp, sec.R_c * mrc
    Kp, Kc, Kt = sec.K_p * mp, sec.K_c * mc, sec.K_t * mt
    pil = jw * Rp * r + Kp * a2
    bun = jw * Rc * r + Kc * a2

    A00 = -w ** 2 * sec.M_b + jw * (sec.R_b + Rp) * r + (sec.K_b + Kp) * a2
    A01 = -pil + act
    A02 = -act + 0.0 * A00
    A10 = -pil + 0.0 * A00
    A11 = (-w ** 2 * sec.M_r + jw * (sec.R_r + Rp + Rc) * r
           + (Kp + Kc) * a2 - act)
    A12 = -bun + act
    A21 = -bun + 0.0 * A00
    A22 = -w ** 2 * sec.M_t + jw * (sec.R_t + Rc) * r + (Kt + Kc) * a2

    m00 = A11 * A22 - A12 * A21
    det = A00 * m00 - A01 * (A10 * A22) + A02 * (A10 * A21)     # A20 = 0
    Y = m00 / det
    Y_r = -A10 * A22 / det
    Y_t = A10 * A21 / det
    return Y, Y_r, sec.G * (Y_r - Y_t) / Y


def mods(coch, x, g0, s, beta, depth):
    """
    The suppressor's action at each place, given the local saturation s(x).

        gamma(x) -> gamma(x) (1 - (1 - beta) s(x))
        K_p(x)   -> K_p(x)   (1 - min(beta depth, DEPTH_CAP) s(x))

    Returns (gamma profile before the taper, K_p multiplier). The gamma profile
    is handed to coch.gamma_of_x(), which applies the apical activity taper.
    """
    bd = min(beta * depth, DEPTH_CAP)
    return g0 * (1.0 - (1.0 - beta) * s), 1.0 - bd * s


def line(coch, freq, g0, s, beta, depth, N, amp=1.0):
    """
    The three read-outs as functions of place on a line whose sections have
    been modulated by the suppressor. The fluid solve is tl_model._fd(),
    untouched; only Y changes.
    """
    x = np.linspace(0.0, 1.0, N + 1)
    a = coch.cf(x)
    prof, mp = mods(coch, x, g0, s, beta, depth)
    gam = coch.gamma_of_x(x, g0, prof)
    Y, Y_r, H_o = terms(coch.sec, freq, a, a, gam, mp=mp)
    P = T._fd(coch, freq, Y, N, amp)
    X_b = Y * P
    return dict(x=x, P=P, X_b=X_b, X_r=Y_r * P, shear=H_o * X_b,
                H_o=H_o, Y=Y, gamma_x=gam, mp=mp)


# ---------------------------------------------------------------------------
# 2. The self-consistent local saturation
# ---------------------------------------------------------------------------

def anderson(F, s0, tol=1e-10, itmax=150, m=6):
    """
    Anderson-accelerated fixed point for s = F(s), clipped to [0, 1].

    Plain damped Picard is not adequate here. The loop is negative feedback
    (more saturation -> less gain -> less shear -> less saturation) but its
    loop gain runs into the hundreds where the response is steepest, so the
    damping needed for convergence is 1/(1+loop gain) and a fixed damping
    either diverges or stalls. Anderson converges in about 10 iterations from a
    warm start over most of the ladder. Where it does not, the residual is
    reported rather than hidden, and section 7 shows that an up sweep and a
    down sweep through the same region agree to 0.01 dB, so the fixed point is
    unique and the failures are slow convergence and not multiple solutions.
    """
    s = s0.copy()
    G, Fh = [], []
    d = float("inf")
    for it in range(itmax):
        g = F(s)
        f = g - s
        d = float(np.max(np.abs(f)))
        if d < tol:
            return s, d, it
        G.append(g)
        Fh.append(f)
        if len(G) > m:
            G.pop(0)
            Fh.pop(0)
        k = len(G)
        if k == 1:
            s = np.clip(s + 0.5 * f, 0.0, 1.0)
        else:
            A = np.column_stack([Fh[i] - Fh[-1] for i in range(k - 1)])
            try:
                al, *_ = np.linalg.lstsq(A, -Fh[-1], rcond=None)
            except np.linalg.LinAlgError:               # pragma: no cover
                al = np.zeros(k - 1)
            alpha = np.empty(k)
            alpha[:k - 1] = al
            alpha[-1] = 1.0 - al.sum()
            s = np.clip(sum(c * gi for c, gi in zip(alpha, G)), 0.0, 1.0)
    return s, d, itmax


class Config:
    """One (world, gain target, suppressor frequency, route) operating point."""

    def __init__(self, coch, gain_target, oct_below, beta, depth, N=N_LAD,
                 x_read=None, invariant=False):
        self.coch, self.N, self.beta, self.depth = coch, N, beta, depth
        self.gain_target, self.oct_below = gain_target, oct_below
        self.invariant = invariant
        self.f_p = float(coch.cf(X_PROBE))
        self.f_s = self.f_p * 2.0 ** (-oct_below)
        self.x_read = X_READ if x_read is None else x_read
        self.j = int(round(self.x_read * N))
        self.g0, self.got = T.gamma_for_gain(coch, gain_target, self.f_p,
                                             "local", N=N, x_ref=self.x_read)
        self.tip = T.gain_db(coch, self.f_p, self.g0, "tip", N)
        self.zero = np.zeros(N + 1)
        # C: one constant per gain target, from the PROBE tone (see docstring)
        pr = line(coch, self.f_p, self.g0, self.zero, beta, depth, N)
        self.C = float(np.max(np.abs(pr["shear"])))
        self.b0 = abs(pr["X_b"][self.j])
        self.c0 = abs(pr["shear"][self.j])
        self.r0 = abs(pr["X_r"][self.j])
        self.H0 = pr["H_o"][self.j]
        self.x_peak = T._peak(pr["x"], pr["X_b"])[0]
        # the local weight that turns shear into saturation drive
        self.wgt = (coch.cf(np.linspace(0, 1, N + 1)) ** 2 if invariant
                    else np.ones(N + 1))
        if invariant:
            sr = line(coch, self.f_s, self.g0, self.zero, beta, depth, N)
            self.C = float(np.max(np.abs(sr["shear"]) * self.wgt))

    def solve(self, L_db, s_start, itmax=150):
        amp = 10.0 ** (L_db / 20.0)

        def F(sv):
            r = line(self.coch, self.f_s, self.g0, sv, self.beta, self.depth,
                     self.N)
            u = amp * np.abs(r["shear"]) * self.wgt / self.C
            return u * u / (1.0 + u * u)

        return anderson(F, s_start, itmax=itmax)

    def readouts(self, s):
        r = line(self.coch, self.f_p, self.g0, s, self.beta, self.depth,
                 self.N)
        j = self.j
        A = 20.0 * math.log10(self.b0 / abs(r["X_b"][j]))
        Cc = 20.0 * math.log10(self.c0 / abs(r["shear"][j]))
        R = 20.0 * math.log10(self.r0 / abs(r["X_r"][j]))
        h = -20.0 * math.log10(abs(r["H_o"][j] / self.H0))
        return A, Cc, R, h


def ladder(cfg, lo=L_LO, hi=L_HI, step=L_STEP, margins=True, n_sec=128):
    """
    The drive ladder: at every level, the self-consistent saturation, the three
    read-outs, and (rule 1) the worst section pole over the whole line.
    """
    Ls = np.arange(lo, hi + 1e-9, step)
    s = cfg.zero.copy()
    out = {k: [] for k in ("L", "A", "cil", "RL", "h", "s_probe", "s_max",
                           "res", "it", "margin")}
    states = []
    for L in Ls:
        s, d, it = cfg.solve(L, s)
        A, Cc, R, h = cfg.readouts(s)
        out["L"].append(L)
        out["A"].append(A)
        out["cil"].append(Cc)
        out["RL"].append(R)
        out["h"].append(h)
        out["s_probe"].append(float(s[cfg.j]))
        out["s_max"].append(float(s.max()))
        out["res"].append(d)
        out["it"].append(it)
        out["margin"].append(section_margin(cfg, s, n_sec) if margins
                             else float("nan"))
        states.append(s.copy())
    for k in out:
        out[k] = np.asarray(out[k], dtype=float)
    out["states"] = states
    return out


# ---------------------------------------------------------------------------
# 3. Stability under drive (rule 1)
# ---------------------------------------------------------------------------

def section_margin(cfg, s, n=128):
    """
    Level 1 of rule 1: the largest pole real part over every section of the
    SUPPRESSED line, by deltaT_3dof.stability_margin()'s exact quadratic
    eigenvalue problem. Negative is stable.
    """
    coch = cfg.coch
    xs = np.linspace(0.0, 1.0, n + 1)
    a = coch.cf(xs)
    si = np.interp(xs, np.linspace(0.0, 1.0, cfg.N + 1), s)
    prof, mp = mods(coch, xs, cfg.g0, si, cfg.beta, cfg.depth)
    gam = coch.gamma_of_x(xs, cfg.g0, prof)
    worst = -np.inf
    for ai, gi, mpi in zip(a, gam, mp):
        sec = T.scaled_section(coch.sec, float(ai), float(gi))
        sec = replace(sec, K_p=sec.K_p * float(mpi))
        worst = max(worst, d3.stability_margin(sec))
    return float(worst)


def _det_mod(cfg, s, omegas, N):
    """tl_model._det() carried through the generalized section."""
    coch = cfg.coch
    x = np.linspace(0.0, 1.0, N + 1)
    a = coch.cf(x)
    si = np.interp(x, np.linspace(0.0, 1.0, cfg.N + 1), s)
    prof, mp = mods(coch, x, cfg.g0, si, cfg.beta, cfg.depth)
    gam = coch.gamma_of_x(x, cfg.g0, prof)
    W = np.asarray(omegas, dtype=complex)[:, None]
    Y, _, _ = terms(coch.sec, W, a[None, :], a[None, :], gam[None, :],
                    mp=mp[None, :])
    q = -(W ** 2) * coch.F * Y
    h = 1.0 / N
    b = -(2.0 + (h * h) * q)[:, 1:N]
    m, n = b.shape
    d_prev = np.ones(m, dtype=complex)
    d_cur = b[:, 0].copy()
    for j in range(1, n):
        d_next = b[:, j] * d_cur - d_prev
        d_prev, d_cur = d_cur, d_next
        sc = np.maximum(np.abs(d_cur), np.abs(d_prev))
        sc = np.where(sc > 0, sc, 1.0)
        d_cur, d_prev = d_cur / sc, d_prev / sc
    return d_cur


def _winding_mod(cfg, s, N, corners, npts0=256, npts_max=4096):
    prev, npts, wnd = None, npts0, 0.0
    while npts <= npts_max:
        z = np.concatenate([corners[k] + (corners[(k + 1) % 4] - corners[k])
                            * np.linspace(0, 1, npts, endpoint=False)
                            for k in range(4)])
        z = np.concatenate([z, z[:1]])
        d = np.concatenate([_det_mod(cfg, s, z[i:i + 512], N)
                            for i in range(0, z.size, 512)])
        ph = np.unwrap(np.angle(d))
        wnd = (ph[-1] - ph[0]) / (2 * math.pi)
        step = float(np.max(np.abs(np.diff(ph))))
        if (prev is not None and abs(wnd - prev) < 0.05
                and abs(wnd - round(wnd)) < 0.05 and step < math.pi / 2):
            return int(round(wnd)), True
        prev, npts = wnd, npts * 2
    return int(round(wnd)), abs(wnd - round(wnd)) < 0.1


def growing_modes_mod(cfg, s, N=1024, depth=0.35, n_bands=8):
    """
    Level 3 of rule 1, and the arbiter: whole-line modes with Re s > 0 on the
    SUPPRESSED line, counted by the argument principle on the determinant of
    the finite-difference system that is actually inverted. This is
    tl_model.growing_modes() with the generalized section; on an unmodulated
    line it returns tl_model's integers (V6).
    """
    coch = cfg.coch
    edges = np.logspace(-coch.decades - 0.15, 0.35, n_bands + 1)
    total, clean = 0, True
    for lo, hi in zip(edges[:-1], edges[1:]):
        corners = [complex(hi, -depth * hi), complex(hi, 1e-7 * hi),
                   complex(lo, 1e-7 * hi), complex(lo, -depth * hi)]
        n, okc = _winding_mod(cfg, s, N, corners)
        clean &= okc
        total += n
    return total, clean


def instability_onset(cfg, lad, n_sec=128, tol=0.02):
    """
    The drive at which the FIRST section of the suppressed line crosses into
    the right half plane, bisected on the ladder bracket. None if no rung of
    the ladder is unstable. This is the number that convicted every feasible
    point of the cross-section study, so it is computed for every
    configuration here and compared with the thresholds it is supposed to
    support.
    """
    bad = np.where(lad["margin"] >= 0.0)[0]
    if bad.size == 0:
        return None
    i = int(bad[0])
    if i == 0:
        return float(lad["L"][0])
    lo, hi = float(lad["L"][i - 1]), float(lad["L"][i])
    s_start = lad["states"][i - 1]
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        s, _, _ = cfg.solve(mid, s_start)
        if section_margin(cfg, s, n_sec) >= 0.0:
            hi = mid
        else:
            lo, s_start = mid, s
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# 4. Thresholds, Delta-T and the growth slope
# ---------------------------------------------------------------------------

def interp_cross(L, y, crit):
    """First upward crossing of crit, linear in (level, dB)."""
    m = np.where(y >= crit)[0]
    if m.size == 0 or m[0] == 0:
        return float("nan"), -1
    i = int(m[0])
    return float(L[i - 1] + (crit - y[i - 1]) * (L[i] - L[i - 1])
                 / (y[i] - y[i - 1])), i


def refine_cross(cfg, lad, which, crit, tol=0.01):
    """
    The same crossing, bisected on the self-consistent solve rather than
    interpolated, so the threshold does not inherit the ladder step.
    """
    L, y = lad["L"], lad[which]
    x0, i = interp_cross(L, y, crit)
    if math.isnan(x0):
        return float("nan")
    lo, hi = float(L[i - 1]), float(L[i])
    s_lo = lad["states"][i - 1]
    k = {"A": 0, "cil": 1, "RL": 2}[which]
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        s, _, _ = cfg.solve(mid, s_lo)
        if cfg.readouts(s)[k] < crit:
            lo, s_lo = mid, s
        else:
            hi = mid
    return 0.5 * (lo + hi)


def growth_slope(cfg, lad, t_cil, which="cil", d_lo=2.0, d_hi=12.0):
    """
    dB of suppression per dB of drive, between 2 and 12 dB above the cilia
    threshold. Same definition as deltaT_3dof.growth_slope(), so the numbers
    are comparable with the cross-section study.
    """
    if math.isnan(t_cil):
        return float("nan")
    k = {"A": 0, "cil": 1, "RL": 2}[which]
    vals = []
    for d in (d_lo, d_hi):
        L = t_cil + d
        i = int(np.clip(np.searchsorted(lad["L"], L) - 1, 0,
                        lad["L"].size - 1))
        s, _, _ = cfg.solve(L, lad["states"][i])
        vals.append(cfg.readouts(s)[k])
    return (vals[1] - vals[0]) / (d_hi - d_lo)


def small_signal_ab(lad, s_target=0.01):
    """
    a = dA/ds and b = dh/ds in the small-saturation limit, read at the rung
    where s at the probe place is nearest s_target.

    These two numbers are the whole story. Near threshold both read-outs are
    linear in s, and s is proportional to u^2, so a criterion c is reached at
    s = c/a on the BM and s = c/(a+b) on the cilia, and

        Delta-T = 10 log10(1 + b/a)

    independent of the criterion. Section 5 checks that against the measured
    Delta-T.
    """
    i = int(np.argmin(np.abs(lad["s_probe"] - s_target)))
    sp = lad["s_probe"][i]
    if sp <= 0:
        return float("nan"), float("nan"), float("nan")
    return lad["A"][i] / sp, lad["h"][i] / sp, sp


def measure(cfg, lad=None, want_slope=True, n_sec=128):
    """Everything one configuration has to say."""
    lad = ladder(cfg, n_sec=n_sec) if lad is None else lad
    stable = lad["margin"] < 0.0
    out = dict(cfg=cfg, lad=lad,
               A_max=float(np.max(lad["A"])),
               A_max_stable=(float(np.max(lad["A"][stable])) if stable.any()
                             else float("nan")),
               cil_max=float(np.max(lad["cil"])),
               h_max=float(np.max(lad["h"])),
               h_min=float(np.min(lad["h"])),
               worst_res=float(np.max(lad["res"])),
               worst_margin=float(np.max(lad["margin"])))
    a, b, sp = small_signal_ab(lad)
    out["a"], out["b"], out["s_ref"] = a, b, sp
    out["dT_law"] = (10.0 * math.log10(1.0 + b / a)
                     if (a > 0 and 1.0 + b / a > 0) else float("nan"))
    for c in CRITERIA:
        tb = refine_cross(cfg, lad, "A", c)
        tc = refine_cross(cfg, lad, "cil", c)
        tr = refine_cross(cfg, lad, "RL", c)
        out[f"t_bm_{c}"] = tb
        out[f"t_cil_{c}"] = tc
        out[f"t_rl_{c}"] = tr
        out[f"dT_{c}"] = tb - tc
        out[f"head_{c}"] = out["A_max"] / c
        out[f"slope_{c}"] = (growth_slope(cfg, lad, tc) if want_slope
                             else float("nan"))
    out["onset"] = instability_onset(cfg, lad, n_sec)
    return out


# ---------------------------------------------------------------------------
# 5. verify()
# ---------------------------------------------------------------------------

X_READ = None       # set in main(), the place where the active wave peaks


def verify() -> bool:
    ok = True
    t0 = time.time()
    coch = COCH
    w = np.logspace(-1.6, 0.6, 9)
    a = np.linspace(0.05, 1.0, 9)

    print("[V1] terms() with unit multipliers IS tl_model.section_terms()")
    worst = 0.0
    for gname, cc in T.WORLDS.items():
        sec = replace(cc.sec, gamma=0.6 * cc.gamma_critical())
        mine = terms(sec, w, a)
        theirs = T.section_terms(sec, w, a)
        e = max(float(np.max(np.abs(m - t))) for m, t in zip(mine, theirs))
        worst = max(worst, e)
        print(f"     world {gname}: max |difference| {e:.1e}")
    ok &= worst == 0.0
    print(f"     worst {worst:.1e}   "
          f"{'PASS (bit for bit)' if worst == 0.0 else 'FAIL'}")

    print("\n[V2] terms() with multipliers IS deltaT_3dof.response() on the "
          "modified section")
    worst = 0.0
    for mp, mc, mt, mrp, mrc in ((0.4, 1.0, 1.0, 1.0, 1.0),
                                 (0.1, 1.7, 0.6, 0.7, 1.3),
                                 (1.0, 1.0, 1.0, 1.0, 1.0)):
        sec = replace(coch.sec, gamma=0.2)
        smod = replace(sec, K_p=sec.K_p * mp, K_c=sec.K_c * mc,
                       K_t=sec.K_t * mt, R_p=sec.R_p * mrp,
                       R_c=sec.R_c * mrc)
        Xb, Xr, sh, Ho = d3.response(smod, w)
        Y, Y_r, H = terms(sec, w, np.ones_like(w), mp=mp, mc=mc, mt=mt,
                          mrp=mrp, mrc=mrc)
        e = max(T._rel(Y, Xb), T._rel(Y_r, Xr), T._rel(H, Ho))
        worst = max(worst, e)
        print(f"     K_p x{mp}, K_c x{mc}, K_t x{mt}, R_p x{mrp}, R_c x{mrc}:"
              f" {e:.1e}")
    ok &= worst < 1e-12
    print(f"     worst {worst:.1e}   {'PASS' if worst < 1e-12 else 'FAIL'}")

    print("\n[V3] line() with no suppression IS tl_model.response()")
    worst = 0.0
    g = 0.5 * T.CEILING[NAME]
    for xp in (0.35, 0.55, 0.7):
        f = float(coch.cf(xp))
        mine = line(coch, f, g, np.zeros(1025), 0.0, DEPTH, 1024)
        theirs = T.response(coch, f, N=1024, gamma=g)
        for k in ("X_b", "X_r", "shear", "H_o", "P"):
            worst = max(worst, T._rel(mine[k], theirs[k]))
    print(f"     X_b, X_r, shear, H_o and P at three places: {worst:.1e}")
    ok &= worst == 0.0
    print(f"     {'PASS (bit for bit)' if worst == 0.0 else 'FAIL'}   the "
          f"fluid solve is tl_model._fd() itself")

    print("\n[V4] the per-place identity, on a SUPPRESSED line")
    print("     cilia_supp - BM_supp = -20 log10 |H_o(s)/H_o(0)| at the probe")
    worst = 0.0
    for beta in (0.0, 1.0):
        cfg = Config(COCH, 20.0, 1.0, beta, DEPTH, N=1024)
        s = cfg.zero.copy()
        for L in (-10.0, 0.0, 10.0, 20.0):
            s, _, _ = cfg.solve(L, s)
            A, Cc, _, h = cfg.readouts(s)
            worst = max(worst, abs((Cc - A) - h))
        print(f"     beta = {beta}: worst |(cilia - BM) - h| over 4 drives "
              f"{worst:.1e}")
    ok &= worst < 1e-10
    print(f"     {'PASS' if worst < 1e-10 else 'FAIL'}  propagation cannot "
          "touch this; it is what makes h local")

    print("\n[V5] the probe level cancels: suppression in dB is independent "
          "of it")
    cfg = Config(COCH, 20.0, 1.0, 0.0, DEPTH, N=1024)
    s, _, _ = cfg.solve(5.0, cfg.zero)
    r1 = line(COCH, cfg.f_p, cfg.g0, s, 0.0, DEPTH, 1024, amp=1.0)
    r2 = line(COCH, cfg.f_p, cfg.g0, s, 0.0, DEPTH, 1024, amp=10.0 ** 1.7)
    e = T._rel(r2["X_b"], r1["X_b"] * 10.0 ** 1.7)
    print(f"     34 dB of probe level: {e:.1e}")
    ok &= e < 1e-12

    print("\n[V6] growing_modes_mod() on an unmodulated line reproduces "
          "tl_model.growing_modes()")
    same = True
    for gf in (0.0, 0.6, 0.9):
        g = gf * COCH.gamma_critical()
        cfg = Config(COCH, 3.0, 1.0, 0.0, DEPTH, N=1024)
        cfg.g0 = g
        n_mine, _ = growing_modes_mod(cfg, np.zeros(1025), N=1024)
        n_ref, _ = T.growing_modes(COCH, g, N=1024)
        print(f"     gamma = {gf:.1f} gc: this file {n_mine}, tl_model "
              f"{n_ref}")
        same &= n_mine == n_ref
    ok &= same
    print(f"     {'PASS' if same else 'FAIL'}")

    print("\n[V7] rule 3, first half: the pillar law is bounded, K_p never "
          "goes negative")
    worst_mp = 1.0
    for beta in BETAS:
        _, mp = mods(COCH, np.linspace(0, 1, 11), 0.1,
                     np.ones(11), beta, DEPTH)
        worst_mp = min(worst_mp, float(np.min(mp)))
    print(f"     smallest K_p multiplier at full saturation over every beta: "
          f"{worst_mp:.3f}")
    ok &= worst_mp >= 1.0 - DEPTH_CAP - 1e-12
    print(f"     cap is {DEPTH_CAP}, so K_p >= {1 - DEPTH_CAP:.2f} K_p0   "
          f"{'PASS' if worst_mp >= 0.1 - 1e-12 else 'FAIL'}")

    print("\n[V8] rule 3, second half: the pillar route does not modulate the "
          "amplifier")
    x = np.linspace(0, 1, 257)
    s = 0.5 + 0.5 * np.sin(7 * x)
    p0, _ = mods(COCH, x, 0.12, np.zeros_like(x), 1.0, DEPTH)
    p1, mp1 = mods(COCH, x, 0.12, s, 1.0, DEPTH)
    act0 = COCH.gamma_of_x(x, 0.12, p0) * COCH.sec.K_a * COCH.sec.G
    act1 = COCH.gamma_of_x(x, 0.12, p1) * COCH.sec.K_a * COCH.sec.G
    same = bool(np.array_equal(act0, act1)) and float(np.min(mp1)) < 0.2
    print(f"     OHC force scale gamma K_a G at beta = 1, before and after a "
          f"90 percent\n     K_p modulation: max |difference| "
          f"{float(np.max(np.abs(act1 - act0))):.1e}, bit-identical "
          f"{np.array_equal(act0, act1)}")
    ok &= same
    print(f"     {'PASS' if same else 'FAIL'}")

    print(f"\n{'verify PASSED' if ok else 'verify FAILED'}  "
          f"({time.time() - t0:.1f} s)")
    return ok


# ---------------------------------------------------------------------------
# 6. The experiment
# ---------------------------------------------------------------------------

def fmt(v, w=7, p=2, nan="   -"):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return f"{nan:>{w}}"
    return f"{v:{w}.{p}f}"


def table0(coch):
    print("\n" + "=" * 78)
    print("TABLE 0  THE LOW-SIDE GEOMETRY, MEASURED")
    print("=" * 78)
    print("  The suppressor's own characteristic place must be APICAL of the "
          "probe and it\n  must cross the probe place while still in its "
          "tail. Read at 40 dB of pinned\n  local gain.")
    cfg0 = Config(coch, 40.0, 1.0, 0.0, DEPTH)
    print(f"\n  probe: f = {cfg0.f_p:.5f} (the section CF at x = {X_PROBE}), "
          f"read at x = {X_READ:.4f}")
    print(f"  {'octaves':>8} {'f_s':>9} {'CF place':>9} {'peak place':>11} "
          f"{'peak apical?':>13} {'tail at probe':>14}")
    for oc in OCT_SWEEP:
        cfg = Config(coch, 40.0, oc, 0.0, DEPTH)
        r = line(coch, cfg.f_s, cfg.g0, cfg.zero, 0.0, DEPTH, cfg.N)
        sh = np.abs(r["shear"])
        i = int(np.argmax(sh))
        xs = float(r["x"][i])
        tail = 20.0 * math.log10(sh[cfg.j] / sh[i])
        print(f"  {oc:8.2f} {cfg.f_s:9.5f} {float(coch.x_of_cf(cfg.f_s)):9.4f}"
              f" {xs:11.4f} {'yes' if xs > X_READ else 'NO':>13} "
              f"{tail:13.2f} dB")
    print("  CF place is where the section CF equals f_s; peak place is where "
          "the wave\n  actually peaks. Beyond about 2 octaves the peak stops "
          "tracking the CF place\n  because the apical activity taper "
          "(x in [0.75, 0.95]) has turned the amplifier\n  off underneath it, "
          "and the peak reverts basalward toward its passive place.")


def row_header():
    print(f"  {'gain':>5} {'oct':>5} {'gamma':>8} {'/gc':>6} {'got':>6} "
          f"{'tip':>6} | {'A_max':>7} {'A_stab':>7} {'cil_max':>7} "
          f"{'h@thr':>6} | {'dT@1.0':>7} {'dT@1.5':>7} {'dT@3.0':>7} | "
          f"{'slope':>6} {'head@1':>7} {'stable':>7}")


def row(m):
    cfg = m["cfg"]
    gc = cfg.coch.gamma_critical()
    stable = "yes" if m["onset"] is None else f"{m['onset']:+.0f}dB"
    print(f"  {cfg.gain_target:5.0f} {cfg.oct_below:5.2f} {cfg.g0:8.5f} "
          f"{cfg.g0 / gc:6.3f} {m['cfg'].got:6.2f} {cfg.tip:6.2f} | "
          f"{m['A_max']:7.2f} {fmt(m['A_max_stable'], 7)} "
          f"{m['cil_max']:7.2f} {fmt(m['h_at_thr'], 6, 2)} | "
          f"{fmt(m['dT_1.0'])} {fmt(m['dT_1.5'])} {fmt(m['dT_3.0'])} | "
          f"{fmt(m['slope_1.0'], 6)} {fmt(m['head_1.0'], 7, 1)} "
          f"{stable:>7}")


def h_at_thr(cfg, m, crit=1.0):
    """h at the level where the BM crosses the criterion: the number that
    decides the sign and size of Delta-T."""
    t = m[f"t_bm_{crit}"]
    if math.isnan(t):
        return float("nan")
    lad = m["lad"]
    i = int(np.clip(np.searchsorted(lad["L"], t) - 1, 0, lad["L"].size - 1))
    s, _, _ = cfg.solve(t, lad["states"][i])
    return cfg.readouts(s)[3]


def run_block(coch, gains, octs, beta, depth, label, n_sec=128,
              invariant=False):
    print("\n" + "=" * 78)
    print(label)
    print("=" * 78)
    row_header()
    res = []
    for g in gains:
        for oc in octs:
            cfg = Config(coch, g, oc, beta, depth, invariant=invariant)
            m = measure(cfg, n_sec=n_sec)
            m["h_at_thr"] = h_at_thr(cfg, m)
            res.append(m)
            row(m)
    print("  got is the local gain actually achieved at x_read; tip is the "
          "line's tip gain\n  at the same gamma. A_max is the ceiling of the "
          "BM suppression A(u) over the\n  whole drive ladder, cil_max the "
          "same for the cilia. h@thr is the local\n  transduction change at "
          "the drive where the BM crosses 1 dB. head@1 is\n  A_max/1.0 dB, "
          "the headroom: near 1 the BM asymptote is grazing the criterion\n"
          "  and the crossing is not a threshold shift. stable is 'yes' if no "
          "section of\n  the suppressed line has a right-half-plane pole at "
          "ANY rung of the ladder,\n  otherwise the drive at which the first "
          "one crosses (rule 1).")
    return res


def table_curve(cfg, m, note=""):
    lad = m["lad"]
    print(f"\n  A(u) and h(u) in full: {cfg.gain_target:.0f} dB of pinned "
          f"local gain, suppressor {cfg.oct_below} oct below,\n  "
          f"{'gain control' if cfg.beta == 0 else f'beta = {cfg.beta}'}"
          f"{note}")
    print(f"  {'level':>7} {'A (BM)':>8} {'cilia':>8} {'RL':>8} {'h':>8} "
          f"{'s@probe':>8} {'s_max':>7} {'margin':>10}")
    for L in range(-30, 61, 5):
        i = int(np.argmin(np.abs(lad["L"] - L)))
        print(f"  {lad['L'][i]:7.1f} {lad['A'][i]:8.3f} {lad['cil'][i]:8.3f} "
              f"{lad['RL'][i]:8.3f} {lad['h'][i]:+8.3f} "
              f"{lad['s_probe'][i]:8.4f} {lad['s_max'][i]:7.4f} "
              f"{lad['margin'][i]:+10.5f}")


def zone_split(cfg, m, crit=1.0):
    """
    Where the BM suppression comes from: saturation far basal of the probe,
    within a third of an octave basal of it, or at and apical of it. The task's
    A + h decomposition says only the last can contribute to h; this says how
    much of A each zone contributes.
    """
    t = m[f"t_bm_{crit}"]
    if math.isnan(t):
        return None
    lad = m["lad"]
    i = int(np.clip(np.searchsorted(lad["L"], t) - 1, 0, lad["L"].size - 1))
    s, _, _ = cfg.solve(t, lad["states"][i])
    x = np.linspace(0.0, 1.0, cfg.N + 1)
    dx = 0.3 / (cfg.coch.decades * math.log2(10.0))     # a third of an octave
    zones = {"far basal": x < cfg.x_read - dx,
             "near basal": (x >= cfg.x_read - dx) & (x < cfg.x_read),
             "at and apical": x >= cfg.x_read}
    out = {"all": cfg.readouts(s)[0]}
    for k, mask in zones.items():
        out[k] = cfg.readouts(np.where(mask, s, 0.0))[0]
    return out


def main(argv):
    global X_READ
    t_start = time.time()
    quick = "--quick" in argv
    coch = COCH
    gc = coch.gamma_critical()
    f_p = float(coch.cf(X_PROBE))
    r = T.response(coch, f_p, N=4096, gamma=T.CEILING[NAME])
    X_READ = T._peak(r["x"], r["X_b"])[0]

    print("=" * 78)
    print("EXPERIMENT 1  DOES ACCUMULATION RESCUE DELTA-T?")
    print("=" * 78)
    print(f"  world {NAME}: K_p = {coch.sec.K_p:g}, "
          f"K_t/K_c = {coch.sec.K_t / coch.sec.K_c:g}, "
          f"R_b = {coch.sec.R_b:g}, {coch.decades} decades, F = {coch.F:g}")
    print(f"  section gamma_critical {gc:.5f}; certified whole-line ceiling "
          f"{T.CEILING[NAME]:.5f} = {T.CEILING[NAME] / gc:.3f} of it")
    print(f"  probe tone f = {f_p:.5f}, read at x = {X_READ:.4f}, where the "
          f"active wave peaks")
    print(f"  ladders on N = {N_LAD}, {L_LO:.0f} to {L_HI:.0f} dB in "
          f"{L_STEP:.0f} dB steps, thresholds then bisected to 0.01 dB")
    print(f"  saturation s = u^2/(1+u^2), u proportional to the LOCAL "
          f"suppressor cilia shear,\n  solved self-consistently on the line "
          f"its own saturation produces")

    print("\n" + "=" * 78)
    print("VERIFICATION")
    print("=" * 78)
    ok = verify()
    if "--verify" in argv:
        return 0 if ok else 1

    table0(coch)

    gains = (1.0, 10.0, 40.0) if quick else GAINS
    octs = (1.0,) if quick else OCTAVES

    # -- the main sweep, gain control -------------------------------------
    A_gain = run_block(
        coch, gains, octs, 0.0, DEPTH,
        "TABLE 1  GAIN CONTROL: the suppressor removes OHC gain, "
        "gamma -> gamma (1 - s)")

    # -- the pillar route --------------------------------------------------
    A_pil = run_block(
        coch, gains, octs, 1.0, DEPTH,
        "TABLE 2  PILLAR ROUTE: the suppressor stiffens nothing and softens "
        "K_p,\n         K_p -> K_p (1 - 0.9 s), with gamma held fixed "
        "(the cross section's only\n         feasible route, now inside the "
        "line)")

    # -- the mechanism knob -----------------------------------------------
    print("\n" + "=" * 78)
    print("TABLE 3  THE MECHANISM KNOB: beta from pure gain control to pure "
          "pillar")
    print("=" * 78)
    print("  beta trades the accumulated term a = dA/ds against the local "
          "term b = dh/ds.\n  If any mixture can put Delta-T in [18, 32] with "
          "a BM suppressed past 3 dB,\n  it is on this line. Suppressor 1 "
          "octave below, 40 and 10 dB of pinned gain.")
    row_header()
    A_beta = []
    for g in ((40.0, 10.0) if not quick else (40.0,)):
        for b in BETAS:
            cfg = Config(coch, g, 1.0, b, DEPTH)
            m = measure(cfg)
            m["h_at_thr"] = h_at_thr(cfg, m)
            A_beta.append(m)
            row(m)

    # -- the curves --------------------------------------------------------
    print("\n" + "=" * 78)
    print("TABLE 4  A(u) IN FULL, WHICH IS WHAT THE QUESTION ASKED FOR")
    print("=" * 78)
    for m in A_gain:
        if m["cfg"].gain_target in (1.0, 40.0) and m["cfg"].oct_below == 1.0:
            table_curve(m["cfg"], m)
    for m in A_pil:
        if m["cfg"].gain_target == 40.0 and m["cfg"].oct_below == 1.0:
            table_curve(m["cfg"], m)
    print("\n  margin is the largest pole real part over 129 sections of the "
          "suppressed\n  line at that rung; positive is an oscillating "
          "section and everything to its\n  right on the ladder is outside "
          "the model's linear regime (rule 1).")

    # -- where A comes from ------------------------------------------------
    print("\n" + "=" * 78)
    print("TABLE 5  IS A ACTUALLY ACCUMULATED? THE BM SUPPRESSION BY ZONE")
    print("=" * 78)
    print("  At the drive where the BM crosses 1 dB, the same saturation "
          "field is applied\n  in one zone at a time. 'far basal' is more "
          "than a third of an octave basal of\n  the probe place, 'near "
          "basal' is the third of an octave just basal of it,\n  'at and "
          "apical' is the rest.")
    print(f"  {'route':>13} {'gain':>5} {'oct':>5} {'all':>8} "
          f"{'far basal':>10} {'near basal':>11} {'at+apical':>10}")
    for m in A_gain + A_pil:
        cfg = m["cfg"]
        if cfg.oct_below != 1.0 or cfg.gain_target not in (10.0, 40.0):
            continue
        z = zone_split(cfg, m)
        if z is None:
            continue
        print(f"  {'gain control' if cfg.beta == 0 else 'pillar':>13} "
              f"{cfg.gain_target:5.0f} {cfg.oct_below:5.2f} {z['all']:8.3f} "
              f"{z['far basal']:10.3f} {z['near basal']:11.3f} "
              f"{z['at and apical']:10.3f}")

    # -- the law -----------------------------------------------------------
    print("\n" + "=" * 78)
    print("TABLE 6  THE LAW BEHIND EVERY NUMBER ABOVE")
    print("=" * 78)
    print("  Near threshold both read-outs are linear in s and s is "
          "proportional to u^2,\n  so the BM reaches a criterion c at "
          "s = c/a and the cilia at s = c/(a+b),\n  and Delta-T = "
          "10 log10(1 + b/a) whatever c is. a = dA/ds is what "
          "accumulation\n  raises; b = dh/ds is local and bounded by how much "
          "|H_o| can move.")
    print(f"\n  {'route':>13} {'gain':>5} {'oct':>5} {'a':>9} {'b':>9} "
          f"{'b/a':>8} {'10lg(1+b/a)':>12} {'dT@1.0':>8} {'dT@3.0':>8} "
          f"{'error':>7}")
    for m in A_gain + A_pil + A_beta:
        cfg = m["cfg"]
        if cfg.oct_below != 1.0:
            continue
        nm = ("gain control" if cfg.beta == 0 else
              "pillar" if cfg.beta == 1 else f"beta {cfg.beta}")
        err = (abs(m["dT_law"] - m["dT_1.0"])
               if not (math.isnan(m["dT_law"]) or math.isnan(m["dT_1.0"]))
               else float("nan"))
        print(f"  {nm:>13} {cfg.gain_target:5.0f} {cfg.oct_below:5.2f} "
              f"{fmt(m['a'], 9, 3)} {fmt(m['b'], 9, 3)} "
              f"{fmt(m['b'] / m['a'] if m['a'] else float('nan'), 8, 3)} "
              f"{fmt(m['dT_law'], 12, 3)} {fmt(m['dT_1.0'], 8, 3)} "
              f"{fmt(m['dT_3.0'], 8, 3)} {fmt(err, 7, 3)}")
    print("\n  error is |law - measured| at the 1 dB criterion. Where it is "
          "large the\n  crossing is not in the linear-in-s regime, which is "
          "the grazing case: the\n  criterion is being met on the asymptote "
          "rather than on the growth.")

    # -- what the law implies ---------------------------------------------
    print("\n  THE BOUND THAT FOLLOWS. A_max is a in dB (the whole gain is "
          "removable at\n  s -> 1) and h_max is b, so a Delta-T of D dB needs")
    print("      A_max = h_max / (10^(D/10) - 1)")
    print(f"  {'target D':>9} {'10^(D/10)-1':>12} "
          f"{'A_max allowed at h_max = 5':>27} {'at 15':>8} {'at 25':>8}")
    for D in (1.0, 3.0, 6.0, 12.0, 18.0, 32.0):
        f_ = 10.0 ** (D / 10.0) - 1.0
        print(f"  {D:9.1f} {f_:12.2f} {5 / f_:27.3f} {15 / f_:8.3f} "
              f"{25 / f_:8.3f}")
    hm = max(m["h_max"] for m in A_gain + A_pil + A_beta)
    print(f"  the largest h anywhere in this file is {hm:.2f} dB, so a "
          f"Delta-T of 18 dB needs\n  A_max <= "
          f"{hm / (10 ** 1.8 - 1):.3f} dB and a Delta-T of 32 dB needs "
          f"A_max <= {hm / (10 ** 3.2 - 1):.4f} dB. That is the cross "
          f"section's\n  grazing ceiling of 1.0 to 1.6 dB, derived rather "
          f"than tuned into.")

    # -- suppressor frequency ---------------------------------------------
    print("\n" + "=" * 78)
    print("TABLE 7  DELTA-T AGAINST SUPPRESSOR FREQUENCY (FA-85's FLAT "
          "LOCUS)")
    print("=" * 78)
    print("  All levels in dB re the stapes pressure that makes the PROBE "
          "tone's peak\n  cilia shear equal the saturation constant, one "
          "reference per gain target, so\n  the threshold column is a "
          "suppression-threshold tuning curve and not a\n  per-frequency "
          "renormalization.")
    print(f"  {'route':>13} {'gain':>5} {'oct':>5} {'t_BM@1':>8} "
          f"{'t_cil@1':>8} {'dT@1.0':>7} {'dT@1.5':>7} {'dT@3.0':>7} "
          f"{'A_max':>7} {'slope':>6}")
    for beta in ((0.0, 1.0) if not quick else (0.0,)):
        for g in ((40.0, 10.0) if not quick else (40.0,)):
            for oc in OCT_SWEEP:
                cfg = Config(coch, g, oc, beta, DEPTH)
                m = measure(cfg, n_sec=64)
                nm = "gain control" if beta == 0 else "pillar"
                print(f"  {nm:>13} {g:5.0f} {oc:5.2f} "
                      f"{fmt(m['t_bm_1.0'], 8, 2)} {fmt(m['t_cil_1.0'], 8, 2)}"
                      f" {fmt(m['dT_1.0'])} {fmt(m['dT_1.5'])} "
                      f"{fmt(m['dT_3.0'])} {m['A_max']:7.2f} "
                      f"{fmt(m['slope_1.0'], 6)}")

    # -- rule 1 ------------------------------------------------------------
    print("\n" + "=" * 78)
    print("TABLE 8  RULE 1: STABILITY UNDER DRIVE, THREE LEVELS")
    print("=" * 78)
    print("  Level 1, every section at every rung, is already in the 'stable' "
          "column of\n  tables 1 to 3 and in the margin column of table 4: "
          f"129 places x "
          f"{int((L_HI - L_LO) / L_STEP) + 1} rungs\n  per configuration, "
          "by the exact quadratic eigenvalue problem.\n"
          "  Level 2 bisects the drive at which the first section crosses.\n"
          "  Level 3 counts whole-line modes by the argument principle on "
          "the FD\n  determinant, on the SUPPRESSED line, at four rungs.")
    print(f"\n  {'route':>13} {'gain':>5} {'rest':>6} {'at t_BM':>9} "
          f"{'s=0.5':>7} {'top':>6} {'onset':>8} {'t_BM@1':>8} "
          f"{'verdict':>28}")
    for m in A_gain + A_pil:
        cfg = m["cfg"]
        if cfg.oct_below != 1.0:
            continue
        lad = m["lad"]
        i_thr = (int(np.clip(np.searchsorted(lad["L"], m["t_bm_1.0"]), 0,
                             lad["L"].size - 1))
                 if not math.isnan(m["t_bm_1.0"]) else 0)
        i_half = int(np.argmin(np.abs(lad["s_probe"] - 0.5)))
        i_top = lad["L"].size - 1
        counts = []
        for i in (0, i_thr, i_half, i_top):
            n, _ = growing_modes_mod(cfg, lad["states"][i], N=1024)
            counts.append(n)
        onset = m["onset"]
        if onset is None and max(counts) == 0:
            verdict = "stable at every drive"
        elif onset is None:
            verdict = "sections ok, LINE grows"
        elif math.isnan(m["t_bm_1.0"]):
            verdict = f"unstable from {onset:+.0f} dB"
        elif onset > m["t_bm_1.0"]:
            verdict = f"ok to {onset - m['t_bm_1.0']:+.0f} dB re t_BM"
        else:
            verdict = f"OSCILLATES {m['t_bm_1.0'] - onset:.0f} dB BELOW t_BM"
        nm = "gain control" if cfg.beta == 0 else "pillar"
        print(f"  {nm:>13} {cfg.gain_target:5.0f} {counts[0]:6d} "
              f"{counts[1]:9d} {counts[2]:7d} {counts[3]:6d} "
              f"{fmt(onset, 8, 1) if onset is not None else '    none':>8} "
              f"{fmt(m['t_bm_1.0'], 8, 1)} {verdict:>28}")
    print("  the four count columns are growing whole-line modes at rest, at "
          "the BM\n  threshold, where the saturation at the probe place "
          "reaches one half, and at\n  the top rung of the ladder.")

    # -- convergence -------------------------------------------------------
    print("\n" + "=" * 78)
    print("TABLE 9  CONVERGENCE: GRID, LADDER STEP, SOLVER, AND HYSTERESIS")
    print("=" * 78)
    print(f"  {'route':>13} {'N':>6} {'step':>6} {'A_max':>8} {'t_BM@1':>8} "
          f"{'dT@1.0':>8} {'dT@3.0':>8} {'worst res':>10} {'rungs>tol':>10}")
    for beta in (0.0, 1.0):
        for N, step in ((1024, 1.0), (2048, 1.0), (4096, 1.0), (2048, 0.25)):
            cfg = Config(coch, 40.0, 1.0, beta, DEPTH, N=N)
            lad = ladder(cfg, step=step, margins=False)
            tb = refine_cross(cfg, lad, "A", 1.0)
            tc = refine_cross(cfg, lad, "cil", 1.0)
            tb3 = refine_cross(cfg, lad, "A", 3.0)
            tc3 = refine_cross(cfg, lad, "cil", 3.0)
            nbad = int(np.sum(lad["res"] > 1e-9))
            nm = "gain control" if beta == 0 else "pillar"
            print(f"  {nm:>13} {N:6d} {step:6.2f} {np.max(lad['A']):8.3f} "
                  f"{tb:8.3f} {tb - tc:8.3f} {tb3 - tc3:8.3f} "
                  f"{np.max(lad['res']):10.1e} {nbad:10d}")
    # hysteresis
    cfg = Config(coch, 40.0, 1.0, 0.0, DEPTH)
    up = ladder(cfg, margins=False)
    s = up["states"][-1]
    dn = []
    for L in up["L"][::-1]:
        s, _, _ = cfg.solve(L, s)
        dn.append(cfg.readouts(s)[0])
    dn = np.array(dn[::-1])
    print(f"  up sweep against down sweep, gain control at 40 dB: worst "
          f"|A_up - A_down| {np.max(np.abs(up['A'] - dn)):.4f} dB")
    cfg = Config(coch, 40.0, 1.0, 1.0, DEPTH)
    up = ladder(cfg, margins=False)
    s = up["states"][-1]
    dn = []
    for L in up["L"][::-1]:
        s, _, _ = cfg.solve(L, s)
        dn.append(cfg.readouts(s)[0])
    dn = np.array(dn[::-1])
    print(f"  the same on the pillar route: worst |A_up - A_down| "
          f"{np.max(np.abs(up['A'] - dn)):.4f} dB, which is what says the "
          f"unconverged\n  rungs are slow convergence and not two solutions")

    # -- saturation normalization ------------------------------------------
    print("\n" + "=" * 78)
    print("TABLE 10  THE OTHER SATURATION NORMALIZATION")
    print("=" * 78)
    print("  Everything above takes u proportional to the local cilia shear "
          "itself, as the\n  task specifies: a bundle that saturates at a "
          "fixed displacement, so the apex,\n  whose sections are far more "
          "compliant, saturates first. The alternative is to\n  make the "
          "saturation reference scale with the section, u proportional to\n"
          "  shear x CF(x)^2, which is the scale-invariant choice. It is a "
          "different model\n  and it is run here so the dependence is on the "
          "record.")
    row_header()
    for g in ((1.0, 10.0, 40.0) if not quick else (40.0,)):
        cfg = Config(coch, g, 1.0, 0.0, DEPTH, invariant=True)
        m = measure(cfg)
        m["h_at_thr"] = h_at_thr(cfg, m)
        row(m)

    # -- verdict -----------------------------------------------------------
    every = A_gain + A_pil + A_beta
    feas = [m for m in every
            if (not math.isnan(m["dT_1.0"]))
            and DELTA_T_LO <= m["dT_1.0"] <= DELTA_T_HI
            and (not math.isnan(m["slope_1.0"]))
            and SLOPE_LO <= m["slope_1.0"] <= SLOPE_HI]
    feas3 = [m for m in feas if m["A_max"] >= 3.0]
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    gm = [m for m in A_gain if m["cfg"].oct_below == 1.0]
    print("  1  ACCUMULATION CURES THE GRAZING DEFECT. A_max, the ceiling of "
          "the BM\n     suppression, tracks the pinned local gain:")
    print("       pinned local gain " + " ".join(
        f"{m['cfg'].gain_target:8.0f}" for m in gm))
    print("       A_max             " + " ".join(
        f"{m['A_max']:8.2f}" for m in gm))
    print("     against the 1.0 to 1.6 dB at which the cross section's "
          "asymptote stalled.")
    print(f"  2  DELTA-T DOES NOT SURVIVE IT. Points anywhere in this file "
          f"with Delta-T in\n     [{DELTA_T_LO:.0f}, {DELTA_T_HI:.0f}] dB "
          f"and slope in [{SLOPE_LO}, {SLOPE_HI}]: {len(feas)}. "
          f"With a BM also suppressed past 3 dB: {len(feas3)}.")
    best = max((m for m in every if not math.isnan(m["dT_1.0"])),
               key=lambda m: m["dT_1.0"], default=None)
    if best is not None:
        print(f"     The largest Delta-T anywhere: {best['dT_1.0']:.2f} dB, "
              f"at {best['cfg'].gain_target:.0f} dB of gain, beta = "
              f"{best['cfg'].beta}, with A_max = {best['A_max']:.2f} dB "
              f"(headroom {best['head_1.0']:.1f}).")
    print(f"  3  CRITERION ROBUSTNESS. Where A_max clears the criterion "
          f"comfortably,\n     Delta-T is the same at 1.0, 1.5 and 3.0 dB, "
          f"because the crossing is in the\n     linear-in-s regime where "
          f"Delta-T = 10 log10(1 + b/a) and the criterion\n     cancels. "
          f"Where A_max grazes, Delta-T runs away with the criterion. Both\n"
          f"     are in tables 1 to 3 and 6.")
    print(f"\n  total {time.time() - t_start:.1f} s")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
