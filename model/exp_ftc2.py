"""exp_ftc2.py -- cat tuning-curve campaign, second design.

Two changes from exp_ftc1, both prompted by the failure of that
campaign and by Shera's apical-basal transition:

1. THE GRADIENT HAS A TRANSITION. exp_ftc1 imposed a single power law
   on the damping scale across the whole cochlea, r = c_R a^(1+alpha).
   The Liberman-Delgutte population is not a single power law: its
   binned medians run Q10 1.97, 2.36, 2.97 through 1.7 kHz and then
   4.48, 5.10, 6.26 above 3.4 kHz, and its one-octave tip-to-tail sits
   at 31, 31, 33, 35 dB through 3.4 kHz and then 56, 60 dB above
   6.8 kHz. The break is at about 3 kHz, which is the apical-basal
   transition CF that Shera and colleagues obtain for cat from SFOAE
   delays and identify with a change in the shapes of neural tuning
   curves. So the damping scale here is a BROKEN power law,

     log10 r = log10 c_R + (1 + alpha_a) log10 a
               + (alpha_b - alpha_a) w log10(1 + (a/a_t)^(1/w)),

   with slope 1 + alpha_a apical of the transition CF a_t, slope
   1 + alpha_b basal of it, and w the transition width in decades.
   alpha_a = 0 is exact scale invariance apically. The transition CF is
   free, so the fitted value is a prediction to compare against the
   3 kHz OAE estimate.

   The activity is graded too, by the SAME parameterization the human
   fit uses: distance from local critical force nu = nu0 a^alpha_nu,
   so alpha_nu < 0 makes the apex less active and therefore more
   broadly tuned. This is not a new mechanism for the cat; it is the
   human model's symmetry-breaking lever applied to the second
   species, and the population's apical-to-basal Q10 growth (2.0 to
   6.3) is what asks for it.

2. THE SIX OBSERVATIONS GATE THE FIT. exp_ftc1 fitted tip placement,
   Q10 and tip-to-tail only, and its winner broke observation 4's tail
   slope and observation 5's transduction phase step. Tuning-curve
   shape is secondary to the observations, so here the observation
   windows are a hard gate and the population fit is minimized only
   within them. Those two are cheap to evaluate (both come from the quiet
   linear tuning curve at the probe place), so they enter as hinge
   penalties outside their scoring windows, together with the
   tip-to-tail and map windows they must also hold. Observation 6's
   forward-gain bound (Allen and Fahey 1992) is also cheap and is
   gated too: the first phase-2 run of this campaign scored 5 of 6 on
   the official sheet, failing observation 6 at 13.94 dB against its
   13 dB ceiling, because reducing basal damping raises the
   traveling-wave gain basal of the best place, which is exactly what
   that observation constrains. The expensive suppression
   observations (1, 2, 3) stay out of the loop for phase 2.

Every improvement is certified by argument-principle winding, as in
exp_ftc1; a dirty or nonzero certificate rejects the candidate.

Checkpoint ftc2_ckpt.json. Run: python3 exp_ftc2.py [minutes].
"""
import sys, os, json, math, time
sys.path.insert(0, '.'); sys.path.insert(0, '..')
import numpy as np
from dataclasses import replace
from scipy.optimize import minimize
import tl_model as T
import exp_accumulation as E
import deltaT_3dof as d3
import exp_ftc1 as F1
import obs_metrics as MS

CKPT = 'ftc2_ckpt.json'
N = F1.N
SEC, COCH, X, A, TAPER = F1.SEC, F1.COCH, F1.X, F1.A, F1.TAPER
PLACES = F1.PLACES
TG = F1.TG

# scoring windows for the four cheap tuning-curve observables
W_TT = (32.0, 48.0)          # obs 4 tip-to-tail, dB
W_SL = (3.0, 9.0)            # obs 4 tail slope
W_MP = (-0.75, -0.25)        # obs 5 second-filter map, octaves
W_PH = (math.pi - 0.7, math.pi + 0.7)   # obs 5 phase step, radians
W_FG = (0.0, 13.0)           # obs 6 forward CA gain, dB (0-10 target,
#                              tolerance 3); the Allen-Fahey bound
# The six observations outrank tuning-curve shape: the penalty weight
# is set so that ANY violation of an observation window is strictly
# worse than any feasible tuning-curve fit (feasible J_ftc runs 2 to
# 15), making the windows a hard gate that still leaves the optimizer a
# gradient back into the feasible set.
PEN_W = 50.0

# Population threshold criterion, 'vel' (default, shear velocity) or
# 'disp' (hair-bundle shear displacement); see exp_ftc1.CRIT.
# Environment variable FTC_CRIT.
CRIT = F1.CRIT


def profiles(z):
    """Broken-power-law damping and the graded activity profile."""
    alpha_a, alpha_b, c_R = z[0], z[1], math.exp(z[2])
    f_t, w = math.exp(z[3]), z[4]
    nu0, alpha_nu = math.exp(z[5]), z[11]
    base = math.exp(z[9])
    a_t = f_t / base                    # transition CF, dimensionless
    u = np.log10(A / a_t)
    # w*log10(1 + 10^(u/w)) without overflow
    bend = w * np.logaddexp(0.0, u * math.log(10.0) / w) / math.log(10.0)
    log_r = math.log10(c_R) + (1.0 + alpha_a) * np.log10(A) \
        + (alpha_b - alpha_a) * bend
    r = 10.0 ** log_r
    nu = np.clip(nu0 * A ** alpha_nu, 0.02, 0.60)
    gam = (1.0 - nu) * F1.gc_dense(r / A) * TAPER
    return r, gam


def obs45(r, gam):
    """obs 4 (tip-to-tail, tail slope) and obs 5 (map, phase step) from
    the quiet linear tuning curve at the probe place, using
    obs_metrics's own arithmetic."""
    import exp_obs46 as O
    from exp_encoded1 import build_rep
    orig = T.response

    def resp(coch, freq, level_db=0.0, N=None, gamma=None,
             profile=None, engine='fd', **kw):
        Ne = coch.N if N is None else int(N)
        xe = np.linspace(0.0, 1.0, Ne + 1)
        ae = coch.cf(xe)
        re = np.interp(xe, X, r)
        if profile is not None:
            p = profile(xe) if callable(profile) else np.interp(
                xe, X, np.asarray(profile, float))
            g = p * coch.taper(xe)
        elif gamma is not None and float(gamma) == 0.0:
            g = np.zeros_like(ae)
        else:
            g = np.interp(xe, X, gam)
        Y, Y_r, H_o = E.terms(coch.sec, freq, ae, re, g)
        amp = 10.0 ** (level_db / 20.0)
        P = T._fd(coch, freq, Y, Ne, amp)
        X_b = Y * P
        with np.errstate(divide='ignore'):
            lg = np.log10(np.abs(P))
        return dict(x=xe, cf=ae, P=P, X_b=X_b, X_r=Y_r * P,
                    shear=H_o * X_b, H_o=H_o, Y=Y,
                    Z_p=1.0 / (1j * 2 * np.pi * freq * Y),
                    gamma_x=g, lg_P=lg)

    T.response = resp
    # FIX 2026-11-07: the point-source excursion (O.af_function) assembles its
    # matrix with O.system, which used gamma = 1.0 times the taper and r = a.
    # Build it on the profiled line instead (same r and gamma as resp).
    sys_orig = O.system

    def system_prof(coch, freq, gamma=None, profile=None, N=None):
        Ne = coch.N if N is None else int(N)
        rr = resp(coch, freq, N=Ne, gamma=gamma, profile=profile)
        h = 1.0 / Ne
        q = -(freq ** 2) * coch.F * rr['Y']
        ab = np.zeros((3, Ne - 1), dtype=complex)
        ab[0, 1:] = 1.0
        ab[1, :] = -(2.0 + (h * h) * q[1:Ne])
        ab[2, :-1] = 1.0
        return dict(x=rr['x'], cf=rr['cf'], Y=rr['Y'], Y_r=None,
                    H_o=rr['H_o'], ab=ab, h=h, N=Ne, freq=freq)

    O.system = system_prof
    try:
        tc, rep = build_rep(COCH, 1.0)   # nonzero: the patch
        # supplies the profile; 0 would request the passive line
        o4 = MS.obs4(rep); o5 = MS.obs5(rep)
        fg = O.fwd_primary(O.forward_gain(COCH, 1.0))
        exc = O.af_function(COCH, 1.0)['excursion']
        return (o4.metrics[0].value, o4.metrics[1].value,
                o5.metrics[0].value, o5.metrics[1].value,
                float(fg), float(exc))
    finally:
        T.response = orig
        O.system = sys_orig


def objective_value(mm):
    """Population fit with shape weighted above absolute placement.

    exp_ftc1 weighted tip threshold (tol 6 dB), Q10 (tol 25 percent)
    and tip-to-tail (tol 6 dB) roughly equally. Absolute tip placement
    is set largely by the coarse middle-ear stage and the criterion
    offset, both nuisance parameters, and letting it outvote shape is
    what allowed a flat-Q10 fit to score better than one reproducing
    the population's Q10 growth. Here the tolerances are 10 dB on tip
    placement, 15 percent on Q10 and 5 dB on tip-to-tail, so the
    CF-dependence of sharpness and tail depth carries the fit."""
    # BUG FIX 2026-08-29: np.interp requires ASCENDING xp. The
    # original conditional skipped sorting when the metrics arrived in
    # descending-frequency order (the natural PLACES order), so
    # np.interp returned a single blended value for every bin and the
    # objective measured aggregate level, not CF dependence. Sort
    # ascending unconditionally, as refit_vel.J_of always did.
    idx = np.argsort(mm[:, 0]); mm = mm[idx]; mf = mm[:, 0]
    J, terms = 0.0, 0
    for ftip, tip, q10, tt in TG:
        if ftip < mf.min() or ftip > mf.max():
            continue
        J += ((np.interp(ftip, mf, mm[:, 1]) - tip) / 10.0) ** 2
        m_q = np.interp(ftip, mf, mm[:, 2])
        if np.isfinite(q10) and np.isfinite(m_q):
            J += ((m_q - q10) / (0.15 * q10)) ** 2
        m_tt = np.interp(ftip, mf, mm[:, 3])
        if np.isfinite(tt) and np.isfinite(m_tt):
            J += ((m_tt - tt) / 5.0) ** 2
        terms += 3
    return J / max(terms, 1)


def hinge(v, lo, hi, tol):
    if v is None or not np.isfinite(v):
        return 4.0
    if v < lo:
        return ((lo - v) / tol) ** 2
    if v > hi:
        return ((v - hi) / tol) ** 2
    return 0.0


W_EX = (2.0, 14.0)           # obs 6 Allen-Fahey excursion, dB

# The published state should be robust, not merely inside the windows.
# The project's own composite criterion is the WORST normalized margin
# over the scored metrics, where 1 is centred in a window and 0 is on
# its edge. The champion (no gradient) scores +0.064, itself limited by
# observation 6. MARGIN_TARGET asks the fit to beat that comfortably;
# falling short is penalized quadratically, so the optimizer buys
# robustness first and tuning-curve fit with what is left.
MARGIN_TARGET = 0.12
MARGIN_W = 40.0

_WINS = (W_TT, W_SL, W_MP, W_PH, W_FG, W_EX)


def margins(m):
    """Normalized margin of each cheap metric: 1 centred, 0 on edge."""
    out = []
    for v, (lo, hi) in zip(m, _WINS):
        if v is None or not np.isfinite(v):
            out.append(-1.0); continue
        half = (hi - lo) / 2.0
        out.append(1.0 - abs(v - 0.5 * (lo + hi)) / half)
    return out


def penalty(m):
    tt, sl, mp, ph, fg, exc = m
    hard = (hinge(tt, *W_TT, 4.0) + hinge(sl, *W_SL, 1.0)
            + hinge(mp, *W_MP, 0.1) + hinge(ph, *W_PH, 0.2)
            + hinge(fg, *W_FG, 1.0) + hinge(exc, *W_EX, 1.0))
    worst = min(margins(m))
    soft = max(0.0, MARGIN_TARGET - worst) ** 2
    return hard + (MARGIN_W / PEN_W) * soft


def main():
    minutes = float(sys.argv[1]) if len(sys.argv) > 1 else 8.0
    deadline = time.time() + 60.0 * minutes
    if os.path.exists(CKPT):
        ck = json.load(open(CKPT))
        z0, best, nev0 = ck['z'], ck['J'], ck.get('nev', 0)
        stall = ck.get('stall', 0)
    else:
        # alpha_a, alpha_b, log c_R, log f_t(kHz), w, log nu0,
        # log f_hp, log f_lp, nlp, log base(kHz), C0, alpha_nu
        z0 = [0.0, -0.35, 0.0, math.log(3.0), 0.35, math.log(0.0968),
              math.log(0.35), math.log(12.0), 1.0, math.log(45.0),
              40.0, 0.0]
        best, nev0, stall = 1e9, 0, 0
    state = dict(best=best, nev=nev0)

    def obj(z):
        if time.time() > deadline:
            raise KeyboardInterrupt
        alpha_a, alpha_b, w = z[0], z[1], z[4]
        nu0, alpha_nu = math.exp(z[5]), z[11]
        f_t, f_hp, f_lp = math.exp(z[3]), math.exp(z[6]), math.exp(z[7])
        nlp, base, C0 = z[8], math.exp(z[9]), z[10]
        if not (-0.6 < alpha_a < 0.6 and -1.5 < alpha_b < 0.6
                and 0.08 < w < 1.2 and 0.02 < nu0 < 0.30
                and -1.2 < alpha_nu < 0.6
                and 0.3 < f_t < 20.0 and 0.05 < f_hp < 1.5
                and 3.0 < f_lp < 40.0 and 0.3 < nlp < 3.0
                and 25.0 < base < 70.0):
            return 60.0
        try:
            r, gam = profiles(z)
            mm = F1.ftc_metrics(r, gam, base, f_hp, f_lp, nlp, C0, crit=CRIT)
            J_ftc = objective_value(mm)
            m45 = obs45(r, gam)
            J = J_ftc + PEN_W * penalty(m45)
        except Exception:
            return 60.0
        state['nev'] += 1
        if J < state['best']:
            if F1.certify(r, gam):
                state['best'] = J
                json.dump(dict(z=list(map(float, z)), J=float(J),
                               J_ftc=float(J_ftc),
                               pen=float(penalty(m45)),
                               obs45=[float(v) for v in m45],
                               worst_margin=float(min(margins(m45))),
                               nev=state['nev'], certified=True,
                               stall=stall, metrics=mm.tolist(),
                               targets=TG.tolist()),
                          open(CKPT, 'w'))
                print('  incumbent J %.4f (ftc %.4f, pen %.4f, '
                      'worst margin %+.3f) eval %d'
                      % (J, J_ftc, penalty(m45), min(margins(m45)),
                         state['nev']), flush=True)
            else:
                print('  candidate J %.4f REJECTED by certificate' % J,
                      flush=True)
                return J + 5.0
        return J

    def restart(zc, k):
        zc = np.asarray(zc, float)
        sc = 0.02 * (1 + 2 * k)
        sim = [zc.copy()]
        for i in range(len(zc)):
            zi = zc.copy()
            zi[i] += (1.0 if (i + k) % 2 == 0 else -1.0) * sc * max(
                abs(zi[i]), 0.5)
            sim.append(zi)
        return np.array(sim)

    print('exp_ftc2: budget %.0f min, %d bins, seed J %s, stall %d'
          % (minutes, len(TG),
             ('%.4f' % best) if best < 1e8 else 'none', stall),
          flush=True)
    while time.time() < deadline - 5.0:
        run_start = state['best']
        try:
            minimize(obj, z0, method='Nelder-Mead',
                     options=dict(maxfev=100000, xatol=1e-3, fatol=1e-4,
                                  initial_simplex=restart(z0, stall)))
        except KeyboardInterrupt:
            break
        stall = 0 if state['best'] < run_start - 1e-4 else stall + 1
        if os.path.exists(CKPT):
            ck = json.load(open(CKPT))
            z0 = ck['z']; ck['stall'] = stall
            json.dump(ck, open(CKPT, 'w'))
        if time.time() < deadline - 5.0:
            print('restart: stall %d, best J %.4f'
                  % (stall, state['best']), flush=True)
    if os.path.exists(CKPT):
        ck = json.load(open(CKPT)); ck['stall'] = stall
        json.dump(ck, open(CKPT, 'w'))
    print('done: best J %.4f after %d evals, stall %d'
          % (state['best'], state['nev'], stall), flush=True)


if __name__ == '__main__':
    main()
