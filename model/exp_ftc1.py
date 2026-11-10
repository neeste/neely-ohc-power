"""Cat FTC gradient campaign (exp_ftc1).

Question: can the cat model, extended with a basal damping gradient and a
fixed middle-ear stage, reproduce the Liberman-Delgutte neural tuning-curve
population (tc.plt, 92 fibers): the growth of Q10, the deepening of the
tail, and the absolute tip placement across CF, while holding the
operating point at a uniform fraction of exact local critical force?

Model per candidate:
  - cat section (best_d3 params, frozen in phase 1);
  - r(x) = c_R a^(1+alpha_R): damping scale and gradient;
  - gamma(x) = frac * gc_exact(rho(x)) * taper: uniform criticality
    fraction against EXACT per-place gamma_c (80-point dense bisection);
  - middle ear (Guinan-Peake based): threshold correction in dB,
    ME(f) = -40 log10( (f/f_hp) / sqrt(1+(f/f_hp)^2) )
            + 10 nlp log10( 1 + (f/f_lp)^2 ),
    second-order high pass below f_hp, nlp-order rolloff above f_lp;
  - map base CF base_khz maps model frequency to kHz;
  - criterion offset C0 places absolute thresholds.

Free parameters (8): alpha_R, log c_R, frac, log f_hp, log f_lp, nlp,
log base_khz, C0. Objective: weighted squared error of model tip
threshold, Q10, and one-octave tip-to-tail against binned medians of the
92 data curves, evaluated at matched tip frequencies. Stability: every
place subcritical by construction (frac < 1); the incumbent's whole line
is certified on improvement by argument-principle winding (10 log bands,
N=1024); dirty or nonzero certificates reject the candidate.

Checkpoint ftc1_ckpt.json. Run: python3 exp_ftc1.py [minutes].
Phase 2 (separate): re-score the six observations at the winner.
"""
import sys, os, json, math, time
sys.path.insert(0, '.'); sys.path.insert(0, '..')
import numpy as np
from dataclasses import replace
from scipy.optimize import minimize
import tl_model as T
import deltaT_3dof as d3
import exp_accumulation as E

CKPT = 'ftc1_ckpt.json'
N = 1024

B = json.load(open('best_d3.json'))['params']
SECK = ('K_t', 'M_t', 'R_t', 'R_c', 'R_b', 'K_c', 'K_p', 'M_r')
SEC = replace(T.WORLDS['B'].sec, **{k: B[k] for k in SECK})
COCH = replace(T.WORLDS['B'], sec=SEC)
X = np.linspace(0.0, 1.0, N + 1)
A = COCH.cf(X)
TAPER = COCH.taper(X)
# 2026-08-31: basal place added so the objective's interpolation range
# reaches the two highest-CF measured fibres (13.4 and 28.2 kHz), which
# were previously outside the model's tip range and silently skipped.
# An apical place is added for the same reason at the low end, so the
# 0.217 kHz fibre stays covered as the base rises.  Neither is plotted:
# fig_ftc8_rec.py builds its own grid and is unaffected.
PLACES = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)

# ---- data targets: binned medians of the 92 curves ----
def data_targets():
    curves, cur, cf = [], [], None
    for line in open('tc.plt'):
        line = line.strip()
        if line.startswith(';'):
            if 'CF=' in line:
                cf = float(line.split('CF=')[1].split()[0])
        elif line == 'plot':
            if cur and cf: curves.append(np.array(cur))
            cur, cf = [], None
        else:
            p = line.split()
            if len(p) == 2:
                try: cur.append((float(p[0]), float(p[1])))
                except ValueError: pass
    rows = []
    for d in curves:
        f, th = d[:, 0], d[:, 1]
        i0 = int(np.argmin(th)); ftip = f[i0]; tip = th[i0]
        t10 = tip + 10.0
        lo = hi = None
        for i in range(i0, 0, -1):
            if th[i-1] >= t10 >= th[i]:
                lo = np.interp(t10, [th[i], th[i-1]], [f[i], f[i-1]]); break
        for i in range(i0, len(f)-1):
            if th[i] <= t10 <= th[i+1]:
                hi = np.interp(t10, [th[i], th[i+1]], [f[i], f[i+1]]); break
        q10 = ftip/(hi-lo) if (lo and hi) else np.nan
        tt = np.interp(ftip/2, f, th) - tip if f[0] <= ftip/2 else np.nan
        rows.append((ftip, tip, q10, tt))
    rows = np.array(rows)
    edges = np.array([0.15, 0.3, 0.6, 1.2, 2.4, 4.8, 9.6, 19.2, 45.0])
    tg = []
    for lo_, hi_ in zip(edges[:-1], edges[1:]):
        m = (rows[:, 0] >= lo_) & (rows[:, 0] < hi_)
        if m.sum() >= 3:
            med = np.nanmedian(rows[m], axis=0)
            tg.append(med)
    return np.array(tg)          # columns: ftip, tip, q10, tt

import os as _os, json as _json
if _os.path.exists('tc.plt'):
    TG = data_targets()
else:   # the individual curves are not redistributed; binned medians only
    TG = np.array(_json.load(open('../data/tc_bins.json'))['bins'])

_gc_cache = {}
def gc_dense(rho):
    key = round(math.log(rho.min()), 3), round(math.log(rho.max()), 3)
    if key not in _gc_cache:
        rg = np.geomspace(max(rho.min()*0.98, 1e-3), rho.max()*1.02, 80)
        gg = []
        for r_ in rg:
            s2 = replace(SEC, gamma=0.0, R_b=SEC.R_b*r_, R_r=SEC.R_r*r_,
                         R_t=SEC.R_t*r_, R_p=SEC.R_p*r_, R_c=SEC.R_c*r_)
            gg.append(d3.gamma_critical(s2))
        _gc_cache[key] = (np.log(rg), np.asarray(gg))
        if len(_gc_cache) > 500: _gc_cache.clear()
    lx, gy = _gc_cache[key]
    return np.interp(np.log(rho), lx, gy)


def me_db(f_khz, f_hp, f_lp, nlp):
    r = np.asarray(f_khz) / f_hp
    hp = -40.0 * np.log10(r / np.sqrt(1.0 + r * r))
    lp = 10.0 * nlp * np.log10(1.0 + (np.asarray(f_khz) / f_lp) ** 2)
    return hp + lp


def model_state(alpha_R, c_R, frac):
    r = c_R * A ** (1.0 + alpha_R)
    rho = r / A
    gam = frac * gc_dense(rho) * TAPER
    return r, gam


# Population threshold criterion switch.
#   'vel'  (default): constant shear VELOCITY (see below); the read-out
#          constants are those of base_ckpt.json (C0 79.47 dB, f_hp 0.050,
#          f_lp 40.0, n_lp 0.30), refitted for this criterion.
#   'disp' (switch FTC_CRIT=disp): constant hair-bundle
#          shear DISPLACEMENT, L_thr = C0 - 20 log10|xi_hb| + ME(f), with the
#          read-out constants of base_ckpt_disp.json (score_base selects the file).
#   'vel'  : constant shear VELOCITY, |xi_hb| replaced by omega |xi_hb|,
#          with omega/omega_ref = f/(1 kHz) (any constant factor is absorbed
#          by C0), L_thr = C0 - 20 log10(|xi_hb| f/1 kHz) + ME(f).
# Select with the crit argument or the environment variable FTC_CRIT.
CRIT = os.environ.get('FTC_CRIT', 'vel')


def crit_factor(f_khz, crit=None):
    c = CRIT if crit is None else crit
    if c == 'disp':
        return np.ones_like(np.asarray(f_khz, float))
    if c == 'vel':
        return np.asarray(f_khz, float) / 1.0
    raise ValueError('FTC_CRIT must be disp or vel, not %r' % c)


def ftc_metrics(r, gam, base_khz, f_hp, f_lp, nlp, C0, crit=None):
    out = []
    for x0 in PLACES:
        j = int(np.argmin(np.abs(X - x0)))
        cfp = float(A[j])
        fr = cfp * np.logspace(math.log10(0.10), math.log10(1.1), 60)
        resp = []
        for f in fr:
            Y, Y_r, H_o = E.terms(SEC, float(f), A, r, gam)
            P = T._fd(COCH, float(f), Y, N, 1.0)
            resp.append(abs((H_o * Y * P)[j]))
        th = C0 - 20*np.log10(np.array(resp) * crit_factor(fr * base_khz, crit))
        th = th + me_db(fr * base_khz, f_hp, f_lp, nlp)
        i0 = int(np.argmin(th)); ftip = fr[i0] * base_khz; tip = th[i0]
        rel = fr / fr[i0]
        t10 = tip + 10.0
        lo = hi = None
        for i in range(i0, 0, -1):
            if th[i-1] >= t10 >= th[i]:
                lo = np.interp(t10, [th[i], th[i-1]], [rel[i], rel[i-1]]); break
        for i in range(i0, len(rel)-1):
            if th[i] <= t10 <= th[i+1]:
                hi = np.interp(t10, [th[i], th[i+1]], [rel[i], rel[i+1]]); break
        q10 = 1.0/(hi-lo) if (lo and hi) else np.nan
        tt = np.interp(0.5, rel, th) - tip if rel[0] <= 0.5 else np.nan
        out.append((ftip, tip, q10, tt))
    return np.array(out)


def objective_value(mm):
    """Interpolate model metrics at the data bins' tip frequencies."""
    mf = mm[:, 0]
    if not np.all(np.diff(mf) < 0):        # places go apex->base? ensure sort
        idx = np.argsort(mf); mm = mm[idx]; mf = mm[:, 0]
    J, terms = 0.0, 0
    for ftip, tip, q10, tt in TG:
        if ftip < mf.min() or ftip > mf.max():
            continue
        m_tip = np.interp(ftip, mf, mm[:, 1])
        m_q = np.interp(ftip, mf, mm[:, 2])
        m_tt = np.interp(ftip, mf, mm[:, 3])
        J += ((m_tip - tip) / 6.0) ** 2
        if np.isfinite(q10) and np.isfinite(m_q):
            J += ((m_q - q10) / (0.25 * q10)) ** 2
        if np.isfinite(tt) and np.isfinite(m_tt):
            J += ((m_tt - tt) / 6.0) ** 2
        terms += 3
    return J / max(terms, 1)


def certify(r, gam):
    F = COCH.F
    h = 1.0 / N
    edges = np.logspace(-2.9, 0.35, 11)
    for lo, hi in zip(edges[:-1], edges[1:]):
        depth = 0.35 * hi
        corners = [complex(hi, -depth), complex(hi, 1e-6),
                   complex(lo, 1e-6), complex(lo, -depth),
                   complex(hi, -depth)]
        pts = []
        for c0, c1 in zip(corners[:-1], corners[1:]):
            tt_ = np.linspace(0, 1, 1000, endpoint=False)
            pts.extend(c0 + (c1 - c0) * tt_)
        tot, ok, prev = 0.0, True, None
        for w in pts:
            Y, _, _ = E.terms(SEC, w, A, r, gam)
            q = -(w * w) * F * Y
            AA = -(2.0 + h * h * q)
            d0, d1 = 1.0 + 0j, AA[1]
            for jj in range(2, N):
                d2 = AA[jj] * d1 - d0
                m_ = abs(d2) or 1e-300
                d0, d1 = d1 / m_, d2 / m_
            cur = np.angle(d1)
            if prev is not None:
                dph = cur - prev
                while dph > math.pi: dph -= 2 * math.pi
                while dph < -math.pi: dph += 2 * math.pi
                if abs(dph) > math.pi / 2: ok = False
                tot += dph
            prev = cur
        if (not ok) or abs(tot / (2 * math.pi)) > 0.1:
            return False
    return True


def main():
    minutes = float(sys.argv[1]) if len(sys.argv) > 1 else 8.0
    deadline = time.time() + 60.0 * minutes
    if os.path.exists(CKPT):
        ck = json.load(open(CKPT))
        z0 = ck['z']; best = ck['J']; nev0 = ck.get('nev', 0)
        stall = ck.get('stall', 0)
    else:
        # alpha_R, log c_R, frac, log f_hp, log f_lp, nlp, log base, C0
        z0 = [-0.15, 0.0, 0.9032, math.log(0.35), math.log(12.0), 1.0,
              math.log(45.0), 40.0]
        best, nev0, stall = 1e9, 0, 0
    state = dict(best=best, nev=nev0)

    def obj(z):
        if time.time() > deadline:
            raise KeyboardInterrupt
        alpha_R, lcR, frac = z[0], z[1], z[2]
        f_hp, f_lp, nlp = math.exp(z[3]), math.exp(z[4]), z[5]
        base, C0 = math.exp(z[6]), z[7]
        if not (-1.2 < alpha_R < 0.3 and 0.70 < frac < 0.945
                and 0.05 < f_hp < 1.5 and 3.0 < f_lp < 40.0
                and 0.3 < nlp < 3.0 and 25.0 < base < 70.0):
            return 50.0
        try:
            r, gam = model_state(alpha_R, math.exp(lcR), frac)
            mm = ftc_metrics(r, gam, base, f_hp, f_lp, nlp, C0)
            J = objective_value(mm)
        except Exception:
            return 50.0
        state['nev'] += 1
        if J < state['best']:
            if certify(r, gam):
                state['best'] = J
                json.dump(dict(z=list(map(float, z)), J=float(J),
                               nev=state['nev'], certified=True,
                               metrics=mm.tolist(),
                               targets=TG.tolist()),
                          open(CKPT, 'w'))
                print('  incumbent J %.4f (certified), eval %d'
                      % (J, state['nev']), flush=True)
            else:
                print('  candidate J %.4f REJECTED by certificate' % J,
                      flush=True)
                return J + 5.0
        return J

    print('exp_ftc1: budget %.0f min, %d data bins, seed J %s, stall %d'
          % (minutes, len(TG),
             ('%.4f' % best) if best < 1e8 else 'none', stall), flush=True)

    def restart_simplex(zc, k):
        """Deterministic simplex around zc, widened with stall count k."""
        zc = np.asarray(zc, float)
        n = len(zc)
        sc = 0.02 * (1 + 2 * k)
        sim = [zc.copy()]
        for i in range(n):
            zi = zc.copy()
            sgn = 1.0 if (i + k) % 2 == 0 else -1.0
            zi[i] += sgn * sc * max(1.0, abs(zi[i]))
            sim.append(zi)
        return np.array(sim)

    while time.time() < deadline - 5.0:
        run_start = state['best']
        try:
            minimize(obj, z0, method='Nelder-Mead',
                     options=dict(maxfev=100000, xatol=1e-3, fatol=1e-4,
                                  initial_simplex=restart_simplex(z0, stall)))
        except KeyboardInterrupt:
            break
        # NM converged before the deadline: restart with a wider simplex
        if state['best'] < run_start - 1e-4:
            stall = 0
        else:
            stall += 1
        if os.path.exists(CKPT):
            ck = json.load(open(CKPT))
            z0 = ck['z']
            ck['stall'] = stall
            json.dump(ck, open(CKPT, 'w'))
        if time.time() < deadline - 5.0:
            print('restart: stall %d, best J %.4f' % (stall, state['best']),
                  flush=True)
    # persist stall for the next session
    if os.path.exists(CKPT):
        ck = json.load(open(CKPT))
        ck['stall'] = stall
        json.dump(ck, open(CKPT, 'w'))
    print('done: best J %.4f after %d evals, stall %d'
          % (state['best'], state['nev'], stall), flush=True)


if __name__ == '__main__':
    main()
