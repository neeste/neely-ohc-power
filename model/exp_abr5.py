"""Stability-constrained wave-V refit campaign (JASA amplifier paper).

Refits the eight symmetry-breaking parameters (alpha_R, c_R, mu0,
alpha_mu, k, F, p_sat, S0) of the human model against the Neely et al.
(1988) wave-V surface through the onset read-out, with the coupled
line's stability constraints inside the loop:

  - gamma_c computed EXACTLY per candidate (160-point dense bisection
    over the candidate's true rho range; no interpolation clip);
  - activity realized as the intended mu(x) profile with mu capped at
    CAP = 0.945 for x < 0.08, smoothstep ramp to the intended profile
    by x = 0.14 (the certified geometry of the 2026-08-27 audit);
  - the incumbent is certified on every improvement by an
    argument-principle winding of the FD determinant over ten log
    bands at N = 1024 (clean, integer-zero windings required); a
    dirty or nonzero certificate rejects the candidate.

Checkpoint: abr5_ckpt.json (best params, readout, rms, cert status,
eval count). Resumes from the checkpoint if present; seeds from
abr4_ckpt.json otherwise. Run:  python3 exp_abr5.py [minutes]
Default budget 8 minutes per invocation.
"""
import sys, os, json, math, time
sys.path.insert(0, '.'); sys.path.insert(0, '..')
import numpy as np
from dataclasses import replace
from scipy.optimize import minimize
import deltaT_3dof as d3
import exp_abr2 as A2
import exp_abr4 as A4
import exp_accumulation as E
from exp_abr1 import F_BASE_HZ

CKPT = 'abr5_ckpt.json'
CAP, X0, X1 = 0.945, 0.08, 0.14
TH_GRID = (0.05, 0.068, 0.09)
TAU_GRID = (3.0, 5.0, 8.0)
PR_GRID = (1.5, 2.0)


def gc_exact_dense(H):
    sec = H.coch.sec
    rho = H.r / H.a
    rg = np.geomspace(rho.min() * 0.98, rho.max() * 1.02, 160)
    gg = []
    for r_ in rg:
        s2 = replace(sec, gamma=0.0, R_b=sec.R_b * r_, R_r=sec.R_r * r_,
                     R_t=sec.R_t * r_, R_p=sec.R_p * r_, R_c=sec.R_c * r_)
        gg.append(d3.gamma_critical(s2))
    return np.interp(np.log(rho), np.log(rg), np.asarray(gg))


def capped_gamma(H, mu0, alpha_mu, gc_ex):
    nu = np.clip(mu0 * H.a ** alpha_mu, 3e-3, 1.0)
    mu = 1.0 - nu
    t = np.clip((H.x - X0) / (X1 - X0), 0.0, 1.0)
    w = 1.0 - (3 * t * t - 2 * t * t * t)
    mu = (1 - w) * mu + w * np.minimum(mu, CAP)
    return mu * gc_ex * H.coch.taper(H.x)


def certify(H, gam, N=1024):
    """Argument-principle winding over ten log bands; True iff every
    band winds to a clean integer zero."""
    F = H.coch.F
    x = np.linspace(0.0, 1.0, N + 1)
    a = np.interp(x, H.x, H.a); rr = np.interp(x, H.x, H.r)
    g = np.interp(x, H.x, gam)
    h = 1.0 / N
    edges = np.logspace(-3.15, 0.35, 11)
    for lo, hi in zip(edges[:-1], edges[1:]):
        depth = 0.35 * hi
        corners = [complex(hi, -depth), complex(hi, 1e-6),
                   complex(lo, 1e-6), complex(lo, -depth),
                   complex(hi, -depth)]
        pts = []
        for c0, c1 in zip(corners[:-1], corners[1:]):
            tt = np.linspace(0, 1, 1000, endpoint=False)
            pts.extend(c0 + (c1 - c0) * tt)
        tot, ok, prev = 0.0, True, None
        for w in pts:
            Y, _, _ = E.terms(H.coch.sec, w, a, rr, g)
            q = -(w * w) * F * Y
            A = -(2.0 + h * h * q)
            d0, d1 = 1.0 + 0j, A[1]
            for j in range(2, N):
                d2 = A[j] * d1 - d0
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
        wind = tot / (2 * math.pi)
        if (not ok) or abs(wind) > 0.1:
            return False
    return True


def unpack(z):
    return dict(alpha_R=z[0], c_R=math.exp(z[1]), mu0=math.exp(-z[2]),
                alpha_mu=z[3], k=z[4], F=math.exp(z[5]),
                p_sat=math.exp(z[6])), z[7]


def pack(p, S0):
    return [p['alpha_R'], math.log(p['c_R']), -math.log(p['mu0']),
            p['alpha_mu'], p['k'], math.log(p['F']),
            math.log(p['p_sat']), S0]


def main():
    minutes = float(sys.argv[1]) if len(sys.argv) > 1 else 8.0
    deadline = time.time() + 60.0 * minutes
    stall = 0
    if os.path.exists(CKPT):
        ck = json.load(open(CKPT))
        seed_p = {k: ck['params'][k] for k in
                  ('alpha_R', 'c_R', 'mu0', 'alpha_mu', 'k', 'F', 'p_sat')}
        seed_p['sec_over'] = ck['params'].get('sec_over')
        S0 = ck['S0']; best_rms = ck['rms']; nev0 = ck.get('nev', 0)
        # stall bookkeeping: if the previous invocation ended within 1e-4
        # of where it started, widen the restart simplex
        stall = ck.get('stall', 0)
        prev = ck.get('run_start_rms')
        if prev is not None and prev - best_rms < 1e-4:
            stall += 1
        ck['run_start_rms'] = best_rms
        ck['stall'] = stall
        json.dump(ck, open(CKPT, 'w'))
    else:
        a4 = json.load(open('abr4_ckpt.json'))
        seed_p = dict(a4['params'])
        S0 = a4['S0']; best_rms = 1e9; nev0 = 0
    sec_over = seed_p.pop('sec_over', None)

    state = dict(best=best_rms, nev=nev0)

    def obj(z):
        if time.time() > deadline:
            raise KeyboardInterrupt
        pd, S0_ = unpack(z)
        if not (10.0 < S0_ < 90.0):
            return 10.0
        pd2 = dict(pd); pd2['sec_over'] = sec_over
        try:
            H = A2.Human(**pd2)
            gc_ex = gc_exact_dense(H)
            gam = capped_gamma(H, pd['mu0'], pd['alpha_mu'], gc_ex)
            H.gam0 = gam; H.stable = True
            rms, info = A4.joint_rms(H, S0_, TH_GRID, TAU_GRID, PR_GRID)
        except Exception:
            return 10.0
        state['nev'] += 1
        if rms < state['best']:
            if certify(H, gam):
                state['best'] = rms
                out = dict(params=dict(pd, sec_over=sec_over), S0=S0_,
                           rms=rms, readout=dict(pr=info['pr'],
                           theta=info['theta'], tau_a=info['tau_a'],
                           a=info['a']), certified=True,
                           nev=state['nev'],
                           lat=info['lat'], keys=info['keys'])
                json.dump(out, open(CKPT, 'w'))
                print('  new incumbent: rms %.3f ms (certified), eval %d'
                      % (rms, state['nev']), flush=True)
            else:
                print('  candidate rms %.3f REJECTED by certificate'
                      % rms, flush=True)
                return rms + 0.5
        return rms

    z0 = pack(seed_p, S0)
    # widened deterministic restart simplex: step scale grows with the
    # stall count and alternates sign by index so restarts explore new
    # directions without randomness
    scale = 0.02 * (1 + 2 * stall)
    simplex = [list(z0)]
    for i in range(len(z0)):
        zi = list(z0)
        step = scale * max(abs(zi[i]), 0.5)
        zi[i] += step if (i + stall) % 2 == 0 else -step
        simplex.append(zi)
    print('exp_abr5: budget %.0f min, seed rms %s, stall %d, simplex '
          'scale %.3f' % (minutes,
          ('%.3f' % best_rms) if best_rms < 1e8 else 'none',
          stall, scale), flush=True)
    try:
        minimize(obj, z0, method='Nelder-Mead',
                 options=dict(maxfev=100000, xatol=1e-4, fatol=1e-4,
                              initial_simplex=simplex))
    except KeyboardInterrupt:
        pass
    print('done: best rms %.3f after %d evals total'
          % (state['best'], state['nev']), flush=True)


if __name__ == '__main__':
    main()
