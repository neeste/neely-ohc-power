"""exp_sfoae8.py -- basal delay bound campaign (human).

Question: can ANY certified profile in the family lengthen the model's
round-trip emission delay at 8 kHz toward the measured SFOAE trend,
without giving up the wave-V fit?  The 8 kHz shortfall (model 2N about
11 cycles at 20 dB SPL against a measured trend near 20) is currently
enforced by the collective stability ceiling, but that ceiling was
established for ONE back-off geometry (cap 0.945, smoothstep from
x = 0.08 to 0.14).  Here the geometry is freed.

Framed as constrained maximization:
  maximize  N8 = 2 f gd(8 kHz, 20 dB SPL)   [cycles]
  subject to  wave-V rms <= RMS_GATE = 0.60 ms  (converged fit 0.571)
              argument-principle certification on every incumbent.

Parameters: the eight abr5 refit parameters plus the back-off
geometry: z[8] = CAP, z[9] = X0, z[10] = DX (ramp width).  At the
seed the family reduces exactly to the converged certified state.
Outcome either lengthens the delay or establishes a certified bound,
which upgrades the paper's 8 kHz caveat into a quantified prediction.

Checkpoint sfoae8_ckpt.json. Run: python3 exp_sfoae8.py [minutes].
"""
import sys, os, json, math, time
sys.path.insert(0, '.'); sys.path.insert(0, '..')
import numpy as np
from scipy.optimize import minimize
import exp_abr2 as A2
import exp_abr4 as A4
import exp_abr5 as A5
import exp_accumulation as E
from exp_abr1 import F_BASE_HZ

CKPT = 'sfoae8_ckpt.json'
RMS_GATE = 0.60
TH_GRID, TAU_GRID, PR_GRID = A5.TH_GRID, A5.TAU_GRID, A5.PR_GRID
F8_KHZ = 8.0
SPL8 = 20.0


def capped_gamma_geo(H, mu0, alpha_mu, gc_ex, cap, x0, x1):
    nu = np.clip(mu0 * H.a ** alpha_mu, 3e-3, 1.0)
    mu = 1.0 - nu
    t = np.clip((H.x - x0) / max(x1 - x0, 1e-3), 0.0, 1.0)
    w = 1.0 - (3 * t * t - 2 * t * t * t)
    mu = (1 - w) * mu + w * np.minimum(mu, cap)
    return mu * gc_ex * H.coch.taper(H.x)


def gd8(H, S0, f_khz=F8_KHZ, spl=SPL8, eps=0.01):
    """Steady-state group delay (ms) at the peak place for f_khz at
    spl, at the self-consistent saturation state, and the round-trip
    delay in cycles 2*N = 2*gd*f."""
    f_dim0 = f_khz * 1000.0 / F_BASE_HZ
    amp = 10.0 ** ((spl - S0) / 20.0)
    r0 = H.line2(f_dim0, np.zeros(len(H.x)))
    C = float(np.max(np.abs(r0['shear'])))
    p_s = H.p_sat

    def FF(sv):
        r = H.line2(f_dim0, sv)
        u = (amp * np.abs(r['shear']) / C) ** p_s
        return u / (1.0 + u)

    sp, _, _ = E.anderson(FF, np.zeros(len(H.x)))
    rm = H.line2(f_dim0, sp)
    j = int(np.argmax(np.abs(rm['shear'])))
    ph = []
    for fd in (f_dim0 * (1 - eps), f_dim0 * (1 + eps)):
        ph.append(np.angle(H.line2(fd, sp)['shear'][j]))
    dph = ph[1] - ph[0]
    while dph > math.pi: dph -= 2 * math.pi
    while dph < -math.pi: dph += 2 * math.pi
    df_hz = 2 * eps * f_khz * 1000.0
    gd_ms = -dph / (2 * math.pi * df_hz) * 1000.0
    return gd_ms, 2.0 * gd_ms * f_khz


def unpack(z):
    pd, S0 = A5.unpack(z[:8])
    return pd, S0, z[8], z[9], z[10]


def main():
    minutes = float(sys.argv[1]) if len(sys.argv) > 1 else 8.0
    deadline = time.time() + 60.0 * minutes
    stall = 0
    if os.path.exists(CKPT):
        ck = json.load(open(CKPT))
        z0 = ck['z']; best = ck['J']; nev0 = ck.get('nev', 0)
        stall = ck.get('stall', 0)
        sec_over = ck['params'].get('sec_over')
    else:
        a5 = json.load(open('abr5_ckpt.json'))
        seed_p = {k: a5['params'][k] for k in
                  ('alpha_R', 'c_R', 'mu0', 'alpha_mu', 'k', 'F',
                   'p_sat')}
        sec_over = a5['params'].get('sec_over')
        z0 = A5.pack(seed_p, a5['S0']) + [A5.CAP, A5.X0, A5.X1 - A5.X0]
        best, nev0 = 1e9, 0
    state = dict(best=best, nev=nev0)

    def obj(z):
        if time.time() > deadline:
            raise KeyboardInterrupt
        pd, S0_, cap, x0, dx = unpack(z)
        if not (10.0 < S0_ < 90.0 and 0.90 < cap < 0.99
                and 0.02 < x0 < 0.20 and 0.02 < dx < 0.25):
            return 10.0
        pd2 = dict(pd); pd2['sec_over'] = sec_over
        try:
            H = A2.Human(**pd2)
            gc_ex = A5.gc_exact_dense(H)
            gam = capped_gamma_geo(H, pd['mu0'], pd['alpha_mu'],
                                   gc_ex, cap, x0, x0 + dx)
            H.gam0 = gam; H.stable = True
            gd_ms, n8 = gd8(H, S0_)
            rms, info = A4.joint_rms(H, S0_, TH_GRID, TAU_GRID,
                                     PR_GRID)
        except Exception:
            return 10.0
        over = max(0.0, rms - RMS_GATE)
        J = -n8 + 200.0 * (over / 0.02) ** 2
        state['nev'] += 1
        if J < state['best']:
            if A5.certify(H, gam):
                state['best'] = J
                json.dump(dict(z=list(map(float, z)),
                               params=dict(pd, sec_over=sec_over),
                               S0=S0_, cap=cap, x0=x0, dx=dx,
                               J=float(J), rms=float(rms),
                               gd8_ms=float(gd_ms), N8=float(n8),
                               readout=dict(pr=info['pr'],
                                            theta=info['theta'],
                                            tau_a=info['tau_a'],
                                            a=info['a']),
                               certified=True, nev=state['nev'],
                               stall=stall),
                          open(CKPT, 'w'))
                print('  incumbent N8 %.2f cycles (gd %.3f ms, rms '
                      '%.3f, cap %.3f, x0 %.3f) eval %d'
                      % (n8, gd_ms, rms, cap, x0, state['nev']),
                      flush=True)
            else:
                print('  candidate N8 %.2f REJECTED by certificate'
                      % n8, flush=True)
                return J + 2.0
        return J

    def restart(zc, k):
        zc = np.asarray(zc, float)
        sc = 0.02 * (1 + 2 * k)
        sim = [zc.copy()]
        for i in range(len(zc)):
            zi = zc.copy()
            zi[i] += (1.0 if (i + k) % 2 == 0 else -1.0) * sc * max(
                abs(zi[i]), 0.05)
            sim.append(zi)
        return np.array(sim)

    print('exp_sfoae8: budget %.0f min, seed J %s, stall %d'
          % (minutes, ('%.3f' % best) if best < 1e8 else 'none',
             stall), flush=True)
    while time.time() < deadline - 5.0:
        run_start = state['best']
        try:
            minimize(obj, z0, method='Nelder-Mead',
                     options=dict(maxfev=100000, xatol=1e-4,
                                  fatol=1e-3,
                                  initial_simplex=restart(z0, stall)))
        except KeyboardInterrupt:
            break
        stall = 0 if state['best'] < run_start - 1e-3 else stall + 1
        if os.path.exists(CKPT):
            ck = json.load(open(CKPT))
            z0 = ck['z']; ck['stall'] = stall
            json.dump(ck, open(CKPT, 'w'))
        if time.time() < deadline - 5.0:
            print('restart: stall %d, best J %.3f'
                  % (stall, state['best']), flush=True)
    if os.path.exists(CKPT):
        ck = json.load(open(CKPT)); ck['stall'] = stall
        json.dump(ck, open(CKPT, 'w'))
    print('done: best J %.3f after %d evals, stall %d'
          % (state['best'], state['nev'], stall), flush=True)


if __name__ == '__main__':
    main()
