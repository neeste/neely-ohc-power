"""Joint human refit at N = 8192: wave-V latency and tuning sharpness.
Tuning target: Q_ERB of low-level (quiet-line) hair-bundle-shear
isodisplacement tuning curves, as defined by Neely and Rasetshwane (2017,
Table II, active model), interpolated log-log to 0.5, 1, 2, 4 kHz:
7.95, 11.0, 14.8, 21.4. Objective J = rms_latency [ms] + W * mean|ln(Q/Qt)|.
Free: the 8 abr5 parameters + the nu floor; widened read-out grid.
Usage: [LAM=0.708] [INIT=start.json] joint_h.py W maxfev tag"""
import sys, os, json, math, time
import numpy as np
from scipy.optimize import minimize
sys.path.insert(0, '.')
import exp_abr2 as A2, exp_abr4 as A4, exp_abr5 as A5, exp_accumulation as E, tl_model as T
from cert2 import certify
LAM = os.environ.get('LAM')      # force split: fraction of the OHC force on the RL (unset = 1, the BM-RL pair)
if LAM is not None:
    import geom, deltaT_3dof as d3
    geom.LAM = float(LAM); geom.install()
    d3.gamma_critical = lambda sec, hi=200.0, tol=1e-6: geom.gamma_critical(sec, 1.0, hi, tol)
A2.N = 8192
A5.TH_GRID = tuple(np.round(np.geomspace(0.02, 5.0, 15), 4)); A5.TAU_GRID = (1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 18.0); A5.PR_GRID = (1.0, 1.5, 2.0, 3.0)
QF = np.array([0.5, 1.0, 2.0, 4.0]); QT = np.array([7.95, 11.0, 14.8, 21.4])
W = float(sys.argv[1]); TAG = sys.argv[3]; OUT = 'out/joint_h_%s.json' % TAG
ck = json.load(open('earlier/human_n8192_ckpt.json')); p0 = dict(ck['params']); sec_over = p0.pop('sec_over'); S0 = ck['S0']
def build(pd, nfl):
    H = A2.Human(**dict(pd, sec_over=sec_over)); gc = A5.gc_exact_dense(H)
    nu = np.clip(pd['mu0']*H.a**pd['alpha_mu'], nfl, 1.0); mu = 1-nu
    t = np.clip((H.x-A5.X0)/(A5.X1-A5.X0), 0, 1); w = 1-(3*t*t-2*t**3)
    mu = (1-w)*mu + w*np.minimum(mu, A5.CAP)
    H.gam0 = mu*gc*H.coch.taper(H.x); H.stable = True
    return H
def qerb(H):
    out = []
    for fk in QF:
        f0 = fk/20.7
        Y, Yr, Ho = E.terms(H.coch.sec, f0, H.a, H.r, H.gam0); P = T._fd(H.coch, f0, Y, A2.N, 1.0)
        j = int(np.argmax(np.abs(Ho*Y*P)))
        fr = f0*np.geomspace(0.3, 1.6, 140); g = []
        for f in fr:
            Y, Yr, Ho = E.terms(H.coch.sec, f, H.a, H.r, H.gam0); P = T._fd(H.coch, f, Y, A2.N, 1.0)
            g.append(abs((Ho*Y*P)[j])**2)
        g = np.array(g); i = int(np.argmax(g))
        out.append(fr[i]/np.trapezoid(g/g[i], fr))
    return np.array(out)
best = dict(J=1e9)
def unpack(z):
    pd, S0_ = A5.unpack(z[:8]); return pd, S0_, math.exp(z[8])
def obj(z):
    pd, S0_, nfl = unpack(z)
    if not (10 < S0_ < 90) or not (1e-4 < nfl < 0.5): return 50.0
    try:
        H = build(pd, nfl)
        rms, info = A4.joint_rms(H, S0_, A5.TH_GRID, A5.TAU_GRID, A5.PR_GRID, reject_onset=True, refine=True)
        if info is None: return 50.0
        q = qerb(H)
    except Exception:
        return 50.0
    J = rms + W*float(np.mean(np.abs(np.log(q/QT))))
    if J < best['J'] - 1e-4:
        x = np.linspace(0, 1, 2049)
        ok, _ = certify(H.coch.sec, H.coch.F, np.interp(x, H.x, H.a), np.interp(x, H.x, H.r), np.interp(x, H.x, H.gam0), 2048, lgw=(-3.15, 0.35))
        if not ok: return J + 1.0
        best.update(J=J, rms=rms, q=q.tolist(), params=dict(pd, sec_over=sec_over), S0=S0_, nu_floor=nfl,
                    readout={k: info[k] for k in ('theta', 'tau_a', 'pr', 'a')}, lat=[float(v) for v in info['lat']], keys=info['keys'])
        json.dump(best, open(OUT, 'w'), indent=1)
        print(time.strftime('%H:%M:%S'), 'J %.3f rms %.3f Q %s nu_floor %.4f %s' % (J, rms, np.round(q, 1).tolist(), nfl, best['readout']), flush=True)
    return J
z0 = list(A5.pack({k: p0[k] for k in ('alpha_R', 'c_R', 'mu0', 'alpha_mu', 'k', 'F', 'p_sat')}, S0)) + [math.log(0.003)]
INIT = os.environ.get('INIT')    # optional starting state (a joint_h json) when OUT does not exist
if INIT and not os.path.exists(OUT):
    c = json.load(open(INIT)); pp = dict(c['params']); pp.pop('sec_over', None)
    z0 = list(A5.pack({k: pp[k] for k in ('alpha_R', 'c_R', 'mu0', 'alpha_mu', 'k', 'F', 'p_sat')}, c['S0'])) + [math.log(c['nu_floor'])]
    print('starting from', INIT, c['J'], flush=True)
if os.path.exists(OUT):
    c = json.load(open(OUT)); pp = dict(c['params']); pp.pop('sec_over', None)
    z0 = list(A5.pack({k: pp[k] for k in ('alpha_R', 'c_R', 'mu0', 'alpha_mu', 'k', 'F', 'p_sat')}, c['S0'])) + [math.log(c['nu_floor'])]
    print('resuming', c['J'], flush=True)
t0 = time.time(); obj(z0); print('first eval %.0f s' % (time.time()-t0), flush=True)
# start the floor search wide: initial simplex step of 1.5 in ln(nu_floor)
simplex = [z0] + [list(np.add(z0, d)) for d in np.diag([0.05, 0.1, 0.3, 0.05, 0.001, 0.1, 0.1, 2.0, 1.5])]
minimize(obj, z0, method='Nelder-Mead', options=dict(maxfev=int(sys.argv[2]), xatol=1e-4, fatol=1e-4, initial_simplex=simplex))
print('done', best.get('J'), flush=True)
