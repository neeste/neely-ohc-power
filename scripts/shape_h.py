"""Human tuning-curve shape at a joint_h state (same metrics and criterion as s2_c708/shape_cmp.py).
Usage: LAM=0.708 python3 shape_h.py state.json out.json"""
import sys, os, json, math
import numpy as np
sys.path.insert(0, '.')
if os.environ.get('LAM') is not None:
    import geom, deltaT_3dof as d3
    geom.LAM = float(os.environ['LAM']); geom.install()
    d3.gamma_critical = lambda sec, hi=200.0, tol=1e-6: geom.gamma_critical(sec, 1.0, hi, tol)
import exp_abr2 as A2, exp_abr5 as A5, exp_accumulation as E, tl_model as T
A2.N = int(os.environ.get('NH', 8192))
c = json.load(open(sys.argv[1])); pd = dict(c['params']); sec_over = pd.pop('sec_over'); nfl = c['nu_floor']
H = A2.Human(**dict(pd, sec_over=sec_over)); gc = A5.gc_exact_dense(H)
nu = np.clip(pd['mu0']*H.a**pd['alpha_mu'], nfl, 1.0); mu = 1-nu
t = np.clip((H.x-A5.X0)/(A5.X1-A5.X0), 0, 1); w = 1-(3*t*t-2*t**3); mu = (1-w)*mu + w*np.minimum(mu, A5.CAP)
g = mu*gc*H.coch.taper(H.x); N = len(H.x)-1
def metrics(fr, th):
    i = int(np.argmin(th)); th = th - th[i]; ft = fr[i]; lx = np.log2(fr/ft)
    def cross(side):
        rng = range(i, -1, -1) if side < 0 else range(i, len(fr)); prev = i
        for k in rng:
            if th[k] >= 10: tt = (10-th[prev])/(th[k]-th[prev]); return lx[prev]+tt*(lx[k]-lx[prev])
            prev = k
    lo, hi = cross(-1), cross(1)
    q10 = 1/(2**hi - 2**lo) if lo is not None and hi is not None else float('nan')
    at = lambda o: float(np.interp(o, lx, th))
    return dict(ftip=ft, q10=q10, tt=at(-1.0), low_slope=(at(-0.5)-at(-1.0))/0.5, hi_slope=(10/hi if hi else float('nan')),
                r07=at(math.log2(0.7)), r08=at(math.log2(0.8)), r09=at(math.log2(0.9)), curve=(lx.tolist(), th.tolist()))
out = {}
for fk in (0.5, 1, 2, 4, 8):
    f0 = fk/20.7
    Y, Yr, Ho = E.terms(H.coch.sec, f0, H.a, H.r, g); P = T._fd(H.coch, f0, Y, N, 1.0)
    j = int(np.argmax(np.abs(Ho*Y*P)))
    fr = f0*np.geomspace(0.12, 1.6, 150); th = []
    for f in fr:
        Y, Yr, Ho = E.terms(H.coch.sec, f, H.a, H.r, g); P = T._fd(H.coch, f, Y, N, 1.0)
        th.append(-20*math.log10(abs((Ho*Y*P)[j])*f))
    m = metrics(fr*20.7, np.array(th)); m['x'] = float(H.x[j]); out[fk] = m
    print('human %4.1f kHz tip %.2f Q10 %.2f tt %.1f lowslope %.1f hislope %.0f  re0.7 %.1f re0.8 %.1f re0.9 %.1f' % (fk, m['ftip'], m['q10'], m['tt'], m['low_slope'], m['hi_slope'], m['r07'], m['r08'], m['r09']), flush=True)
json.dump(out, open(sys.argv[2], 'w'))
