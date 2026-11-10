"""Step 6(a) at the adopted human state: sensitivity of Q_ERB (0.5-4 kHz) and latency rms to the
activity profile (nu floor and the nu slope alpha_mu), one at a time, read-out refit each time."""
import sys, json, math, numpy as np
sys.path.insert(0, '.')
import exp_abr2 as A2, exp_abr4 as A4, exp_abr5 as A5, exp_accumulation as E, tl_model as T
TH = tuple(np.round(np.geomspace(0.02, 5.0, 15), 4)); TAU = (1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 18.0); PR = (1.0, 1.5, 2.0, 3.0)
ck = json.load(open('abr5_ckpt.json')); p0 = dict(ck['params']); S0 = ck['S0']; nfl0 = ck['nu_floor']
QT = np.array([7.95, 11.0, 14.8, 21.4])
def build(pd, nfl):
    H = A2.Human(**pd); gc = A5.gc_exact_dense(H)
    nu = np.clip(pd['mu0']*H.a**pd['alpha_mu'], nfl, 1.0); mu = 1-nu
    t = np.clip((H.x-A5.X0)/(A5.X1-A5.X0), 0, 1); w = 1-(3*t*t-2*t**3); mu = (1-w)*mu + w*np.minimum(mu, A5.CAP)
    H.gam0 = mu*gc*H.coch.taper(H.x); H.stable = True; return H
def qerb(H):
    out = []
    for fk in (0.5, 1.0, 2.0, 4.0):
        f0 = fk/20.7; Y, Yr, Ho = E.terms(H.coch.sec, f0, H.a, H.r, H.gam0); P = T._fd(H.coch, f0, Y, A2.N, 1.0)
        j = int(np.argmax(np.abs(Ho*Y*P))); fr = f0*np.geomspace(0.3, 1.6, 140); g = []
        for f in fr:
            Y, Yr, Ho = E.terms(H.coch.sec, f, H.a, H.r, H.gam0); P = T._fd(H.coch, f, Y, A2.N, 1.0); g.append(abs((Ho*Y*P)[j])**2)
        g = np.array(g); i = int(np.argmax(g)); out.append(fr[i]/np.trapezoid(g/g[i], fr))
    return np.array(out)
rows = []
def run(tag, pd, nfl):
    H = build(pd, nfl); rms, info = A4.joint_rms(H, S0, TH, TAU, PR, reject_onset=True, refine=True); q = qerb(H)
    rows.append(dict(tag=tag, rms=rms, q=q.tolist(), misfit=float(np.mean(np.abs(np.log(q/QT)))), nu_floor=nfl, alpha_mu=pd['alpha_mu'], mu0=pd['mu0']))
    print('%-22s rms %.3f Q %s mean|lnQ/QT| %.3f' % (tag, rms, np.round(q, 1).tolist(), rows[-1]['misfit']), flush=True)
run('state', p0, nfl0)
for s in (0.7, 0.85, 1.2, 1.5): run('nu floor x%.2f' % s, p0, nfl0*s)
for d in (-0.1, -0.05, 0.05, 0.1): run('alpha_mu %+.2f' % d, dict(p0, alpha_mu=p0['alpha_mu']+d), nfl0)
json.dump(rows, open('out/nu_scan.json', 'w'), indent=1)
