"""Human Q_ERB of the quiet-line hair-bundle shear at the places of greatest response to 0.5, 1, 2, 4 kHz
(the tuning target of the human refit, joint_hs_cnt.py qerb(), same arithmetic), at the state in
abr5_ckpt.json (or the state file given). Targets: 7.95, 11.0, 14.8, 21.4.
Usage (from model/): python3 ../scripts/run_human.py ../scripts/qerb_h.py [state.json] [out/qerb_h.json]"""
import sys, os, json
import numpy as np
sys.path.insert(0, '.')
import exp_abr2 as A2, exp_abr5 as A5, exp_accumulation as E, tl_model as T
A2.N = 8192
JS = sys.argv[1] if len(sys.argv) > 1 else 'abr5_ckpt.json'; OUT = sys.argv[2] if len(sys.argv) > 2 else 'out/qerb_h.json'
c = json.load(open(JS)); pd = dict(c['params']); sec_over = pd.pop('sec_over'); nfl = c['nu_floor']
H = A2.Human(**dict(pd, sec_over=sec_over)); gc = A5.gc_exact_dense(H)
nu = np.clip(pd['mu0']*H.a**pd['alpha_mu'], nfl, 1.0); mu = 1 - nu
t = np.clip((H.x - A5.X0)/(A5.X1 - A5.X0), 0, 1); w = 1 - (3*t*t - 2*t**3)
mu = (1 - w)*mu + w*np.minimum(mu, A5.CAP)
H.gam0 = mu*gc*H.coch.taper(H.x)
QF = (0.5, 1.0, 2.0, 4.0); q = []
for fk in QF:
    f0 = fk/20.7
    Y, Yr, Ho = E.terms(H.coch.sec, f0, H.a, H.r, H.gam0); P = T._fd(H.coch, f0, Y, A2.N, 1.0)
    j = int(np.argmax(np.abs(Ho*Y*P)))
    fr = f0*np.geomspace(0.3, 1.6, 140); g = []
    for f in fr:
        Y, Yr, Ho = E.terms(H.coch.sec, f, H.a, H.r, H.gam0); P = T._fd(H.coch, f, Y, A2.N, 1.0)
        g.append(abs((Ho*Y*P)[j])**2)
    g = np.array(g); i = int(np.argmax(g))
    q.append(float(fr[i]/np.trapezoid(g/g[i], fr)))
    print('%.1f kHz: x %.4f Q_ERB %.2f' % (fk, H.x[j], q[-1]), flush=True)
json.dump(dict(state=os.path.basename(JS), f_khz=list(QF), q_erb=q, stored=c.get('q')), open(OUT, 'w'), indent=1)
