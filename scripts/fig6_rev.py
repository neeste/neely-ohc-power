"""Fig. 6 data at the human abr5 state, 1-kHz tone. m1: gain and BM group
delay versus the nu floor (quiet line), each state certified. m2: the same
versus level at abr5 (self-consistent saturation, frozen line for delay).
Gains are re the passive line (gamma = 0) at the place of greatest
low-level response of the quantity named."""
import sys, json, math, numpy as np
sys.path.insert(0, '.')
import exp_abr2 as A2, exp_abr5 as A5, exp_accumulation as E, tl_model as T
from cert2 import certify
ck = json.load(open('abr5_ckpt.json')); p = dict(ck['params']); S0 = ck['S0']; NFL = ck.get('nu_floor', 0.003)   # the state's own nu floor
H = A2.Human(**p); gc = A5.gc_exact_dense(H); N = len(H.x) - 1
f = 1000.0/A2.F_BASE_HZ
def gam_floor(nfl):
    nu = np.clip(p['mu0']*H.a**p['alpha_mu'], nfl, 1.0); mu = 1-nu
    t = np.clip((H.x-A5.X0)/(A5.X1-A5.X0), 0, 1); w = 1-(3*t*t-2*t**3)
    mu = (1-w)*mu + w*np.minimum(mu, A5.CAP)
    return mu*gc*H.coch.taper(H.x)
def lin(g, ff):
    Y, Yr, Ho = E.terms(H.coch.sec, ff, H.a, H.r, g)
    P = T._fd(H.coch, ff, Y, N, 1.0); Xb = Y*P
    return Xb, Ho*Xb
xbp, shp = lin(0*H.a, f)
def metrics(g):
    xb, sh = lin(g, f)
    j = int(np.argmax(np.abs(sh)))
    gs = 20*np.log10(abs(sh[j])/abs(shp[j])); gb = 20*np.log10(abs(xb[j])/abs(xbp[j]))
    fs = np.linspace(f*0.99, f*1.01, 9); ph = []
    for ff in fs:
        b, _ = lin(g, ff); ph.append(np.angle(b[j]))
    tau = -np.polyfit(fs, np.unwrap(ph), 1)[0]/(2*np.pi)   # units 1/B
    return dict(x=float(H.x[j]), gain_shear=float(gs), gain_bm=float(gb), delay_cycles=float(tau*f))
m1 = []
for nfl in sorted({0.3, 0.2, 0.14, 0.1, 0.07, 0.05, 0.03, 0.02, 0.01, 0.005, 0.003, NFL}, reverse=True):
    g = gam_floor(nfl); ok, _ = certify(H.coch.sec, H.coch.F, H.a, H.r, g, N, lgw=(-3.15, 0.35))
    m = metrics(g); m.update(nu=nfl, certified=ok); m1.append(m); print('m1', m, flush=True)
g0 = gam_floor(NFL); H.gam0 = g0
m2 = []
r0 = H.line2(f, np.zeros(N+1)); C = float(np.max(np.abs(r0['shear']))); s = np.zeros(N+1)
for spl in range(20, 101, 10):
    A = 10**((spl-S0)/20)
    def FF(sv):
        rr = H.line2(f, sv); u = (A*np.abs(rr['shear'])/C)**H.p_sat; return u/(1+u)
    s, _, _ = E.anderson(FF, s)
    m = metrics(g0*(1-s)); m.update(spl=spl, smax=float(s.max())); m2.append(m); print('m2', m, flush=True)
json.dump(dict(m1=m1, m2=m2), open('out/fig6_rev.json', 'w'), indent=1)
