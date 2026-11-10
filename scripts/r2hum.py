"""run2 human helper: state builder for a joint_hs json (shared cat section, lambda), one-factor variations.
Run under run_human.py in a human run dir (geom installed, N from NPTS).
gamma(x) = gscale * mu(x) * gc(rho; lam) * taper, mu as joint_hs.build (nu floor, basal cap), rho = dscale*r/a.
gc from exp_abr5.gc_exact_dense (160 points), rebuilt for the varied lambda / damping."""
import sys, json, math
import numpy as np
sys.path.insert(0, '.')
import geom, tl_model as T, exp_accumulation as E, exp_abr2 as A2, exp_abr5 as A5
BASE = 20.7       # kHz per dimensionless frequency unit (joint_hs: f0 = fk/20.7)
class State:
    def __init__(self, js, N=None, lam=None, gscale=1.0, dscale=1.0):
        c = json.load(open(js)) if isinstance(js, str) else js
        self.c = c; pd = dict(c['params']); sec_over = pd.pop('sec_over'); nfl = c['nu_floor']
        self.lam = float(c['lam']) if lam is None else float(lam); self.gscale = gscale; self.dscale = dscale
        if N is not None: A2.N = N
        self.N = A2.N
        old = geom.LAM; geom.LAM = self.lam
        H = A2.Human(**dict(pd, sec_over=sec_over)); H.r = H.r*dscale
        gc = A5.gc_exact_dense(H); geom.LAM = old
        nu = np.clip(pd['mu0']*H.a**pd['alpha_mu'], nfl, 1.0); mu = 1 - nu
        t = np.clip((H.x - A5.X0)/(A5.X1 - A5.X0), 0, 1); w = 1 - (3*t*t - 2*t**3)
        mu = (1 - w)*mu + w*np.minimum(mu, A5.CAP)
        self.H = H; self.sec = H.coch.sec; self.coch = H.coch; self.S0 = c['S0']; self.p_sat = H.p_sat
        self.x, self.a, self.r, self.gc, self.taper = H.x, H.a, H.r, gc, H.coch.taper(H.x)
        self.ratio = gscale*mu; self.gam = self.ratio*gc*self.taper
    def _set(self): geom.LAM = self.lam
    def fields(self, f, gam=None):
        self._set(); g = self.gam if gam is None else gam
        Y, Yr, Ho = E.terms(self.sec, f, self.a, self.r, g); P = T._fd(self.coch, f, Y, self.N, 1.0)
        Xb = Y*P; Xr = Yr*P; Xt = Xr - Ho*Xb/self.sec.G
        return dict(Y=Y, P=P, Xb=Xb, Xr=Xr, Xt=Xt, sh=Ho*Xb, Ho=Ho)
    def power(self, f, F, gam=None):
        self._set(); g = self.gam if gam is None else gam
        D, p = geom.power(self.sec, f, self.a, self.r, g, F['Xb'], F['Xr'], F['Xt'])
        Z = 1/(1j*f*F['Y']); pf = 0.5*Z.real*np.abs(1j*f*F['Xb'])**2
        return D, p, pf
    def shear_peak(self, f, gam=None): return int(np.argmax(np.abs(self.fields(f, gam)['sh'])))
    def nsf(self, f, gam=None):
        js = self.shear_peak(f, gam); fs = np.linspace(f*0.99, f*1.01, 9)
        ph = np.unwrap([np.angle(self.fields(ff, gam)['sh'][js]) for ff in fs])
        return float(2*(-np.polyfit(fs, ph, 1)[0]/(2*np.pi))*f), js
    def qerb(self, f, nf=140):
        """joint_hs.qerb definition (shear displacement power, 0.3-1.6 of f, place of the shear peak)."""
        j = self.shear_peak(f); fr = f*np.geomspace(0.3, 1.6, nf)
        g = np.array([abs(self.fields(ff)['sh'][j])**2 for ff in fr]); i = int(np.argmax(g))
        return float(fr[i]/np.trapezoid(g/g[i], fr)), float(fr[i]*BASE), j
    def share(self, f, gam=None, win=2.0):
        F = self.fields(f, gam); D, p, pf = self.power(f, F, gam)
        err = float(np.max(np.abs(D - pf - p))/np.max(D))
        j0 = int(0.02*self.N); jb = int(np.argmax(np.abs(F['Xb'][j0:]))) + j0; jsh = int(np.argmax(np.abs(F['sh'])))
        ob = np.log2(self.a/self.a[jb]); osh = np.log2(self.a/self.a[jsh]); s = p/np.maximum(D, 1e-300)
        w = (np.abs(ob) < win) & (self.a > 1.5e-4); idx = np.where(w)[0]; js = idx[np.argmax(s[w])]; jp = idx[np.argmax(p[w])]
        band = ob[w][s[w] > 1.0]; g = self.gam if gam is None else gam
        return dict(share_max=float(s[js]), share_max_oct_bm=float(ob[js]), share_max_oct_shear=float(osh[js]),
                    p_ohc_max_oct_bm=float(ob[jp]), p_ohc_max_oct_shear=float(osh[jp]), share_at_bm_peak=float(s[jb]),
                    share_gt1=bool(band.size > 0), band_gt1_oct_bm=[float(band.min()), float(band.max())] if band.size else None,
                    x_bm_peak=float(self.x[jb]), x_shear_peak=float(self.x[jsh]), shear_re_bm_oct=float(np.log2(self.a[jsh]/self.a[jb])),
                    gamma_ratio_at_bm_peak=float(g[jb]/(self.gc[jb]*self.taper[jb])), identity_err=err)
def certify_h(S, log=print, full=True):
    """cert_h_lam.py style: quiet + saturated (0.5 kHz 100, 2 kHz 40 and 100, 8 kHz 70 and 100 dB SPL),
    N 2048 (depth 0.35, 1.0) and 8192 (0.35). Stops at the first dirty certificate."""
    import time
    from cert2_h8k import certify
    N0 = S.N; states = {'quiet': np.zeros(N0 + 1)}
    for fk, spl in ((0.5, 100), (2.0, 40), (2.0, 100), (8.0, 70), (8.0, 100)):
        f = fk/BASE; amp = 10.0**((spl - S.S0)/20.0); C = float(np.max(np.abs(S.fields(f)['sh'])))
        def FF(sv):
            u = (amp*np.abs(S.fields(f, S.gam*(1 - sv))['sh'])/C)**S.p_sat; return u/(1.0 + u)
        s, _, _ = E.anderson(FF, np.zeros(N0 + 1)); states['%g kHz %d dB SPL' % (fk, spl)] = s
    res = {}; S._set()
    for name, s in states.items():
        gam = S.gam*(1 - s)
        for N in (2048, 8192):
            x = np.linspace(0, 1, N + 1); a = np.interp(x, S.x, S.a); r = np.interp(x, S.x, S.r); g = np.interp(x, S.x, gam)
            for depth in ((0.35, 1.0) if N <= 2048 else (0.35,)):
                t0 = time.time(); ok, bands = certify(S.sec, S.coch.F, a, r, g, N, lgw=(-3.15, 0.35), depth=depth)
                res['%s | N=%d depth=%g' % (name, N, depth)] = dict(clean=ok, smax=float(np.max(s)))
                log('   hcert %-20s N=%d depth=%g clean=%s (%.0fs)' % (name, N, depth, ok, time.time() - t0))
                if not ok: return sum(v['clean'] for v in res.values()), len(res), res
    return sum(v['clean'] for v in res.values()), len(res), res
