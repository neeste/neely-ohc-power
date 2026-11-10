"""run2 cat helper: state builder at the c708 state with one-factor variations.
Run under run_cat.py (installs lambda, 24-point gamma_c table, lambda energy accounting).
gamma(x) = gscale * (1 - nu(a)) * gc(rho; lam) * taper, rho = dscale*r(a)/a.
At (lam0, 1, 1) this is score_base.profiles exactly. For a changed lambda or damping scale the
gamma_c table is rebuilt (24 points, as run_cat.py) so that gamma/gamma_c is held fixed."""
import sys, json, math
import numpy as np
from dataclasses import replace
sys.path.insert(0, '.')
import geom, deltaT_3dof as d3, tl_model as T, exp_accumulation as E
import score_base as SB
LAM0 = float(geom.LAM); SEC = SB.COCH.sec; COCH = SB.COCH; BASE = SB.BASE
_tab = {}
def gc_table(lam, dscale):
    key = (round(lam, 6), round(dscale, 6))
    if key not in _tab:
        if abs(lam - LAM0) < 1e-9 and abs(dscale - 1) < 1e-12:
            _tab[key] = (SB._LX.copy(), SB._GY.copy())
        else:
            old = geom.LAM; geom.LAM = lam
            x = np.linspace(0, 1, 1025); a = COCH.cf(x); rho = dscale*SB._damping(a)/a
            rg = np.geomspace(max(rho.min()*0.98, 1e-3), rho.max()*1.02, 24)
            gg = np.array([geom.gamma_critical(replace(SEC, gamma=0.0, R_b=SEC.R_b*q, R_r=SEC.R_r*q, R_t=SEC.R_t*q, R_p=SEC.R_p*q, R_c=SEC.R_c*q), 1.0) for q in rg])
            geom.LAM = old; _tab[key] = (np.log(rg), gg)
    return _tab[key]
class State:
    def __init__(self, N=2048, lam=None, gscale=1.0, dscale=1.0):
        self.sec, self.coch = SEC, COCH; self.lam = LAM0 if lam is None else float(lam); self.N = N; self.gscale = gscale; self.dscale = dscale
        x = np.linspace(0, 1, N+1); a = COCH.cf(x); r = dscale*SB._damping(a)
        LX, GY = gc_table(self.lam, dscale)
        gc = np.interp(np.log(r/a), LX, GY); nu = SB._nu(a); tp = COCH.taper(x)
        self.x, self.a, self.r, self.gc, self.taper = x, a, r, gc, tp
        self.ratio = gscale*(1 - nu)            # gamma/gamma_c before the taper
        self.gam = self.ratio*gc*tp
    def _set(self): geom.LAM = self.lam
    def solve(self, f, gam=None):
        self._set(); g = self.gam if gam is None else gam
        Y, Yr, Ho = E.terms(SEC, f, self.a, self.r, g); P = T._fd(COCH, f, Y, self.N, 1.0)
        return Y, Yr, Ho, P
    def fields(self, f, gam=None):
        Y, Yr, Ho, P = self.solve(f, gam); Xb = Y*P; Xr = Yr*P; Xt = Xr - Ho*Xb/SEC.G
        return dict(Y=Y, P=P, Xb=Xb, Xr=Xr, Xt=Xt, sh=Ho*Xb, Ho=Ho)
    def power(self, f, F, gam=None):
        self._set(); g = self.gam if gam is None else gam
        D, p = geom.power(SEC, f, self.a, self.r, g, F['Xb'], F['Xr'], F['Xt'])
        Z = 1/(1j*f*F['Y']); pf = 0.5*Z.real*np.abs(1j*f*F['Xb'])**2
        return D, p, pf
    def shear_peak(self, f, gam=None):
        return int(np.argmax(np.abs(self.fields(f, gam)['sh'])))
    def nsf(self, f, gam=None):
        """N_SF = 2 tau f, tau = shear group delay at the shear peak (sfoae_cat.py)."""
        js = self.shear_peak(f, gam); fs = np.linspace(f*0.99, f*1.01, 9)
        ph = np.unwrap([np.angle(self.fields(ff, gam)['sh'][js]) for ff in fs])
        tau = -np.polyfit(fs, ph, 1)[0]/(2*np.pi)        # 1/B units
        return float(2*tau*f), js
    def qerb(self, f, nf=400):
        """Q_ERB of the low-level shear-velocity threshold curve at the place of the shear peak for f
        (tuning_ratio.py definition, ERB from 10^(-(thr-thr_min)/10))."""
        j = self.shear_peak(f); fr = self.a[j]*np.geomspace(0.1, 1.6, nf)
        resp = np.array([abs(self.fields(ff)['sh'][j])*ff for ff in fr])
        thr = -20*np.log10(resp); i = int(np.argmin(thr)); g2 = 10**(-(thr - thr[i])/10)
        return float(fr[i]/np.trapezoid(g2, fr)), float(fr[i]*BASE), j
    def bm_gain(self, f, nf=120):
        """Active re passive BM displacement at the active BM peak place for f, max over 0.3-1.3 of a(j) (bmgain_refit.bm_gain style)."""
        Xb = self.fields(f)['Xb']; j0 = int(0.02*self.N); j = int(np.argmax(np.abs(Xb[j0:]))) + j0
        fr = self.a[j]*np.geomspace(0.3, 1.3, nf); d = []
        for ff in fr:
            d.append(20*np.log10(abs(self.fields(ff)['Xb'][j])/abs(self.fields(ff, 0*self.gam)['Xb'][j])))
        i = int(np.argmax(d)); return float(d[i]), float(fr[i]/self.a[j])
    def share(self, f, gam=None, win=2.0):
        """OHC share p_ohc/D near the peak (share_s2.py): max share and its place (oct re BM and re shear peak),
        max p_ohc place, band with share > 1, identity error."""
        F = self.fields(f, gam); D, p, pf = self.power(f, F, gam)
        err = float(np.max(np.abs(D - pf - p))/np.max(D))
        jb = int(np.argmax(np.abs(F['Xb']))); jsh = int(np.argmax(np.abs(F['sh'])))
        ob = np.log2(self.a/self.a[jb]); osh = np.log2(self.a/self.a[jsh]); s = p/np.maximum(D, 1e-300)
        w = np.abs(ob) < win; idx = np.where(w)[0]; js = idx[np.argmax(s[w])]; jp = idx[np.argmax(p[w])]
        band = ob[w][s[w] > 1.0]
        g = self.gam if gam is None else gam
        return dict(share_max=float(s[js]), share_max_oct_bm=float(ob[js]), share_max_oct_shear=float(osh[js]),
                    p_ohc_max_oct_bm=float(ob[jp]), p_ohc_max_oct_shear=float(osh[jp]), share_at_bm_peak=float(s[jb]),
                    share_gt1=bool(band.size > 0), band_gt1_oct_bm=[float(band.min()), float(band.max())] if band.size else None,
                    x_bm_peak=float(self.x[jb]), x_shear_peak=float(self.x[jsh]), shear_re_bm_oct=float(np.log2(self.a[jsh]/self.a[jb])),
                    gamma_ratio_at_bm_peak=float(g[jb]/(self.gc[jb]*self.taper[jb])), identity_err=err)
