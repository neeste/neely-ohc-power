"""scope_geom.py: amplifier-side scoping. For each species and each OHC
force geometry (pair = model; bm = same force on BM only), at the same
nu(x) profile (gamma_c recomputed per geometry): whole-line certificate,
negative-resistance band and OHC-share peak relative to the response peak
(octaves, + basal), and for human the wave-V rms with read-out constants
re-optimized on the same grid. Usage: python3 scope_geom.py human|cat"""
import sys, json, math, time
import numpy as np
sys.path.insert(0, '.')
import geom
from cert2 import certify

def bands(x, pm_x, pm_f, f_khz, X_b, Y, w, D, p):
    Z = 1.0/(1j*w*Y)
    ipk = int(np.argmax(np.abs(X_b)))
    # local CF from the low-level place map (pm_x decreasing in f)
    o = np.argsort(pm_x)
    cf = np.exp(np.interp(x, np.asarray(pm_x)[o], np.log(np.asarray(pm_f)[o])))
    octs = np.log2(cf/cf[ipk])
    neg = np.where(Z.real < 0)[0]
    share = p/np.maximum(D, 1e-300)
    win = np.abs(octs) < 2.0
    js = np.where(win)[0][np.argmax(share[win])]
    out = dict(f_khz=f_khz, x_peak=float(x[ipk]),
               share_peak=float(share[ipk]), share_max=float(share[js]),
               share_max_oct=float(octs[js]))
    if neg.size:
        # contiguous runs near the peak
        runs = np.split(neg, np.where(np.diff(neg) > 1)[0] + 1)
        out['neg_bands_oct'] = [(float(octs[r[-1]]), float(octs[r[0]]))
                                for r in runs if np.min(np.abs(octs[r])) < 2]
    else:
        out['neg_bands_oct'] = []
    return out

def human():
    import exp_abr2 as A2, exp_abr4 as A4, exp_abr5 as A5
    ck = json.load(open('abr5_ckpt.json')); p = dict(ck['params']); S0 = ck['S0']
    prof = json.load(open('out/profiles.json'))['human']
    res = {}
    for g in ('pair', 'bm'):
        geom.GEOM = g; geom.install()
        H = A2.Human(**p)
        rho = H.r/H.a
        gc = geom.gc_of_rho(H.coch.sec, rho, n=160)
        gam = A5.capped_gamma(H, p['mu0'], p['alpha_mu'], gc)
        H.gam0 = gam; H.stable = True
        N = len(H.x) - 1
        ok, bb = certify(H.coch.sec, H.coch.F, H.a, H.r, gam, N,
                         lgw=(-3.15, 0.35))
        r = dict(gc_mid=float(np.median(gc)), certified=ok,
                 windings=[b[2] for b in bb], bands=[])
        print(g, 'certified', ok, [b[2] for b in bb], flush=True)
        for fk in (0.5, 1.0, 2.0, 4.0):
            w = fk/20.7
            R = H.line2(w, np.zeros(N+1))
            X_b, X_r = R['X_b'], R['X_r']; X_t = X_r - R['H_o']*X_b/H.coch.sec.G
            D, pw = geom.power(H.coch.sec, w, H.a, H.r, gam, X_b, X_r, X_t)
            Z = 1/(1j*w*R['Y']); pf = 0.5*Z.real*np.abs(1j*w*X_b)**2
            err = float(np.max(np.abs(D - pf - pw))/np.max(D))
            bd = bands(H.x, prof['pm_x'], prof['pm_f_khz'], fk, X_b, R['Y'],
                       w, D, pw); bd['identity_err'] = err
            r['bands'].append(bd); print(g, bd, flush=True)
        t0 = time.time()
        rms, info = A4.joint_rms(H, S0, A5.TH_GRID, A5.TAU_GRID, A5.PR_GRID)
        r['rms'] = rms; r['readout'] = {k: info[k] for k in
                                        ('theta', 'tau_a', 'pr', 'a')}
        print(g, 'rms', rms, r['readout'], f'{time.time()-t0:.0f}s', flush=True)
        res[g] = r
    json.dump(res, open('out/scope_human.json', 'w'), indent=1)

def cat():
    import score_base as SB
    sb = SB._sb_bmk
    prof = json.load(open('out/profiles.json'))['cat']
    res = {}
    for g in ('pair', 'bm'):
        geom.GEOM = g; geom.install(sb=sb)
        for N in (1024, 2048):
            x = np.linspace(0, 1, N+1); a = SB.COCH.cf(x)
            r_ = SB._damping(a); rho = r_/a
            gc = geom.gc_of_rho(SB.COCH.sec, rho, n=80, sb=1.0)
            SB._prof_cache[N] = (r_, (1 - SB._nu(a))*gc)
        N = 2048
        x = np.linspace(0, 1, N+1); a = SB.COCH.cf(x)
        r_, g0 = SB._prof_cache[N]
        gam = g0*SB.COCH.taper(x)
        ok, bb = certify(SB.COCH.sec, SB.COCH.F, a, r_, gam, N)
        out = dict(certified=ok, windings=[b[2] for b in bb], bands=[])
        print(g, 'certified', ok, [b[2] for b in bb], flush=True)
        for fk in (1.0, 4.0, 10.0):
            w = fk/SB.BASE
            R = SB.line_grad(SB.COCH, w, None, np.zeros(N+1), 0.0, 0.0, N)
            X_b, X_r = R['X_b'], R['X_r']; X_t = X_r - R['H_o']*X_b/SB.COCH.sec.G
            D, pw = geom.power(SB.COCH.sec, w, a, r_, gam, X_b, X_r, X_t)
            Z = 1/(1j*w*R['Y']); pf = 0.5*Z.real*np.abs(1j*w*X_b)**2
            err = float(np.max(np.abs(D - pf - pw))/np.max(D))
            bd = bands(x, prof['pm_x'], prof['pm_f_khz'], fk, X_b, R['Y'],
                       w, D, pw); bd['identity_err'] = err
            out['bands'].append(bd); print(g, bd, flush=True)
        res[g] = out
    json.dump(res, open('out/scope_cat.json', 'w'), indent=1)

if __name__ == '__main__':
    {'human': human, 'cat': cat}[sys.argv[1]]()
