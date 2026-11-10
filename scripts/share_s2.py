"""OHC power share p_ohc/D at the cat stage-2 state (place-dependent damping and activity).
Usage: share_s2.py ckpt.json out.json  (run in stage2/ for lambda = 0.708 section, verify_bg/ for lambda = 1)"""
import sys, json
import numpy as np
sys.path.insert(0, '.')
import geom, deltaT_3dof as d3, tl_model as T, exp_accumulation as E
bd = json.load(open('best_d3.json')); LAM = bd.get('lam', 1.0)
geom.LAM = LAM; geom.install()
T.section_terms = lambda sec, w, a, r=None, gamma=None: geom.terms(sec, w, a, r, gamma)
d3.gamma_critical = lambda sec, hi=200.0, tol=1e-6: geom.gamma_critical(sec, 1.0, hi, tol)
import exp_ftc2 as X, exp_ftc1 as F1
c = json.load(open(sys.argv[1])); z = c['z']
r, gam = X.profiles(z)
A, SEC, COCH, N = F1.A, F1.SEC, F1.COCH, F1.N
ratio = gam/np.maximum(F1.gc_dense(r/A)*F1.TAPER, 1e-30)
rows = []
for x0 in (0.3, 0.5, 0.7):
    j0 = int(np.argmin(np.abs(F1.X - x0))); f = float(A[j0])
    Y, Y_r, H_o = E.terms(SEC, f, A, r, gam); P = T._fd(COCH, f, Y, N, 1.0)
    X_b = Y*P; X_r = Y_r*P; X_t = X_r - H_o*X_b/SEC.G
    D, p = geom.power(SEC, f, A, r, gam, X_b, X_r, X_t)
    Z = 1/(1j*f*Y); pf = 0.5*Z.real*np.abs(1j*f*X_b)**2
    err = float(np.max(np.abs(D - pf - p))/np.max(D))
    ipk = int(np.argmax(np.abs(X_b))); octs = np.log2(A/A[ipk]); share = p/np.maximum(D, 1e-300)
    win = np.abs(octs) < 2; js = np.where(win)[0][np.argmax(share[win])]
    band = octs[win][share[win] > 1.0]
    rows.append(dict(x0=x0, share_max=float(share[js]), share_max_oct=float(octs[js]), share_at_peak=float(share[ipk]),
                     band_gt1_oct=[float(band.min()), float(band.max())] if band.size else None,
                     gamma_ratio_at_peak=float(ratio[ipk]), identity_err=err))
out = dict(source=sys.argv[1], lam=LAM, J=c['J'], worst_margin=c.get('worst_margin'), rows=rows)
json.dump(out, open(sys.argv[2], 'w'), indent=1)
print('lam %.3f' % LAM)
for rw in rows: print('  ', {k: (round(v, 3) if isinstance(v, float) else v) for k, v in rw.items()})
