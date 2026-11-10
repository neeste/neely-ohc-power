"""Run 6 item 2 check: wave power flux Pi(x)/Pi(shear peak) at selected places, cat 1 kHz quiet (run under run_cat.py)."""
import sys, numpy as np, json
sys.path.insert(0, '.')
import exp_accumulation as E, tl_model as T, geom, score_base as SB
N = 4096; x = np.linspace(0, 1, N+1); a = SB.COCH.cf(x); r, g0 = SB.profiles(N); g = g0*SB.COCH.taper(x); SEC = SB.COCH.sec
out = {}
for fk in (1.0, 4.0):
    w = fk/SB.BASE; Y, Yr, Ho = E.terms(SEC, w, a, r, g); P = T._fd(SB.COCH, w, Y, N, 1.0); Xb = Y*P
    pf = 0.5*np.real(1/(1j*w*Y))*np.abs(1j*w*Xb)**2
    seg = 0.5*(pf[1:] + pf[:-1])/N; Pi = np.concatenate([np.cumsum(seg[::-1])[::-1], [0.0]])
    j = int(np.argmax(np.abs(Ho*Xb))); o = np.log2(a[j]/a)
    pts = [-1.0, -0.5, -0.2, 0.0, 0.03, 0.1, 0.2, 0.3, 0.36, 0.4, 0.5, 0.7, 1.0]
    row = {('%+.2f' % p): float(Pi[int(np.argmin(np.abs(o - p)))]/Pi[j]) for p in pts}; row['Pi0_over_Pi_peak'] = float(Pi[0]/Pi[j])
    out['%g kHz' % fk] = row; print(fk, {k: round(v, 3) for k, v in row.items()})
json.dump(out, open(sys.argv[1], 'w'), indent=1)
