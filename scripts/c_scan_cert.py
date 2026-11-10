"""Quiet-line stability of the cat lambda scan states (C), N 1024, depths 0.35 and 1.0; plus the 40-dB probes for
the states that pass, to locate the stability boundary relative to the apical/basal switch."""
import sys, json, numpy as np
import r2cat as R, r2cert as RC
OUT = 'out/c_scan_cert_cat.json'; res = {}
for lam in sorted(set(np.round(np.arange(0.0, 1.0001, 0.05), 3).tolist() + [0.8561939116132914])):
    S = R.State(N=1024, lam=float(lam)); S._set(); ok = True
    for depth in (0.35, 1.0):
        o, b = RC.certify(R.SEC, R.COCH.F, S.a, S.r, S.gam, 1024, depth=depth); ok = ok and o
    sat = None
    if ok:
        nc, nt, _ = RC.certify_state(lambda N: R.State(N=N, lam=float(lam)), R.SEC, R.COCH.F, Ns=(1024,), quick=True, log=lambda s: None)
        sat = '%d/%d' % (nc, nt)
    res['%.4f' % lam] = dict(quiet_clean=ok, quick_sat=sat); print('lam %.2f quiet clean %s; quiet+40dB probes N1024: %s' % (lam, ok, sat), flush=True)
    json.dump(res, open(OUT, 'w'), indent=1)
