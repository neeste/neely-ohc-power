"""Human lambda-scan stability (C): quiet line N 2048 depths 0.35 and 1.0, and the 2-kHz 100-dB SPL saturated profile, N 2048 depth 0.35.
Usage (model/, run_human.py): c_scan_cert_h.py state.json"""
import sys, json, numpy as np
import r2hum as R, exp_accumulation as E
from cert2_h8k import certify
JS = sys.argv[1]; OUT = 'out/c_scan_cert_human.json'
import os; res = json.load(open(OUT)) if os.path.exists(OUT) else {}
for lam in np.round(np.arange(0.0, 1.0001, 0.05), 3):
    if '%.2f' % lam in res: continue
    S = R.State(JS, lam=float(lam)); S._set(); x = np.linspace(0, 1, 2049)
    a = np.interp(x, S.x, S.a); r = np.interp(x, S.x, S.r); ok = True
    for depth in (0.35, 1.0):
        o, _ = certify(S.sec, S.coch.F, a, r, np.interp(x, S.x, S.gam), 2048, lgw=(-3.15, 0.35), depth=depth); ok = ok and o
    sat = None
    if ok:
        f = 2.0/R.BASE; amp = 10.0**((100 - S.S0)/20.0); C = float(np.max(np.abs(S.fields(f)['sh'])))
        def FF(sv):
            u = (amp*np.abs(S.fields(f, S.gam*(1 - sv))['sh'])/C)**S.p_sat; return u/(1.0 + u)
        s, _, _ = E.anderson(FF, np.zeros(S.N + 1)); S._set()
        sat, _ = certify(S.sec, S.coch.F, a, r, np.interp(x, S.x, S.gam*(1 - s)), 2048, lgw=(-3.15, 0.35), depth=0.35)
    res['%.2f' % lam] = dict(quiet_clean=ok, quick_sat=None if sat is None else ('1/1' if sat else '0/1'))
    print('lam %.2f quiet clean %s; 2 kHz 100 dB: %s' % (lam, ok, res['%.2f' % lam]['quick_sat']), flush=True)
    json.dump(res, open(OUT, 'w'), indent=1)
