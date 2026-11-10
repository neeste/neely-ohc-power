import sys, json, numpy as np
sys.path.insert(0, '.')
import score_base as SB, exp_accumulation as E, tl_model as T
from exp_place6 import accounting
N = 2048
x = np.linspace(0, 1, N+1); a = SB.COCH.cf(x); r, g0 = SB.profiles(N); gam = g0*SB.COCH.taper(x)
res = []
for fk in np.geomspace(0.2, 40, 24):
    w = fk/SB.BASE
    Y, Yr, Ho = E.terms(SB.COCH.sec, w, a, r, gam)
    P = T._fd(SB.COCH, w, Y, N, 1.0); Xb = Y*P
    D, pf, po = accounting(SB.COCH.sec, a, r, gam, w, Xb, Yr*P, Ho, 1/(1j*w*Y))
    with np.errstate(invalid='ignore', divide='ignore'): sh = po/D
    ok = D > 1e-6*np.nanmax(D)
    j = int(np.nanargmax(np.where(ok, sh, -1)))
    ish = int(np.argmax(np.abs(Ho*Xb)))
    res.append((fk, float(sh[j]), float(np.log2(a[ish]/a[j])), bool(np.any((pf < 0) & ok))))
    print('%6.2f kHz  max share %.4f at %+.2f oct re shear peak; Re Z<0 anywhere: %s' % res[-1], flush=True)
json.dump(res, open('out/share_scan_cat.json', 'w'))
