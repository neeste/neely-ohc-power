"""Run 6 item 4, grid check of s_line(lambda), cat: the uniform scale at which the quiet line first fails the winding test,
found separately for each grid N = 1024, 2048, 4096 (contour depth 0.35), bisection to 0.005 in [S_LO, s_cap], s_cap =
0.99/max gamma/gamma_c. A failure whose s rises with N (or disappears) is a discretization artifact of the FD line, not a
line instability. Run from model/ under run_cat.py. Usage: lam_conv.py out.json [lambda ...]"""
import sys, os, json, math, time
import numpy as np
sys.path.insert(0, '.'); import geom, r2cert as RC, r2cat as R
OUT = sys.argv[1]
LAMS = [float(v) for v in sys.argv[2:]] or [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.55]
res = json.load(open(OUT)) if os.path.exists(OUT) else dict(rows={}, depth=0.35, tol=0.005)
def one(lam, s, N):
    S = R.State(N=N, lam=lam, gscale=s); geom.LAM = lam; t0 = time.time()
    ok, bands = RC.certify(S.sec, S.coch.F, S.a, S.r, S.gam, N, depth=0.35, stop_early=True)
    w = float(sum(b[2] for b in bands))
    print('  lam %.2f s %.4f N %d clean %s winding %+.2f (%.0f s)' % (lam, s, N, ok, w, time.time() - t0), flush=True)
    return dict(clean=bool(ok), winding=round(w, 3), nb=len(bands))
for lam in LAMS:
    k = '%.2f' % lam; row = res['rows'].setdefault(k, dict(lam=lam))
    S1 = R.State(N=4096, lam=lam, gscale=1.0); cap = 0.99/float(np.max(S1.ratio*S1.taper)); row['s_cap'] = cap
    for N in (1024, 2048, 4096):
        nk = 'N%d' % N
        if nk in row and 's_line' in row[nk]: continue
        rr = row.setdefault(nk, dict(tests={}))
        def test(s):
            kk = '%.5f' % s
            if kk not in rr['tests']:
                rr['tests'][kk] = one(lam, s, N); json.dump(res, open(OUT, 'w'), indent=1)
            return rr['tests'][kk]['clean']
        if test(cap):
            rr.update(s_line=cap, capped=True)
        else:
            lo = 0.5
            while not test(lo): lo *= 0.5
            hi = cap
            while hi - lo > 0.005:
                mid = round(0.5*(lo + hi), 5); lo, hi = (mid, hi) if test(mid) else (lo, mid)
            rr.update(s_line=hi, s_pass=lo, capped=False, winding_at_s_line=rr['tests']['%.5f' % hi]['winding'])
        json.dump(res, open(OUT, 'w'), indent=1)
        print('== lam %.2f N %d: s_line %.4f%s' % (lam, N, rr['s_line'], ' (capped)' if rr['capped'] else ''), flush=True)
print('CONV DONE', flush=True)
