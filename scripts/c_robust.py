"""Queue C (run2): robustness of the location result (OHC share / p_ohc maximum re the BM and shear peaks).
One factor at a time from the adopted state: lambda in {0, 0.25, 0.5, (cat: 0.7085), adopted, 1}; gamma/gamma_c in
{0.85, 0.90, adopted} (uniform gamma scale T/g0, see VAR); damping scale in {0.7, 1, 1.4}. gamma/gamma_c held fixed when lambda or damping changes.
Each state certified (cat: cert_s2 set, 40; human: cert_h_lam set, 18). Then a location-only lambda scan (step 0.05).
Usage: cat:   (model/)  python3 run_cat.py c_robust.py cat
       human: (model/)  python3 run_human.py c_robust.py human joint_hs_hs856.json"""
import sys, os, json, time, numpy as np
MODEL = sys.argv[1]
if MODEL == 'cat':
    import r2cat as R, r2cert as RC
    LAM0 = R.LAM0; make = lambda N=2048, **k: R.State(N=N, **k)
    def cert(k):
        return RC.certify_state(lambda N: R.State(N=N, **k), R.SEC, R.COCH.F, log=lambda s: print(s, flush=True))
    NT = 40
else:
    import r2hum as R
    JS = sys.argv[2]; LAM0 = float(json.load(open(JS))['lam']); make = lambda N=None, **k: R.State(JS, **k)
    def cert(k):
        return R.certify_h(R.State(JS, **k), log=lambda s: print(s, flush=True))
    NT = 18
OUT = 'out/c_robust_%s.json' % MODEL
res = json.load(open(OUT)) if os.path.exists(OUT) else dict(model=MODEL, lam0=LAM0, rows=[], lam_scan=[])
if MODEL != 'cat': res['state'] = JS
done = {r['name'] for r in res['rows']}
# gamma/gamma_c rows scale gamma uniformly by T/g0, g0 = gamma/gamma_c of the state at its peak region:
# cat 0.9501 (stage-1 ratio of c856); human 1 - nu_floor (0.9758 at hs856). The cat set includes the lambda 0.7085 state.
if MODEL == 'cat':
    G0 = 0.9501; LAMS = (0.0, 0.25, 0.5, 0.7084610343120661, 1.0)
else:
    G0 = 1.0 - float(json.load(open(JS))['nu_floor']); LAMS = (0.0, 0.25, 0.5, 1.0)
VAR = [('adopted', {})] + [('lam %.2f' % l, dict(lam=l)) for l in LAMS] + \
      [('g/gc %.2f' % T, dict(gscale=T/G0)) for T in (0.85, 0.90)] + [('damp x%.1f' % d, dict(dscale=d)) for d in (0.7, 1.4)]
def shares(S):
    out = {}
    for fk in (1.0, 4.0):
        out[str(fk)] = S.share(fk/R.BASE)
    return out
for name, k in VAR:
    if name in done: continue
    t0 = time.time(); print('== %s %s' % (name, k), flush=True)
    nc, nt, det = cert(k)
    S = make(**k); row = dict(name=name, kw=k, cert_clean=nc, cert_total=nt, stable=(nc == nt == NT), cf=shares(S))
    for fk, sh in row['cf'].items():
        print('  %s kHz stable=%s share max %.3f at %+.3f oct re BM, %+.3f re shear; p_ohc max %+.3f re BM, %+.3f re shear; >1 band %s; g/gc %.3f; id err %.1e' %
              (fk, row['stable'], sh['share_max'], sh['share_max_oct_bm'], sh['share_max_oct_shear'], sh['p_ohc_max_oct_bm'], sh['p_ohc_max_oct_shear'],
               sh['band_gt1_oct_bm'], sh['gamma_ratio_at_bm_peak'], sh['identity_err']), flush=True)
    res['rows'].append(row); json.dump(res, open(OUT, 'w'), indent=1); print('   (%.0f s)' % (time.time() - t0), flush=True)
if not res['lam_scan']:
    for lam in np.round(np.arange(0.0, 1.0001, 0.05), 3):
        S = make(lam=float(lam)); rr = dict(lam=float(lam))
        for fk, sh in shares(S).items():
            rr[fk] = {q: sh[q] for q in ('share_max', 'share_max_oct_bm', 'share_max_oct_shear', 'p_ohc_max_oct_bm', 'p_ohc_max_oct_shear', 'identity_err')}
        res['lam_scan'].append(rr)
        print('  scan lam %.2f  1k: share-max %+.3f oct, p-max %+.3f | 4k: share-max %+.3f, p-max %+.3f' % (lam, rr['1.0']['share_max_oct_bm'], rr['1.0']['p_ohc_max_oct_bm'], rr['4.0']['share_max_oct_bm'], rr['4.0']['p_ohc_max_oct_bm']), flush=True)
    cross = {}
    for fk in ('1.0', '4.0'):
        for key in ('share_max_oct_bm', 'p_ohc_max_oct_bm'):
            L = [r['lam'] for r in res['lam_scan']]; v = [r[fk][key] for r in res['lam_scan']]; c = None
            for i in range(len(v) - 1):
                if v[i] > 0 and v[i+1] <= 0: c = L[i] + (L[i+1] - L[i])*v[i]/(v[i] - v[i+1])
            cross['%s %s' % (fk, key)] = c
    res['lam_crossing_basal_to_apical'] = cross; print('crossing', cross, flush=True)
    json.dump(res, open(OUT, 'w'), indent=1)
print('C DONE', MODEL, flush=True)
