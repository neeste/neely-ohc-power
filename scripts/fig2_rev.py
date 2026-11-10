"""Fig. 2 at the cat state (score_base: best_d3 + base_ckpt): eight
measured tuning curves at matched CFs against the model curve whose tip is
nearest. Writes fig2_curves.json (PLT generator format) with per-pair rms
over the overlap, below the tip, and above the tip, and high-side
thresholds at 1.2 and 1.35 x tip."""
import json, math, sys
import numpy as np
sys.path.insert(0, '.')
import score_base as SB, exp_accumulation as E, tl_model as T
N = 1024
X = np.linspace(0, 1, N+1); A = SB.COCH.cf(X); r, g0 = SB.profiles(N); gam = g0*SB.COCH.taper(X)
base, C0 = SB.BASE, SB.Z[10]
import os
STAGE2 = os.environ.get('FTC_DEF', 'stage2') == 'stage2'   # stage-2 threshold: displacement shear criterion + middle ear (exp_ftc1.ftc_metrics)
if STAGE2:
    import exp_ftc1 as F1
    F_HP, F_LP, NLP = math.exp(SB.Z[6]), math.exp(SB.Z[7]), SB.Z[8]
    # Run 10: FTC_CRIT=disp (default) constant shear displacement, FTC_CRIT=vel constant shear velocity
    # (|xi_hb| f/1 kHz, exp_ftc1.crit_factor); C0 and the middle ear from base_ckpt.json either way.
    print('[fig2_rev] stage-2 read-out, criterion %s, C0 %.3f f_hp %.4f f_lp %.3f n_lp %.4f' % (F1.CRIT, C0, F_HP, F_LP, NLP), flush=True)
CEIL = 90.0
curves, cur, cf = [], [], None
for line in open('tc.plt'):
    line = line.strip()
    if line.startswith(';'):
        if 'CF=' in line: cf = float(line.split('CF=')[1].split()[0])
    elif line == 'plot':
        if cur and cf: curves.append((cf, np.array(cur, float)))
        cur, cf = [], None
    else:
        p = line.split()
        if len(p) == 2:
            try: cur.append((float(p[0]), float(p[1])))
            except ValueError: pass
if cur and cf: curves.append((cf, np.array(cur, float)))
cfs = np.array([c[0] for c in curves])
def model_curve(x0):
    j = int(np.argmin(np.abs(X - x0))); cfp = float(A[j])
    fr = cfp*np.logspace(math.log10(0.10), math.log10(1.6), 120)
    resp = []
    for f in fr:
        Y, Yr, Ho = E.terms(SB.COCH.sec, float(f), A, r, gam)
        P = T._fd(SB.COCH, float(f), Y, N, 1.0)
        resp.append(abs((Ho*Y*P)[j])*(float(F1.crit_factor(f*base)) if STAGE2 else f))
    th = C0 - 20*np.log10(np.array(resp))
    if STAGE2: th = th + F1.me_db(fr*base, F_HP, F_LP, NLP)
    return fr*base, th
def segs(x, y):
    m = y <= CEIL; out = []; cx, cy = [], []
    for xi, yi, mi in zip(x, y, m):
        if mi: cx.append(float(xi)); cy.append(float(yi))
        elif cx: out.append([cx, cy]); cx, cy = [], []
    if cx: out.append([cx, cy])
    return out
grid = np.linspace(0.06, 0.86, 41)
MC = [model_curve(x0) for x0 in grid]
tips = np.array([MC[i][0][int(np.argmin(MC[i][1]))] for i in range(len(MC))])
pairs = []
for tcf in [0.4, 0.7, 1.2, 2.0, 3.4, 5.7, 8.0, 11.3]:
    i = int(np.argmin(np.abs(np.log(cfs) - np.log(tcf)))); dcf, dd = curves[i]
    k = int(np.argmin(np.abs(np.log(tips) - np.log(dcf)))); fk, th = MC[k]
    lo, hi = max(fk.min(), dd[:, 0].min()), min(fk.max(), dd[:, 0].max())
    m = (dd[:, 0] >= lo) & (dd[:, 0] <= hi) & (dd[:, 1] <= CEIL)
    mi = np.interp(np.log(dd[m, 0]), np.log(fk), th); res = dd[m, 1] - mi
    dtip = dd[np.argmin(dd[:, 1]), 0]
    below = dd[m, 0] < dtip; above = dd[m, 0] > dtip
    def at(xq, xs, ys): return float(np.interp(np.log(xq), np.log(xs), ys))
    hs = {}
    for fac in (1.2, 1.35):
        hs[str(fac)] = dict(data=at(fac*dtip, dd[:, 0], dd[:, 1]) - dd[:, 1].min(),
                            model=at(fac*tips[k], fk, th) - th.min())
    pairs.append(dict(target=tcf, data_cf=float(dcf), model_tip=float(tips[k]),
                      rms=float(np.sqrt(np.mean(res**2))),
                      rms_below=float(np.sqrt(np.mean(res[below]**2))) if below.any() else None,
                      rms_above=float(np.sqrt(np.mean(res[above]**2))) if above.any() else None,
                      high_side=hs,
                      data=segs(dd[:, 0], dd[:, 1]), model=segs(fk, th)))
    pp = pairs[-1]
    print('CF %.2f tip %.2f rms %.1f below %.1f above %s  1.2x: data %.1f model %.1f  1.35x: data %.1f model %.1f'
          % (dcf, tips[k], pp['rms'], pp['rms_below'] or -1, pp['rms_above'],
             hs['1.2']['data'], hs['1.2']['model'], hs['1.35']['data'], hs['1.35']['model']), flush=True)
json.dump(dict(pairs=pairs), open('out/fig2_curves.json', 'w'))
