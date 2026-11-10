"""Run 6 item 4: lambda scan at a fixed distance from the line's own instability.
Cat: c708 section and profiles (run2/code/r2cat.State: gamma = gscale (1 - nu) gamma_c(rho; lambda) taper, 24-point
gamma_c table rebuilt for each lambda as combo.py does; gscale 1 = gamma/gamma_c 0.955 at the peak region). Run in
a state directory of the lambda 0.7085 state (scripts/mkstate.py earlier/combo_c708.json ../state_c708) under run_cat.py.  Human: hs708 profiles (run2/code/r2hum.State, gamma_c from
gc_exact_dense rebuilt for lambda), run from model/ under run_human.py.
s_line(lambda): uniform scale of gamma at which the quiet line first fails the argument-principle test (cat: N 1024,
2048 x depth 0.35, 1.0, bands over [-2.9, 0.35]; human: N 2048 depth 0.35 and 1.0, N 8192 depth 0.35, [-3.15, 0.35]),
bisection to TOL = 0.005. Cap: s_cap = 0.99/max_x(gamma/gamma_c at gscale 1); if the line passes at s_cap, s_line = s_cap
(marked capped). If the start (s = 1) fails, bisect downward in [S_LO, 1].
Operating points: gscale = 0.97 s_line and 0.94 s_line. At each, at 1 and 4 kHz: OHC share p_ohc/D and p_ohc/max(D)
against place re the BM peak (octaves, APICAL POSITIVE), every local maximum of the share (scipy find_peaks, prominence
>= 0.002) within 2 octaves, the global maximum, and every band where the share exceeds 1.
Usage: lam_fixed.py cat|hum out.json [lambda ...]"""
import sys, os, json, math, time
sys.path.insert(1, __import__('os').path.dirname(__import__('os').path.abspath(__file__))); import cat_state
import numpy as np
from scipy.signal import find_peaks
sys.path.insert(0, '.'); import geom, r2cert as RC
kind, OUT = sys.argv[1], sys.argv[2]
TOL = 0.005
if kind == 'cat':
    import r2cat as R
    LAMS = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.7084610343120661, 0.80, 0.8564463479156329, 1.00]
    SETS = [(1024, 0.35), (2048, 0.35), (1024, 1.0), (2048, 1.0)]; LGW = (-2.9, 0.35); NF = 4096; BASE = R.BASE
    def make(lam, g, N): return R.State(N=N, lam=lam, gscale=g)
    def line(lam, g, N):
        S = make(lam, g, N); return S.sec, S.coch.F, S.a, S.r, S.gam
else:
    import r2hum as RH
    JS = json.load(open(cat_state.HUMAN))
    LAMS = [0.0, 0.30, 0.50, 0.60, 0.7084610343120661, 1.00]
    SETS = [(2048, 0.35), (2048, 1.0), (8192, 0.35)]; LGW = (-3.15, 0.35); NF = 8192; BASE = RH.BASE
    _cache = {}
    def make(lam, g, N=8192):
        key = round(lam, 6)
        if key not in _cache: _cache[key] = RH.State(JS, N=8192, lam=lam, gscale=1.0)
        S0 = _cache[key]
        class SS: pass
        S = SS(); S.__dict__.update(S0.__dict__); S.gscale = g; S.ratio = g*S0.ratio; S.gam = g*S0.gam
        S.fields = lambda f, gam=None, S=S: RH.State.fields(S, f, gam); S.power = lambda f, F, gam=None, S=S: RH.State.power(S, f, F, gam)
        S._set = lambda S=S: setattr(geom, 'LAM', S.lam)
        return S
    def line(lam, g, N):
        S = make(lam, g); x = np.linspace(0, 1, N + 1)
        return S.sec, S.coch.F, np.interp(x, S.x, S.a), np.interp(x, S.x, S.r), np.interp(x, S.x, S.gam)
if len(sys.argv) > 3: LAMS = [float(v) for v in sys.argv[3:]]
res = json.load(open(OUT)) if os.path.exists(OUT) else dict(kind=kind, sets=SETS, lgw=LGW, tol=TOL,
          sign='octaves re BM peak, apical positive', rows={})
def save(): json.dump(res, open(OUT, 'w'), indent=1)
def test(lam, s, row, full=False):
    key = '%.5f' % s
    if key in row['tests'] and not (full and row['tests'][key].get('partial')): return row['tests'][key]['clean']
    geom.LAM = lam; clean = True; det = {}
    for N, depth in SETS:
        sec, F, a, r, g = line(lam, s, N); geom.LAM = lam; t0 = time.time()
        bands = RC.certify(sec, F, a, r, g, N, lgw=LGW, depth=depth, stop_early=not full)
        bands = bands[1] if isinstance(bands, tuple) else bands
        ok = all(abs(b[2]) < 0.1 and b[3] < math.pi/2 for b in bands)
        det['%d|%.2f' % (N, depth)] = dict(clean=bool(ok), winding=round(float(sum(b[2] for b in bands)), 3), nb=len(bands), secs=round(time.time() - t0, 1))
        print('    lam %.4f s %.4f N %d depth %.2f clean %s winding %+.3f (%.0f s)' % (lam, s, N, depth, ok, det['%d|%.2f' % (N, depth)]['winding'], time.time() - t0), flush=True)
        if not ok:
            clean = False
            if not full: break
    row['tests'][key] = dict(clean=clean, settings=det, partial=(not clean and not full)); save()
    return clean
def lobes(o, y, win, wa=None):
    w = np.where(win)[0]; i0, i1 = int(w.min()), int(w.max())        # contiguous window; end points are not peaks
    k, pr = find_peaks(y[i0:i1 + 1], prominence=0.002)
    return [dict(oct=float(o[i0 + i]), height=float(y[i0 + i]), prominence=float(p), w_over_a=(float(wa[i0 + i]) if wa is not None else None))
            for i, p in zip(k, pr['prominences'])]
def bands(o, s, win):
    m = win & (s > 1); out = []; i = 0; n = len(s)
    while i < n:
        if m[i]:
            i0 = i
            while i + 1 < n and m[i + 1]: i += 1
            out.append([float(min(o[i0], o[i])), float(max(o[i0], o[i]))])
        i += 1
    return out
for lam in LAMS:
    lk = '%.5f' % lam
    row = res['rows'].setdefault(lk, dict(lam=lam, tests={})); t0 = time.time()
    if 's_line' not in row:
        S1 = make(lam, 1.0, NF); rmax = float(np.max(S1.ratio*S1.taper)); cap = 0.99/rmax; row.update(rmax_at_g1=rmax, s_cap=cap)
        print('== lambda %.4f: max gamma/gamma_c %.4f at gscale 1, cap %.4f' % (lam, rmax, cap), flush=True)
        if test(lam, 1.0, row):
            if test(lam, cap, row): row.update(s_line=cap, capped=True, s_pass=cap)
            else:
                lo, hi = 1.0, cap
                while hi - lo > TOL:
                    mid = round(0.5*(lo + hi), 5); lo, hi = (mid, hi) if test(lam, mid, row) else (lo, mid)
                row.update(s_line=hi, s_pass=lo, capped=False)
        else:
            lo = 0.5
            while not test(lam, lo, row): lo *= 0.5
            hi = 1.0
            while hi - lo > TOL:
                mid = round(0.5*(lo + hi), 5); lo, hi = (mid, hi) if test(lam, mid, row) else (lo, mid)
            row.update(s_line=hi, s_pass=lo, capped=False)
        if not row['capped']:
            test(lam, row['s_line'], row, full=True)
            row['failing_side'] = row['tests']['%.5f' % row['s_line']]['settings']
        save()
    print('   s_line %.4f%s (passes %.4f)' % (row['s_line'], ' (capped)' if row['capped'] else '', row['s_pass']), flush=True)
    row.setdefault('ops', {})
    for op in (0.97, 0.94):
        ok_ = '%.2f' % op
        if ok_ in row['ops']: continue
        g = op*row['s_line']; S = make(lam, g, NF); geom.LAM = lam; d = dict(gscale=g)
        for fk in (1.0, 4.0):
            f = fk/BASE; F = S.fields(f); D, p, pf = S.power(f, F)
            err = float(np.max(np.abs(D - pf - p))/D.max())
            j0 = int(0.02*(len(S.x) - 1)); jb = int(np.argmax(np.abs(F['Xb'][j0:]))) + j0; jsh = int(np.argmax(np.abs(F['sh'])))
            o = np.log2(S.a[jb]/S.a); ok = D > 1e-6*D.max(); s = np.where(ok, p/np.where(ok, D, 1.0), 0.0)
            win = ok & (np.abs(o) < 2.0) & (S.a > 1.5e-4)
            js = int(np.argmax(np.where(win, s, -1e9))); jp = int(np.argmax(np.where(win, p, -1e9)))
            pl = np.abs(o) <= 1.5; step = max(1, int(pl.sum()/300))
            idx = np.where(pl)[0][::step]
            d['%g kHz' % fk] = dict(x_bm_peak=float(S.x[jb]), shear_peak_oct=float(o[jsh]),
                                   ratio_at_bm_peak=float(S.ratio[jb]*S.taper[jb]),
                                   share_max=float(s[js]), share_max_oct=float(o[js]), p_ohc_max_oct=float(o[jp]),
                                   share_lobes=lobes(o, s, win, f/S.a), p_ohc_lobes=lobes(o, p/D.max(), win, f/S.a),
                                   w_over_a=dict(bm_peak=float(f/S.a[jb]), shear_peak=float(f/S.a[jsh]), share_max=float(f/S.a[js]), p_ohc_max=float(f/S.a[jp])),
                                   bm_lobes=lobes(o, np.abs(F['Xb'])/np.abs(F['Xb']).max(), win, f/S.a), shear_lobes=lobes(o, np.abs(F['sh'])/np.abs(F['sh']).max(), win, f/S.a),
                                   bands_gt1=bands(o, s, win), share_at_bm_peak=float(s[jb]), identity_err=err,
                                   profile=dict(oct=o[idx].tolist(), share=s[idx].tolist(), p_ohc=(p/D.max())[idx].tolist(), D=(D/D.max())[idx].tolist()))
            q = d['%g kHz' % fk]
            print('   op %.2f (g %.4f) %g kHz: share max %.3f at %+.3f oct; lobes %s; bands>1 %s; p_ohc max %+.3f oct; err %.1e' % (
                op, g, fk, q['share_max'], q['share_max_oct'], [(round(l['oct'], 3), round(l['height'], 3)) for l in q['share_lobes']],
                [[round(b0, 3), round(b1, 3)] for b0, b1 in q['bands_gt1']], q['p_ohc_max_oct'], err), flush=True)
        row['ops'][ok_] = d; save()
    print('   (%.0f s)' % (time.time() - t0), flush=True)
print('LAMFIX DONE', flush=True)
