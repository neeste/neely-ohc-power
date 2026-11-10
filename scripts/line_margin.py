"""Run 6 item 3: the line's own distance from instability.
Uniform scale s on gamma (all places), quiet line, argument-principle test as in the paper's certification:
cat = the quiet members of the 40-certificate set (stage2/tools/cert_s2.py: N 1024 and 2048, depth 0.35 and 1.0,
10 bands over log10 omega in [-2.9, 0.35]); human = the quiet members of cert_h_lam.py (N 2048 depth 0.35 and 1.0,
N 8192 depth 0.35, bands over [-3.15, 0.35]). A setting FAILS when any band has |winding| >= 0.1 or a phase step
>= pi/2 after refinement. The line fails at s when any setting fails.
Search: s in [1.00, min(1.15, s_cap)], s_cap = 0.99/max_x(gamma/gamma_c) (growing_modes is not usable within about
1 percent of the section's own critical gamma). Bisection to TOL on the full quiet test.
Modes: cat_prof (stage-2 profiled line, X.profiles(z)), cat_si (stage-1 scale-invariant line, uniform gamma p[8],
r = a, gamma_c place independent), hum (joint_hs state in abr5_ckpt.json).
Cat: run from model/ under run_cat.py. Human: under run_human.py (NPTS 8192, LAM).
Usage: line_margin.py mode out.json [tol] [s_lo s_hi]"""
import sys, os, json, math, time
sys.path.insert(1, __import__('os').path.dirname(__import__('os').path.abspath(__file__))); import cat_state
import numpy as np
from dataclasses import replace
sys.path.insert(0, '.')
mode, OUT = sys.argv[1], sys.argv[2]
TOL = float(sys.argv[3]) if len(sys.argv) > 3 else 0.002
import geom, deltaT_3dof as d3
if mode.startswith('cat'):
    import exp_ftc1 as F1, exp_ftc2 as X, tl_model as T
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cert_s2.py')).read().split('ck = json.load')[0]
    ns = {}; exec(compile(src, 'cert_s2_head', 'exec'), ns); certify = ns['certify']
    bd = json.load(open('best_d3.json')); z = json.load(open(cat_state.COMBO))['z']
    SEC, COCH, A1 = F1.SEC, F1.COCH, F1.A; N1 = F1.N; x1 = np.linspace(0, 1, N1 + 1); BASE = math.exp(z[9])
    if mode == 'cat_prof':
        r1, g1 = X.profiles(z); gc1 = F1.gc_dense(r1/A1)
    else:
        g = float(bd['params']['gamma']); r1 = A1.copy(); g1 = g*COCH.taper(x1)
        gc1 = np.full(N1 + 1, d3.gamma_critical(replace(SEC, gamma=0.0)))
    F = COCH.F
    SETS = [(1024, 0.35), (2048, 0.35), (1024, 1.0), (2048, 1.0)]
    LGW = (-2.9, 0.35)
    def line(N):
        x = np.linspace(0, 1, N + 1); return COCH.cf(x), np.interp(x, x1, r1), x
    def gc_exact(j):
        q = r1[j]/A1[j]
        return d3.gamma_critical(replace(SEC, gamma=0.0, R_b=SEC.R_b*q, R_r=SEC.R_r*q, R_t=SEC.R_t*q, R_p=SEC.R_p*q, R_c=SEC.R_c*q))
    import exp_accumulation as E
    def peaks():
        out = {}
        for fk in (1.0, 4.0):
            w = fk/BASE; Y, Yr, Ho = E.terms(SEC, w, A1, r1, g1); P = T._fd(COCH, w, Y, N1, 1.0)
            out['%g kHz' % fk] = int(np.argmax(np.abs(Y*P)))
        return out
    sec_for_cert = SEC
else:
    import exp_abr2 as A2, exp_abr5 as A5, exp_accumulation as E, tl_model as T
    from cert2_h8k import certify
    A2.N = 8192
    c = json.load(open('abr5_ckpt.json')); pd = dict(c['params'])
    H = A2.Human(**pd); gc = A5.gc_exact_dense(H)
    H.gam0 = A5.capped_gamma(H, pd['mu0'], pd['alpha_mu'], gc); H.stable = True
    x1, A1, r1, g1, gc1 = H.x, H.a, H.r, H.gam0, gc; BASE = 20.7; F = H.coch.F; sec_for_cert = H.coch.sec
    SETS = [(2048, 0.35), (8192, 0.35), (2048, 1.0)]
    LGW = (-3.15, 0.35)
    def line(N):
        x = np.linspace(0, 1, N + 1); return np.interp(x, x1, A1), np.interp(x, x1, r1), x
    def gc_exact(j):
        sec = H.coch.sec; q = r1[j]/A1[j]
        return d3.gamma_critical(replace(sec, gamma=0.0, R_b=sec.R_b*q, R_r=sec.R_r*q, R_t=sec.R_t*q, R_p=sec.R_p*q, R_c=sec.R_c*q))
    def peaks():
        out = {}
        for fk in (1.0, 4.0):
            w = fk/BASE; Y, Yr, Ho = E.terms(sec_for_cert, w, A1, r1, g1); P = T._fd(H.coch, w, Y, len(x1) - 1, 1.0)
            out['%g kHz' % fk] = int(np.argmax(np.abs(Y*P)))
        return out

ratio = g1/gc1; jm = int(np.argmax(ratio)); rmax = float(ratio[jm])
gce = gc_exact(jm); rmax_ex = float(g1[jm]/gce)
s_cap = 0.99/max(rmax, rmax_ex)
pk = peaks()
res = json.load(open(OUT)) if os.path.exists(OUT) else dict(mode=mode, cwd=os.getcwd(), lam=geom.LAM, tests={})
res.update(rmax_table=rmax, rmax_exact=rmax_ex, x_rmax=float(x1[jm]), cf_rmax_khz=float(A1[jm]*BASE), s_cap=s_cap,
           ratio_at_peaks={k: dict(x=float(x1[j]), cf_khz=float(A1[j]*BASE), ratio=float(ratio[j])) for k, j in pk.items()},
           sets=SETS, lgw=LGW, tol=TOL)
print('%s: max gamma/gamma_c %.4f (exact %.4f) at x %.4f (CF %.2f kHz); s_cap %.4f; peaks %s' % (mode, rmax, rmax_ex, x1[jm], A1[jm]*BASE, s_cap,
      {k: round(float(ratio[j]), 4) for k, j in pk.items()}), flush=True)
def one(s, N, depth):
    key = '%.4f|%d|%.2f' % (s, N, depth)
    if key in res['tests']: return res['tests'][key]
    a, r, x = line(N); g = np.interp(x, x1, s*g1); t0 = time.time()
    ok, bands = certify(sec_for_cert, F, a, r, g, N, lgw=LGW, depth=depth)
    wsum = float(sum(b[2] for b in bands))
    d = dict(clean=bool(ok), winding=round(wsum, 3), bands=bands, secs=round(time.time() - t0, 1))
    res['tests'][key] = d; json.dump(res, open(OUT, 'w'), indent=1)
    print('  s %.4f N %d depth %.2f clean %s winding %+.3f (%.0f s)' % (s, N, depth, ok, wsum, d['secs']), flush=True)
    return d
def full(s):
    """True when the quiet line passes every setting; the fast settings first, stop at the first failure."""
    for N, depth in SETS:
        if not one(s, N, depth)['clean']: return False
    return True
lo = 1.0; hi = min(1.15, s_cap)
if len(sys.argv) > 5: lo, hi = float(sys.argv[4]), float(sys.argv[5])
assert full(lo), 'line fails at the lower end'
if full(hi):
    res['result'] = dict(s_line_ge=hi, capped=hi < 1.15, note='stable at the cap; the line instability is not reached before the sections own (cap at max gamma/gamma_c = 0.99)' if hi < 1.15 else 'stable at 1.15')
    print('STABLE AT UPPER END s = %.4f' % hi, flush=True)
else:
    while hi - lo > TOL:
        mid = round(0.5*(lo + hi), 5)
        if full(mid): lo = mid
        else: hi = mid
    # failing side: all settings at hi (complete the record); passing side recorded already
    fails = {'%d|%.2f' % (N, dp): one(hi, N, dp) for N, dp in SETS}
    res['result'] = dict(s_line=hi, s_pass=lo, failing_side={k: dict(clean=v['clean'], winding=v['winding']) for k, v in fails.items()},
                         max_ratio_at_s_line=hi*rmax, max_ratio_exact_at_s_line=hi*rmax_ex,
                         ratio_at_peaks_at_s_line={k: hi*float(ratio[j]) for k, j in pk.items()})
    print('s_line %.4f (passes at %.4f)' % (hi, lo), flush=True)
json.dump(res, open(OUT, 'w'), indent=1)
print('LM DONE', flush=True)
