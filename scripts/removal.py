"""Run 6 item 1: local-inactivation (removal) test with the BM and hair-bundle-shear read-outs.
Generalizes step3/fisher_analog.py (final-state version, which gave the manuscript numbers) to c708, c85
(cat; run from model/ under run_cat.py, N = 4096, graded profile SB.profiles) and hs708
(human; run from model/ under run_human.py, N = 8192). Quiet (linear) level.
Read-out place j = argmax |shear| of the control (as fisher_analog.py). Cycles from the BM phase, re j,
positive basal. Places in octaves re the read-out place: log2(a(j)/a(x)), APICAL POSITIVE.
Each manipulation sets gamma = 0 over a set of places; BM and shear at j are recomputed from scratch
(shear = H_o X_b with H_o of the manipulated line).
Diagnostic added in run 6: for every manipulation whose set contains j, the same manipulation with gamma kept at j
('keep j'), and gamma = 0 at j alone ('read-out place only'). H_o is a property of the local section, so when j is
not in the set the shear and BM change by the same factor, and when j is in the set the shear change contains the
local factor |H_o(j; 0)/H_o(j; gamma_j)|.
Usage: removal.py cat|hum out.json"""
import sys, os, json, math, numpy as np
sys.path.insert(0, '.')
import exp_accumulation as E, tl_model as T, geom
from exp_place6 import accounting      # replaced by the lambda version (acc_lam) by the wrapper
kind, OUT = sys.argv[1], sys.argv[2]
if kind == 'cat':
    import score_base as SB
    N = int(os.environ.get('NREM', 4096)); x = np.linspace(0, 1, N+1); a = SB.COCH.cf(x); r, g0 = SB.profiles(N); gam = g0*SB.COCH.taper(x)
    SEC, COCH, BASE, FREQS = SB.COCH.sec, SB.COCH, SB.BASE, (9.0, 2.0)
else:
    import exp_abr2 as A2, exp_abr5 as A5
    A2.N = 8192
    c = json.load(open('abr5_ckpt.json')); pd = dict(c['params'])
    H = A2.Human(**pd); gc = A5.gc_exact_dense(H); gam = A5.capped_gamma(H, pd['mu0'], pd['alpha_mu'], gc)
    x, a, r = H.x, H.a, H.r; N = len(x) - 1; SEC, COCH, BASE, FREQS = H.coch.sec, H.coch, 20.7, (4.0, 1.0)
idx = np.arange(N+1)
def run(g, w):
    Y, Yr, Ho = E.terms(SEC, w, a, r, g); P = T._fd(COCH, w, Y, N, 1.0); Xb = Y*P
    return Xb, Ho*Xb, Yr*P, Ho, Y, P
def db(u, v): return float(20*np.log10(abs(u)/abs(v)))
out = dict(kind=kind, lam=geom.LAM, N=N, cwd=os.getcwd(), sign='octaves re read-out place, apical positive; cycles positive basal')
for fk in FREQS:
    w = fk/BASE
    xb, sh, xr, Ho, Y, P0 = run(gam, w); j = int(np.argmax(np.abs(sh))); jb = int(np.argmax(np.abs(xb)))
    D, pf, po = accounting(SEC, a, r, gam, w, xb, xr, Ho, 1/(1j*w*Y))
    ok = D > 1e-6*D.max()
    share = np.where(ok, po/np.where(ok, D, 1), np.nan)
    ph = np.unwrap(np.angle(xb)); cyc = (ph - ph[j])/(2*np.pi)
    octv = np.log2(a[j]/a)
    # share bands
    above = ok & (share > 1)
    band = None; bas = None
    if above.any():
        # the band containing the largest share; contiguous run
        k0 = int(np.nanargmax(np.where(above, share, -np.inf))); i0 = k0; i1 = k0
        while i0 - 1 >= 0 and above[i0-1]: i0 -= 1
        while i1 + 1 <= N and above[i1+1]: i1 += 1
        band = (i0, i1)
        # basal of the band (smaller x): contiguous run adjacent to the band's basal edge with 0.5 <= share <= 1
        k = i0 - 1; b1 = k
        while k >= 0 and ok[k] and 0.5 <= share[k] <= 1.0: k -= 1
        bas = (k + 1, b1) if b1 >= k + 1 else None
        bas_all = ok & (share >= 0.5) & (share <= 1.0) & (idx < i0)
    segs = {}
    def setof(name):
        if name == 'everywhere': return np.ones(N+1, bool)
        if name == '0-1 cycle basal': return (cyc >= 0) & (cyc <= 1) & (idx <= j)
        if name == '1-3 cycles basal': return (cyc >= 1) & (cyc <= 3) & (idx <= j)
        if name == '0.9-1.1 cycles basal': return (cyc >= 0.9) & (cyc <= 1.1) & (idx <= j)
        if name == '0.1-0.15 cycles basal': return (cyc >= 0.1) & (cyc <= 0.15) & (idx <= j)
        if name == '0-1 cycle apical': return (cyc <= 0) & (cyc >= -1) & (idx >= j)
        if name == '(a) share > 1 band':
            m = np.zeros(N+1, bool)
            if band: m[band[0]:band[1]+1] = True
            return m
        if name == '(b) basal of band, 0.5 <= share <= 1':
            m = np.zeros(N+1, bool)
            if bas: m[bas[0]:bas[1]+1] = True
            return m
        if name == '(b) all places basal of band with 0.5 <= share <= 1':
            return bas_all if band else np.zeros(N+1, bool)
        if name == 'read-out place only': m = np.zeros(N+1, bool); m[j] = True; return m
    names = ['everywhere', '0-1 cycle basal', '1-3 cycles basal', '0.9-1.1 cycles basal', '0.1-0.15 cycles basal',
             '0-1 cycle apical', '(a) share > 1 band', '(b) basal of band, 0.5 <= share <= 1',
             '(b) all places basal of band with 0.5 <= share <= 1', 'read-out place only']
    def apply(m):
        g = gam.copy(); g[m] = 0.0; xb2, sh2, _, Ho2, Y2, P2 = run(g, w)
        return dict(bm_db=db(xb2[j], xb[j]), shear_db=db(sh2[j], sh[j]), bm_db_at_bm_peak=db(xb2[jb], xb[jb]),
                    Ho_local_db=db(Ho2[j], Ho[j]), Y_local_db=db(Y2[j], Y[j]), P_db=db(P2[j], P0[j]))
    for nm in names:
        m = setof(nm)
        if m is None or not m.any(): segs[nm] = None; continue
        d = apply(m)
        d.update(n_places=int(m.sum()), contains_readout=bool(m[j]),
                 oct_range=[float(octv[m].min()), float(octv[m].max())], x_range=[float(x[m].min()), float(x[m].max())],
                 share_range=[float(np.nanmin(share[m])), float(np.nanmax(share[m]))] if np.isfinite(share[m]).any() else None)
        if m[j] and nm != 'read-out place only':
            m2 = m.copy(); m2[j] = False; d2 = apply(m2)
            d['keep_j'] = dict(bm_db=d2['bm_db'], shear_db=d2['shear_db'], P_db=d2['P_db'])
        segs[nm] = d
    # non-additivity: basal cycle + apical cycle vs everywhere
    b, ap, ev = segs['0-1 cycle basal'], segs['0-1 cycle apical'], segs['everywhere']
    nonadd = dict(bm_sum=b['bm_db'] + ap['bm_db'], bm_everywhere=ev['bm_db'],
                  shear_sum=b['shear_db'] + ap['shear_db'], shear_everywhere=ev['shear_db'],
                  bm_sum_keep_j=b['keep_j']['bm_db'] + ap['keep_j']['bm_db'], shear_sum_keep_j=b['keep_j']['shear_db'] + ap['keep_j']['shear_db'],
                  everywhere_keep_j_shear=ev['keep_j']['shear_db'])
    # read-out-place check: shear change = BM change + local H_o change
    chk = max(max(abs(v['shear_db'] - v['bm_db'] - v['Ho_local_db']), abs(v['bm_db'] - v['P_db'] - v['Y_local_db'])) for v in segs.values() if v)
    smax = float(np.nanmax(share)); kmax = int(np.nanargmax(share))
    out['%g kHz' % fk] = dict(x_readout=float(x[j]), x_bm_peak=float(x[jb]), oct_bm_peak_re_readout=float(octv[jb]),
                              cycles_stapes_to_readout=float(cyc[0]), share_max=smax, share_max_oct=float(octv[kmax]),
                              band_oct=[float(octv[band[1]]), float(octv[band[0]])] if band else None,
                              basal_region_oct=[float(octv[bas[1]]), float(octv[bas[0]])] if bas else None,
                              share_at_readout=float(share[j]), segments=segs, nonadditivity=nonadd,
                              identity_check_db=chk)
    print('%g kHz read-out x %.4f (BM peak %+.3f oct), %.2f cycles; share max %.3f at %+.3f oct; band %s; basal region %s' % (
        fk, x[j], octv[jb], cyc[0], smax, octv[kmax], out['%g kHz' % fk]['band_oct'], out['%g kHz' % fk]['basal_region_oct']), flush=True)
    for nm, v in segs.items():
        if v is None: print('   %-52s (empty)' % nm); continue
        kj = ('  keep-j BM %+6.2f shear %+6.2f' % (v['keep_j']['bm_db'], v['keep_j']['shear_db'])) if 'keep_j' in v else ''
        print('   %-52s BM %+6.2f shear %+6.2f  P %+6.2f Y(j) %+6.2f Ho(j) %+6.2f  oct %+.2f..%+.2f%s' % (nm, v['bm_db'], v['shear_db'], v['P_db'], v['Y_local_db'], v['Ho_local_db'],
              v['oct_range'][0], v['oct_range'][1], kj), flush=True)
    print('   non-additivity', {k: round(v, 2) for k, v in nonadd.items()}, 'identity check %.1e dB' % chk, flush=True)
json.dump(out, open(OUT, 'w'), indent=1)
print('REMOVAL DONE', flush=True)
