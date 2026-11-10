"""Model analog of Fisher et al. (2012): remove OHC force (gamma = 0) over
segments basal to the peak place of a 9-kHz tone and report BM and shear
at the peak re control. Segment lengths are set in cycles of the local
wave (phase of BM displacement), cat record, low level, N = 4096."""
import sys, json, math, numpy as np
sys.path.insert(0, '.')
import score_base as SB, exp_accumulation as E, tl_model as T
import geom
N = 4096
x = np.linspace(0, 1, N+1); a = SB.COCH.cf(x); r, g0 = SB.profiles(N); gam = g0*SB.COCH.taper(x)
def run(g, f):
    Y, Yr, Ho = E.terms(SB.COCH.sec, f, a, r, g); P = T._fd(SB.COCH, f, Y, N, 1.0); Xb = Y*P
    return Xb, Ho*Xb
out = {}
for geo in ('pair', 'bm'):
    if geo == 'bm':
        _lam_state = geom.LAM; geom.LAM = None   # BM-only comparison (lambda = 0 geometry)
        geom.GEOM = 'bm'; geom.install(sb=getattr(SB, '_sb_bmk', None))
        gc = geom.gc_of_rho(SB.COCH.sec, r/a, n=80); gam = (1-SB._nu(a))*gc*SB.COCH.taper(x)
    for fk in (9.0, 2.0):
        f = fk/SB.BASE
        xb, sh = run(gam, f); j = int(np.argmax(np.abs(sh)))
        ph = np.unwrap(np.angle(xb)); cyc = (ph - ph[j])/(2*np.pi)    # cycles re peak (positive basal)
        def seg(c0, c1):
            m = (cyc >= c0) & (cyc <= c1) & (np.arange(N+1) <= j)
            g = gam.copy(); g[m] = 0.0
            xb2, sh2 = run(g, f)
            return dict(bm_db=float(20*np.log10(abs(xb2[j])/abs(xb[j]))), shear_db=float(20*np.log10(abs(sh2[j])/abs(sh[j]))),
                        x0=float(x[m].min()) if m.any() else None, x1=float(x[m].max()) if m.any() else None)
        res = {'0-1 cycle basal': seg(0.0, 1.0), '0.9-1.1 cycle basal': seg(0.9, 1.1),
               '0.1-0.15 cycle basal': seg(0.1, 0.15), '1-3 cycles basal': seg(1.0, 3.0),
               'apical of peak (0 to -1 cycle)': None}
        m = (cyc <= 0) & (cyc >= -1.0) & (np.arange(N+1) >= j)
        g = gam.copy(); g[m] = 0; xb2, sh2 = run(g, f)
        res['apical of peak (0 to -1 cycle)'] = dict(bm_db=float(20*np.log10(abs(xb2[j])/abs(xb[j]))), shear_db=float(20*np.log10(abs(sh2[j])/abs(sh[j]))))
        g = 0*gam; xb2, sh2 = run(g, f)
        res['all (passive)'] = dict(bm_db=float(20*np.log10(abs(xb2[j])/abs(xb[j]))), shear_db=float(20*np.log10(abs(sh2[j])/abs(sh[j]))))
        out['%s %g kHz' % (geo, fk)] = dict(x_peak=float(x[j]), cycles_to_peak=float(cyc[0]), segments=res)
        print(geo, fk, 'kHz peak x %.3f, %.1f cycles stapes->peak' % (x[j], cyc[0]))
        for k, v in res.items(): print('   %-32s BM %+6.1f dB  shear %+6.1f dB' % (k, v['bm_db'], v['shear_db']))
out['note'] = "'pair' entries use the state's own force split (lambda in geom at start); 'bm' is the BM-only geometry"
out['lam_state'] = _lam_state if '_lam_state' in dir() else geom.LAM
json.dump(out, open('out/fisher_analog.json', 'w'), indent=1)
