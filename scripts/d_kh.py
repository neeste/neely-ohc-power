"""Queue D (run2): limited kH coth kH fluid test at fixed parameters, cat c708 and the adopted human state.
Usage: cat:   (model/) python3 run_cat.py d_kh.py cat
       human: (model/)  NPTS=8192 python3 run_human.py d_kh.py human state.json
H/L: cat 1/25 (H 1 mm, L 25 mm, units_cat.json and the manuscript); human 1/35 (H 1 mm, L 35 mm).
Sensitivity: H 0.5 and 2 mm (peak place and level only)."""
import sys, os, json, time, numpy as np
import r2kh as K
MODEL = sys.argv[1]
if MODEL == 'cat':
    import r2cat as R; mk = lambda N: R.State(N=N); L_mm = 25.0
else:
    import r2hum as R; JS = sys.argv[2]; mk = lambda N: R.State(JS, N=N); L_mm = 35.0
OUT = 'out/d_kh_%s.json' % MODEL
res = dict(model=MODEL, L_mm=L_mm, H_mm=1.0, tol_db=0.1, itmax=10, rows={}, grid={}, H_sens={}, itmax30={})
if MODEL != 'cat': res['state'] = JS
def peaks(F, S):
    jb = int(np.argmax(np.abs(F['Xb'][int(0.02*S.N):]))) + int(0.02*S.N); js = int(np.argmax(np.abs(F['sh'])))
    return dict(x_bm=float(S.x[jb]), cf_bm_khz=float(S.a[jb]*R.BASE), bm_db=float(20*np.log10(abs(F['Xb'][jb]))),
                x_shear=float(S.x[js]), cf_shear_khz=float(S.a[js]*R.BASE), shear_db=float(20*np.log10(abs(F['sh'][js])))), jb, js
def metrics(S, f):
    F = S.fields(f); pk, jb, js = peaks(F, S); q, ftip, jq = S.qerb(f); n, _ = S.nsf(f); sh = S.share(f)
    return dict(peaks=pk, Q_ERB=q, f_tip_khz=ftip, N_SF=n, share=sh)
N0 = 8192
S = mk(N0); kh = K.KH(S, 1.0/L_mm)
for fk in (1.0, 4.0):
    f = fk/R.BASE; t0 = time.time()
    m0 = metrics(S, f)
    kh.install(); kh.stats.update(calls=0, plain=0, failed=0, iters=[]); F1 = S.fields(f); main = dict(kh.last); m1 = metrics(S, f); st = dict(kh.stats); kh.remove()
    k_at = float(main['k'][int(np.argmax(np.abs(F1['Xb'])))]); fac_at = float(main['fac'][int(np.argmax(np.abs(F1['Xb'])))])
    row = dict(longwave=m0, corrected=m1, converged=main['converged'], iters=main['iters'], last_change_db=main['last_change_db'],
               k_at_bm_peak=k_at, kH_at_bm_peak=k_at/L_mm, fac_at_bm_peak=fac_at,
               sweep_stats=dict(calls=st['calls'], converged=st['plain'], not_converged=st['failed'], max_iters=max(st['iters'])),
               shift_bm_oct=float(np.log2(m1['peaks']['cf_bm_khz']/m0['peaks']['cf_bm_khz'])),
               shift_shear_oct=float(np.log2(m1['peaks']['cf_shear_khz']/m0['peaks']['cf_shear_khz'])))
    # longwave k at the long-wave peak for reference
    F0 = S.fields(f); _, kk = K.kfac(F0['P'], S.x, 1.0/L_mm); row['k_longwave_at_bm_peak'] = float(kk[int(np.argmax(np.abs(F0['Xb'])))])
    res['rows'][str(fk)] = row
    print('%s %.0f kHz: conv %s (%d it, last %.3f dB; sweeps %d/%d conv) | BM peak %.4f -> %.4f (%+.2f oct), %.1f -> %.1f dB | shear %.1f -> %.1f dB | Q_ERB %.2f -> %.2f | N_SF %.2f -> %.2f | share max %.3f at %+.3f -> %.3f at %+.3f oct | k %.0f -> %.0f | %.0fs' %
          (MODEL, fk, main['converged'], main['iters'], main['last_change_db'], st['plain'], st['calls'], m0['peaks']['x_bm'], m1['peaks']['x_bm'], row['shift_bm_oct'],
           m0['peaks']['bm_db'], m1['peaks']['bm_db'], m0['peaks']['shear_db'], m1['peaks']['shear_db'], m0['Q_ERB'], m1['Q_ERB'], m0['N_SF'], m1['N_SF'],
           m0['share']['share_max'], m0['share']['share_max_oct_bm'], m1['share']['share_max'], m1['share']['share_max_oct_bm'], row['k_longwave_at_bm_peak'], k_at, time.time() - t0), flush=True)
    json.dump(res, open(OUT, 'w'), indent=1)
    # residual of the 10-iteration cap: same point with 30 iterations
    kh30 = K.KH(S, 1.0/L_mm, itmax=30); F30 = kh30.fields(f); it30 = dict(kh30.last)
    kh30.install(); m30 = metrics(S, f); st30 = dict(kh30.stats); kh30.remove()
    res['itmax30'][str(fk)] = dict(metrics=m30, converged=it30['converged'], iters=it30['iters'],
                                   sweep_stats=dict(calls=st30['calls'], converged=st30['plain'], not_converged=st30['failed'], max_iters=max(st30['iters'])))
    print('   cap 30: conv %s (%d it; sweeps %d/%d conv, max %d it) | BM %.4f %.2f dB | Q_ERB %.2f | N_SF %.2f | share max %.3f at %+.3f oct' % (it30['converged'], it30['iters'], st30['plain'], st30['calls'], max(st30['iters']),
          m30['peaks']['x_bm'], m30['peaks']['bm_db'], m30['Q_ERB'], m30['N_SF'], m30['share']['share_max'], m30['share']['share_max_oct_bm']), flush=True)
    # H sensitivity
    for Hmm in (0.5, 2.0):
        khs = K.KH(S, Hmm/L_mm); Fs = khs.fields(f); ps, _, _ = peaks(Fs, S)
        res['H_sens']['%s H%.1f' % (fk, Hmm)] = dict(peaks=ps, converged=khs.last['converged'], iters=khs.last['iters'],
                                                       shift_bm_oct=float(np.log2(ps['cf_bm_khz']/m0['peaks']['cf_bm_khz'])))
        print('   H %.1f mm: BM peak %.4f (%+.2f oct) %.1f dB, conv %s (%d it)' % (Hmm, ps['x_bm'], res['H_sens']['%s H%.1f' % (fk, Hmm)]['shift_bm_oct'], ps['bm_db'], khs.last['converged'], khs.last['iters']), flush=True)
    json.dump(res, open(OUT, 'w'), indent=1)
# grid check at N = 16384 (peaks only)
S2 = mk(16384); kh2 = K.KH(S2, 1.0/L_mm)
for fk in (1.0, 4.0):
    F = kh2.fields(fk/R.BASE); p2, _, _ = peaks(F, S2); res['grid'][str(fk)] = dict(N=16384, peaks=p2, converged=kh2.last['converged'])
    print('   N 16384 %.0f kHz: BM peak %.4f %.2f dB (N 8192: %.4f %.2f)' % (fk, p2['x_bm'], p2['bm_db'], res['rows'][str(fk)]['corrected']['peaks']['x_bm'], res['rows'][str(fk)]['corrected']['peaks']['bm_db']), flush=True)
json.dump(res, open(OUT, 'w'), indent=1); print('D DONE', MODEL, flush=True)
