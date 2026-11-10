"""Queue B (run2): cat delay against distance from criticality at the c708 state.
gamma scaled uniformly by g = T/0.9501 (T = gamma/gamma_c at the 4-kHz peak region; 0.955 adopted).
At each T: N_SF(shear) low level and 40 dB SPL (flat and middle-ear conventions of sfoae_cat_lv.py, OHC
saturation knee C fixed at the adopted state), Q_ERB, tuning ratio, BM gain at CF, OHC share. Certified."""
import sys, os, json, math, time, numpy as np
import r2cat as R, r2cert as RC
import exp_accumulation as E, exp_ftc1 as F1, score_base as SB
OUT = 'out/b_crit.json'; os.makedirs(os.path.dirname(OUT), exist_ok=True)
res = json.load(open(OUT)) if os.path.exists(OUT) else dict(rows=[])
done = {round(r['target'], 3) for r in res['rows']}
CAL = 65.0 - float(np.median(json.load(open('base_battery.json'))['locus']['L']))
Z = SB.Z; fhp, flp, nlp = math.exp(Z[6]), math.exp(Z[7]), Z[8]
S1 = R.State(N=2048); fp = float(SB.COCH.cf(E.X_PROBE)); Cp = float(np.abs(S1.fields(fp)['sh']).max())
me_p = float(F1.me_db(fp*R.BASE, fhp, flp, nlp)); res.update(calibration_db=CAL, probe_khz=fp*R.BASE)
def nsf_level(S, f, L, conv):
    fk = f*R.BASE; dme = float(F1.me_db(fk, fhp, flp, nlp)) - me_p if conv == 'me' else 0.0
    A = 10**((L - dme - CAL)/20)
    def FF(sv):
        u = (A*np.abs(S.fields(f, S.gam*(1 - sv))['sh'])/Cp)**4.0; return u/(1 + u)
    s, _, _ = E.anderson(FF, np.zeros(S.N + 1)); n, _ = S.nsf(f, S.gam*(1 - s)); return n, float(s.max())
for T in (0.9501, 0.93, 0.90, 0.88, 0.85):
    if round(T, 3) in done: continue
    g = T/0.9501; t0 = time.time(); print('== T %.3f gscale %.4f' % (T, g), flush=True)
    nc, nt, det = RC.certify_state(lambda N: R.State(N=N, gscale=g), R.SEC, R.COCH.F, log=lambda s: print(s, flush=True))
    row = dict(target=T, gscale=g, cert_clean=nc, cert_total=nt, stable=(nc == nt == 40), cf={})
    if row['stable']:
        S = R.State(N=2048, gscale=g)
        for fk in (1.0, 4.0, 10.0):
            f = fk/R.BASE; n0, js = S.nsf(f); q, ftip, jq = S.qerb(f); gdb, fpk = S.bm_gain(f); sh = S.share(f)
            c = dict(N_SF_low=n0, Q_ERB=q, f_tip_khz=ftip, ratio=q/n0, bm_gain_db=gdb, bm_gain_f_re_cf=fpk, share=sh)
            for conv in ('flat', 'me'):
                n40, smax = nsf_level(S, f, 40.0, conv); c['N_SF_40_' + conv] = n40; c['smax40_' + conv] = smax
            row['cf'][str(fk)] = c
            print('  %4.1f kHz  N_SF low %.2f 40dB flat %.2f me %.2f | Q_ERB %.2f r %.2f | gain %.1f dB | share max %.3f at %+.3f oct (BM) | g/gc %.3f' %
                  (fk, n0, c['N_SF_40_flat'], c['N_SF_40_me'], q, q/n0, gdb, sh['share_max'], sh['share_max_oct_bm'], sh['gamma_ratio_at_bm_peak']), flush=True)
    res['rows'].append(row); json.dump(res, open(OUT, 'w'), indent=1); print('   (%.0f s)' % (time.time() - t0), flush=True)
print('B DONE', flush=True)
