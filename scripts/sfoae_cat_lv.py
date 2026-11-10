"""Cat SFOAE delay N_SF = 2 tau f (shear group delay at the shear peak) at 20, 30, 40 dB SPL on the
self-consistent saturated line, probe alone (as sfoae_h / exp_sfoae8.gd8 do for human).
Calibration as tf.py cat: drive A = 10^((L - C0_supp)/20) against the knee C = peak |shear| of the
unsaturated line at the probe frequency (u = 1 there at C0_supp dB SPL), s = u^4/(1+u^4).
Two level conventions: 'flat' (no middle-ear shape, as tf.py) and 'me' (L reduced by the stage-2
middle-ear correction relative to the probe frequency, exp_ftc1.me_db). Also tuning ratio Q_ERB/N_SF
at 40 dB with Q_ERB from out/tuning_ratio_cat.json (low level). Writes out/sfoae_cat_levels.json."""
import sys, json, math, numpy as np
sys.path.insert(0, '.')
import score_base as SB, exp_accumulation as E, tl_model as T, exp_ftc1 as F1
N = 2048
x = np.linspace(0, 1, N+1); a = SB.COCH.cf(x); r, g0 = SB.profiles(N); gam = g0*SB.COCH.taper(x)
CAL = 65.0 - float(np.median(json.load(open('base_battery.json'))['locus']['L']))
Z = SB.Z; base = SB.BASE; fhp, flp, nlp = math.exp(Z[6]), math.exp(Z[7]), Z[8]
def shear(f, s):
    Y, Yr, Ho = E.terms(SB.COCH.sec, f, a, r, gam*(1-s)); P = T._fd(SB.COCH, f, Y, N, 1.0); return Ho*Y*P
fp = float(SB.COCH.cf(E.X_PROBE)); Cp = float(np.max(np.abs(shear(fp, np.zeros(N+1)))))
me_p = float(F1.me_db(fp*base, fhp, flp, nlp))
res = dict(calibration_db=CAL, probe_khz=fp*base, conventions={})
for conv in ('flat', 'me'):
    rows = []
    for fk in np.geomspace(0.25, 32, 22):
        f = fk/base; rec = dict(f_khz=float(fk))
        dme = float(F1.me_db(fk, fhp, flp, nlp)) - me_p if conv == 'me' else 0.0
        s = np.zeros(N+1)
        for L in (20, 30, 40):
            A = 10**((L - dme - CAL)/20)
            def FF(sv):
                u = (A*np.abs(shear(f, sv))/Cp)**4.0; return u/(1+u)
            s, _, _ = E.anderson(FF, s)
            sh = shear(f, s); j = int(np.argmax(np.abs(sh)))
            fs = np.linspace(f*0.99, f*1.01, 9); ph = np.unwrap([np.angle(shear(ff, s)[j]) for ff in fs])
            tau = -np.polyfit(fs, ph, 1)[0]/(2*np.pi)/base          # ms
            rec['N_SF_%d' % L] = float(2*tau*fk); rec['smax_%d' % L] = float(s.max())
        rows.append(rec)
        print(conv, '%6.2f kHz  N_SF 20/30/40 dB: %5.2f %5.2f %5.2f  smax40 %.3f' % (fk, rec['N_SF_20'], rec['N_SF_30'], rec['N_SF_40'], rec['smax_40']), flush=True)
    res['conventions'][conv] = rows
tr = json.load(open('out/tuning_ratio_cat.json'))
for conv in ('flat', 'me'):
    R = res['conventions'][conv]; ff = np.log([q['f_khz'] for q in R]); n40 = [q['N_SF_40'] for q in R]
    res['ratio40_' + conv] = [dict(cf_khz=t['cf_khz'], q_erb=t['q_erb'], n_sf40=float(np.interp(math.log(t['cf_khz']), ff, n40)),
                                   ratio=float(t['q_erb']/np.interp(math.log(t['cf_khz']), ff, n40))) for t in tr]
    print(conv, 'ratio at 40 dB:', [(round(v['cf_khz'], 2), round(v['ratio'], 2)) for v in res['ratio40_' + conv]], flush=True)
json.dump(res, open('out/sfoae_cat_levels.json', 'w'), indent=1)
