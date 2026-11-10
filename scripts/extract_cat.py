"""Run 8: manuscript numbers from the cat outputs (tf_cat, fig2_curves, fig3_curves, share_s2, tm_cat, sfoae, tuning ratio).
Usage (from model/): python3 ../scripts/extract_cat.py out out/extract_cat.json"""
import sys, json, math
import numpy as np
D = sys.argv[1].rstrip('/') + '/'
o = {}
# ---- Table V and Sec. III.B text: transfer functions ----
T = json.load(open(D + 'tf_cat.json')); rows = []
for p in T['places']:
    f = np.array(p['f_khz']); cf = p['cf_khz']; bm = {r['level']: r for r in p['bm']}; rl = {r['level']: r for r in p['rl']}
    def at(r, ff, key='mag_db'): return float(np.interp(math.log(ff), np.log(f), r[key]))
    pas = bm['passive']; row = dict(x=p['x'], cf_khz=cf)
    row['bm_gain_20'] = at(bm[20], cf) - at(pas, cf)
    row['bm_cf_re_0.1cf_20'] = at(bm[20], cf) - at(bm[20], 0.1*cf)
    row['rl_gain_20'] = at(rl[20], cf) - at(rl['passive'], cf) if 'passive' in rl else None
    row['bm_gain_levels'] = {L: at(bm[L], cf) - at(pas, cf) for L in (40, 60, 80, 100)}
    row['bm_phase_cf_re_0.1cf_20'] = at(bm[20], cf, 'phase_cyc') - at(bm[20], 0.1*cf, 'phase_cyc')
    row['bm_phase_cf_re_0.1cf_passive'] = at(pas, cf, 'phase_cyc') - at(pas, 0.1*cf, 'phase_cyc')
    row['max_diff_20_40'] = float(np.max(np.abs(np.array(bm[20]['mag_db']) - np.array(bm[40]['mag_db']))))
    row['max_diff_20_60'] = float(np.max(np.abs(np.array(bm[20]['mag_db']) - np.array(bm[60]['mag_db']))))
    # above CF: notch near 1.1 CF, plateau, passive at 1.6 CF
    a = np.array(bm[20]['mag_db']) - at(bm[20], cf); pa = np.array(pas['mag_db']) - at(pas, cf)
    m = (f > 1.0*cf) & (f <= 1.6*cf); i = np.where(m)[0]
    jn = i[int(np.argmin(a[i]))]
    after = i[i > jn]; jp = after[int(np.argmax(a[after]))] if len(after) else None
    row['notch'] = dict(f_re_cf=float(f[jn]/cf), db_re_cf=float(a[jn]))
    row['plateau'] = dict(f_re_cf=float(f[jp]/cf), db_re_cf=float(a[jp])) if jp is not None else None
    row['passive_at_1.6cf_re_cf'] = float(np.interp(math.log(1.6*cf), np.log(f), pa))
    row['act_minus_pas_at_1.6cf'] = at(bm[20], 1.6*cf) - at(pas, 1.6*cf)
    rows.append(row)
o['tf'] = rows
# ---- Fig. 3 per-curve numbers (model and data comparison metrics only) ----
import os
F = json.load(open(D + 'fig2_curves.json'))['pairs'] if os.path.exists(D + 'fig2_curves.json') else []   # needs the individual curves; fc = []
fc = []
def metr(curves):
    xs, ys = [np.array(v) for v in max(curves, key=lambda c: len(c[0]))]; k = int(np.argmin(ys)); ft, tt = xs[k], ys[k]
    lo, hi = None, None
    for j in range(k, -1, -1):
        if ys[j] >= tt + 10: lo = np.exp(np.interp(tt + 10, [ys[j+1], ys[j]], np.log([xs[j+1], xs[j]]))); break
    for j in range(k, len(ys)):
        if ys[j] >= tt + 10: hi = np.exp(np.interp(tt + 10, [ys[j-1], ys[j]], np.log([xs[j-1], xs[j]]))); break
    q10 = ft/(hi - lo) if lo and hi else None
    tail = float(np.interp(math.log(ft/2), np.log(xs), ys)) - tt if xs[0] <= ft/2 else None
    return dict(tip_khz=float(ft), tip_db=float(tt), q10=q10, tt=tail)
for p in F:
    r = dict(data_cf=p['data_cf'], model_tip=p['model_tip'], rms=p['rms'], rms_below=p['rms_below'], rms_above=p['rms_above'], high_side=p['high_side'])
    r['model'] = metr(p['model']); r['data_tip_db'] = metr(p['data'])['tip_db']; r['data_q10'] = metr(p['data'])['q10']; r['data_tt'] = metr(p['data'])['tt']
    fc.append(r)
o['fig3'] = fc
# ---- energy: fig3_curves (cat 1 and 4 kHz, re shear peak, apical positive) and share_s2 ----
E = json.load(open(D + 'fig3_curves.json')); en = {}
for fk, v in E['cat_share'].items():
    x = np.array(v['x']); s = np.array(v['share']); k = int(np.argmax(s)); ab = x[s > 1]
    en[fk] = dict(share_max=float(s[k]), at_oct=float(x[k]), band=[float(ab.min()), float(ab.max())] if len(ab) else None,
                  share_at_shear_peak=float(np.interp(0.0, x, s)), x_shear=v.get('x_shear_peak'), x_bm=v.get('x_bm_peak'))
lp = E['left']; x = np.array(lp['x']); Dd = np.array(lp['D']); po = np.array(lp['po'])
en['left'] = dict(D_peak_oct=float(x[int(np.argmax(Dd))]), po_peak_oct=float(x[int(np.argmax(po))]))
# integral of p_ohc over D, +-0.8 oct (whole, apical, basal halves)
def frac(m): return float(np.trapezoid(po[m], x[m])/np.trapezoid(Dd[m], x[m]))
en['left']['ohc_frac'] = dict(whole=frac(np.abs(x) <= 0.8), apical=frac((x >= 0) & (x <= 0.8)), basal=frac((x <= 0) & (x >= -0.8)))
o['fig8'] = en
S2 = json.load(open(D + 'share_s2.json')); o['share_s2'] = S2
# ---- SFOAE, tuning ratio ----
L = json.load(open(D + 'sfoae_cat_levels.json')); fl = L['conventions']['flat']
o['sfoae40'] = [(c['f_khz'], c['N_SF_40'], c.get('smax40')) for c in fl]
o['sfoae_low'] = [(c['f_khz'], c.get('N_SF_20', c.get('N_SF_low'))) for c in fl]
S = json.load(open(D + 'sfoae_cat.json')); o['sfoae_cat'] = S
TR = json.load(open(D + 'tuning_ratio_cat.json')); o['tuning_ratio'] = TR
json.dump(o, open(sys.argv[2], 'w'), indent=1, default=float)
for r in o['tf']:
    print('TF x %.3f CF %.2f: BMgain20 %.1f, BM re 0.1CF %+.1f, RL gain %.1f, levels %s, phase %.2f (pas %.2f), d20/40 %.3f d20/60 %.2f, notch %s plateau %s, pas1.6 %.1f, act-pas1.6 %.1f' % (
        r['x'], r['cf_khz'], r['bm_gain_20'], r['bm_cf_re_0.1cf_20'], r['rl_gain_20'] or float('nan'), {k: round(v, 1) for k, v in r['bm_gain_levels'].items()},
        r['bm_phase_cf_re_0.1cf_20'], r['bm_phase_cf_re_0.1cf_passive'], r['max_diff_20_40'], r['max_diff_20_60'], r['notch'], r['plateau'], r['passive_at_1.6cf_re_cf'], r['act_minus_pas_at_1.6cf']))
for r in o['fig3']:
    m = r['model']
    print('FTC CF %.2f tip %.2f rms %.1f (below %.1f above %.1f) 1.2x d %.1f m %.1f 1.35x d %.1f m %.1f | model tip %.1f dB Q10 %s tt %s | data tip %.1f' % (
        r['data_cf'], r['model_tip'], r['rms'], r['rms_below'], r['rms_above'], r['high_side']['1.2']['data'], r['high_side']['1.2']['model'],
        r['high_side']['1.35']['data'], r['high_side']['1.35']['model'], m['tip_db'], m['q10'] and round(m['q10'], 2), m['tt'] and round(m['tt'], 1), r['data_tip_db']))
print('FIG8', json.dumps(o['fig8'], default=float))
print('SFOAE40', [(round(a, 2), round(b, 2), c) for a, b, c in o['sfoae40']])
