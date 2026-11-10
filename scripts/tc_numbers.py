"""Run 10 item 3: every cat tuning-curve number of Sec. III.A, Table IV and the Fig. 3 caption, displacement (c856 as published)
and velocity (c856 mechanics, velocity criterion, refitted C0 and middle ear) side by side.
Inputs per criterion: repo model/out/fig2_curves.json (eight pairs; measured curves from local tc.plt), model/out/bmdef_shoulder.json
(nine-place threshold rows and shoulder parts), Table IV file from run8 ftcbins_model.py, run-6-definition shoulder json.
Usage (from model/): python3 ../scripts/tc_numbers.py out/tc_numbers.json"""
import sys, json, math
import numpy as np
# Inputs (from model/, written by the documented sequence): velocity default and the _disp switch outputs.
# out/fig2_curves.json needs the individual measured curves (tc.plt, not redistributed): run fig2_rev.py with
# and without FTC_CRIT=disp and copy the latter to out/fig2_curves_disp.json.
SRC = dict(disp=dict(fig2='out/fig2_curves_disp.json', bmdef='out/bmdef_shoulder_disp.json',
                     tab4='out/ftcbins_disp.json', sh='out/shoulder_ref_disp.json'),
           vel=dict(fig2='out/fig2_curves.json', bmdef='out/bmdef_shoulder.json',
                    tab4='out/ftcbins.json', sh='out/shoulder_ref.json'))

def metr(curves):        # run 8 extract1.py metr(), verbatim arithmetic
    xs, ys = [np.array(v) for v in max(curves, key=lambda c: len(c[0]))]; k = int(np.argmin(ys)); ft, tt = xs[k], ys[k]
    lo, hi = None, None
    for j in range(k, -1, -1):
        if ys[j] >= tt + 10: lo = np.exp(np.interp(tt + 10, [ys[j+1], ys[j]], np.log([xs[j+1], xs[j]]))); break
    for j in range(k, len(ys)):
        if ys[j] >= tt + 10: hi = np.exp(np.interp(tt + 10, [ys[j-1], ys[j]], np.log([xs[j-1], xs[j]]))); break
    q10 = ft/(hi - lo) if lo and hi else None
    tail = float(np.interp(math.log(ft/2), np.log(xs), ys)) - tt if xs[0] <= ft/2 else None
    return dict(tip_khz=float(ft), tip_db=float(tt), q10=q10, tt=tail)

def joined(curves):
    xs = np.concatenate([c[0] for c in curves]); ys = np.concatenate([c[1] for c in curves]); i = np.argsort(xs); return xs[i], ys[i]

out = {}
for crit, S in SRC.items():
    F = json.load(open(S['fig2']))['pairs']; o = dict(pairs=[])
    for p in F:
        m = metr(p['model']); d = metr(p['data'])
        # slope of model minus data below the data tip, over the common range (data points <= 90 dB), dB/octave
        dx, dy = joined(p['data']); mx, my = joined(p['model'])
        lo, hi = max(mx.min(), dx.min()), min(mx.max(), dx.max())
        sel = (dx >= lo) & (dx <= hi) & (dy <= 90.0) & (dx < d['tip_khz'])
        res = dy[sel] - np.interp(np.log(dx[sel]), np.log(mx), my)          # data minus model at the data frequencies
        slope = float(-np.polyfit(np.log2(dx[sel]), res, 1)[0]) if sel.sum() >= 3 else None   # model minus data
        sel_all = (dx >= lo) & (dx <= hi) & (dy <= 90.0)
        res_all = dy[sel_all] - np.interp(np.log(dx[sel_all]), np.log(mx), my)
        slope_all = float(-np.polyfit(np.log2(dx[sel_all]), res_all, 1)[0]) if sel_all.sum() >= 3 else None
        o['pairs'].append(dict(target=p['target'], data_cf=p['data_cf'], model_tip_khz=p['model_tip'], rms=p['rms'], rms_below=p['rms_below'],
                               rms_above=p['rms_above'], high_side=p['high_side'], model=m, data=d, tip_model_minus_data=m['tip_db'] - d['tip_db'],
                               slope_below_tip_model_minus_data_db_oct=slope, n_below=int(sel.sum()),
                               slope_common_range_model_minus_data_db_oct=slope_all))
    P = o['pairs']
    lcf = np.log2([p['data_cf'] for p in P]); dt = np.array([p['tip_model_minus_data'] for p in P])
    o['tip_trend_db_per_oct_8cf'] = float(np.polyfit(lcf, dt, 1)[0])
    o['tip_trend_db_per_oct_8cf_model_tips_only'] = float(np.polyfit(np.log2([p['model_tip_khz'] for p in P]), [p['model']['tip_db'] for p in P], 1)[0])
    o['data_tip_trend_db_per_oct_8cf'] = float(np.polyfit(lcf, [p['data']['tip_db'] for p in P], 1)[0])
    hi6 = [p for p in P if p['data_cf'] >= 1.2]
    o['rms_6cf_ge_1.2'] = [min(p['rms'] for p in hi6), max(p['rms'] for p in hi6)]
    o['rms_0.72_0.41'] = [P[1]['rms'], P[0]['rms']]
    o['rms_above_below_0.72_0.41'] = dict(above=[P[1]['rms_above'], P[0]['rms_above']], below=[P[1]['rms_below'], P[0]['rms_below']])
    o['rms_mean_8'] = float(np.mean([p['rms'] for p in P]))
    o['model_tip_db_range'] = [P[0]['model']['tip_db'], P[-1]['model']['tip_db']]
    o['q10_pairs_range'] = [min(p['model']['q10'] for p in P), max(p['model']['q10'] for p in P)]
    o['tt_pairs_range'] = [min(p['model']['tt'] for p in P if p['model']['tt'] is not None), max(p['model']['tt'] for p in P if p['model']['tt'] is not None)]
    o['hs1.2_range'] = [min(p['high_side']['1.2']['model'] for p in P), max(p['high_side']['1.2']['model'] for p in P)]
    four = [p for p in P if 1.9 <= p['data_cf'] <= 8.0]
    o['hs1.35_4cf_2_to_7.9'] = [min(p['high_side']['1.35']['model'] for p in four), max(p['high_side']['1.35']['model'] for p in four)]
    o['hs1.35_11.3'] = P[-1]['high_side']['1.35']['model']
    o['hs1.2_0.41_0.72'] = [P[0]['high_side']['1.2']['model'], P[1]['high_side']['1.2']['model']]
    # Table IV
    T4 = json.load(open(S['tab4'])); o['table4'] = T4['rows']; o['table4_q10_endpoint_exponent'] = T4.get('q10_endpoint_exponent')
    o['table4_q10_model_over_data'] = T4.get('q10_model_over_data')
    rows = [r for r in T4['rows'] if r['model']]
    lb = np.log2([r['bin_khz'] for r in rows]); dtb = np.array([r['model'][0] - r['data'][0] for r in rows])
    o['tip_trend_db_per_oct_table4'] = float(np.polyfit(lb, dtb, 1)[0])
    o['table4_tt_model_minus_data'] = [(r['bin_khz'], r['model'][2] - r['data'][2]) for r in rows if r['data'][2] == r['data'][2]]
    # shoulder (bmdef rows: rise within 0.2 oct, slopes) as run 8 shoulder_text.py
    Bm = json.load(open(S['bmdef'])); sht = []
    for r in Bm['shoulder']:
        oc = np.array(r['oct']); th = np.array(r['th_rel']); Pp = np.array(r['parts_rel'])
        at = lambda v: float(np.interp(v, oc, th)); sl = lambda a, b, y: float((np.interp(b, oc, y) - np.interp(a, oc, y))/(b - a))
        seg = [sl(a, a + 0.1, th) for a in np.arange(0.2, 0.6, 0.1)]
        sht.append({'x': r['x'], 'tip_khz': r['tip_khz'], 'rise_0.2': at(0.2), 'slope_0.2_0.6': sl(0.2, 0.6, th), 'seg_0.2_0.6': seg,
                    'slope_0.7_0.8': sl(0.7, 0.8, th), 'slope_0.6_0.8': sl(0.6, 0.8, th),
                    'parts_0.2_0.6': {nm: sl(0.2, 0.6, Pp[:, i]) for i, nm in enumerate(['fluid', 'bm', 'divider', 'me', 'crit'][:Pp.shape[1]])},
                    'fluid_0.7_0.8': sl(0.7, 0.8, Pp[:, 0]),
                    'parts_seg_0.2_0.6': {nm: [sl(a, a + 0.1, Pp[:, i]) for a in np.arange(0.2, 0.6, 0.1)] for i, nm in enumerate(['fluid', 'bm', 'divider', 'me', 'crit'][:Pp.shape[1]])}})
    o['shoulder_text'] = sht
    o['bm_threshold_rows'] = [dict(x=r['x'], tip_khz=r['tip_khz'], thr=r['thr_db_spl'], q10=r['q10'], tt=r['tt'],
                                   bm_um_s=r['H1.0mm_G25']['bm_um_s'], bm_nm=r['H1.0mm_G25']['bm_nm'], shear_um_s=r['H1.0mm_G25']['shear_um_s'],
                                   range_um_s=[min(v['bm_um_s'] for k, v in r.items() if k.startswith('H')), max(v['bm_um_s'] for k, v in r.items() if k.startswith('H'))])
                              for r in Bm['threshold']]
    SH = json.load(open(S['sh'])); o['shoulder_run6'] = {k: dict(cf_khz=v['cf_khz'], shoulder_index_db=v['shoulder_index_db'], at_f_re_cf=v['shoulder_f_re_cf'],
                                                                 thr_re_tip=v['thr_re_tip'], bm_notch=v['bm_notch']) for k, v in SH['places'].items()}
    o['shoulder_run6_worst'] = SH.get('worst')
    out[crit] = o
json.dump(out, open(sys.argv[1], 'w'), indent=1)
f1 = lambda v: '--' if v is None else ('%.1f' % v)
print('%-6s %-6s | %-34s | %-34s' % ('CF', 'tip', 'disp: rms below above tip  Q10  tt', 'vel: rms below above tip  Q10  tt'))
for a, b in zip(out['disp']['pairs'], out['vel']['pairs']):
    print('%5.2f  %5.2f | %5.1f %5.1f %5.1f %5.1f %4.2f %s | %5.1f %5.1f %5.1f %5.1f %4.2f %s | data tip %.1f | slope<tip %s %s' % (
        a['data_cf'], a['model_tip_khz'], a['rms'], a['rms_below'], a['rms_above'], a['model']['tip_db'], a['model']['q10'], f1(a['model']['tt']),
        b['rms'], b['rms_below'], b['rms_above'], b['model']['tip_db'], b['model']['q10'], f1(b['model']['tt']), a['data']['tip_db'],
        f1(a['slope_below_tip_model_minus_data_db_oct']), f1(b['slope_below_tip_model_minus_data_db_oct'])))
for k in ('tip_trend_db_per_oct_8cf', 'tip_trend_db_per_oct_table4', 'rms_6cf_ge_1.2', 'rms_0.72_0.41', 'rms_mean_8', 'model_tip_db_range', 'q10_pairs_range', 'tt_pairs_range',
          'hs1.2_range', 'hs1.35_4cf_2_to_7.9', 'hs1.35_11.3', 'table4_q10_endpoint_exponent'):
    print('%-32s disp %s   vel %s' % (k, out['disp'][k], out['vel'][k]))
for c in ('disp', 'vel'):
    print(c, 'shoulder run6', {k: round(v['shoulder_index_db'], 2) for k, v in out[c]['shoulder_run6'].items()})
    for s in out[c]['shoulder_text']: print('   x %.1f rise0.2 %.1f slope0.2-0.6 %.0f seg %s 0.7-0.8 %.0f fluid0.7-0.8 %.0f parts %s' % (s['x'], s['rise_0.2'], s['slope_0.2_0.6'], [round(t) for t in s['seg_0.2_0.6']], s['slope_0.7_0.8'], s['fluid_0.7_0.8'], {k: round(v) for k, v in s['parts_0.2_0.6'].items()}))
