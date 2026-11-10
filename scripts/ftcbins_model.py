"""Run 8: Table IV model columns. Interpolate the nine-place ftc_metrics (bmdef_shoulder.json 'threshold' rows:
tip_khz, thr_db_spl, q10, tt) in log tip frequency to the median tip frequency of each data bin (tc_bins.json),
as the population objective does; bins outside the model tip range get no entry.
Usage (from model/): ftcbins_model.py out/bmdef_shoulder.json ../data/tc_bins.json out/ftcbins.json"""
import sys, json, math
import numpy as np
b = json.load(open(sys.argv[1]))['threshold']; t = json.load(open(sys.argv[2]))['bins']
b = sorted(b, key=lambda r: r['tip_khz']); lf = np.log([r['tip_khz'] for r in b])
rows = []
for f, thr, q, tt in t:
    if not (lf[0] <= math.log(f) <= lf[-1]):
        rows.append(dict(bin_khz=f, data=[thr, q, tt], model=None)); continue
    m = [float(np.interp(math.log(f), lf, [r[k] for r in b])) for k in ('thr_db_spl', 'q10', 'tt')]
    rows.append(dict(bin_khz=f, data=[thr, q, tt], model=m))
out = dict(rows=rows, model_tip_range_khz=[b[0]['tip_khz'], b[-1]['tip_khz']], places=[dict(x=r['x'], tip_khz=r['tip_khz'], thr=r['thr_db_spl'], q10=r['q10'], tt=r['tt']) for r in b])
qs = [(r['bin_khz'], r['model'][1], r['data'][1]) for r in rows if r['model'] and r['data'][1] == r['data'][1]]
if len(qs) >= 2:
    (f0, m0, d0), (f1, m1, d1) = qs[0], qs[-1]
    out['q10_endpoint_exponent'] = dict(model=math.log(m1/m0)/math.log(f1/f0), data=math.log(d1/d0)/math.log(f1/f0))
    out['q10_model_over_data'] = [(f, m/d) for f, m, d in qs]
json.dump(out, open(sys.argv[3], 'w'), indent=1)
print('%7s %7s %7s %6s %6s %6s %6s' % ('bin', 'thr_d', 'thr_m', 'q_d', 'q_m', 'tt_d', 'tt_m'))
for r in rows:
    m = r['model'] or [float('nan')]*3
    print('%7.2f %7.1f %7.1f %6.2f %6.2f %6.1f %6.1f' % (r['bin_khz'], r['data'][0], m[0], r['data'][1], m[1], r['data'][2], m[2]))
print('model tip range', out['model_tip_range_khz']); print(out.get('q10_endpoint_exponent')); print(out.get('q10_model_over_data'))
