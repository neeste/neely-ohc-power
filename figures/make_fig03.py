"""make_fig03.py - run 10: Fig. 3 (cat tuning curves, measured gray, model red) with the velocity read-out, in the style of
run 8 make_figs8.py (same header, series, msgs helpers, sizes and canvas; prelude of figs_s3/make_rev_figs.py).
Usage: make_fig03.py fig2_curves.json table2.json outdir [tag]
fig2_curves.json is the repo fig2_rev.py output of the state (FTC_CRIT=vel for the velocity read-out; it holds the measured
curves, so Fig03.plt and this input stay local). Writes outdir/Fig03.plt and outdir/Fig03_model_curves.json (model curves only).
Render: tplt -r -m5 Fig03.plt > Fig03.eps ; tplt -m8 Fig03.plt > Fig03.svg ; topng8.py."""
import json, math, os, sys
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'make_rev_figs.py')).read(); pre = src[:src.index('# ---------------- profiles')]
exec(pre)
F2, T2, O = sys.argv[1], sys.argv[2], sys.argv[3].rstrip('/') + '/'
TAG = sys.argv[4] if len(sys.argv) > 4 else 'model c856, velocity criterion'
os.makedirs(O, exist_ok=True)
D2 = json.load(open(F2))['pairs']
TB = json.load(open(T2))['x_transition_cf_khz']
L = []
head(L, 'Fig03', 'cat tuning curves, measured (gray) and %s (red)' % TAG, sz, (0.1, 30), (0, 90), 'frequency (kHz)', 'threshold (dB SPL)', xannot='0.1 0.3 1 3 10 30', yint='9.1')
for p_ in D2:
    for xs, ys in p_['data']: series(L, xs, ys, 7, 2, 0, 1, '1.2', 'measured CF %.2f' % p_['data_cf'])
for p_ in D2:
    for xs, ys in p_['model']: series(L, xs, ys, RED, 2, 0, 1, '2.0', 'model tip %.2f' % p_['model_tip'])
L += ['pltcol=%d : pltype=2 : lintype=1 : pltlwt=1' % GRAY, '%.3f 0' % TB, '%.3f 90' % TB, 'plot']
szw = '%.3f' % cap(9.4, DOUBLE)
L = [('xllc=1.6 : yllc=1.4 : xlen=7.6 : ylen=4.0' if ln.startswith('xllc=') else ('annsiz=-%s : labsiz=-%s' % (szw, szw) if ln.startswith('annsiz=') else ln)) for ln in L]
open(O + 'Fig03.plt', 'w').write('\n'.join(L) + '\n')
json.dump(dict(pairs=[dict(target=p_['target'], data_cf=p_['data_cf'], model_tip=p_['model_tip'], model=p_['model']) for p_ in D2], transition_cf_khz=TB,
               criterion=TAG), open(O + 'Fig03_model_curves.json', 'w'))
print('wrote', O + 'Fig03.plt', 'transition CF %.3f kHz' % TB)
