"""make_figs8.py - run 8: cat figures of amplifier_rev.tex at c856, in the style of figs_s3/make_rev_figs.py
(same header, series, msgs helpers and sizes). Writes PLT files to figures/plt/ named by figure number. Render: tplt -r -m5 X.plt > X.eps ; tplt -m8 X.plt > X.svg ; topng.py.
Figures with both species (Fig02a, Fig02b, Fig08b) carry the cat series only; a marked comment line shows where
the human series of run 9 go (same pltcol/lintype as before: BLUE, lintype 3 in Fig02, BLUE in Fig08b)."""
import json, math, os
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'make_rev_figs.py')).read(); pre = src[:src.index('# ---------------- profiles')]
exec(pre)
D8 = '../model/out/'; O = os.environ.get('FIGOUT', 'plt/'); os.makedirs(O, exist_ok=True); F9 = '../model/out/fig9/'
def save8(name, L): open(O + name + '.plt', 'w').write('\n'.join(L) + '\n')
LAMC = 0.8561939116132914

# ---------------- Fig02a, Fig02b: place profiles (cat; human series to be added) ----------------
P = json.load(open(D8 + 'profiles.json'))
for nm, key, ylab, yr, yint in (('Fig02a', 'rho', 'damping per cycle, ^r^', (0, 2), '4.1'), ('Fig02b', 'nu', 'distance from criticality, ^n^', (-3, 0), '3')):
    L = []
    head(L, nm, 'place profiles, cat c856 (human series to be added from run 9)', sz, (0.1, 50), yr, 'local scale a(x) x B (kHz)', ylab,
         xannot='0.1 0.3 1 3 10 30', yint=yint)
    L.append('yfmt=f3.1' if key == 'rho' else 'yannot= -3,0.001 -2,0.01 -1,0.1 0,1')
    d = P['cat']; ys = d[key]
    if key == 'nu': ys = [math.log10(max(v, 1e-6)) if t > 0.01 else None for v, t in zip(ys, d['taper'])]
    pts = [(x, y) for x, y in zip(d['map_khz'], ys) if y is not None and 0.1 <= x <= 50]
    series(L, [p[0] for p in pts], [p[1] for p in pts], RED, 2, 0, 1, '2.0', 'cat c856')
    L.append('; HUMAN SERIES (run 9) GO HERE: pltcol=%d : pltype=2 : lintype=3 : symbol=1 : pltlwt=2.0, x = a(x) B (kHz), y = %s' % (BLUE, 'rho (values <= 2.2)' if key == 'rho' else 'log10 nu where taper > 0.01'))
    msgs(L, sz, [(RED, 10, (0.95 if key == 'rho' else -0.75), 'cat'), (BLUE, (5.0 if key == 'rho' else 0.15), (1.6 if key == 'rho' else -2.25), 'human')])
    save8(nm, L)
    json.dump(dict(x_khz=[p[0] for p in pts], y=[p[1] for p in pts], key=key), open(O + nm + '_cat_panel.json', 'w'))

# Fig03 (cat tuning curves, velocity read-out): make_fig03.py ../model/out/fig2_curves.json ../model/out/table2.json plt
# (needs the individual measured curves, which are not redistributed).

# ---------------- Fig06: cat transfer functions (stacked) ----------------
T_ = json.load(open(D8 + 'tf_cat.json')); p = T_['places'][1]
cf = p['cf_khz']; f = [v/cf for v in p['f_khz']]
cols = {'passive': GRAY, 20: RED, 60: GREEN, 80: BLUE, 100: BLACK}
allv = [v for r in p['bm'] for v in r['mag_db']] + list(p['rl'][1]['mag_db'])
top = 10*math.ceil(max(allv)/10) + 10; lo = top - 60
s2 = '%.3f' % cap(7.6, SINGLE)
L = ['; Fig06.plt - cat c856 BM and RL displacement re stapes pressure (top) and BM phase (bottom), CF %.2f kHz' % cf, ';',
     'head=0 : clip=1', 'annsiz=-%s : labsiz=-%s' % (s2, s2), 'xmin=0.1 : xmax=1.6 : xcyc=%.4f' % math.log10(16), 'xannot= 0.1 0.2 0.5 1',
     '; lower panel: phase', 'xllc=1.9 : yllc=1.2 : xlen=5.7 : ylen=2.4', 'ymin=-4 : ymax=0 : yint=4.1 : yhor=yes', 'xlabel=frequency / CF', 'ylabel=phase (cycles)']
order = {'passive': 0, 100: 1, 80: 2, 60: 3, 20: 4}
for r in sorted([r for r in p['bm'] if r['level'] != 40], key=lambda r: order[r['level']]):
    ph = r['phase_cyc']; off = round(ph[0]); series(L, f, [v - off for v in ph], cols[r['level']], 2, 3 if r['level'] == 'passive' else 0, 1, '1.8', 'BM phase %s' % r['level'])
L += ['msgdat=yes : mhal=lef : mval=half : msgsiz=-%s' % s2]
for k, (col, txt) in enumerate(((RED, 'BM, 20 dB SPL'), (GREEN, 'BM, 60'), (BLUE, 'BM, 80'), (BLACK, 'BM, 100'))):
    L += ['msgcol=%d' % col, '%g %g "%s"' % (0.11, -1.9 - 0.55*k, txt)]
L += ['newframe', '; upper panel: magnitude', 'xllc=1.9 : yllc=4.0 : xlen=5.7 : ylen=3.0', 'xanskp=-1 : xlabel= ', 'ymin=%g : ymax=%g : yint=6 : yhor=yes' % (lo, top), 'ylabel=displacement (dB)']
for r in sorted([r for r in p['bm'] if r['level'] != 40], key=lambda r: order[r['level']]):
    series(L, f, r['mag_db'], cols[r['level']], 2, 3 if r['level'] == 'passive' else 0, 1, '1.8', 'BM %s' % r['level'])
series(L, f, p['rl'][1]['mag_db'], RED, 2, 1, 1, '1.4', 'RL 20 dB SPL')
L += ['msgdat=yes : mhal=lef : mval=half : msgsiz=-%s' % s2, 'msgcol=%d' % RED, '%g %g "RL, 20 dB SPL (dotted)"' % (0.11, top - 4),
      'msgcol=%d' % GRAY, '%g %g "passive (dashed)"' % (0.11, top - 12)]
save8('Fig06', L)

# ---------------- Fig08a, Fig08b: energy accounting ----------------
E3 = json.load(open(D8 + 'fig3_curves.json'))
lp = E3['left']; mD = max(lp['D'])
L = []
head(L, 'Fig08a', 'cat c856 1 kHz: dissipation and OHC power vs place', sz, (-0.8, 0.8), (0, 1.0), 'octaves apical of the shear peak', 'power density (normalized)', logx=False, yint='5.1', extra=('4.1',))
L.append('xfmt=f4.1 : yfmt=f3.1')
series(L, lp['x'], [v/mD for v in lp['D']], BLUE, 2, 0, 1, '2.0', 'D'); series(L, lp['x'], [v/mD for v in lp['po']], RED, 2, 0, 1, '2.0', 'p_ohc')
msgs(L, sz, [(BLUE, -0.75, 0.9, 'dissipation'), (RED, -0.75, 0.8, 'OHC power')])
save8('Fig08a', L)
L = []
head(L, 'Fig08b', 'OHC share of dissipation, cat c856 (human series to be added), 1 and 4 kHz', sz, (-0.8, 0.8), (0, 1.6), 'octaves apical of the shear peak', 'OHC power / dissipation', logx=False, yint='8.1', extra=('4.1',))
L.append('xfmt=f4.1 : yfmt=f3.1')
for f_, lt in (('1.0', 0), ('4.0', 3)):
    v = E3['cat_share'][f_]; series(L, v['x'], v['share'], RED, 2, lt, 1, '2.0', 'cat_share %s kHz' % f_)
L.append('; HUMAN SERIES (run 9) GO HERE: pltcol=%d : pltype=2 : lintype=0 (1 kHz) or 3 (4 kHz) : pltlwt=2.0, x = octaves apical of the shear peak, y = p_ohc/D' % BLUE)
L += ['pltcol=%d : pltype=2 : lintype=1 : pltlwt=1' % GRAY, '-0.8 1', '0.8 1', 'plot']
msgs(L, sz, [(BLUE, -0.75, 1.52, 'human'), (RED, -0.75, 1.40, 'cat'), (BLACK, 0.40, 1.52, '1 kHz solid'), (BLACK, 0.40, 1.40, '4 kHz dashed')])
save8('Fig08b', L)
json.dump(E3['cat_share'], open(O + 'Fig08b_cat_panel.json', 'w'))

# ---------------- Fig09a, Fig09b: the c856 section (rho = 1, gamma = q gamma_c(lambda)) ----------------
q = os.environ.get('Q9', '0.950')
import numpy as np
FD = json.load(open(F9 + 'fig09_data_q%s.json' % q))
CYAN = 3
L = []
head(L, 'Fig09a', 'OHC share of the isolated c856 section, gamma = %s gamma_c(lambda), rho = 1' % q, sz, (0.4, 1.2), (-0.6, 2.2), 'frequency / local scale, f/aB', 'OHC power / dissipation', logx=False, yint='7.1', extra=('4.1',))
L.append('xfmt=f3.1 : yfmt=f3.1')
LAB = []
for lam, col, txt, xy in ((LAMC, BLUE, 'l = 0.86 (fitted)', (0.42, 2.08)), (0.3, CYAN, 'l = 0.30', (0.42, 1.88)), (1.0, RED, 'l = 1', (0.89, 2.08)), (0.0, GREEN, 'l = 0', (0.89, 1.88))):
    dat = np.loadtxt(F9 + 'share_local_q%s_l%.3f.dat' % (q, lam)); m = (dat[:, 0] >= 0.4) & (dat[:, 0] <= 1.2)
    m = m & (dat[:, 1] >= -0.6); series(L, dat[m, 0], dat[m, 1], col, 2, 0, 1, '2.0', 'share lambda %.4f' % lam)
    r = [x for x in FD if abs(x['lam'] - (0.8561939116132914 if lam == LAMC else lam)) < 1e-6][0]
    wpk = r['wa_bm']; spk = float(np.interp(wpk, dat[:, 0], dat[:, 1]))
    series(L, [wpk], [spk], col, 1, 0, 1, '2.0', 'line peak lambda %.4f' % lam)
    LAB.append((col, xy[0], xy[1], txt.replace('l =', '^l^ =')))
L += ['pltcol=%d : pltype=2 : lintype=1 : pltlwt=1' % BLACK, '0.4 1', '1.2 1', 'plot']
msgs(L, sz, LAB + [(BLACK, 0.62, -0.45, 'circles: peak of the line')])
save8('Fig09a', L)
L = []
head(L, 'Fig09b', 'c856 section vs lambda: share > 1 band, least-damped modes, line peak (gamma = %s gamma_c)' % q, sz, (0.0, 1.0), (0.4, 1.2), 'force split, ^l^', 'frequency / local scale, f/aB', logx=False, yint='4.1', extra=('5.1',))
L.append('xfmt=f3.1 : yfmt=f3.1')
rows = sorted(FD, key=lambda r: r['lam'])
series(L, [r['lam'] for r in rows if r['band']], [r['band'][0] for r in rows if r['band']], MAG, 2, 0, 1, '1.4', 'band lower edge')
series(L, [r['lam'] for r in rows if r['band']], [r['band'][1] for r in rows if r['band']], MAG, 2, 0, 1, '1.4', 'band upper edge')
up = [(r['lam'], max(r['modes'], key=lambda t: t[0])[0]) for r in rows]
lo_ = [(r['lam'], m[0]) for r in rows for m in r['modes'] if 0.5 < m[0] < 0.8 and m[1] >= 2.5]
series(L, [a for a, b in up], [b for a, b in up], BLACK, 2, 1, 1, '1.4', 'upper (BM) mode')
series(L, [a for a, b in lo_], [b for a, b in lo_], BLACK, 2, 1, 1, '1.4', 'lower (RL-TM) mode, Q >= 2.5')
series(L, [r['lam'] for r in rows], [r['wa_bm'] for r in rows], BLUE, 1, 0, 1, '2.0', 'line peak (scale-invariant line)')
msgs(L, sz, [(BLACK, 0.03, 1.14, 'dotted: section modes'), (MAG, 0.42, 0.70, 'share > 1'), (BLUE, 0.03, 0.46, 'circles: peak of the line')])
save8('Fig09b', L)

# ---------------- Fig12a: cat SFOAE delay at 40 dB SPL ----------------
C = json.load(open(D8 + 'sfoae_cat_levels.json'))['conventions']['flat']
L = []
head(L, 'Fig12a', 'cat c856 N_SF vs frequency, 40 dB SPL, with measured power law', sz, (0.2, 40), (0, 30), 'frequency (kHz)', 'N[SF] (cycles)', xannot='0.3 1 3 10 30', yint='6.1')
series(L, [c['f_khz'] for c in C], [c['N_SF_40'] for c in C], RED, 3, 0, 11, '1.8', 'model 40 dB SPL')
fm = [3.0*(40/3.0)**(k/40) for k in range(41)]
series(L, fm, [3.32*f**0.44 for f in fm], GRAY, 2, 3, 1, '1.8', 'Shera and Guinan 2003, cat, 3.32 f^0.44')
msgs(L, sz, [(RED, 0.25, 28.6, 'cat model, 40 dB SPL'), (GRAY, 0.25, 26.6, 'measured (dashed)')])
save8('Fig12a', L)

# ---------------- Fig11: cat N_SF vs gamma/gamma_c (b_crit) ----------------
if os.path.exists(D8 + 'b_crit.json'):
    B = json.load(open(D8 + 'b_crit.json')); rows = sorted([r for r in B['rows'] if r['stable']], key=lambda r: r['target'])
    if len(rows) >= 2:
        L = []
        head(L, 'Fig11', 'cat N_SF vs gamma/gamma_c, c856 state, gamma scaled uniformly', sz, (0.84, 0.98), (0, 35), 'activity at the peak, ^g^/^g^[c]', 'N[SF] (cycles)',
             logx=False, yint='7.1', extra=('7.1',), xannot='0.84 0.86 0.88 0.90 0.92 0.94 0.96 0.98')
        cols = {'1.0': BLUE, '4.0': RED, '10.0': GREEN}; sym = {'1.0': 1, '4.0': 2, '10.0': 3}; SG = {'1.0': 3.3, '4.0': 6.0, '10.0': 9.2}
        for fk in ('1.0', '4.0', '10.0'):
            xs = [r['cf'][fk]['share']['gamma_ratio_at_bm_peak'] for r in rows]; ys = [r['cf'][fk]['N_SF_low'] for r in rows]
            series(L, xs, ys, cols[fk], 3, 0, sym[fk], '1.8', 'model %s kHz' % fk)
            series(L, [0.845, 0.975], [SG[fk], SG[fk]], cols[fk], 2, 3, 0, '1.2', 'Shera and Guinan 2003 cat, %s kHz' % fk)
        msgs(L, sz, [(BLUE, 0.845, 33.0, 'model 1 kHz (circles)'), (RED, 0.845, 30.4, 'model 4 kHz (triangles)'), (GREEN, 0.845, 27.8, 'model 10 kHz (inverted)'),
                     (BLACK, 0.845, 25.2, 'dashed: measured')])
        save8('Fig11', L)
print('figs written')
