"""make_rev_figs.py - PLT files for the revised manuscript's new and
regenerated figures. Reads JSON written by the revision scripts in
../model/out (run from figures/). Human figures and the shared helpers; the cat figures
of the c856 state are made by make_figs_cat.py. Render: tplt -r -m5 X.plt > X.eps ; tplt -m8 X.plt > X.svg"""
import json, math, os
R = os.environ.get('R', '../model/out/')   # JSON written by the scripts (model/out/)
import os
HUMDIR = os.environ.get('HUMDIR', '../model/')
# run 9 label moves: fig2a "human" x 5 -> 12, fig9b labels x 50 -> 42, fig8b key x 0.40 -> 0.04 (inside the frame)
IPH = int(os.environ.get('IPH', '2'))   # human transfer-function place: the 1-kHz CF node (third place of scripts/tf.py human)
RED, BLUE, GRAY, TAN, GREEN, MAG, BLACK = 4, 1, 8, 6, 2, 5, 0
SINGLE, DOUBLE, TARGET = 3.37, 6.69, 0.100
def cap(nat_w, col): return TARGET*nat_w/col
def series(L, xs, ys, col, pltype=2, lintype=0, sym=1, lwt='1.8', tag=''):
    L.append('pltcol=%d : pltype=%d : lintype=%d : symbol=%d : pltlwt=%s' % (col, pltype, lintype, sym, lwt))
    if tag: L.append('; ' + tag)
    for x, y in zip(xs, ys):
        if y is None or (isinstance(y, float) and math.isnan(y)): continue
        L.append('%.5g\t%.5g' % (x, y))
    L.append('plot')
def head(L, name, desc, sz, xr, yr, xlab, ylab, logx=True, xannot=None, yint='', extra=()):
    L += ['; %s.plt - %s' % (name, desc), ';', 'head=0 : clip=1',
          'xllc=1.9 : yllc=1.5 : xlen=5.7 : ylen=4.0',
          'annsiz=-%s : labsiz=-%s' % (sz, sz)]
    if logx:
        L.append('xmin=%g : xmax=%g : xcyc=%.4f' % (xr[0], xr[1], math.log10(xr[1]/xr[0])))
    else:
        L.append('xmin=%g : xmax=%g : xint=%s' % (xr[0], xr[1], extra[0] if extra else '8'))
    if xannot: L.append('xannot= ' + xannot)
    L.append('ymin=%g : ymax=%g : yint=%s : yhor=yes' % (yr[0], yr[1], yint or '5'))
    L += ['xlabel=' + xlab, 'ylabel=' + ylab]
def msgs(L, sz, items, mhal='lef'):
    L.append('msgdat=yes : mhal=%s : mval=half : msgsiz=-%s' % (mhal, sz))
    for col, x, y, t in items:
        L.append('msgcol=%d' % col); L.append('%g %g "%s"' % (x, y, t))
FIGOUT = os.environ.get('FIGOUT', 'plt/'); os.makedirs(FIGOUT, exist_ok=True)
def save(name, L): open(FIGOUT + name + '.plt', 'w').write('\n'.join(L) + '\n')
sz = "%.3f" % cap(7.6, SINGLE)

# ---------------- profiles (Methods) ----------------
P = json.load(open(R + 'profiles.json'))
for nm, key, ylab, yr, yint, logy in (('fig2a', 'rho', 'damping per cycle, ^r^', (0, 2), '4.1', False),
                                      ('fig2b', 'nu', 'distance from criticality, ^n^', (-3, 0), '3', True)):
    L = []
    head(L, nm, 'place profiles, cat and human', sz, (0.1, 50), yr, 'local scale a(x) x B (kHz)', ylab,
         xannot='0.1 0.3 1 3 10 30', yint=yint)
    L.append('yfmt=f3.1' if key == 'rho' else 'yannot= -3,0.001 -2,0.01 -1,0.1 0,1')
    for sp, col in (('cat', RED), ('human', BLUE)):
        d = P[sp]; xs = [a for a in d['map_khz']]
        ys = d[key]
        if key == 'nu':
            ys = [math.log10(max(v, 1e-6)) if t > 0.01 else None for v, t in zip(ys, d['taper'])]
        if key == 'rho':
            ys = [v if v <= 2.2 else None for v in ys]
        pts = [(x, y) for x, y in zip(xs, ys) if y is not None and 0.1 <= x <= 50]
        series(L, [p[0] for p in pts], [p[1] for p in pts], col, 2, 0 if sp == 'cat' else 3, 1, '2.0', sp)
    msgs(L, sz, [(RED, 10, (0.95 if key == 'rho' else -0.75), 'cat'), (BLUE, (12.0 if key == 'rho' else 0.15), (1.6 if key == 'rho' else -2.25), 'human')])
    save(nm, L)

# ---------------- transfer functions ----------------
def tf_fig(sp, ip, tag):
    T = json.load(open(R + 'tf_%s.json' % sp)); p = T['places'][ip]
    cf = p['cf_khz']; f = [v/cf for v in p['f_khz']]
    cols = {'passive': GRAY, 20: RED, 40: MAG, 60: GREEN, 80: BLUE, 100: BLACK}
    for part in ('mag', 'phase'):
        L = []; nm = 'fig4%s_%s' % (tag, part)
        if part == 'mag':
            allv = [v for r in p['bm'] for v in r['mag_db']] + [v for v in p['rl'][1]['mag_db']]
            top = 10*math.ceil(max(allv)/10); yr = (top-50, top)
            head(L, nm, '%s BM and RL displacement re stapes pressure, CF %.2f kHz' % (sp, cf), sz,
                 (0.1, 1.6), yr, 'frequency / CF', 'displacement (dB)', xannot='0.1 0.2 0.5 1', yint='5')
            for r in p['bm']:
                lv = r['level']
                if lv in (40,): continue
                series(L, f, r['mag_db'], cols[lv], 2, 3 if lv == 'passive' else 0, 1, '1.8', 'BM %s' % lv)
            series(L, f, p['rl'][1]['mag_db'], RED, 2, 1, 1, '1.4', 'RL 20 dB SPL')
            msgs(L, sz, [(BLACK, 0.11, top-3, 'BM: 20, 60, 80, 100 dB SPL'), (RED, 0.11, top-8, 'RL, 20 dB SPL (dotted)'),
                         (GRAY, 0.11, top-13, 'passive (dashed)')])
        else:
            head(L, nm, '%s BM phase, CF %.2f kHz' % (sp, cf), sz, (0.1, 1.6), (-4, 0),
                 'frequency / CF', 'BM phase (cycles)', xannot='0.1 0.2 0.5 1', yint='4.1')
            for r in p['bm']:
                lv = r['level']
                if lv in (40,): continue
                ph = r['phase_cyc']; off = round(ph[0])
                series(L, f, [v - off for v in ph], cols[lv], 2, 3 if lv == 'passive' else 0, 1, '1.8', 'BM %s' % lv)
        save(nm, L)
tf_fig('cat', 1, 'a'); tf_fig('human', IPH, 'b')

# ---------------- human tuning curves ----------------
H = json.load(open(R + 'ftc_human.json'))
L = []
head(L, 'fig5', 'human tuning curves, quiet line, re tip', sz, (0.1, 8), (-10, 70), 'frequency (kHz)',
     'threshold re tip (dB)', xannot='0.1 0.3 1 3', yint='8.1')
for c, col in zip(H[:4], (RED, MAG, GREEN, BLUE)):
    pts = [(f, t) for f, t in zip(c['f'], c['thr']) if t <= 70]
    series(L, [q[0] for q in pts], [q[1] for q in pts], col, 2, 0, 1, '1.8', '%g kHz' % c['f_khz'])
save('fig5', L)

# ---------------- SFOAE delays (Fig. 7): 40-dB model curves against Shera and Guinan (2003) ----------------
C = json.load(open(R + 'sfoae_cat_levels.json'))['conventions']['flat']
L = []
head(L, 'fig6a', 'cat N_SF vs frequency, 40 dB SPL, with measured power law', sz, (0.2, 40), (0, 30), 'frequency (kHz)', 'N[SF] (cycles)',
     xannot='0.3 1 3 10 30', yint='6.1')
series(L, [c['f_khz'] for c in C], [c['N_SF_40'] for c in C], RED, 3, 0, 11, '1.8', 'model 40 dB SPL')
fm = [3.0*(40/3.0)**(k/40) for k in range(41)]
series(L, fm, [3.32*f**0.44 for f in fm], GRAY, 2, 3, 1, '1.8', 'Shera and Guinan 2003, cat, 3.32 f^0.44')
msgs(L, sz, [(RED, 0.25, 28.6, 'cat model, 40 dB SPL'), (GRAY, 0.25, 26.6, 'measured (dashed)')])
save('fig6a', L)
S = json.load(open(R + 'sfoae_human.json'))
n20 = [v[1] for v in S['n20']]; n40 = [v[1] for v in S['n40']]
L = []
head(L, 'fig6b', 'human N_SF vs frequency, 20 and 40 dB SPL, with measured data', sz, (0.4, 10), (0, 35), 'frequency (kHz)', 'N[SF] (cycles)',
     xannot='0.5 1 2 4 8', yint='7.1')
series(L, S['f'], n40, BLUE, 3, 0, 11, '1.8', 'model 40 dB SPL')
series(L, S['f'], n20, BLUE, 2, 1, 1, '1.4', 'model 20 dB SPL')
fm = [1.0*(10/1.0)**(k/40) for k in range(41)]
series(L, fm, [11.0*f**0.37 for f in fm], GRAY, 2, 3, 1, '1.8', 'Shera and Guinan 2003, human, 11.0 f^0.37')
sgf = [0.55, 0.75, 1.1, 1.6, 2.3, 3.2, 4.6, 6.6, 9.2]; sgn = [6.4, 8.4, 11.5, 13.5, 14.6, 16.9, 18.9, 23, 27]
series(L, sgf, sgn, GRAY, 1, 0, 1, '1.5', 'SGO2002 Fig. 2 binned, doubled')
msgs(L, sz, [(BLUE, 0.45, 32.5, 'model 40 dB SPL (solid)'), (BLUE, 0.45, 29.5, 'model 20 dB SPL (dotted)'),
             (GRAY, 0.45, 26.5, 'measured (dashed, circles)')])
save('fig6b', L)

# ---------------- wave V ----------------
import numpy as np
ck = json.load(open(HUMDIR + 'abr5_ckpt.json'))
lat = np.array(ck['lat']); keys = ck['keys']; a = ck['readout']['a']
fs = sorted(set(k[0] for k in keys)); spl = sorted(set(k[1] for k in keys))
szd = "%.3f" % cap(7.6, SINGLE)
L = []
L += ['; fig7.plt - human wave V latency, model (solid) vs Neely et al. 1988 (dashed)', ';',
      'head=0 : clip=1', 'xllc=1.9 : yllc=1.5 : xlen=5.7 : ylen=4.0',
      'annsiz=-%s : labsiz=-%s' % (szd, szd), 'xmin=0.25 : xmax=8 : xcyc=1.505', 'xannot= 0.25 0.5 1 2 4 8',
      'ymin=6 : ymax=22 : yint=8 : yhor=yes', 'xlabel=frequency (kHz)', 'ylabel=wave V latency (ms)']
for s in spl:
    series(L, fs, [5+12.9*5**(-s/100)*f**(-0.413) for f in fs], TAN, 2, 3, 1, '1.1', 'NNGJ %d' % s)
for s in spl:
    series(L, fs, [lat[i]+a for i, k in enumerate(keys) if k[1] == s], BLUE, 2, 0, 1, '1.8', 'model %d' % s)
msgs(L, szd, [(TAN, 7.6, 21, 'Neely et al. 1988 (dashed)'), (BLUE, 7.6, 19.6, 'model (solid)')], mhal='rig')
save('fig7', L)

# ---------------- nu and level ----------------
N6 = json.load(open(R + 'fig6_rev.json'))
m1, m2 = N6['m1'], N6['m2']
u = [-math.log10(m['nu']) for m in m1]
L = []
L += ['; fig9a.plt - shear gain, BM gain, BM group delay vs nu, human 1-kHz place, quiet', ';',
      'head=0 : clip=1', 'xllc=1.9 : yllc=1.5 : xlen=5.7 : ylen=4.0', 'annsiz=-%s : labsiz=-%s' % (sz, sz),
      'xmin=0.5 : xmax=2.6 : xint=7.1', 'ymin=-10 : ymax=50 : yint=6 : yhor=yes',
      'xannot= %.4f,0.3 %.4f,0.1 %.4f,0.03 %.4f,0.01 %.4f,0.003' % tuple(-math.log10(v) for v in (0.3, 0.1, 0.03, 0.01, 0.003)),
      'xlabel=distance from criticality, ^n^', 'ylabel=gain (dB) or delay (cycles)']
series(L, u, [m['gain_shear'] for m in m1], RED, 3, 0, 11, '2.0', 'shear gain')
series(L, u, [m['gain_bm'] for m in m1], RED, 3, 3, 1, '1.6', 'BM gain')
series(L, u, [m['delay_cycles'] for m in m1], BLUE, 3, 0, 10, '2.0', 'delay')
msgs(L, sz, [(RED, 0.6, 46, 'shear gain (dB)'), (RED, 0.6, 41.5, 'BM gain (dB, dashed)'), (BLUE, 0.6, 37, 'BM delay (cycles)')])
save('fig9a', L)
L = []
L += ['; fig9b.plt - the same vs level, human abr5 state', ';', 'head=0 : clip=1',
      'xllc=1.9 : yllc=1.5 : xlen=5.7 : ylen=4.0', 'annsiz=-%s : labsiz=-%s' % (sz, sz),
      'xmin=20 : xmax=100 : xint=8', 'ymin=-10 : ymax=30 : yint=8 : yhor=yes',
      'xlabel=stimulus level (dB SPL)', 'ylabel=gain (dB) or delay (cycles)']
series(L, [m['spl'] for m in m2], [m['gain_shear'] for m in m2], RED, 3, 0, 11, '2.0', 'shear gain')
series(L, [m['spl'] for m in m2], [m['gain_bm'] for m in m2], RED, 3, 3, 1, '1.6', 'BM gain')
series(L, [m['spl'] for m in m2], [m['delay_cycles'] for m in m2], BLUE, 3, 0, 10, '2.0', 'delay')
msgs(L, sz, [(RED, 42, 27, 'shear gain (dB)'), (RED, 42, 23.5, 'BM gain (dB, dashed)'), (BLUE, 42, 20, 'BM delay (cycles)')])
save('fig9b', L)
print('done')

# ---------------- cat tuning curves (manuscript Fig. 3) ----------------
if os.path.exists(R + 'fig2_curves.json'):   # needs the individual measured curves (not redistributed); see make_fig03.py
    D2 = json.load(open(R + 'fig2_curves.json'))['pairs']
    L = []
    head(L, 'fig3tc', 'cat tuning curves, measured (gray) and model (red)', sz, (0.1, 30), (0, 90),
         'frequency (kHz)', 'threshold (dB SPL)', xannot='0.1 0.3 1 3 10 30', yint='9.1')
    for p_ in D2:
        for xs, ys in p_['data']: series(L, xs, ys, 7, 2, 0, 1, '1.2', 'measured CF %.2f' % p_['data_cf'])
    for p_ in D2:
        for xs, ys in p_['model']: series(L, xs, ys, RED, 2, 0, 1, '2.0', 'model tip %.2f' % p_['model_tip'])
    L += ['pltcol=%d : pltype=2 : lintype=1 : pltlwt=1' % GRAY, '2 0', '2 90', 'plot']
    # double-column layout: wider frame, lettering sized for 6.69 in
    szw = '%.3f' % cap(9.4, DOUBLE)
    L = [('xllc=1.6 : yllc=1.4 : xlen=7.6 : ylen=4.0' if ln.startswith('xllc=') else
          ('annsiz=-%s : labsiz=-%s' % (szw, szw) if ln.startswith('annsiz=') else ln)) for ln in L]
    save('fig3tc', L)

# ---------------- energy accounting (manuscript Fig. 8a, 8b) ----------------
E3 = json.load(open(R + 'fig3_curves.json'))
lp = E3['left']; mD = max(lp['D'])
L = []
head(L, 'fig8a', 'cat 1 kHz: dissipation and OHC power vs place', sz, (-0.8, 0.8), (0, 1.0),
     'octaves apical of the shear peak', 'power density (normalized)', logx=False, yint='5.1', extra=('4.1',))
L.append('xfmt=f4.1 : yfmt=f3.1')
series(L, lp['x'], [v/mD for v in lp['D']], BLUE, 2, 0, 1, '2.0', 'D')
series(L, lp['x'], [v/mD for v in lp['po']], RED, 2, 0, 1, '2.0', 'p_ohc')
msgs(L, sz, [(BLUE, -0.75, 0.9, 'dissipation'), (RED, -0.75, 0.8, 'OHC power')])
save('fig8a', L)
L = []
head(L, 'fig8b', 'OHC share of dissipation, cat and human, 1 and 4 kHz', sz, (-0.8, 0.8), (0, 1.6),
     'octaves apical of the shear peak', 'OHC power / dissipation', logx=False, yint='8.1', extra=('4.1',))
L.append('xfmt=f4.1 : yfmt=f3.1')
for sp, col in (('cat_share', RED), ('human_share', BLUE)):
    for f_, lt in (('1.0', 0), ('4.0', 3)):
        v = E3[sp][f_]; series(L, v['x'], v['share'], col, 2, lt, 1, '2.0', '%s %s kHz' % (sp, f_))
L += ['pltcol=%d : pltype=2 : lintype=1 : pltlwt=1' % GRAY, '-0.8 1', '0.8 1', 'plot']
msgs(L, sz, [(BLUE, -0.75, 1.52, 'human'), (RED, -0.75, 1.40, 'cat'), (BLACK, 0.04, 1.52, '1 kHz solid'), (BLACK, 0.04, 1.40, '4 kHz dashed')])
save('fig8b', L)
print('done2')

# ---------------- transfer functions, stacked panels with shared x (Neely & Rasetshwane 2017 style) ----------------
def tf_stack(sp, ip, name):
    T_ = json.load(open(R + 'tf_%s.json' % sp)); p = T_['places'][ip]
    cf = p['cf_khz']; f = [v/cf for v in p['f_khz']]
    cols = {'passive': GRAY, 20: RED, 60: GREEN, 80: BLUE, 100: BLACK}
    allv = [v for r in p['bm'] for v in r['mag_db']] + list(p['rl'][1]['mag_db'])
    top = 10*math.ceil(max(allv)/10) + 10; lo = top - 60
    s2 = '%.3f' % cap(7.6, SINGLE)
    L = ['; %s.plt - %s BM and RL displacement re stapes pressure (top) and BM phase (bottom), CF %.2f kHz' % (name, sp, cf), ';',
         'head=0 : clip=1', 'annsiz=-%s : labsiz=-%s' % (s2, s2),
         'xmin=0.1 : xmax=1.6 : xcyc=%.4f' % math.log10(16), 'xannot= 0.1 0.2 0.5 1',
         '; lower panel: phase', 'xllc=1.9 : yllc=1.2 : xlen=5.7 : ylen=2.4',
         'ymin=-4 : ymax=0 : yint=4.1 : yhor=yes', 'xlabel=frequency / CF', 'ylabel=phase (cycles)']
    for r in sorted([r for r in p['bm'] if r['level'] != 40], key=lambda r: {'passive': 0, 100: 1, 80: 2, 60: 3, 20: 4}[r['level']]):
        lv = r['level']
        ph = r['phase_cyc']; off = round(ph[0])
        series(L, f, [v - off for v in ph], cols[lv], 2, 3 if lv == 'passive' else 0, 1, '1.8', 'BM phase %s' % lv)
    L += ['msgdat=yes : mhal=lef : mval=half : msgsiz=-%s' % s2]
    for k, (col, txt) in enumerate(((RED, 'BM, 20 dB SPL'), (GREEN, 'BM, 60'), (BLUE, 'BM, 80'), (BLACK, 'BM, 100'))):
        L += ['msgcol=%d' % col, '%g %g "%s"' % (0.11, -1.9 - 0.55*k, txt)]
    L += ['newframe', '; upper panel: magnitude', 'xllc=1.9 : yllc=4.0 : xlen=5.7 : ylen=3.0',
          'xanskp=-1 : xlabel= ', 'ymin=%g : ymax=%g : yint=6 : yhor=yes' % (lo, top), 'ylabel=displacement (dB)']
    order = {'passive': 0, 100: 1, 80: 2, 60: 3, 20: 4}
    for r in sorted([r for r in p['bm'] if r['level'] != 40], key=lambda r: order[r['level']]):
        lv = r['level']
        series(L, f, r['mag_db'], cols[lv], 2, 3 if lv == 'passive' else 0, 1, '1.8', 'BM %s' % lv)
    series(L, f, p['rl'][1]['mag_db'], RED, 2, 1, 1, '1.4', 'RL 20 dB SPL')
    L += ['msgdat=yes : mhal=lef : mval=half : msgsiz=-%s' % s2,
          'msgcol=%d' % RED, '%g %g "RL, 20 dB SPL (dotted)"' % (0.11, top - 4),
          'msgcol=%d' % GRAY, '%g %g "passive (dashed)"' % (0.11, top - 12)]
    save(name, L)
tf_stack('cat', 1, 'fig4cat'); tf_stack('human', IPH, 'fig4hum')
print('done3')
