"""make_tm_figs.py (run 3): PLT files for the TM transfer functions, in the style of Figs. 4 and 5
(figs_s3/make_rev_figs.py tf_stack; same frame geometry, lettering size cap(7.6, SINGLE), colours).
  figS_tmcat, figS_tmhum: TM displacement re stapes pressure, magnitude (top) and phase (bottom), at
      20, 40, 60, 80, 100 dB SPL plus passive.
  fig4cat_tm, fig4hum_tm: fig4cat / fig4hum with one added curve, TM at 20 dB SPL (dash-dot), legend updated.
Reads ../model/out/tm/tm_<species>.json (written by scripts/tm_tf.py). Run from figures/.
Render: tplt -r -m5 X.plt > X.eps ; tplt -m8 X.plt > X.svg"""
import json, math, os
D = '../model/out/tm/'
RED, BLUE, GRAY, TAN, GREEN, MAG, BLACK = 4, 1, 8, 6, 2, 5, 0
SINGLE, TARGET = 3.37, 0.100
def cap(nat_w, col): return TARGET*nat_w/col
s2 = '%.3f' % cap(7.6, SINGLE)
COLS = {'passive': GRAY, 20: RED, 40: MAG, 60: GREEN, 80: BLUE, 100: BLACK}
DASHDOT = 'lintype=5 : lindot=1'
def series(L, xs, ys, col, lintype=0, lwt='1.8', tag='', extra=''):
    L.append('pltcol=%d : pltype=2 : lintype=%d : symbol=1 : pltlwt=%s' % (col, lintype, lwt) + (' : ' + extra if extra else ''))
    if tag: L.append('; ' + tag)
    for x, y in zip(xs, ys): L.append('%.5g\t%.5g' % (x, y))
    L.append('plot')
    if extra: L.append('lindot=0')
def frame_top(L, lo, top):
    L += ['newframe', '; upper panel: magnitude', 'xllc=1.9 : yllc=4.0 : xlen=5.7 : ylen=3.0',
          'xanskp=-1 : xlabel= ', 'ymin=%g : ymax=%g : yint=6 : yhor=yes' % (lo, top), 'ylabel=displacement (dB)']
def frame_bot(L, name, desc, ymin):
    L += ['; %s.plt - %s' % (name, desc), ';', 'head=0 : clip=1', 'annsiz=-%s : labsiz=-%s' % (s2, s2),
          'xmin=0.1 : xmax=1.6 : xcyc=%.4f' % math.log10(16), 'xannot= 0.1 0.2 0.5 1',
          '; lower panel: phase', 'xllc=1.9 : yllc=1.2 : xlen=5.7 : ylen=2.4',
          'ymin=%g : ymax=0 : yint=%g.1 : yhor=yes' % (ymin, -ymin), 'xlabel=frequency / CF', 'ylabel=phase (cycles)']
def msg(L, items):
    L.append('msgdat=yes : mhal=lef : mval=half : msgsiz=-%s' % s2)
    for col, x, y, t in items: L += ['msgcol=%d' % col, '%g %g "%s"' % (x, y, t)]
FIGOUT = os.environ.get('FIGOUT', 'plt/'); os.makedirs(FIGOUT, exist_ok=True)
def save(name, L): open(FIGOUT + name + '.plt', 'w').write('\n'.join(L) + '\n')
def load(sp):
    p = json.load(open(D + 'tm_%s.json' % sp))['places'][0]; cf = p['cf_khz']
    return p, cf, [v/cf for v in p['f_khz']]
def by(p, k, lv): return [r for r in p[k] if r['level'] == lv][0]
def unwr(ph): off = round(ph[0]); return [v - off for v in ph]

ORDER = ('passive', 100, 80, 60, 40, 20)          # draw 20 dB last (on top), as in Fig. 4
LAY = {'cat': dict(ymin=-5, leg_y0=-2.5, top_msgs=None), 'human': dict(ymin=-4, leg_y0=-1.55, top_msgs=None)}
def tm_supp(sp, name):
    p, cf, f = load(sp)
    allv = [v for r in p['tm'] for v in r['mag_db']]
    top = 10*math.ceil(max(allv)/10) + 10; lo = top - 60
    ymin = LAY[sp]['ymin']; L = []
    frame_bot(L, name, '%s TM displacement re stapes pressure, magnitude (top) and phase (bottom), CF %.2f kHz' % (sp, cf), ymin)
    for lv in ORDER:
        series(L, f, unwr(by(p, 'tm', lv)['phase_cyc']), COLS[lv], 3 if lv == 'passive' else 0, '1.8', 'TM phase %s' % lv)
    y0 = LAY[sp]['leg_y0']; dy = 0.55
    msg(L, [(COLS[lv], 0.11, y0 - dy*k, t) for k, (lv, t) in enumerate(((20, 'TM, 20 dB SPL'), (40, 'TM, 40'), (60, 'TM, 60'), (80, 'TM, 80'), (100, 'TM, 100')))])
    frame_top(L, lo, top)
    for lv in ORDER:
        series(L, f, by(p, 'tm', lv)['mag_db'], COLS[lv], 3 if lv == 'passive' else 0, '1.8', 'TM %s' % lv)
    msg(L, [(BLACK, 0.11, top - 4, 'TM displacement'), (GRAY, 0.11, top - 12, 'passive (dashed)')])
    save(name, L); return top, lo

def fig4_tm(sp, name):
    """figs_s3/make_rev_figs.py tf_stack verbatim, plus TM at 20 dB SPL (dash-dot)."""
    p, cf, f = load(sp)
    cols = {'passive': GRAY, 20: RED, 60: GREEN, 80: BLUE, 100: BLACK}
    allv = [v for r in p['bm'] for v in r['mag_db']] + list(p['rl'][1]['mag_db'])
    top = 10*math.ceil(max(allv)/10) + 10; lo = top - 60
    L = ['; %s.plt - %s BM, RL and TM displacement re stapes pressure (top) and BM phase (bottom), CF %.2f kHz' % (name, sp, cf), ';',
         'head=0 : clip=1', 'annsiz=-%s : labsiz=-%s' % (s2, s2),
         'xmin=0.1 : xmax=1.6 : xcyc=%.4f' % math.log10(16), 'xannot= 0.1 0.2 0.5 1',
         '; lower panel: phase', 'xllc=1.9 : yllc=1.2 : xlen=5.7 : ylen=2.4',
         'ymin=-4 : ymax=0 : yint=4.1 : yhor=yes', 'xlabel=frequency / CF', 'ylabel=phase (cycles)']
    order = {'passive': 0, 100: 1, 80: 2, 60: 3, 20: 4}
    for r in sorted([r for r in p['bm'] if r['level'] != 40], key=lambda r: order[r['level']]):
        lv = r['level']; series(L, f, unwr(r['phase_cyc']), cols[lv], 3 if lv == 'passive' else 0, '1.8', 'BM phase %s' % lv)
    msg(L, [(col, 0.11, -1.9 - 0.55*k, txt) for k, (col, txt) in enumerate(((RED, 'BM, 20 dB SPL'), (GREEN, 'BM, 60'), (BLUE, 'BM, 80'), (BLACK, 'BM, 100')))])
    frame_top(L, lo, top)
    for r in sorted([r for r in p['bm'] if r['level'] != 40], key=lambda r: order[r['level']]):
        lv = r['level']; series(L, f, r['mag_db'], cols[lv], 3 if lv == 'passive' else 0, '1.8', 'BM %s' % lv)
    series(L, f, p['rl'][1]['mag_db'], RED, 1, '1.4', 'RL 20 dB SPL')
    series(L, f, by(p, 'tm', 20)['mag_db'], RED, 5, '1.4', 'TM 20 dB SPL (dash-dot)', extra='lindot=1')
    msg(L, [(RED, 0.11, top - 4, 'RL, 20 dB SPL (dotted)'), (RED, 0.11, top - 12, 'TM, 20 dB SPL (dash-dot)'),
            (GRAY, 0.11, top - 20, 'passive (dashed)')])
    save(name, L); return top, lo

if __name__ == '__main__':
    print(tm_supp('cat', 'figS_tmcat'), tm_supp('human', 'figS_tmhum'))
    print(fig4_tm('cat', 'fig4cat_tm'), fig4_tm('human', 'fig4hum_tm'))
