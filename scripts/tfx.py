"""Run 9: Table V (human rows) and Sec. III.C human numbers from a tf_human.json (step3/tf.py output), as results_facts 2.3:
gains = active minus passive BM displacement re stapes pressure at the grid frequency nearest CF; BM at CF re BM at 0.1 CF
(20 dB SPL); RL gain = RL active (20 dB) minus RL passive (RL minus BM passive in brackets); BM phase at CF re 0.1 CF;
active minus passive BM at 1.6 CF; BM at CF re passive by level.  Usage: tfx9.py tf_human.json [out.json]"""
import sys, json, numpy as np
T = json.load(open(sys.argv[1])); rows = []
for p in T['places']:
    f = np.array(p['f_khz'])/p['cf_khz']; c = int(np.argmin(np.abs(f - 1))); lo = int(np.argmin(np.abs(f - 0.1))); h = int(np.argmin(np.abs(f - 1.6)))
    bm = {r['level']: r for r in p['bm']}; rl = {r['level']: r for r in p['rl']}
    m = lambda d, L, i: d[L]['mag_db'][i]; ph = lambda d, L, i: d[L]['phase_cyc'][i]
    rows.append(dict(x=p['x'], cf_khz=p['cf_khz'], f_cf_grid=float(f[c]),
        bm_gain={str(L): m(bm, L, c) - m(bm, 'passive', c) for L in (20, 40, 60, 80, 100)},
        bm_cf_re_01cf_20=m(bm, 20, c) - m(bm, 20, lo), rl_gain_20=m(rl, 20, c) - m(rl, 'passive', c), rl_re_bmpass_20=m(rl, 20, c) - m(bm, 'passive', c),
        bm_phase_cf_re_01cf={'20': ph(bm, 20, c) - ph(bm, 20, lo), 'passive': ph(bm, 'passive', c) - ph(bm, 'passive', lo)},
        act_minus_pas_16cf_20=m(bm, 20, h) - m(bm, 'passive', h)))
for r in rows:
    print('x %.4f CF %.2f kHz | BM gain 20 %.1f | BM CF re 0.1CF %+.1f | RL gain %.1f [%.1f] | BM gain 40/60/80/100 %s | phase 20 %.2f passive %.2f | 1.6CF act-pas %.1f' % (
        r['x'], r['cf_khz'], r['bm_gain']['20'], r['bm_cf_re_01cf_20'], r['rl_gain_20'], r['rl_re_bmpass_20'], '/'.join('%.1f' % r['bm_gain'][k] for k in ('40', '60', '80', '100')),
        r['bm_phase_cf_re_01cf']['20'], r['bm_phase_cf_re_01cf']['passive'], r['act_minus_pas_16cf_20']))
if len(sys.argv) > 2: json.dump(rows, open(sys.argv[2], 'w'), indent=1)
