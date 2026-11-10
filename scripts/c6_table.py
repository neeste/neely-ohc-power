"""Run 8: Supplement Tables SVI and SVII at c856 (the 18 metrics along the uniform gamma scan, g = T/0.9501, with the
fixed obs45) and the zero crossings, from out/c6_scan.json. Writes out/c6_table.md and out/c6_cross.json."""
import json
d = json.load(open('out/c6_scan.json')); rows = sorted(d['rows'], key=lambda r: -r['target'])
W = dict(tt=(32, 48), sl=(3, 9), mp=(-0.75, -0.25), ph=(2.442, 3.842), fwd=(0, 12.5), exc=(2, 14), afd=(2, 14), gr33=(1.4, 2.6),
         mdt=(18, 32), vsl=(1.4, 2.6), vdt=(14, 36), bmg=(20, 60), s2_tt=(32, 48), s2_sl=(3, 9), s2_mp=(-0.75, -0.25),
         s2_ph=(2.442, 3.842), s2_fg=(0, 13), s2_exc=(2, 14))
NM = dict(tt='Tip-to-tail difference, dB (4, SI)', sl='Tail slope, dB/oct (4, SI)', mp='Second-map place, oct re BF (5, SI)',
          ph='Phase step, rad (5, SI)', fwd='Forward gain, dB (6, SI)', exc='Point-source excursion, dB (6, SI)',
          afd='Distributed-source excursion, dB (6, SI)', gr33='Suppression growth slope, dB/dB (2, SI)',
          mdt='BM minus neural suppression threshold, probe 40 or 50 dB above threshold, dB (3, SI)',
          vsl='Vertical-protocol slope, dB/dB (2, SI)', vdt='Vertical-protocol threshold difference, dB (3, SI)',
          bmg='BM-gain metric, dB (SI, x = 0.55)', s2_tt='Tip-to-tail difference, dB (4, P)', s2_sl='Tail slope, dB/oct (4, P)',
          s2_mp='Second-map place, oct re BF (5, P)', s2_ph='Phase step, rad (5, P)', s2_fg='Forward gain, dB (6, P)',
          s2_exc='Point-source excursion, dB (6, P), corrected obs45')
K = list(NM)
L = ['# Supplement Tables SVI and SVII at c856 (run 8)', '',
     'State c856 (combo_r7b.json, lambda 0.8562) with the OHC activity scaled uniformly by g = T/0.9501 (0.9501 = the uniform stage-1 gamma/gamma_c of c856; T = 0.9501 is the state itself). Stage-1 gamma p[8] and the stage-2 profile both scaled; lambda and the 24-point gamma_c table fixed. Scored as combo.py (run4 c6_state.py with the fixed obs45; the profiled-line point-source excursion is the corrected value). Margins in parentheses; -4 marks a read-out without a finite value (n.r.).', '',
     '| Metric | Window | ' + ' | '.join('T = %.4f' % r['target'] if i == 0 else '%.3f' % r['target'] for i, r in enumerate(rows)) + ' |',
     '|---|---|' + '---|'*len(rows)]
for k in K:
    cells = []
    for r in rows:
        v = r['values'].get(k); m = r['margins'][k]
        cells.append('n.r. (-4)' if m <= -3.999 else '%.4g (%+.3f)' % (v, m))
    L.append('| %s | %g to %g | %s |' % (NM[k], W[k][0], W[k][1], ' | '.join(cells)))
L.append('| Score | | ' + ' | '.join('%+.4f' % r['score'] for r in rows) + ' |')
L.append('| attained by | | ' + ' | '.join(', '.join(w[0] for w in r['worst'] if abs(w[1] - r['score']) < 0.002) for r in rows) + ' |')
L.append('| J_ftc | | ' + ' | '.join('%.2f' % r['J_ftc'] for r in rows) + ' |')
L.append('| J_c | | ' + ' | '.join('%.4f' % r['obj'] for r in rows) + ' |')
L.append('| Stage-1 gamma/gamma_c | | ' + ' | '.join('%.4f' % r['stage1_ratio'] for r in rows) + ' |')
L.append('| Stable (quiet profiled line) | | ' + ' | '.join('yes' if r['cert'] else 'NO' for r in rows) + ' |')
L += ['', '## Zero crossings (linear interpolation in T between scan rows)', '', '| Metric | At T = 0.9501 | Crossing |', '|---|---|---|']
cross = {}
for k in K:
    m0 = rows[0]['margins'][k]; st = 'inside (%+.3f)' % m0 if m0 >= 0 else 'outside (%+.3f)' % m0; c = None
    for a, b in zip(rows[:-1], rows[1:]):
        ma, mb = a['margins'][k], b['margins'][k]
        if (ma >= 0) != (mb >= 0):
            if mb <= -3.999 or ma <= -3.999: c = 'read-out fails, %.3f to %.3f' % (a['target'], b['target'])
            else:
                t = a['target'] + (b['target'] - a['target'])*ma/(ma - mb); c = ('leaves' if ma >= 0 else 'enters') + ' at T = %.3f' % t
            break
    if c is None: c = 'stays inside' if m0 >= 0 else 'stays outside'
    cross[k] = c; L.append('| %s | %s | %s |' % (NM[k], st, c))
open('out/c6_table.md', 'w').write('\n'.join(L) + '\n'); json.dump(cross, open('out/c6_cross.json', 'w'), indent=1)
print('\n'.join(L))
