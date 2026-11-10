"""tm_post.py (run 3): TM/BM ratio at the peak, from tm_cat.json and tm_human.json.
Neither BM displacement re stapes pressure has a dominant tip maximum at these places (the low-frequency
tail is higher), so the peak is taken as the TM tip: the maximum of |X_t| for f >= 0.5 CF. Reported per
level: f_peak/CF, |X_t| and |X_b| there, TM/BM ratio (dB and phase in cycles), and the same ratio at the
grid frequency nearest CF and at the BM local (tip) maximum nearest CF when one exists in 0.7 to 1.3 CF.
Writes tm_peaks.json and tm_peaks.md."""
import json, math
import numpy as np
D = 'out/tm/'
out = {}; md = ['| species (CF) | level | f_peak / CF | TM peak (dB) | BM at TM peak (dB) | TM/BM at peak (dB) | TM/BM phase (cyc) | TM/BM at CF (dB) | BM tip max: f/CF, TM/BM (dB) |', '|---|---|---|---|---|---|---|---|---|']
for sp in ('cat', 'human'):
    p = json.load(open(D + 'tm_%s.json' % sp))['places'][0]
    cf = p['cf_khz']; f = np.array(p['f_khz'])/cf; rows = []
    for b, t in zip(p['bm'], p['tm']):
        mb, mt = np.array(b['mag_db']), np.array(t['mag_db']); pb, pt = np.array(b['phase_cyc']), np.array(t['phase_cyc'])
        w = np.where(f >= 0.5)[0]; i = int(w[np.argmax(mt[w])]); edge = bool(i == w[0])
        c = int(np.argmin(np.abs(f - 1.0)))
        loc = [k for k in range(1, len(f) - 1) if 0.7 <= f[k] <= 1.3 and mb[k] > mb[k-1] and mb[k] > mb[k+1]]
        kb = min(loc, key=lambda k: abs(math.log(f[k]))) if loc else None
        ph = lambda k: float(((pt[k] - pb[k] + 0.5) % 1.0) - 0.5)
        r = dict(level=b['level'], f_peak_re_cf=float(f[i]), f_peak_khz=float(f[i]*cf), peak_at_window_edge=edge,
                 tm_peak_db=float(mt[i]), bm_at_peak_db=float(mb[i]), tm_bm_db=float(mt[i] - mb[i]), tm_bm_phase_cyc=ph(i),
                 tm_bm_at_cf_db=float(mt[c] - mb[c]), f_cf_grid=float(f[c]),
                 bm_tip=None if kb is None else dict(f_re_cf=float(f[kb]), bm_db=float(mb[kb]), tm_bm_db=float(mt[kb] - mb[kb])))
        rows.append(r)
        lv = 'passive' if b['level'] == 'passive' else '%d dB SPL' % b['level']
        md.append('| %s (%.2f kHz) | %s | %.2f%s | %.1f | %.1f | %+.1f | %+.3f | %+.1f | %s |' % (
            sp, cf, lv, f[i], ' (edge)' if edge else '', mt[i], mb[i], mt[i] - mb[i], ph(i), mt[c] - mb[c],
            'none' if kb is None else '%.2f, %+.1f' % (f[kb], mt[kb] - mb[kb])))
    out[sp] = dict(cf_khz=cf, x=p['x'], rows=rows)
json.dump(out, open(D + 'tm_peaks.json', 'w'), indent=1); open(D + 'tm_peaks.md', 'w').write('\n'.join(md) + '\n')
print('\n'.join(md))
