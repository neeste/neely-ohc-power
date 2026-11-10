"""Markdown tables for queue C (c_robust_<model>.json, c_scan_cert_<model>.json) and D (d_kh_<model>.json)."""
import json, os, sys
O = 'out/'
def ctab(model):
    C = json.load(open(O + 'c_robust_%s.json' % model)); L = []
    sc = json.load(open(O + 'c_scan_cert_%s.json' % model)) if os.path.exists(O + 'c_scan_cert_%s.json' % model) else {}
    L += ['## %s%s' % (model, (' (state %s)' % os.path.basename(C['state'])) if 'state' in C else ''), '',
          '| variation | stable (certificates) | CF (kHz) | g/gc at BM peak | share max | share max place, oct re BM / re shear | p_ohc max place, oct re BM / re shear | share > 1 band, oct re BM | share at BM peak | energy identity error |',
          '|---|---|---|---|---|---|---|---|---|---|']
    for r in C['rows']:
        for fk, s in r['cf'].items():
            b = s['band_gt1_oct_bm']
            L.append('| %s | %s (%d/%d) | %s | %.3f | %.3f | %+.2f / %+.2f | %+.3f / %+.3f | %s | %.3f | %.0e |' % (
                r['name'], 'yes' if r['stable'] else 'NO', r['cert_clean'], r['cert_total'], fk, s['gamma_ratio_at_bm_peak'], s['share_max'],
                s['share_max_oct_bm'], s['share_max_oct_shear'], s['p_ohc_max_oct_bm'], s['p_ohc_max_oct_shear'],
                ('%+.2f to %+.2f' % tuple(b)) if b else 'none', s['share_at_bm_peak'], s['identity_err']))
    L += ['', 'Lambda scan (location only; stability from the quiet line, N 1024, depths 0.35 and 1.0, and the 40-dB probes):', '',
          '| lambda | quiet stable | 40-dB probes | 1 kHz share max, oct re BM | 1 kHz p_ohc max | 4 kHz share max | 4 kHz p_ohc max |', '|---|---|---|---|---|---|---|']
    for r in C['lam_scan']:
        c = sc.get('%.2f' % r['lam'], {})
        L.append('| %.2f | %s | %s | %+.3f | %+.3f | %+.3f | %+.3f |' % (r['lam'], {True: 'yes', False: 'NO'}.get(c.get('quiet_clean'), 'n/a'), c.get('quick_sat') or '-',
                 r['1.0']['share_max_oct_bm'], r['1.0']['p_ohc_max_oct_bm'], r['4.0']['share_max_oct_bm'], r['4.0']['p_ohc_max_oct_bm']))
    return L
def dtab(model):
    D = json.load(open(O + 'd_kh_%s.json' % model)); L = ['## %s%s, H %.1f mm, L %.0f mm' % (model, (' (state %s)' % os.path.basename(D['state'])) if 'state' in D else '', D['H_mm'], D['L_mm']), '',
        '| quantity | 1 kHz long-wave | 1 kHz corrected | 4 kHz long-wave | 4 kHz corrected |', '|---|---|---|---|---|']
    R = D['rows']; get = lambda fk, w, f: f(R[fk][w])
    items = [('BM peak place x', lambda m: '%.4f' % m['peaks']['x_bm']), ('BM peak CF (kHz)', lambda m: '%.2f' % m['peaks']['cf_bm_khz']),
             ('BM peak level (dB re model unit)', lambda m: '%.1f' % m['peaks']['bm_db']), ('shear peak place x', lambda m: '%.4f' % m['peaks']['x_shear']),
             ('shear peak level (dB)', lambda m: '%.1f' % m['peaks']['shear_db']), ('Q_ERB', lambda m: '%.2f' % m['Q_ERB']), ('N_SF (shear)', lambda m: '%.2f' % m['N_SF']),
             ('share max', lambda m: '%.3f' % m['share']['share_max']), ('share max place, oct re BM peak', lambda m: '%+.2f' % m['share']['share_max_oct_bm']),
             ('share max place, oct re shear peak', lambda m: '%+.2f' % m['share']['share_max_oct_shear']), ('p_ohc max place, oct re BM peak', lambda m: '%+.2f' % m['share']['p_ohc_max_oct_bm'])]
    for nm, f in items:
        L.append('| %s | %s | %s | %s | %s |' % (nm, f(R['1.0']['longwave']), f(R['1.0']['corrected']), f(R['4.0']['longwave']), f(R['4.0']['corrected'])))
    L += ['', 'Convergence and checks:', '']
    for fk in ('1.0', '4.0'):
        r = R[fk]; s = r['sweep_stats']; c30 = D['itmax30'][fk]
        L.append('- %s kHz: at the cap of 10 iterations converged=%s (last change %.3f dB); in the Q_ERB/N_SF sweeps %d of %d solves met 0.1 dB within 10. With a cap of 30: converged in %d iterations; Q_ERB %.2f, N_SF %.2f, BM peak %.2f dB. Long-wave |k| at the BM peak %.0f per cochlear length; corrected %.0f (kH %.1f, correction factor kH coth kH %.1f). BM peak shift %+.2f oct (positive = basal). N 16384: BM peak x %.4f, %.2f dB.' % (
            fk, r['converged'], r['last_change_db'], s['converged'], s['calls'], c30['iters'], c30['metrics']['Q_ERB'], c30['metrics']['N_SF'], c30['metrics']['peaks']['bm_db'],
            r['k_longwave_at_bm_peak'], r['k_at_bm_peak'], r['kH_at_bm_peak'], r['fac_at_bm_peak'], r['shift_bm_oct'], D['grid'][fk]['peaks']['x_bm'], D['grid'][fk]['peaks']['bm_db']))
        for H in ('0.5', '2.0'):
            h = D['H_sens']['%s H%s' % (fk, H)]
            L.append('  - H %s mm: BM peak shift %+.2f oct, level %.1f dB (converged %s in %d)' % (H, h['shift_bm_oct'], h['peaks']['bm_db'], h['converged'], h['iters']))
    return L
if __name__ == '__main__':
    models = sys.argv[1:] or ['cat']
    C = ['# Queue C: robustness of the location result (run 2)', '',
         'One factor at a time from the adopted state. Lambda and damping changes hold gamma/gamma_c fixed (gamma_c rebuilt for the new lambda or damping); gamma/gamma_c changes scale gamma uniformly by T/g0, g0 = gamma/gamma_c of the state (0.955 cat; 1 - nu_floor human, 0.9758 at hs856, c_robust9.py). Share = p_ohc/D within 2 oct of the BM peak; places in octaves re the BM and shear peaks of the same tone, negative = apical. Stable = all certificates clean (cat: quiet + 9 saturated profiles x N 1024, 2048 x depths 0.35, 1.0 = 40; human: cert_h_lam set, 18); certification stops at the first dirty certificate. Energy identity D = p_f + p_ohc checked with the lambda accounting (geom.power, acc_lam).', '']
    Dd = ['# Queue D: limited kH coth kH fluid test at fixed parameters (run 2)', '',
          'The long-wave line is re-solved with F_eff(x) = F kH coth(kH) (H_eff = H tanh(kH)/(kH)), |k| from the previous iterate pressure (k = -dphi/dx + i dln|P|/dx), iterated to 0.1 dB in the BM and shear magnitudes within 30 dB of their peaks, cap 10. Limits: WKB-local (the correction uses the local wavenumber of the previous iterate and the 1-D equation, not a 2-D solution); no pressure focusing near the partition and no change of the WKB amplitude factor; the corrected state is not certified (the argument-principle test applies to the long-wave line); Y(x) unchanged. The share p_ohc/D is a local quantity (both terms scale with |P|^2), so its place in x does not change; only its place relative to the response peaks changes.', '']
    for m in models:
        if os.path.exists(O + 'c_robust_%s.json' % m): C += ctab(m) + ['']
        if os.path.exists(O + 'd_kh_%s.json' % m): Dd += dtab(m) + ['']
    open(O + 'C_table.md', 'w').write('\n'.join(C) + '\n'); open(O + 'D_table.md', 'w').write('\n'.join(Dd) + '\n')
    print('\n'.join(C)); print('\n'.join(Dd))
