"""place_map9.py (run 9): human place-map numbers that had no script (inventory #11, #17-26, #44-49,
#51-54, #95 and MF 1.4, 1.6, 1.7, 3.6, 4.3).
Run in a human run directory under the wrapper (abr5_ckpt.json is the state):
  from model/: python3 ../scripts/run_human.py ../scripts/place_map_h.py out/place_map_h.json
Definitions:
- shear-peak place of f: node of max |H_o Y P| on the quiet line at unit stapes pressure (as step3/ftc_h.py).
- fixed-place sweep at x: |shear| and |X_b| at node round(xN) for f = 0.2 to 1.6 a(x)B (MF 1.6). The place
  is untuned when the largest response is at the lowest frequency tested. Interior local maxima are listed.
- rho = c_R a^alpha_R at the exact x; gamma_c from exp_abr5.gc_exact_dense (160-point table, interpolated
  in log rho) read at the nearest node; the table spans 0.98 min(rho) to 1.02 max(rho)."""
import sys, json, math
import numpy as np
sys.path.insert(0, '.')
import exp_abr2 as A2, exp_abr5 as A5, exp_accumulation as E, tl_model as T
OUT = sys.argv[1]; B = 20.7
ck = json.load(open('abr5_ckpt.json')); p = dict(ck['params'])
H = A2.Human(**p); N = len(H.x) - 1
gc = A5.gc_exact_dense(H); g = A5.capped_gamma(H, p['mu0'], p['alpha_mu'], gc)
k, cR, aR = p['k'], p['c_R'], p['alpha_R']
a_of = lambda x: max(10.0**(-3.0*x) - k, 1e-4)
rho_of = lambda x: cR*a_of(x)**aR

def solve(f):
    Y, Yr, Ho = E.terms(H.coch.sec, f, H.a, H.r, g); P = T._fd(H.coch, f, Y, N, 1.0)
    return np.abs(Ho*Y*P), np.abs(Y*P)

def peak_place(fk):
    sh, bm = solve(fk/B)
    return float(H.x[int(np.argmax(sh))]), float(H.x[int(np.argmax(bm))])

def fixed_place(x, n=241):
    j = int(round(x*N)); fr = a_of(x)*np.geomspace(0.2, 1.6, n)
    S, Bm = np.array([[v[j] for v in solve(f)] for f in fr]).T
    rec = dict(x=x, aB_khz=a_of(x)*B)
    for name, r in (('shear', S), ('bm', Bm)):
        i = int(np.argmax(r)); loc = [m for m in range(1, n-1) if r[m] > r[m-1] and r[m] > r[m+1]]
        fi = fr[i]
        if 0 < i < n-1:   # parabolic refinement of the maximum in log f (grid step 0.87 percent)
            y0, y1, y2 = np.log(r[i-1:i+2]); fi = fr[i]*(fr[1]/fr[0])**(0.5*(y0-y2)/(y0-2*y1+y2))
        rec[name] = dict(argmax_f_khz=float(fi*B), argmax_re_aB=float(fi/a_of(x)),
                         untuned=bool(i == 0), local_max_f_khz=[float(fr[m]*B) for m in loc],
                         local_max_rise_db=[float(20*math.log10(r[m]/r[0])) for m in loc])
    return rec

res = dict(state_J=ck.get('J'), lam=float(__import__('geom').LAM), N=N, k=k, c_R=cR, alpha_R=aR)
res['x_a_floor'] = -math.log10(1e-4 + k)/3.0 if 1e-4 + k > 0 and -math.log10(1e-4 + k)/3.0 <= 1 else None   # #11 (None: a never reaches 1e-4, e.g. k < 0)
res['a_apex'] = float(H.a[-1])
pk = {fk: peak_place(fk) for fk in (0.25, 0.5, 1, 2, 4, 8)}
res['shear_peak_x'] = {str(fk): v[0] for fk, v in pk.items()}       # #12-16, #24
res['bm_peak_x'] = {str(fk): v[1] for fk, v in pk.items()}
res['cf_over_aB'] = {str(fk): fk/(B*a_of(v[0])) for fk, v in pk.items()}   # #17-21
res['dx_025_05'] = pk[0.25][0] - pk[0.5][0]                         # #25
res['fixed_place'] = [fixed_place(x) for x in (0.1, 0.3, 0.5, 0.7, 0.9)]   # #26
xs = (0, 0.2, 0.3, 0.348, 0.388, 0.4, 0.47, 0.5, 0.6, 0.7)
res['rho'] = {'%.4g' % x: rho_of(x) for x in xs}                    # #22-23, #44-48, #95 (0.388 = 0.5-kHz place)
res['rho_plateau'] = cR*max(1e-4, float(H.a.min()))**aR              # #49 (at the smallest a: 1e-4 if clipped, else the apex)
rho = H.r/H.a
res['gc_table_rho'] = [0.98*float(rho.min()), 1.02*float(rho.max())]   # #51
nd = lambda x: int(round(x*N))
res['gc'] = {'0': float(gc[0]), '0.348': float(gc[nd(0.348)]), 'plateau': float(gc[-1]),
             'range': [float(gc.min()), float(gc.max())]}             # #52-54
res['nu_at'] = {'%.3g' % x: float(1 - g[nd(x)]/(gc[nd(x)]*H.coch.taper(H.x)[nd(x)])) for x in (0, 0.08, 0.14, 0.5)}
json.dump(res, open(OUT, 'w'), indent=1)
print('x(a = 1e-4)', res['x_a_floor'], 'a at apex %.5g' % res['a_apex'])
for fk, v in pk.items():
    print('%5.2f kHz: shear x %.4f  BM x %.4f  CF/(aB) %.3f' % (fk, v[0], v[1], res['cf_over_aB'][str(fk)]))
print('x(0.25) - x(0.5) = %.4f' % res['dx_025_05'])
for r in res['fixed_place']:
    print('x %.1f (aB %.3f kHz): shear max %.3f kHz untuned %s loc %s | BM max %.3f kHz untuned %s loc %s' % (
        r['x'], r['aB_khz'], r['shear']['argmax_f_khz'], r['shear']['untuned'],
        ['%.3f' % f for f in r['shear']['local_max_f_khz']], r['bm']['argmax_f_khz'], r['bm']['untuned'],
        ['%.3f' % f for f in r['bm']['local_max_f_khz']]))
print('rho', {k_: round(v, 3) for k_, v in res['rho'].items()}, 'plateau %.1f' % res['rho_plateau'])
print('gc table rho %.4f to %.1f; gc %s' % (*res['gc_table_rho'], {k_: v for k_, v in res['gc'].items()}))
print('nu', res['nu_at'])
