"""Run 9: human place-map numbers of the manuscript (Sec. II.A, II.D, Table III) at the state in abr5_ckpt.json,
quiet line, N 8192, under run_human.py (LAM, state construction).  Definitions as in methods_facts 1.4 to 1.8, 3.6, 4.3:
shear-peak place = argmax |H_o X_b| of the quiet line at the tone; CF/(aB) = f/(a(x)B); CF at a place = frequency of the
greatest |shear| (and |X_b|) over 0.2 to 1.6 a(x)B (200 log points); a place is untuned when that maximum is at the lowest
frequency; rho = c_R a^alpha_R; gamma_c from A5.gc_exact_dense; octave lengths for L = 35 mm.  Usage: map9.py out.json"""
import sys, json, math, numpy as np
sys.path.insert(0, '.')
import exp_abr2 as A2, exp_abr5 as A5, exp_accumulation as E, tl_model as T
ck = json.load(open('abr5_ckpt.json')); p = dict(ck['params']); B = 20.7
H = A2.Human(**p); gc = A5.gc_exact_dense(H); H.gam0 = A5.capped_gamma(H, p['mu0'], p['alpha_mu'], gc); N = len(H.x)-1
x, a = H.x, H.a; rho = p['c_R']*a**p['alpha_R']
def fields(f):
    Y, Yr, Ho = E.terms(H.coch.sec, f, a, H.r, H.gam0); P = T._fd(H.coch, f, Y, N, 1.0); Xb = Y*P; return Xb, Ho*Xb
out = dict(N=N, k=p['k'], F=p['F'], lam=ck['lam'])
pk = {}
for fk in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
    Xb, Sh = fields(fk/B); j = int(np.argmax(np.abs(Sh))); jb = int(np.argmax(np.abs(Xb)))
    pk[str(fk)] = dict(x_shear=float(x[j]), x_bm=float(x[jb]), cf_over_aB=float(fk/(a[j]*B)), g_over_gc=float(H.gam0[j]/gc[j]), nu=float(1-H.gam0[j]/(gc[j]*H.coch.taper(x)[j])))
out['peaks'] = pk
cfp = {}
for xp in (0.1, 0.3, 0.5, 0.7, 0.9):
    j = int(round(xp*N)); fr = a[j]*np.geomspace(0.2, 1.6, 200); sh = []; bm = []
    for f in fr:
        Xb, Sh = fields(f); sh.append(abs(Sh[j])); bm.append(abs(Xb[j]))
    i, ib = int(np.argmax(sh)), int(np.argmax(bm))
    cfp[str(xp)] = dict(aB_khz=float(a[j]*B), cf_shear_khz=float(fr[i]*B), cf_bm_khz=float(fr[ib]*B), untuned_shear=(i == 0), untuned_bm=(ib == 0))
out['cf_at_place'] = cfp
out['rho'] = {str(xp): float(np.interp(xp, x, rho)) for xp in (0.0, 0.1, 0.2, 0.3, 0.4, 0.45, 0.47, 0.5, 0.7, 0.764)}
out['x_floor'] = float(x[np.argmax(a <= 1e-4 + 1e-12)])
out['rho_floor'] = float(rho[-1])
out['rho_at_peaks'] = {fk: float(np.interp(v['x_shear'], x, rho)) for fk, v in pk.items()}
out['gc'] = dict(stapes=float(gc[0]), x0348=float(np.interp(0.348, x, gc)), apex=float(gc[-1]), max=float(gc.max()),
                 table_rho=[float(rho.min()*0.98), float(rho.max()*1.02)])
out['mu0_a_alpha_max'] = float(np.max(p['mu0']*a**p['alpha_mu']))
L = 35.0; xs = {fk: v['x_shear'] for fk, v in pk.items()}
out['mm_per_oct_1to4'] = (xs['1.0'] - xs['4.0'])/2*L; out['mm_per_oct_0.5to1'] = (xs['0.5'] - xs['1.0'])*L
out['kh_factor'] = dict(oct_1to4=4.7/out['mm_per_oct_1to4'], oct_05to1=4.1/out['mm_per_oct_0.5to1'])
m = {H_: 2*1000*0.035**2/(H_*1e-3*p['F']) for H_ in (0.5, 1.0, 2.0)}
out['m_kg_m2'] = m; out['Kb0_Pa_m'] = {h: v*(2*math.pi*B*1e3)**2 for h, v in m.items()}
nu = 1 - H.gam0/(gc*np.maximum(H.coch.taper(x), 1e-12))
out['nu_profile'] = {str(xp): float(np.interp(xp, x, nu)) for xp in (0.0, 0.04, 0.08, 0.1, 0.14, 0.2, 0.4, 0.6, 0.74)}
json.dump(out, open(sys.argv[1], 'w'), indent=1); print(json.dumps(out, indent=1))
