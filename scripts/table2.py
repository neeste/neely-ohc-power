"""Run 8 item 2: Table II parameters and the Sec. II numbers of the cat state (place map, damping, activity,
critical gain). Run under run_cat.py in the state's run dir:  python3 run_cat.py table2.py out.json
(from model/: python3 ../scripts/run_cat.py ../scripts/table2.py out/table2.json)."""
import sys, json, math
import numpy as np
from dataclasses import replace
sys.path.insert(0, '.')
import geom, score_base as SB, tl_model as T, deltaT_3dof as d3, exp_ftc1 as F1
OUT = sys.argv[1]
bd = json.load(open('best_d3.json')); p = bd['params']; Z = SB.Z
lam = float(geom.LAM); BASE = SB.BASE; S = SB.COCH.sec
o = dict(lam=lam, params=p, z=Z)
o['profile'] = dict(c_R=SB.C_R, alpha_a=SB.ALPHA_A, alpha_b=SB.ALPHA_B, exp_apical=1 + SB.ALPHA_A, exp_basal=1 + SB.ALPHA_B,
                    f_t_khz=SB.F_T, a_t=SB.A_T, w_dec=SB.WID, nu0=SB.NU0, alpha_nu=SB.ALPHA_NU,
                    f_hp_khz=math.exp(Z[6]), f_lp_khz=math.exp(Z[7]), n_lp=Z[8], B_khz=BASE, C0_db=Z[10])
xa = lambda x: SB.COCH.cf(np.asarray(x, float))
o['x_transition'] = float(-math.log10(SB.A_T)/2.75)
# damping per cycle and activity
xs = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.465, 0.4686, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0])
a = xa(xs); rho = SB._damping(a)/a; nu = SB._nu(a)
o['rho_at'] = dict(zip(['%.4f' % v for v in xs], rho.tolist()))
o['nu_at'] = dict(zip(['%.4f' % v for v in xs], nu.tolist()))
xx = np.linspace(0, 1, 2049); aa = xa(xx); rr = SB._damping(aa)/aa; nn = SB._nu(aa)
m6 = xx >= 0.6
o['rho_range_x0.6_1'] = [float(rr[m6].min()), float(rr[m6].max())]
o['rho_range_line'] = [float(rr.min()), float(rr.max())]
o['nu_range_line'] = [float(nn.min()), float(nn.max())]
o['nu_limits_reached'] = bool((nn <= 0.02 + 1e-12).any() or (nn >= 0.60 - 1e-12).any())
tp = SB.COCH.taper(xx); mt = tp > 0.01
o['nu_range_taper_gt_0.01'] = [float(nn[mt].min()), float(nn[mt].max())]
o['g0_over_gc_at'] = {k: 1 - v for k, v in o['nu_at'].items()}
# 24-point gamma_c table as installed by run_cat.py
o['gc_table'] = dict(rho_min=float(math.exp(SB._LX[0])), rho_max=float(math.exp(SB._LX[-1])), gc_min=float(SB._GY.min()), gc_max=float(SB._GY.max()), n=len(SB._LX))
def gc_rho(r_, lam_=None):
    if lam_ is not None: geom.LAM = lam_
    s2 = replace(S, gamma=0.0, R_b=S.R_b*r_, R_r=S.R_r*r_, R_t=S.R_t*r_, R_p=S.R_p*r_, R_c=S.R_c*r_)
    g = d3.gamma_critical(s2); geom.LAM = lam; return float(g)
err = {}
for xv in (0.1, 0.3, 0.5, 0.7):
    r_ = float(SB._damping(xa(xv))/xa(xv)); gi = float(np.interp(math.log(r_), SB._LX, SB._GY)); gd = gc_rho(r_)
    err['%.1f' % xv] = dict(rho=r_, interp=gi, direct=gd, rel=abs(gi - gd)/gd)
o['gc_interp_err'] = err; o['gc_interp_err_max'] = max(v['rel'] for v in err.values())
o['gc_rho1_vs_lam'] = {'%.4f' % L: gc_rho(1.0, L) for L in (0.0, 0.25, 0.5, 0.7084610343120661, 0.8561939116132914, 1.0)}
o['gc_rho1_vs_lam']['ratio_lam_to_1'] = o['gc_rho1_vs_lam']['%.4f' % lam]/o['gc_rho1_vs_lam']['1.0000']
o['stage1_gamma'] = p['gamma']; o['stage1_ratio_rho1'] = p['gamma']/gc_rho(1.0)
# single sign change of the largest real part over gamma in [0, 200] at the extremes of rho
sc = {}
for r_ in (float(rr.min()), 0.5, float(rr.max())):
    s2 = replace(S, R_b=S.R_b*r_, R_r=S.R_r*r_, R_t=S.R_t*r_, R_p=S.R_p*r_, R_c=S.R_c*r_)
    gs = np.linspace(0, 200, 400); mr = [geom.margin(replace(s2, gamma=g)) for g in gs]
    sg = np.sign(mr); sc['%.3f' % r_] = int(np.sum(sg[1:] != sg[:-1]))
o['sign_changes_400'] = sc
# taper places
o['taper_x0.75_khz'] = None
# place map: low-level BM displacement peak on the quiet profiled line, N = 2048
N = 2048; x = np.linspace(0, 1, N + 1); r, g0 = SB.profiles(N)
fr = np.geomspace(0.0011, 0.62, 160); xp = []
for f in fr:
    R = SB.line_grad(SB.COCH, f, g0, np.zeros(N + 1), 0.0, 0.0, N); xp.append(float(x[int(np.argmax(np.abs(R['X_b'])))]))
xp = np.array(xp); fk = fr*BASE
o['placemap'] = dict(f_khz=fk.tolist(), x=xp.tolist())
srt = np.argsort(xp)
def cf_at(xv):          # CF of place x: invert the monotone part of the map
    m = np.r_[True, np.diff(xp) > 0]           # keep strictly increasing in x as f decreases
    xs_, fs_ = xp[::-1], np.log(fk[::-1])
    u, idx = np.unique(xs_, return_index=True)
    return float(np.exp(np.interp(xv, u, fs_[idx])))
def x_of(fkhz): return float(np.interp(math.log(fkhz), np.log(fk), xp))
o['cf_at_x'] = {'%.1f' % v: cf_at(v) for v in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)}
o['cf_over_aB'] = {k: v/(float(xa(float(k)))*BASE) for k, v in o['cf_at_x'].items()}
o['x_of_f'] = {'%g' % f: x_of(f) for f in (38.5, 12.0, 10.0, 9.0, 4.0, 2.0, 1.0, 0.5, 0.084)}
# extrapolated CF at the stapes: linear in log f against x over the basal points (x < 0.1)
mb = (xp > 0.004) & (xp < 0.1)
cfit = np.polyfit(xp[mb], np.log(fk[mb]), 1); o['cf_stapes_extrap_khz'] = float(math.exp(np.polyval(cfit, 0.0)))
o['f_at_xmin'] = dict(x=float(xp.min()), f_khz=float(fk[np.argmin(xp)]))
o['taper_x0.75_cf_khz'] = cf_at(0.75); o['taper_x0.95_cf_khz'] = cf_at(0.95)
o['x_transition_cf_khz'] = cf_at(o['x_transition'])
# BM velocity and shear peaks at x = 0.5 against the place-map CF
j = int(round(0.5*N)); f0 = cf_at(0.5)/BASE; ff = np.geomspace(0.8*f0, 1.25*f0, 241); bv = []; sh = []
for f in ff:
    R = SB.line_grad(SB.COCH, f, g0, np.zeros(N + 1), 0.0, 0.0, N); bv.append(abs(f*R['X_b'][j])); sh.append(abs(R['shear'][j]))
o['x0.5'] = dict(cf_map_khz=cf_at(0.5), bmvel_peak_khz=float(ff[int(np.argmax(bv))]*BASE), shear_peak_khz=float(ff[int(np.argmax(sh))]*BASE))
j = int(round(0.55*N)); f0 = float(xa(0.55)); ff = f0*2.0**np.linspace(-3.6, 0.6, 169); bv = []
for f in ff:
    R = SB.line_grad(SB.COCH, f, g0, np.zeros(N + 1), 0.0, 0.0, N); bv.append(abs(f*R['X_b'][j]))
o['x0.55'] = dict(bf_bmvel_khz_profiled=float(ff[int(np.argmax(bv))]*BASE), cf_map_khz=cf_at(0.55))
# probe read place (profiled): place of the probe's low-level BM peak
R = SB.line_grad(SB.COCH, float(xa(0.55)), g0, np.zeros(N + 1), 0.0, 0.0, N); o['x_r_profiled_N2048'] = float(x[int(np.argmax(np.abs(R['X_b'])))])
o['probe_khz'] = float(xa(0.55))*BASE
# calibration
b = json.load(open('base_battery.json')); Lm = float(np.median(b['locus']['L'])); csupp = 65.0 - Lm
kn = -20*math.log10(p['u_thr'])
o['calib'] = dict(C_supp_db=csupp, locus_L=b['locus']['L'], locus_slope=b['locus']['slope'], knee_above_thr_db=kn,
                  probe_threshold_db_spl=csupp - kn, knee_db_spl=csupp, gate_floor_db=-20*math.log10(p['s_g']))
json.dump(o, open(OUT, 'w'), indent=1, default=float)
for k in ('profile', 'x_transition', 'x_transition_cf_khz', 'rho_at', 'rho_range_x0.6_1', 'nu_at', 'g0_over_gc_at', 'gc_table', 'gc_interp_err_max',
          'gc_rho1_vs_lam', 'stage1_ratio_rho1', 'sign_changes_400', 'cf_at_x', 'cf_over_aB', 'x_of_f', 'cf_stapes_extrap_khz', 'f_at_xmin',
          'taper_x0.75_cf_khz', 'taper_x0.95_cf_khz', 'x0.5', 'x0.55', 'x_r_profiled_N2048', 'calib'):
    print(k, json.dumps(o[k], default=float), flush=True)
print('TABLE2 DONE')
