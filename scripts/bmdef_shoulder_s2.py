"""Cat stage-2 state: (1) BM-velocity deficit at neural threshold (as repo/scripts/units.py,
rho 1000 kg/m^3, L 25 mm, H 0.5/1/2 mm, middle-ear gain 15/25/30 dB) at the nine fitted places,
using the stage-2 tips (exp_ftc1.ftc_metrics); (2) shoulder diagnosis: high-side threshold
th = C0 - 20log|P| - 20log|Y| - 20log|H_o| + ME [- 20log f for FTC_CRIT=vel, run 10] split into fluid pressure, BM admittance,
divider (shear/BM) and middle-ear parts, slopes in 0.1-oct bins from the tip to +0.8 oct.
Run in a run directory with best_d3.json (section, lam).  Usage: bmdef_shoulder_s2.py ckpt.json out.json"""
import sys, json, math
import numpy as np
sys.path.insert(0, '.')
import geom, deltaT_3dof as d3, tl_model as T
bd = json.load(open('best_d3.json')); geom.LAM = bd.get('lam', 1.0); geom.install()
T.section_terms = lambda sec, w, a, r=None, gamma=None: geom.terms(sec, w, a, r, gamma)
d3.gamma_critical = lambda sec, hi=200.0, tol=1e-6: geom.gamma_critical(sec, 1.0, hi, tol)
import exp_accumulation as E, exp_ftc1 as F1, exp_ftc2 as X
ck = json.load(open(sys.argv[1])); z = ck['z']
base, f_hp, f_lp, nlp, C0 = math.exp(z[9]), math.exp(z[6]), math.exp(z[7]), z[8], z[10]
r, gam = X.profiles(z); SEC, COCH, A, N = F1.SEC, F1.COCH, F1.A, F1.N
mm = F1.ftc_metrics(r, gam, base, f_hp, f_lp, nlp, C0)
RHO, L, B = 1000.0, 0.025, base*1e3
def resp(f, j):
    Y, Yr, Ho = E.terms(SEC, f, A, r, gam); P = T._fd(COCH, f, Y, N, 1.0)
    return P[j], Y[j], Ho[j]
rows = []
for i, xp in enumerate(F1.PLACES):
    j = int(np.argmin(np.abs(F1.X - xp))); ftip, Lthr = mm[i, 0]*1e3, mm[i, 1]
    P, Y, Ho = resp(ftip/B, j); Xb, Sh = abs(Y*P), abs(Ho*Y*P)
    rec = dict(x=xp, tip_khz=ftip/1e3, thr_db_spl=float(Lthr), q10=float(mm[i, 2]), tt=float(mm[i, 3]))
    for H in (0.0005, 0.001, 0.002):
        m = 2*RHO*L**2/(H*COCH.F)
        for G in (15, 25, 30):
            ps = 20e-6*10**(Lthr/20)*10**(G/20); k = m*(2*math.pi*B)**2
            rec['H%.1fmm_G%d' % (H*1e3, G)] = dict(bm_nm=Xb*ps/k*1e9, bm_um_s=2*math.pi*ftip*Xb*ps/k*1e6,
                                                   shear_um_s=2*math.pi*ftip*Sh*ps/k*1e6)
    rows.append(rec)
sh = []
for xp in (0.3, 0.5, 0.7):
    i = F1.PLACES.index(xp); j = int(np.argmin(np.abs(F1.X - xp))); ft = mm[i, 0]/base
    oc = np.arange(0.0, 0.81, 0.05); comp = []
    for o in oc:
        f = ft*2**o; P, Y, Ho = resp(f, j)
        comp.append((-20*math.log10(abs(P)), -20*math.log10(abs(Y)), -20*math.log10(abs(Ho)),
                     float(F1.me_db(f*base, f_hp, f_lp, nlp)), -20*math.log10(float(F1.crit_factor(f*base)))))
    comp = np.array(comp); comp -= comp[0]; th = comp.sum(1)
    bins = []
    for k in range(0, len(oc) - 2, 2):
        d = (comp[k+2] - comp[k])/0.1
        bins.append(dict(oct=[float(oc[k]), float(oc[k+2])], th=float(th[k+2]-th[k])/0.1, fluid=float(d[0]), bm=float(d[1]),
                         divider=float(d[2]), me=float(d[3]), crit=float(d[4])))
    sh.append(dict(x=xp, tip_khz=float(mm[i, 0]), oct=oc.tolist(), th_rel=th.tolist(), parts_rel=comp.tolist(), slopes_db_per_oct=bins))
out = dict(source=sys.argv[1], lam=geom.LAM, criterion=F1.CRIT, parts_columns=['fluid', 'bm', 'divider', 'me', 'crit (0 for disp; -20 log10 f for vel)'], threshold=rows, shoulder=sh)
json.dump(out, open(sys.argv[2], 'w'), indent=1)
print('lam %.3f' % geom.LAM)
for rec in rows:
    v = rec['H1.0mm_G25']
    print('x %.1f tip %5.2f kHz thr %5.1f dB SPL: BM %.3f nm %.3f um/s shear %.3f um/s  [H 1 mm, ME 25 dB]' % (rec['x'], rec['tip_khz'], rec['thr_db_spl'], v['bm_nm'], v['bm_um_s'], v['shear_um_s']))
for s in sh:
    print('shoulder x %.1f tip %.2f kHz' % (s['x'], s['tip_khz']))
    for b in s['slopes_db_per_oct']:
        print('   %.1f-%.1f oct: th %+6.0f  fluid %+6.0f  bm %+6.0f  divider %+6.0f  me %+5.0f dB/oct' % (b['oct'][0], b['oct'][1], b['th'], b['fluid'], b['bm'], b['divider'], b['me']))
