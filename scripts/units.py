"""units.py: physical-unit assignment. rho = 1000 kg/m^3; L, H assumed;
m = 2 rho L^2/(H F). Cat: BM and shear velocity at neural threshold at the
tips of the nine fitted places, for middle-ear pressure gains of 15, 25,
30 dB (Nedzelnitsky 1980: 15-30 dB). Human: BM displacement and velocity
per unit stapes pressure at CF (quiet line, N = 8192)."""
import sys, json, math, numpy as np
sys.path.insert(0, '.')
import score_base as SB, exp_accumulation as E, tl_model as T
RHO = 1000.0
out = dict(assumptions=dict(rho=RHO, L_cat=0.025, L_human=0.035, H=[0.0005, 0.001, 0.002]))
# tip kHz and tip threshold (dB SPL) of the nine fitted places, from bmdef_shoulder_s2.py (run it first;
# the threshold uses the read-out of the state, velocity criterion unless FTC_CRIT=disp)
_bm = json.load(open(sys.argv[1] if len(sys.argv) > 1 else 'out/bmdef_shoulder.json'))['threshold']
mm = np.array([[r['tip_khz'], r['thr_db_spl']] for r in sorted(_bm, key=lambda r: r['x'])])
N = 2048; x = np.linspace(0, 1, N+1); a = SB.COCH.cf(x); r, g0 = SB.profiles(N); gam = g0*SB.COCH.taper(x)
B = SB.BASE*1e3; F = SB.COCH.F
rows = []
for i, xp in enumerate((0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)):
    j = int(round(xp*N)); ftip = mm[i, 0]*1e3; Lthr = mm[i, 1]
    f = ftip/B
    Y, Yr, Ho = E.terms(SB.COCH.sec, f, a, r, gam); P = T._fd(SB.COCH, f, Y, N, 1.0)
    Xb = abs((Y*P)[j]); Sh = abs((Ho*Y*P)[j])
    rec = dict(x=xp, tip_khz=ftip/1e3, thr_db_spl=Lthr)
    for H in out['assumptions']['H']:
        m = 2*RHO*0.025**2/(H*F)
        for Gme in (15, 25, 30):
            ps = 20e-6*10**(Lthr/20)*10**(Gme/20)
            xb = Xb*ps/(m*(2*math.pi*B)**2); sh = Sh*ps/(m*(2*math.pi*B)**2)
            rec['H%.1fmm_G%d' % (H*1e3, Gme)] = dict(bm_nm=xb*1e9, bm_um_s=2*math.pi*ftip*xb*1e6, shear_um_s=2*math.pi*ftip*sh*1e6)
    rows.append(rec)
out['cat_threshold'] = rows
out['cat_m'] = {('H%.1fmm' % (H*1e3)): 2*RHO*0.025**2/(H*F) for H in out['assumptions']['H']}
out['cat_Kb0'] = {k: v*(2*math.pi*B)**2 for k, v in out['cat_m'].items()}
json.dump(out, open('out/units_cat.json', 'w'), indent=1)
print('cat m (kg/m^2):', {k: round(v, 4) for k, v in out['cat_m'].items()}, ' Kb0 (Pa/m):', {k: '%.2e' % v for k, v in out['cat_Kb0'].items()})
for rec in rows:
    v = rec['H1.0mm_G25']
    print('x %.1f tip %5.2f kHz thr %4.1f dB SPL: BM %.3f nm %.2f um/s, shear %.2f um/s  [H 1 mm, ME 25 dB]' % (rec['x'], rec['tip_khz'], rec['thr_db_spl'], v['bm_nm'], v['bm_um_s'], v['shear_um_s']))
