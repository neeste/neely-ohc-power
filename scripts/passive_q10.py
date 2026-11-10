"""Run 6 item 2 (Prediction 5): cat passive BM tuning at the 5- and 20-kHz places (place of the quiet active BM peak).
Passive line (gamma = 0), drive P(0) = 1. Reports, for BM displacement and BM velocity (|f X_b|) re stapes pressure:
peak frequency, peak re the low-frequency end (0.05 f), Q10 (f_peak over the 10-dB bandwidth) where both sides fall
10 dB, Q3 likewise. Run under run_cat.py in the combo run dir. Usage: passive_q10.py out.json"""
import sys, json, math, numpy as np
sys.path.insert(0, '.')
import exp_accumulation as E, tl_model as T, score_base as SB
N = 4096; x = np.linspace(0, 1, N+1); a = SB.COCH.cf(x); r, g0 = SB.profiles(N); gam0 = g0*SB.COCH.taper(x)
SEC, COCH, BASE = SB.COCH.sec, SB.COCH, SB.BASE
def xb(g, w):
    Y, Yr, Ho = E.terms(SEC, w, a, r, g); return Y*T._fd(COCH, w, Y, N, 1.0)
def q(fr, m, dbd):
    l = 20*np.log10(m/m.max()); i = int(np.argmax(m)); lo = np.where(l[:i] <= -dbd)[0]; hi = np.where(l[i:] <= -dbd)[0]
    if not lo.size or not hi.size: return float('nan')
    def cr(k0, k1):
        t = (-dbd - l[k0])/(l[k1] - l[k0]); return math.exp(math.log(fr[k0]) + t*(math.log(fr[k1]) - math.log(fr[k0])))
    return float(fr[i]/(cr(i + hi[0] - 1, i + hi[0]) - cr(lo[-1], lo[-1] + 1)))
out = {}
for fk in (5.0, 20.0):
    w0 = fk/BASE; jb = int(np.argmax(np.abs(xb(gam0, w0))))
    fr = w0*np.geomspace(0.05, 3.0, 700); mp = np.array([abs(xb(0*gam0, f)[jb]) for f in fr]); ma = np.array([abs(xb(gam0, f)[jb]) for f in fr])
    d = dict(x=float(x[jb]), section_scale_khz=float(a[jb]*BASE), rho=float(r[jb]/a[jb]))
    for nm, m in (('disp', mp), ('vel', mp*fr)):
        i = int(np.argmax(m))
        d[nm] = dict(f_peak_khz=float(fr[i]*BASE), peak_re_low_db=float(20*np.log10(m[i]/m[0])), Q10=q(fr, m, 10), Q3=q(fr, m, 3))
    i = int(np.argmax(ma)); d['active_disp'] = dict(f_peak_khz=float(fr[i]*BASE), Q10=q(fr, ma, 10), gain_at_peak_db=float(20*np.log10(ma[i]/mp[i])))
    out[str(fk)] = d; print(fk, json.dumps(d), flush=True)
json.dump(out, open(sys.argv[1], 'w'), indent=1)
