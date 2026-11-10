"""Stage-1 refit with a BM-gain datum: maximize the smallest normalized
margin over the eleven metrics of the six observations plus BM gain at CF
(active re passive, low level, x = 0.55 probe place) with window 20-60 dB.
Uniform gamma, scale-invariant line, as in the stage-1 search. Starts from
best_d3. Writes bmgain_refit.json."""
import sys, json, math, time
import numpy as np
from dataclasses import replace
from scipy.optimize import minimize
sys.path.insert(0, '.')
import exp_best1 as B
import tl_model as T, exp_accumulation as E
B.P_ORD = 4.0
B.WIN['fwd'] = (0.0, 12.5)
LO = dict(K_t=5e-4, M_t=0.02, R_t=5e-4, R_c=0.005, R_b=0.03, K_c=0.01, K_p=0.05, M_r=0.05,
          gamma=0.01, q=0.002, m=0.7, u_thr=5e-4, s_g=1e-4)
HI = dict(K_t=0.05, M_t=0.6, R_t=0.08, R_c=0.2, R_b=0.4, K_c=0.2, K_p=0.5, M_r=0.6,
          gamma=0.30, q=0.5, m=2.3, u_thr=3.0, s_g=0.5)
GWIN = (20.0, 60.0)
def bm_gain(p):
    kt, mt, rt, rc, rb, kc, kp, mr, g = p[:9]
    coch = replace(T.WORLDS["B"], sec=replace(T.WORLDS["B"].sec, K_t=kt, M_t=mt, R_t=rt,
                   R_c=rc, R_b=rb, K_c=kc, K_p=kp, M_r=mr), N=1024)
    x = np.linspace(0, 1, 1025); a = coch.cf(x); j = int(round(0.55*1024))
    best = -1e9; tt = None
    fr = a[j]*np.geomspace(0.3, 1.3, 120)
    act = []; pas = []
    for f in fr:
        for gg, lst in ((g*coch.taper(x), act), (0*x, pas)):
            Y, Yr, Ho = E.terms(coch.sec, f, a, None, gg)
            P = T._fd(coch, f, Y, 1024, 1.0); lst.append(abs(Y[j]*P[j]))
    act = 20*np.log10(act); pas = 20*np.log10(pas)
    d = act - pas; i = int(np.argmax(d))
    return float(d[i]), float(fr[i]/a[j])
best = dict(score=-1e9)
def obj(lp):
    p = [math.exp(v) for v in lp]
    for k, v in zip(B.NAMES, p):
        if not (LO[k] <= v <= HI[k]): return 10.0
    try:
        out = B.evaluate(p)
        if out['primary'] <= -9: return 9.0
        gdb, fpk = bm_gain(p)
    except Exception as e:
        return 10.0
    mg = min(gdb-GWIN[0], GWIN[1]-gdb)/((GWIN[1]-GWIN[0])/2)
    margins = dict(out['margins']); margins['bmg'] = mg
    sc = min(margins.values())
    if sc > best['score']:
        best.update(score=sc, p=p, margins=margins, values=dict(out['values'], bmg=gdb, bmg_fpk=fpk),
                    ratio=out['ratio'])
        json.dump(best, open('out/bmgain_refit.json', 'w'), indent=1)
        worst = sorted(margins.items(), key=lambda kv: kv[1])[:3]
        print(f'{time.strftime("%H:%M:%S")} score {sc:+.3f} bmg {gdb:.1f} dB ratio {out["ratio"]:.3f} worst {worst}', flush=True)
    return -sc
if __name__ == '__main__':
    d3 = json.load(open('best_d3.json'))['params']
    p0 = [d3[k] for k in B.NAMES]
    t0 = time.time(); print('start', -obj([math.log(v) for v in p0]), f'{time.time()-t0:.0f}s', flush=True)
    minimize(obj, [math.log(v) for v in p0], method='Nelder-Mead',
             options=dict(maxfev=int(sys.argv[1]), xatol=1e-3, fatol=1e-4))
    print('done', best['score'], flush=True)
