"""Cat, OHC force on BM only: refit only the activity profile (nu0, alpha_nu,
p_nu) to maximize the smallest normalized margin over observations 4-6
(tt, sl, map, phase, AF excursion), certified states only."""
import sys, json, math, numpy as np
from scipy.optimize import minimize
sys.path.insert(0, '.')
import score_base as SB
import geom
geom.GEOM = 'bm'; geom.install(sb=SB._sb_bmk)
import moh_score as MS
from exp_encoded1 import build_rep
from exp_af_source2 import af_run
GC = {}
for N in (1024, 2048, 4096):
    x = np.linspace(0, 1, N+1); a = SB.COCH.cf(x); r_ = SB._damping(a)
    GC[N] = (a, r_, geom.gc_of_rho(SB.COCH.sec, r_/a, n=80, sb=1.0))
WIN = dict(tt=(32, 48), sl=(3, 9), mp=(-0.75, -0.25), ph=(math.pi-0.7, math.pi+0.7), exc=(2, 14))
def setnu(z):
    SB.NU0, SB.ALPHA_NU, SB.P_NU = math.exp(z[0]), z[1], z[2]
    for N, (a, r_, gc) in GC.items():
        SB._prof_cache[N] = (r_, (1 - SB._nu(a))*gc)
best = dict(score=-1e9)
OUT = 'out/refit_bm_c.json'
def obj(z):
    setnu(z)
    try:
        tc, rep = build_rep(SB.COCH, SB.G)
        o4 = MS.obs4(rep); o5 = MS.obs5(rep)
        m = dict(tt=o4.metrics[0].value, sl=o4.metrics[1].value,
                 mp=o5.metrics[0].value, ph=o5.metrics[1].value)
        rows = af_run(SB.COCH, SB.G, 2048)
        flat = [v for rr, v in rows if rr >= 1.25]; m['exc'] = max(flat) - min(flat)
    except Exception as e:
        return 10.0
    marg = {k: min(m[k]-lo, hi-m[k])/((hi-lo)/2) for k, (lo, hi) in WIN.items()}
    sc = min(marg.values())
    if sc > best['score']:
        if not SB.certify(1.0):
            return -sc + 1.0
        best.update(score=sc, z=list(z), metrics=m, margins=marg)
        json.dump(best, open(OUT, 'w'), indent=1)
        print('incumbent', round(sc, 3), {k: round(v, 2) for k, v in m.items()}, [round(v, 3) for v in z], flush=True)
    return -sc
z0 = [math.log(SB.NU0), SB.ALPHA_NU, SB.P_NU]
obj(z0)
minimize(obj, z0, method='Nelder-Mead', options=dict(maxfev=int(sys.argv[1]), xatol=1e-3, fatol=1e-4,
         initial_simplex=[z0]+[list(np.add(z0, d)) for d in np.diag([0.7, 0.2, 0.3])]))
print('done', best, flush=True)
