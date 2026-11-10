import sys, json, numpy as np
sys.path.insert(0, '.')
import score_base as SB
import geom
geom.GEOM = 'bm'; geom.install(sb=SB._sb_bmk)
for N in (1024, 2048, 4096):
    x = np.linspace(0, 1, N+1); a = SB.COCH.cf(x)
    r_ = SB._damping(a)
    gc = geom.gc_of_rho(SB.COCH.sec, r_/a, n=80, sb=1.0)
    SB._prof_cache[N] = (r_, (1 - SB._nu(a))*gc)
SB.battery()
