"""profiles.py: place profiles and low-level place map for the cat record
(score_base: best_d3.json + base_ckpt.json) and the human certified state
(abr5_ckpt.json). Writes profiles.json. Run from the model directory."""
import sys, json, math
import numpy as np
sys.path.insert(0, '.')
import score_base as SB
import tl_model as T
import exp_accumulation as E
import exp_abr2 as A2
import exp_abr5 as A5
import deltaT_3dof as d3
from dataclasses import replace

out = {}
# ---------------- cat ----------------
N = 1024
x = np.linspace(0, 1, N + 1)
a = SB.COCH.cf(x)
r, gam0 = SB.profiles(N)
rho = r / a
nu = SB._nu(a)
sb = SB._sb_bmk(a) if hasattr(SB, "_sb_bmk") else np.ones_like(a)   # no BM-stiffness gradient at a 12-parameter state
gc_nom = gam0 / (1 - nu)
taper = SB.COCH.taper(x)
# low-level place map: BM |X_b| peak place on the quiet line
fr = np.geomspace(0.0012, 0.55, 40)
xpk = []
for f in fr:
    Rq = SB.line_grad(SB.COCH, f, gam0, np.zeros(N + 1), 0.0, 0.0, N)
    xpk.append(float(x[int(np.argmax(np.abs(Rq["X_b"])))]))
out['cat'] = dict(x=x[::8].tolist(), map_khz=(a * SB.BASE)[::8].tolist(),
                  rho=rho[::8].tolist(), nu=nu[::8].tolist(),
                  S_b=sb[::8].tolist(), taper=taper[::8].tolist(),
                  gam0=gam0[::8].tolist(), gc_nominal=gc_nom[::8].tolist(),
                  pm_f_khz=(fr * SB.BASE).tolist(), pm_x=xpk,
                  B_khz=SB.BASE)
# ---------------- human ----------------
ck = json.load(open(A5.CKPT))
print('abr5 keys', list(ck.keys()))
json.dump(out, open('out/profiles.json', 'w'))
print('cat done; place map (f kHz, x):')
for f, xx in zip(fr[::5] * SB.BASE, xpk[::5]):
    print(f'  {f:8.3f}  {xx:.4f}')
