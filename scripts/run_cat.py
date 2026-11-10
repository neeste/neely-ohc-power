"""Run a script at the final cat state c856 (model/best_d3.json + model/base_ckpt.json; force split
lambda from best_d3.json). The population read-out uses the shear-velocity criterion and the
read-out constants of base_ckpt.json by default; FTC_CRIT=disp selects the displacement criterion
and base_ckpt_disp.json (also substituted for a base_ckpt.json argument of the script). Installs lambda (geom), the 24-point gamma_c(rho) table that the
combined fit (combo.py) used, and the lambda version of the energy accounting, then runs the script.
Usage, from model/:  python3 ../scripts/run_cat.py ../scripts/<script>.py [args]"""
import sys, os, json, runpy
import numpy as np
from dataclasses import replace
sys.path.insert(0, '.'); HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(1, HERE)
import geom, deltaT_3dof as d3, tl_model as T
bd = json.load(open('best_d3.json')); geom.LAM = float(bd.get('lam', 1.0)); geom.install()
T.section_terms = lambda sec, w, a, r=None, gamma=None: geom.terms(sec, w, a, r, gamma)
d3.gamma_critical = lambda sec, hi=200.0, tol=1e-6: geom.gamma_critical(sec, 1.0, hi, tol)
import exp_ftc1 as F1
import score_base as SB
rho = SB._damping(F1.A)/F1.A
rg = np.geomspace(max(rho.min()*0.98, 1e-3), rho.max()*1.02, 24); S = F1.SEC
gg = np.array([d3.gamma_critical(replace(S, gamma=0.0, R_b=S.R_b*q, R_r=S.R_r*q, R_t=S.R_t*q, R_p=S.R_p*q, R_c=S.R_c*q)) for q in rg])
SB._LX, SB._GY = np.log(rg), gg; SB._prof_cache.clear()
F1.gc_dense = lambda rh: np.interp(np.log(rh), np.log(rg), gg)
import acc_lam; acc_lam.install()
print('[run_cat] lam %.4f read-out %s (%s)' % (geom.LAM, F1.CRIT, SB.CKPT_FILE), flush=True)
os.makedirs('out', exist_ok=True)   # outputs go to model/out/
script = sys.argv[1]; sys.argv = [SB.CKPT_FILE if a == 'base_ckpt.json' else a for a in sys.argv[1:]]; sys.path.insert(1, os.path.dirname(os.path.abspath(script)))
runpy.run_path(script, run_name='__main__')
