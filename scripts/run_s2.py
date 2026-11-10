"""Cat stage 2 (place profiles) at the adopted force split.
Section from paper/rev/lam_refit_l0.7.json (via best_d3.json here); start z from the
lambda = 1 stage-2 incumbent. Checkpoint out/ftc2_lam_ckpt.json (the record of the run is earlier/ftc2_lam_ckpt.json).
Usage: python3 run_s2.py [minutes]"""
import sys, json
sys.path.insert(0, '.')
import geom, deltaT_3dof as d3, tl_model as T
LAM = json.load(open('best_d3.json'))['lam']
geom.LAM = LAM; geom.install()
T.section_terms = lambda sec, w, a, r=None, gamma=None: geom.terms(sec, w, a, r, gamma)
d3.gamma_critical = lambda sec, hi=200.0, tol=1e-6: geom.gamma_critical(sec, 1.0, hi, tol)
import exp_ftc2 as X
X.CKPT = 'out/ftc2_lam_ckpt.json'
print('stage 2 at lambda %.4f' % LAM, flush=True)
X.main()
