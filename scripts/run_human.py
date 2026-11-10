"""Run a script at the final human state (model/abr5_ckpt.json = joint_hs_hs856.json: the cat c856
section, lambda and nu floor stored in the file), N = 8192 unless NPTS is set. Installs lambda,
the state's own nu floor in exp_abr5.capped_gamma, and the lambda energy accounting.
Usage, from model/:  python3 ../scripts/run_human.py ../scripts/<script>.py [args]"""
import sys, os, json, runpy
os.environ.setdefault('NPTS', '8192')
import numpy as np
sys.path.insert(0, '.'); HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(1, HERE)
ck = json.load(open('abr5_ckpt.json'))
os.environ.setdefault('LAM', str(ck.get('lam', 1.0)))
import geom, deltaT_3dof as d3, tl_model as T
geom.LAM = float(os.environ['LAM']); geom.install()
d3.gamma_critical = lambda sec, hi=200.0, tol=1e-6: geom.gamma_critical(sec, 1.0, hi, tol)
import exp_abr2 as A2, exp_abr5 as A5
A2.N = int(os.environ['NPTS']); NFL = float(ck.get('nu_floor', 3e-3))
def capped_gamma(H, mu0, alpha_mu, gc_ex):
    nu = np.clip(mu0*H.a**alpha_mu, NFL, 1.0); mu = 1.0 - nu
    t = np.clip((H.x - A5.X0)/(A5.X1 - A5.X0), 0.0, 1.0); w = 1.0 - (3*t*t - 2*t**3)
    mu = (1 - w)*mu + w*np.minimum(mu, A5.CAP)
    return mu*gc_ex*H.coch.taper(H.x)
A5.capped_gamma = capped_gamma
import acc_lam; acc_lam.install()
print('[run_human] lam %.4f N %d nu_floor %.4f' % (geom.LAM, A2.N, NFL), flush=True)
os.makedirs('out', exist_ok=True)   # outputs go to model/out/
script = sys.argv[1]; sys.argv = sys.argv[1:]; sys.path.insert(1, os.path.dirname(os.path.abspath(script)))
runpy.run_path(script, run_name='__main__')
