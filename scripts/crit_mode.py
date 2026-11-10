import sys, json, os
sys.path.insert(1, __import__('os').path.dirname(__import__('os').path.abspath(__file__))); import cat_state
sys.path.insert(0, '.')   # model/ modules
import numpy as np
from dataclasses import replace
import geom, tl_model as T, exp_best1 as B
c = json.load(open(cat_state.COMBO))
SECK = ('K_t','M_t','R_t','R_c','R_b','K_c','K_p','M_r')
sec0 = replace(T.WORLDS['B'].sec, **{k: v for k, v in zip(B.NAMES, c['p']) if k in SECK})
Q = float(sys.argv[1]) if len(sys.argv) > 1 else 0.955
def eig(s):
    M, C, K = geom.matrices(s); Mi = np.linalg.inv(M); Z = np.zeros((3,3))
    A = np.block([[Z, np.eye(3)], [-Mi@K, -Mi@C]]); return np.linalg.eigvals(A)
for lam in [0, 0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.70846, 0.8561939116132914, 1.0]:
    geom.LAM = lam
    gc = geom.gamma_critical(replace(sec0, gamma=0.0))
    e = eig(replace(sec0, gamma=gc*0.999999)); e = e[e.imag > 0]
    i = np.argmax(e.real)
    e0 = eig(replace(sec0, gamma=Q*gc)); e0 = e0[e0.imag > 0]; e0 = e0[np.argsort(e0.imag)]
    print('lam %.3f gc %.4f crit mode w/a %.3f | modes at Q gc: ' % (lam, gc, e[i].imag) +
          ' '.join('%.3f(Q%.1f)' % (z.imag, z.imag/(-2*z.real)) for z in e0))
print('mode shapes (|X_b|,|X_r|,|X_t| normalized; phase of X_r, X_t re X_b deg)')
for lam in [0.0, 0.70846, 0.8561939116132914]:
    geom.LAM = lam
    gc = geom.gamma_critical(replace(sec0, gamma=0.0))
    for gam in (0.0, Q*gc):
        s = replace(sec0, gamma=gam); M, C, K = geom.matrices(s); Mi = np.linalg.inv(M); Z = np.zeros((3,3))
        A = np.block([[Z, np.eye(3)], [-Mi@K, -Mi@C]]); ev, V = np.linalg.eig(A)
        for k in np.argsort(ev.imag):
            if ev[k].imag <= 0: continue
            v = V[:3, k]; v = v/v[np.argmax(abs(v))]
            print(' lam %.3f g/gc %.3f  w/a %.3f Q %.1f  |b r t| %.2f %.2f %.2f  ph r,t re b %.0f %.0f' % (lam, gam/gc, ev[k].imag, ev[k].imag/(-2*ev[k].real), *abs(v), np.degrees(np.angle(v[1]/v[0])), np.degrees(np.angle(v[2]/v[0]))))
