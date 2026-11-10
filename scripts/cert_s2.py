"""Argument-principle certification of a cat stage-2 (combo) state, quiet and at
self-consistent saturated profiles, at N = 1024 and 2048 and contour depths 0.35 and 1.0.
Winding code as in verify8k/cert2_h8k.py (vectorized), with the force split lambda of the state.
Saturation: single probe tone at place x0, s = U^4/(1+U^4), U(x) = A_p |shear(x; s)| / C,
C = peak |shear| of the unsaturated line at that frequency (u = 1 at its own unsaturated peak),
solved by Anderson iteration (exp_accumulation.anderson); probe 0, 20, 40 dB re the knee.
Run from model/ via run_cat.py.  Usage: run_cat.py cert_s2.py base_ckpt.json out/cert_s2.json"""
import sys, json, math, time
import numpy as np
sys.path.insert(0, '.')
import geom, deltaT_3dof as d3, tl_model as T
bd = json.load(open('best_d3.json')); geom.LAM = bd.get('lam', 1.0); geom.install()
T.section_terms = lambda sec, w, a, r=None, gamma=None: geom.terms(sec, w, a, r, gamma)
d3.gamma_critical = lambda sec, hi=200.0, tol=1e-6: geom.gamma_critical(sec, 1.0, hi, tol)
import exp_accumulation as E, exp_ftc1 as F1, exp_ftc2 as X

def winding(sec, F, a, r, gam, N, lo, hi, depth, npts):
    corners = [complex(hi, -depth*hi), complex(hi, 1e-6), complex(lo, 1e-6),
               complex(lo, -depth*hi), complex(hi, -depth*hi)]
    pts = []
    for c0, c1 in zip(corners[:-1], corners[1:]):
        t = np.linspace(0, 1, npts, endpoint=False); pts.append(c0 + (c1 - c0)*t)
    W = np.concatenate(pts + [np.array([corners[0]])]); h = 1.0/N; ph = np.empty(W.size)
    for i0 in range(0, W.size, 400):
        w = W[i0:i0+400][:, None]
        Y, _, _ = E.terms(sec, w, a[None, :], r[None, :], gam[None, :])
        A = -(2.0 + h*h*(-(w*w)*F*Y)); d0 = np.ones(w.shape[0], complex); d1 = A[:, 1].copy()
        for j in range(2, N):
            d2 = A[:, j]*d1 - d0; m_ = np.abs(d2); m_[m_ == 0] = 1e-300; d0, d1 = d1/m_, d2/m_
        ph[i0:i0+400] = np.angle(d1)
    dph = np.diff(ph); dph = (dph + np.pi) % (2*np.pi) - np.pi
    return dph.sum()/(2*np.pi), float(np.abs(dph).max())

def certify(sec, F, a, r, gam, N, lgw=(-2.9, 0.35), depth=0.35, nb=10):
    edges = np.logspace(lgw[0], lgw[1], nb + 1); out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        npts = 1000
        while True:
            wnd, mx = winding(sec, F, a, r, gam, N, lo, hi, depth, npts)
            if mx < math.pi/2 or npts >= 16000: break
            npts *= 4
        out.append((float(lo), float(hi), round(wnd, 3), mx, npts))
    return all(abs(o[2]) < 0.1 and o[3] < math.pi/2 for o in out), out

ck = json.load(open(sys.argv[1])); z = ck['z']
r1, g1 = X.profiles(z); N1 = F1.N; x1 = np.linspace(0, 1, N1 + 1)
SEC, COCH, A1 = F1.SEC, F1.COCH, F1.A

def shear(f, gam):
    Y, Y_r, H_o = E.terms(SEC, f, A1, r1, gam); P = T._fd(COCH, f, Y, N1, 1.0)
    return np.abs(H_o*Y*P)

states = {'quiet': np.zeros(N1 + 1)}
for x0 in (0.3, 0.5, 0.7):
    j0 = int(np.argmin(np.abs(F1.X - x0))); f = float(A1[j0]); C = float(shear(f, g1).max())
    for L in (0.0, 20.0, 40.0):
        Ap = 10**(L/20)
        def Fm(s):
            v = (Ap*shear(f, g1*(1 - s))/C)**4; return v/(1 + v)
        s, _, _ = E.anderson(Fm, np.zeros(N1 + 1))
        states['probe x%.1f L%+.0f' % (x0, L)] = s
res = {}
for name, s in states.items():
    gam = g1*(1 - s)
    for N in (1024, 2048):
        x = np.linspace(0, 1, N + 1); a = COCH.cf(x)
        r = np.interp(x, x1, r1); g = np.interp(x, x1, gam)
        for depth in (0.35, 1.0):
            t0 = time.time(); ok, bands = certify(SEC, COCH.F, a, r, g, N, depth=depth)
            res['%s | N=%d depth=%.2f' % (name, N, depth)] = dict(clean=ok, bands=bands, smax=float(s.max()))
            print('%-20s N=%d depth=%.2f clean=%s smax=%.3f (%.0fs)' % (name, N, depth, ok, s.max(), time.time() - t0), flush=True)
            json.dump(dict(lam=geom.LAM, source=sys.argv[1], results=res), open(sys.argv[2], 'w'), indent=1)
print('clean %d of %d' % (sum(v['clean'] for v in res.values()), len(res)), flush=True)
