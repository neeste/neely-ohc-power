"""cert2.py: whole-line stability certification by the argument principle,
vectorized, at two grids and two contour depths, for the cat record and the
human certified state, quiet and at self-consistent saturated profiles.
Run from the model directory:  python3 cert2.py cat|human"""
import sys, json, math, time
import numpy as np
sys.path.insert(0, '.')
import exp_accumulation as E

def winding(sec, F, a, r, gam, N, lo, hi, depth, npts):
    corners = [complex(hi, -depth*hi), complex(hi, 1e-6), complex(lo, 1e-6),
               complex(lo, -depth*hi), complex(hi, -depth*hi)]
    pts = []
    for c0, c1 in zip(corners[:-1], corners[1:]):
        t = np.linspace(0, 1, npts, endpoint=False)
        pts.append(c0 + (c1 - c0) * t)
    W = np.concatenate(pts + [np.array([corners[0]])])
    h = 1.0 / N
    ph = np.empty(W.size)
    for i0 in range(0, W.size, 400):
        w = W[i0:i0+400][:, None]
        Y, _, _ = E.terms(sec, w, a[None, :], r[None, :], gam[None, :])
        A = -(2.0 + h*h*(-(w*w)*F*Y))          # (m, N+1)
        d0 = np.ones(w.shape[0], complex); d1 = A[:, 1].copy()
        for j in range(2, N):
            d2 = A[:, j]*d1 - d0
            m_ = np.abs(d2); m_[m_ == 0] = 1e-300
            d0, d1 = d1/m_, d2/m_
        ph[i0:i0+400] = np.angle(d1)
    dph = np.diff(ph); dph = (dph + np.pi) % (2*np.pi) - np.pi
    return dph.sum()/(2*np.pi), float(np.abs(dph).max())

def certify(sec, F, a, r, gam, N, lgw=(-2.9, 0.35), depth=0.35, nb=10):
    edges = np.logspace(lgw[0], lgw[1], nb + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        npts = 1000
        while True:
            wnd, mx = winding(sec, F, a, r, gam, N, lo, hi, depth, npts)
            if mx < math.pi/2 or npts >= 16000:
                break
            npts *= 4
        out.append((float(lo), float(hi), round(wnd, 3), mx, npts))
    clean = all(abs(o[2]) < 0.1 and o[3] < math.pi/2 for o in out)
    return clean, out

def regrid(x0, y, N):
    return np.interp(np.linspace(0, 1, N+1), x0, y)

def run_cat():
    import score_base as SB
    import exp_twotone as TT
    from exp_best1 import Gated3
    import tl_model as T
    res = {}
    T.gamma_for_gain = lambda *a, **k: (SB.G, float("nan"))
    TT.P_ORD = 4.0
    Ns = (1024, 2048)
    x2 = np.linspace(0, 1, 2049)
    r2, g2 = SB.profiles(2048)
    tap2 = SB.COCH.taper(x2)
    f_p = float(SB.COCH.cf(E.X_PROBE))
    rq = SB.line_grad(SB.COCH, f_p, SB.G, np.zeros(2049), 0.0, 0.0, 2048)
    E.X_READ = T._peak(rq["x"], rq["X_b"])[0]
    states = {'quiet': np.zeros(2049)}
    TT.OCT_BELOW = 1.8
    tt = Gated3(SB.COCH, 30.0, SB.UT, SB.Q, m=SB.M_G, s_g=SB.SG, N=2048)
    for SL in (20, 40, 60, 80):
        s, _, _ = E.anderson(lambda sv: _F(tt, None, tt.P0 + SL, sv),
                             np.zeros(2049))
        states[f'probe SL{SL}'] = s
    for Ls in (-10.0, 10.0, 30.0):
        s, _, _ = E.anderson(lambda sv: _F(tt, Ls, tt.P0 + 8, sv),
                             np.zeros(2049))
        states[f'supp 1.8oct L{Ls:+.0f} + probe SL8'] = s
    for name, s in states.items():
        gam = g2 * (1 - s) * tap2
        for N in Ns:
            a = SB.COCH.cf(np.linspace(0, 1, N+1))
            r = regrid(x2, r2, N); g = regrid(x2, gam, N)
            for depth in (0.35, 1.0):
                t0 = time.time()
                ok, bands = certify(SB.COCH.sec, SB.COCH.F, a, r, g, N,
                                    depth=depth)
                res[f'{name} | N={N} depth={depth}'] = dict(clean=ok,
                    bands=bands, smax=float(np.max(s)))
                print(f'{name:34s} N={N} depth={depth}: clean={ok} '
                      f'smax={np.max(s):.3f} ({time.time()-t0:.0f}s)',
                      flush=True)
    json.dump(res, open('out/cert2_cat.json', 'w'),
              indent=1)

def _F(tt, Ls, Pp, sv):
    # one fixed-point map, identical to TwoTone.solve's inner F
    As = 10.0**(Ls/20.0) if Ls is not None else 0.0
    Ap = 10.0**(Pp/20.0) if Pp is not None else 0.0
    import exp_twotone as TT
    u2 = 0.0
    if As > 0:
        rs = E.line(tt.coch, tt.f_s, tt.g0, sv, tt.beta, tt.depth, tt.N)
        u2 = u2 + (As*np.abs(rs["shear"])/tt.C)**2
    if Ap > 0:
        rp = E.line(tt.coch, tt.f_p, tt.g0, sv, tt.beta, tt.depth, tt.N)
        u2 = u2 + (Ap*np.abs(rp["shear"])/tt.C)**2
    v = u2**(TT.P_ORD/2.0)
    return v/(1.0 + v)

def run_human():
    import exp_abr2 as A2, exp_abr5 as A5
    ck = json.load(open('abr5_ckpt.json'))
    p = dict(ck['params']); S0 = ck['S0']
    H = A2.Human(**p)
    gc = A5.gc_exact_dense(H)
    gam0 = A5.capped_gamma(H, p['mu0'], p['alpha_mu'], gc)
    H.gam0 = gam0
    N0 = len(H.x) - 1
    states = {'quiet': np.zeros(N0+1)}
    for fk, spl in ((0.5, 100), (2.0, 40), (2.0, 100), (8.0, 70), (8.0, 100)):
        if True:
            f = fk*1000.0/A2.F_BASE_HZ if hasattr(A2, 'F_BASE_HZ') else fk/20.7
            amp = 10.0**((spl - S0)/20.0)
            r0 = H.line2(f, np.zeros(N0+1)); C = float(np.max(np.abs(r0['shear'])))
            def FF(sv):
                rr = H.line2(f, sv)
                u = (amp*np.abs(rr['shear'])/C)**H.p_sat
                return u/(1.0+u)
            s, _, _ = E.anderson(FF, np.zeros(N0+1))
            states[f'{fk} kHz {spl} dB SPL'] = s
    res = {}
    for name, s in states.items():
        gam = gam0*(1 - s)
        for N in (2048, 8192):
            x = np.linspace(0, 1, N+1)
            a = np.interp(x, H.x, H.a); r = np.interp(x, H.x, H.r)
            g = np.interp(x, H.x, gam)
            for depth in ((0.35, 1.0) if N <= 2048 else (0.35,)):
                t0 = time.time()
                ok, bands = certify(H.coch.sec, H.coch.F, a, r, g, N,
                                    lgw=(-3.15, 0.35), depth=depth)
                res[f'{name} | N={N} depth={depth}'] = dict(clean=ok,
                    bands=bands, smax=float(np.max(s)))
                print(f'{name:22s} N={N} depth={depth}: clean={ok} '
                      f'smax={np.max(s):.3f} ({time.time()-t0:.0f}s)',
                      flush=True)
    json.dump(res, open('out/cert2_human.json', 'w'),
              indent=1)

if __name__ == '__main__':
    {'cat': run_cat, 'human': run_human}[sys.argv[1]]()
