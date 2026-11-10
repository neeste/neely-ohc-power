"""Argument-principle winding certification (stage2/tools/cert_s2.py, verbatim winding code) for a run2 state
object with .solve-compatible profiles. certify_state(S_maker, ...) runs quiet and saturated (single probe at
x0 0.3, 0.5, 0.7; 0, 20, 40 dB re the knee) at N 1024 and 2048, contour depths 0.35 and 1.0."""
import math, time, numpy as np
import exp_accumulation as E
def winding(sec, F, a, r, gam, N, lo, hi, depth, npts):
    corners = [complex(hi, -depth*hi), complex(hi, 1e-6), complex(lo, 1e-6), complex(lo, -depth*hi), complex(hi, -depth*hi)]
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
def certify(sec, F, a, r, gam, N, lgw=(-2.9, 0.35), depth=0.35, nb=10, stop_early=True):
    edges = np.logspace(lgw[0], lgw[1], nb + 1); out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        npts = 1000
        while True:
            wnd, mx = winding(sec, F, a, r, gam, N, lo, hi, depth, npts)
            if mx < math.pi/2 or npts >= 16000: break
            npts *= 4
        out.append((float(lo), float(hi), round(wnd, 3), mx, npts))
        if stop_early and not (abs(out[-1][2]) < 0.1 and out[-1][3] < math.pi/2): break
    return all(abs(o[2]) < 0.1 and o[3] < math.pi/2 for o in out), out
def sat_profiles(S, x0s=(0.3, 0.5, 0.7), Ls=(0.0, 20.0, 40.0)):
    """S: state on its own grid; returns dict name -> s(x) on S's grid."""
    out = {'quiet': np.zeros(S.N + 1)}
    for x0 in x0s:
        j0 = int(np.argmin(np.abs(S.x - x0))); f = float(S.a[j0])
        C = float(np.abs(S.fields(f)['sh']).max())
        for L in Ls:
            Ap = 10**(L/20)
            def Fm(s):
                v = (Ap*np.abs(S.fields(f, S.gam*(1 - s))['sh'])/C)**4; return v/(1 + v)
            s, _, _ = E.anderson(Fm, np.zeros(S.N + 1)); out['probe x%.1f L%+.0f' % (x0, L)] = s
    return out
def certify_state(make, sec, F, Ns=(1024, 2048), depths=(0.35, 1.0), quick=False, log=print):
    """make(N) -> state. Saturated profiles from the N=Ns[0] state, interpolated to the others.
    quick=True: quiet and the 40-dB probes only. Returns (n_clean, n_total, details)."""
    S0 = make(Ns[0]); sats = sat_profiles(S0, Ls=(40.0,) if quick else (0.0, 20.0, 40.0)); res = {}
    for N in Ns:
        S = make(N); S._set()
        for name, s in sats.items():
            g = S.gam*(1 - np.interp(S.x, S0.x, s))
            for depth in depths:
                t0 = time.time(); ok, bands = certify(sec, F, S.a, S.r, g, N, depth=depth)
                res['%s | N=%d depth=%.2f' % (name, N, depth)] = dict(clean=ok, smax=float(s.max()), bands=bands)
                log('   cert %-18s N=%d depth=%.2f clean=%s (%.0fs)' % (name, N, depth, ok, time.time() - t0))
                if not ok: return sum(v['clean'] for v in res.values()), len(res), res
    return sum(v['clean'] for v in res.values()), len(res), res
