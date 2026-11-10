"""Energy accounting at the peak: who pays the viscous bill?

  viscous dissipation  =  power taken from the wave  +  power from the OHCs
  D_visc(x)            =  0.5 Re{Z_p} |v_b|^2       +  p_ohc(x)

The left side is computed independently from the five dashpots of the
section, so the identity is a check on the whole accounting, not an
assumption. The OHC share s = p_ohc / D_visc is the quantity of interest:
s < 1 means the wave still loses energy there, s > 1 means it gains.
"""
import sys, json, math
from dataclasses import replace
import numpy as np
sys.path.insert(0, '.')
import tl_model as T
import exp_abr2 as A2

B = json.load(open('best_d3.json'))['params']
SEC = ('K_t', 'M_t', 'R_t', 'R_c', 'R_b', 'K_c', 'K_p', 'M_r')
G = B['gamma']
OUT = {}


def cat(**over):
    s = {k: B[k] for k in SEC}; s.update(over)
    return replace(T.WORLDS['B'], sec=replace(T.WORLDS['B'].sec, **s))


def accounting(sec, a, r, gam, w, Xb, Xr, Ho, Zp):
    """Returns (D_visc, p_fluid, p_ohc) densities."""
    Xt = Xr - Ho * Xb / sec.G
    vb, vr, vt = 1j * w * Xb, 1j * w * Xr, 1j * w * Xt
    D = 0.5 * (sec.R_b * r * np.abs(vb) ** 2
               + sec.R_p * r * np.abs(vr - vb) ** 2
               + sec.R_r * r * np.abs(vr) ** 2
               + sec.R_c * r * np.abs(vr - vt) ** 2
               + sec.R_t * r * np.abs(vt) ** 2)
    p_fluid = 0.5 * np.real(Zp) * np.abs(vb) ** 2
    p_ohc = 0.5 * gam * sec.K_a * a ** 2 * np.real(
        Ho * Xb * np.conj(vr - vb))
    return D, p_fluid, p_ohc


def report(oc, D, pf, po, tag, win=(-1.0, 1.5)):
    m = (oc > win[0]) & (oc < win[1])
    err = float(np.max(np.abs(D[m] - pf[m] - po[m])) / np.max(D[m]))
    o = oc[m][::-1]
    tD = float(np.trapezoid(D[m][::-1], o))
    tO = float(np.trapezoid(po[m][::-1], o))
    share = po[m] / D[m]
    i = int(np.argmax(share))
    print(f'   {tag}: identity error {err:.2e}; over [{win[0]},{win[1]}] oct '
          f'the OHCs supply {100 * tO / tD:.1f}% of the viscous loss; '
          f'peak local share {share[i]:.3f} at {oc[m][i]:+.2f} oct')
    y = np.clip(po[m], 0, None)[::-1]
    c = np.cumsum(y) / np.sum(y)
    lo = float(np.interp(0.10, c, o)); hi = float(np.interp(0.90, c, o))
    cen = float(np.trapezoid(y * o, o) / np.trapezoid(y, o))
    print(f'          injection centroid {cen:+.3f} oct, 80% band '
          f'[{lo:+.2f},{hi:+.2f}] (width {hi - lo:.2f} oct)')
    return dict(err=err, frac=100 * tO / tD, peak_share=float(share[i]),
                at=float(oc[m][i]), cen=cen, lo=lo, hi=hi)


def run_cat():
    coch = cat()
    print('[G1] cat model')
    res = {}
    for f_oct in (-1.0, 0.0, 1.0):
        f = float(coch.cf(0.5)) * 2 ** f_oct
        r = T.response(coch, f, N=8192, gamma=G)
        x, a, Xb, Xr, Ho, Zp = (r['x'], r['cf'], r['X_b'], r['X_r'],
                                r['H_o'], r['Z_p'])
        xp = x[int(np.argmax(np.abs(Xb)))]
        oc = np.log2(a / float(np.interp(xp, x, a)))
        D, pf, po = accounting(coch.sec, a, a, G, f, Xb, Xr, Ho, Zp)
        res[f'{f_oct:+.0f}'] = report(oc, D, pf, po, f'f = CF x 2^{f_oct:+.0f}')
    OUT['cat'] = res


def run_human():
    ck = json.load(open('abr4_ckpt.json'))
    H = A2.Human(**ck['params'])
    print('\n[G2] human model')
    rows = []
    for f_khz in (0.5, 1.0, 2.0, 4.0):
        w = f_khz * 1000.0 / A2.F_BASE_HZ
        rr = H.line2(w, np.zeros(A2.N + 1))
        Xb, Xr, Ho, Y = rr['X_b'], rr['X_r'], rr['H_o'], rr['Y']
        Zp = 1.0 / (1j * w * Y)
        xp = H.x[int(np.argmax(np.abs(Xb)))]
        oc = np.log2(H.a / float(np.interp(xp, H.x, H.a)))
        D, pf, po = accounting(H.coch.sec, H.a, H.r, H.gam0, w, Xb, Xr,
                               Ho, Zp)
        rows.append(report(oc, D, pf, po, f'{f_khz:4.2f} kHz'))
    OUT['human'] = rows


def run_sweep():
    print('\n[G3] TM resonance sweep: strength and placement')
    base = math.sqrt(B['K_t'] / B['M_t'])
    rows = []
    for sc in (0.4, 0.7, 1.0, 1.3, 1.7):
        c = cat(K_t=B['K_t'] * sc ** 2)
        gc = c.gamma_critical()
        gam = 0.9032 * gc
        f = float(c.cf(0.5))
        r = T.response(c, f, N=4096, gamma=gam)
        r0 = T.response(c, f, N=4096, gamma=0.0)
        x, a, Xb, Xr, Ho, Zp = (r['x'], r['cf'], r['X_b'], r['X_r'],
                                r['H_o'], r['Z_p'])
        xp = x[int(np.argmax(np.abs(Xb)))]
        oc = np.log2(a / float(np.interp(xp, x, a)))
        D, pf, po = accounting(c.sec, a, a, gam, f, Xb, Xr, Ho, Zp)
        g = 20 * math.log10(np.max(np.abs(Xb)) / np.max(np.abs(r0['X_b'])))
        s = report(oc, D, pf, po, f'w_t/CF {base * sc:5.3f}')
        s.update(rho=base * sc, gain=g, gc=float(gc))
        print(f'          BM gain {g:+.2f} dB, gamma_c {gc:.4f}')
        rows.append(s)
    OUT['sweep'] = rows


def main():
    print('=' * 74)
    run_cat()
    run_human()
    run_sweep()
    json.dump(OUT, open('place_data5.json', 'w'), indent=1)
    print('\nwrote place_data5.json')


if __name__ == '__main__':
    main()
