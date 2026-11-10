import sys, json, numpy as np
sys.path.insert(0, '.')
import score_base as SB, geom, exp_accumulation as E, tl_model as T
N = 1024
x = np.linspace(0, 1, N+1); a = SB.COCH.cf(x); r = SB._damping(a); tap = SB.COCH.taper(x)
for g in ('pair', 'bm'):
    geom.GEOM = g; geom.install(sb=SB._sb_bmk)
    gc = geom.gc_of_rho(SB.COCH.sec, r/a, n=80)
    for scale_nu in (1.0, 0.5, 0.2, 0.05):
        nu = SB._nu(a)*scale_nu
        gam = (1-nu)*gc*tap
        for xp in (0.2, 0.465):
            j = int(xp*N)
            cf = 0.59*a[j]
            fr = cf*np.geomspace(0.1, 1.5, 300)
            act = []; pas = []; rl = []
            for f in fr:
                for gg, lst in ((gam, act), (0*gam, pas)):
                    Y, Yr, Ho = E.terms(SB.COCH.sec, f, a, r, gg)
                    P = T._fd(SB.COCH, f, Y, N, 1.0)
                    lst.append(abs(Y[j]*P[j]/P[0]))
                    if gg is gam: rl.append(abs(Yr[j]*P[j]/P[0]))
            act = 20*np.log10(act); pas = 20*np.log10(pas); rl = 20*np.log10(rl)
            ia = np.argmax(act[50:])+50
            print(f'{g:4s} nu x{scale_nu:<4} x={xp}: BM active peak {act[ia]:.1f} dB at {fr[ia]/cf:.2f}CF, '
                  f'gain re passive {act[ia]-pas[ia]:.1f} dB, max gain {np.max(act-pas):.1f}, '
                  f'tip-tail(0.1CF) {act[ia]-act[0]:.1f}; RL gain {np.max(rl-pas):.1f}', flush=True)
