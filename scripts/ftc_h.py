"""Human tuning curves at the abr5 state: low-level threshold for a
constant hair-bundle shear-velocity criterion, relative to the tip, at the
places of greatest low-level response to 0.5, 1, 2, 4 and 8 kHz.
Reports tip frequency, Q10, and tip-to-tail (tail read 1 octave below)."""
import sys, json, math, numpy as np
sys.path.insert(0, '.')
import exp_abr2 as A2, exp_abr5 as A5, exp_accumulation as E, tl_model as T
ck = json.load(open('abr5_ckpt.json')); p = dict(ck['params'])
H = A2.Human(**p); gc = A5.gc_exact_dense(H); N = len(H.x)-1
g = A5.capped_gamma(H, p['mu0'], p['alpha_mu'], gc)
out = []
for fk in (0.5, 1, 2, 4, 8):
    f0 = fk/20.7
    Y, Yr, Ho = E.terms(H.coch.sec, f0, H.a, H.r, g); P = T._fd(H.coch, f0, Y, N, 1.0)
    j = int(np.argmax(np.abs(Ho*Y*P)))
    fr = f0*np.geomspace(0.1, 1.6, 160); th = []
    for f in fr:
        Y, Yr, Ho = E.terms(H.coch.sec, f, H.a, H.r, g); P = T._fd(H.coch, f, Y, N, 1.0)
        th.append(-20*math.log10(abs((Ho*Y*P)[j])*f))
    th = np.array(th); i = int(np.argmin(th)); th = th - th[i]; ftip = fr[i]
    lx = np.log(fr)
    def cross(side):
        idx = range(i, -1, -1) if side < 0 else range(i, len(fr))
        prev = i
        for k in idx:
            if th[k] >= 10:
                t = (10 - th[prev])/(th[k]-th[prev]); return math.exp(lx[prev]+t*(lx[k]-lx[prev]))
            prev = k
        return None
    lo, hi = cross(-1), cross(1)
    q10 = ftip/(hi-lo) if lo and hi else None
    tail = float(np.interp(math.log(ftip/2), lx, th))
    out.append(dict(f_khz=fk, x=float(H.x[j]), tip_khz=float(ftip*20.7), q10=q10, tip_to_tail=tail,
                    f=(fr*20.7).tolist(), thr=th.tolist()))
    print(f'{fk} kHz: x {H.x[j]:.3f} tip {ftip*20.7:.3f} kHz Q10 {q10 if q10 is None else round(q10,2)} tip-to-tail(1 oct) {tail:.1f} dB', flush=True)
json.dump(out, open('out/ftc_human.json', 'w'))
