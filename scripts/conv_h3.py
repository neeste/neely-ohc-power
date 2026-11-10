import sys, json, math, numpy as np
sys.path.insert(0, '.')
import exp_abr2 as A2, exp_abr5 as A5, exp_accumulation as E, tl_model as T
ck = json.load(open('abr5_ckpt.json')); p = dict(ck['params']); S0 = ck['S0']
for N in [int(v) for v in sys.argv[1:]]:
    A2.N = N
    H = A2.Human(**p); gc = A5.gc_exact_dense(H); g0 = A5.capped_gamma(H, p['mu0'], p['alpha_mu'], gc)
    out = []
    for fk, spl in ((1.0, 40), (4.0, 40), (1.0, 0)):
        f0 = fk/20.7
        def lin(ff, s):
            Y, Yr, Ho = E.terms(H.coch.sec, ff, H.a, H.r, g0*(1-s)); P = T._fd(H.coch, ff, Y, N, 1.0); return Ho*Y*P
        z = np.zeros(N+1); C = np.max(np.abs(lin(f0, z)))
        if spl > 0:
            A = 10**((spl-S0)/20)
            def FF(sv):
                u = (A*np.abs(lin(f0, sv))/C)**H.p_sat; return u/(1+u)
            s, _, _ = E.anderson(FF, z)
        else: s = z
        sh = lin(f0, s); j = int(np.argmax(np.abs(sh)))
        fs = np.linspace(f0*0.99, f0*1.01, 9); ph = [np.angle(lin(ff, s)[j]) for ff in fs]
        tau = -np.polyfit(fs, np.unwrap(ph), 1)[0]/(2*np.pi)
        out.append('%g kHz %d dB: 2N %.2f, peak %.2f dB' % (fk, spl, 2*tau*f0, 20*math.log10(abs(sh[j]))))
    print('N=%d  ' % N + ' | '.join(out), flush=True)
