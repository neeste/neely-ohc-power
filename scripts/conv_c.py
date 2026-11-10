import sys, math, numpy as np
sys.path.insert(0, '.')
import score_base as SB, exp_accumulation as E, tl_model as T
for N in (1024, 2048, 4096, 8192):
    x = np.linspace(0, 1, N+1); a = SB.COCH.cf(x); r, g0 = SB.profiles(N); g = g0*SB.COCH.taper(x)
    out = []
    for fk in (1.0, 4.0, 15.0):
        f0 = fk/SB.BASE
        def lin(ff):
            Y, Yr, Ho = E.terms(SB.COCH.sec, ff, a, r, g); P = T._fd(SB.COCH, ff, Y, N, 1.0); return Ho*Y*P
        sh = lin(f0); j = int(np.argmax(np.abs(sh)))
        fs = np.linspace(f0*0.99, f0*1.01, 9); ph = [np.angle(lin(ff)[j]) for ff in fs]
        tau = -np.polyfit(fs, np.unwrap(ph), 1)[0]/(2*np.pi)
        out.append('%g kHz: N_SF %.2f peak %.2f' % (fk, 2*tau*f0, 20*math.log10(abs(sh[j]))))
    print('N=%d ' % N + ' | '.join(out), flush=True)
