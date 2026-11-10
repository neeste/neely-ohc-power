"""Cat tuning ratio r = Q_ERB / N_SF (Shera, Guinan and Oxenham 2010).
Q_ERB from the low-level shear-velocity threshold curve at each place,
ERB = integral of 10^(-(thr - thr_min)/10) df; N_SF from shear group
delay at the same CF (sfoae_cat.json)."""
import sys, json, math, numpy as np
sys.path.insert(0, '.')
import score_base as SB, exp_accumulation as E, tl_model as T
N = 1024
X = np.linspace(0, 1, N+1); A = SB.COCH.cf(X); r, g0 = SB.profiles(N); gam = g0*SB.COCH.taper(X)
sf = json.load(open('out/sfoae_cat.json'))
fsf = np.array([c['f_khz'] for c in sf]); nsf = np.array([c['N_SF_shear'] for c in sf])
out = []
for x0 in np.linspace(0.08, 0.8, 13):
    j = int(round(x0*N)); fr = A[j]*np.geomspace(0.1, 1.6, 400); resp = []
    for f in fr:
        Y, Yr, Ho = E.terms(SB.COCH.sec, f, A, r, gam); P = T._fd(SB.COCH, f, Y, N, 1.0)
        resp.append(abs((Ho*Y*P)[j])*f)
    thr = -20*np.log10(resp); i = int(np.argmin(thr)); ftip = fr[i]*SB.BASE
    g2 = 10**(-(thr - thr[i])/10); erb = np.trapezoid(g2, fr*SB.BASE)
    qerb = ftip/erb; n = float(np.interp(math.log(ftip), np.log(fsf), nsf))
    out.append(dict(cf_khz=float(ftip), q_erb=float(qerb), n_sf=n, ratio=float(qerb/n)))
    print('CF %6.2f kHz  Q_ERB %5.2f  N_SF %5.2f  r %.2f' % (ftip, qerb, n, qerb/n), flush=True)
json.dump(out, open('out/tuning_ratio_cat.json', 'w'), indent=1)
