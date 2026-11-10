"""R2.s test: wavenumber Re k = d(phase)/dx and decay Im k of BM
displacement, active vs passive, at fixed places relative to the active
shear peak; cat record (N = 4096), 4 kHz, low level."""
import sys, json, numpy as np
sys.path.insert(0, '.')
import score_base as SB, exp_accumulation as E, tl_model as T
N = 4096
x = np.linspace(0, 1, N+1); a = SB.COCH.cf(x); r, g0 = SB.profiles(N); gam = g0*SB.COCH.taper(x)
out = {}
for fk in (1.0, 4.0, 12.0):
    f = fk/SB.BASE; res = {}
    for nm, g in (('active', gam), ('passive', 0*gam)):
        Y, Yr, Ho = E.terms(SB.COCH.sec, f, a, r, g); P = T._fd(SB.COCH, f, Y, N, 1.0); Xb = Y*P
        ph = np.unwrap(np.angle(Xb)); lm = np.log(np.abs(Xb))
        res[nm] = dict(k=-np.gradient(ph, x), alpha=np.gradient(lm, x), mag=20*np.log10(np.abs(Xb)), sh=np.abs(Ho*Xb))
    j = int(np.argmax(res['active']['sh'])); jp = int(np.argmax(res['passive']['mag'][int(0.02*N):])) + int(0.02*N)
    rows = []
    for dx in (-0.10, -0.05, -0.02, 0.0, 0.01):
        i = j + int(round(dx*N))
        rows.append(dict(dx=dx, k_act=float(res['active']['k'][i]), k_pas=float(res['passive']['k'][i]),
                         mag_act=float(res['active']['mag'][i]), mag_pas=float(res['passive']['mag'][i])))
    out[fk] = dict(x_peak_active=float(x[j]), x_peak_passive_bm=float(x[jp]), rows=rows)
    print(fk, 'kHz: active shear peak x %.3f; passive BM peak x %.3f' % (x[j], x[jp]))
    for rw in rows: print('   dx %+.2f  k act %7.1f  pas %7.1f  (ratio %.2f)  |X_b| act %.1f pas %.1f dB' % (rw['dx'], rw['k_act'], rw['k_pas'], rw['k_act']/rw['k_pas'], rw['mag_act'], rw['mag_pas']))
json.dump(out, open('out/wavenumber.json', 'w'), indent=1)
