import sys, json, math, numpy as np
sys.path.insert(0, '.')
import score_base as SB, exp_accumulation as E, tl_model as T
N = 2048
x = np.linspace(0, 1, N+1); a = SB.COCH.cf(x); r, g0 = SB.profiles(N); gam = g0*SB.COCH.taper(x)
prof = json.load(open('out/profiles.json'))['cat']
o = np.argsort(prof['pm_x']); pmx = np.asarray(prof['pm_x'])[o]; pmf = np.asarray(prof['pm_f_khz'])[o]
def resp(f, g):
    Y, Yr, Ho = E.terms(SB.COCH.sec, f, a, r, g)
    P = T._fd(SB.COCH, f, Y, N, 1.0); return Y*P, Ho*Y*P
out = []
for fk in np.geomspace(0.25, 32, 22):
    f = fk/SB.BASE
    xb, sh = resp(f, gam)
    jb = int(np.argmax(np.abs(xb[int(0.02*N):]))) + int(0.02*N)
    js = int(np.argmax(np.abs(sh)))
    ph = []
    for ff in (f*0.99, f*1.01):
        xb2, sh2 = resp(ff, gam); ph.append((np.angle(xb2[js]), np.angle(sh2[js])))
    d_bm = (ph[1][0]-ph[0][0]+np.pi) % (2*np.pi) - np.pi
    d_sh = (ph[1][1]-ph[0][1]+np.pi) % (2*np.pi) - np.pi
    # phase difference across 2 percent may exceed pi; use finer unwrap
    fs = np.linspace(f*0.99, f*1.01, 9); pb = []; psh = []
    for ff in fs:
        xb2, sh2 = resp(ff, gam); pb.append(np.angle(xb2[js])); psh.append(np.angle(sh2[js]))
    pb = np.unwrap(pb); psh = np.unwrap(psh)
    tau_bm = -np.polyfit(fs, pb, 1)[0]/(2*np.pi)   # in units of 1/B (kHz^-1 -> ms)
    tau_sh = -np.polyfit(fs, psh, 1)[0]/(2*np.pi)
    tb = tau_bm/SB.BASE; ts = tau_sh/SB.BASE      # ms
    xpas, _ = resp(f, 0*gam)
    out.append(dict(f_khz=float(fk), x_shear_peak=float(x[js]), x_bm_peak=float(x[jb]),
                    tau_bm_ms=float(tb), tau_shear_ms=float(ts), N_SF=float(2*tb*fk), N_SF_shear=float(2*ts*fk)))
    print(f'{fk:7.2f} kHz  x_sh {x[js]:.3f} x_bm {x[jb]:.3f}  tau_bm {tb:.3f} ms  N_SF(BM) {2*tb*fk:5.2f}  N_SF(shear) {2*ts*fk:5.2f}', flush=True)
json.dump(out, open('out/sfoae_cat.json', 'w'), indent=1)
