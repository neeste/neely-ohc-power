"""tf.py: BM and RL displacement re stapes pressure, magnitude and phase,
at fixed places, versus frequency and level, plus the passive (gamma=0)
line. Single-tone self-consistent saturation. Usage: tf.py cat|human"""
import sys, json, math
import numpy as np
sys.path.insert(0, '.')
import exp_accumulation as E
import tl_model as T
N = int(__import__("os").environ.get("NPTS", "1024"))

def sweep(lineF, x, places, freqs_of, levels, Afun, Cfun, psat):
    out = []
    for xp in places:
        j = int(round(xp*N)); fr = freqs_of(xp)
        rec = dict(x=xp, f_khz=fr['khz'].tolist(), levels=list(levels), bm=[], rl=[])
        for kind in ('passive',) + tuple(levels):
            mb, pb, mr, pr = [], [], [], []
            s = np.zeros(N+1)
            for f in fr['dim']:
                if kind == 'passive':
                    R = lineF(f, None, passive=True)
                else:
                    A = Afun(kind); C = Cfun(f)
                    def FF(sv):
                        rr = lineF(f, sv)
                        u = (A*np.abs(rr['shear'])/C)**psat
                        return u/(1+u)
                    s, _, _ = E.anderson(FF, s)
                    R = lineF(f, s)
                P0 = R['P'][0]
                xb = R['X_b'][j]/P0; xr = R['X_r'][j]/P0
                mb.append(20*math.log10(abs(xb))); pb.append(float(np.angle(xb)))
                mr.append(20*math.log10(abs(xr))); pr.append(float(np.angle(xr)))
            rec['bm'].append(dict(level=kind, mag_db=mb, phase_cyc=(np.unwrap(pb)/(2*np.pi)).tolist()))
            rec['rl'].append(dict(level=kind, mag_db=mr, phase_cyc=(np.unwrap(pr)/(2*np.pi)).tolist()))
        out.append(rec); print('place', xp, 'done', flush=True)
    return out

def cat():
    import score_base as SB
    r, g0 = SB.profiles(N); x = np.linspace(0, 1, N+1); a = SB.COCH.cf(x)
    tap = SB.COCH.taper(x)
    def lineF(f, s, passive=False):
        if passive:
            Y, Yr, Ho = E.terms(SB.COCH.sec, f, a, r, 0*a)
        else:
            Y, Yr, Ho = E.terms(SB.COCH.sec, f, a, r, g0*tap*(1-s))
        P = T._fd(SB.COCH, f, Y, N, 1.0); Xb = Y*P
        return dict(P=P, X_b=Xb, X_r=Yr*P, shear=Ho*Xb)
    fp = float(SB.COCH.cf(E.X_PROBE))
    Cp = float(np.max(np.abs(lineF(fp, np.zeros(N+1))['shear'])))
    prof = json.load(open('out/profiles.json'))['cat']
    def cf_of(xp):
        o = np.argsort(prof['pm_x'])
        return float(np.exp(np.interp(xp, np.asarray(prof['pm_x'])[o], np.log(np.asarray(prof['pm_f_khz'])[o]))))
    def freqs_of(xp):
        k = cf_of(xp)*np.geomspace(0.1, 1.6, 90)
        return dict(khz=k, dim=k/SB.BASE)
    CAL = 68.31
    import os
    if os.path.exists('base_battery.json'):   # C0_supp = 65 - median neural locus (curves.py), at the run's own state
        CAL = 65.0 - float(np.median(json.load(open('base_battery.json'))['locus']['L']))
    print('cat calibration C0_supp %.2f dB' % CAL, flush=True)
    out = sweep(lineF, x, (0.2, 0.465, 0.7), freqs_of, (20, 40, 60, 80, 100),
                lambda L: 10**((L-CAL)/20), lambda f: Cp, 4.0)
    for rec in out: rec['cf_khz'] = cf_of(rec['x'])
    json.dump(dict(calibration_db=CAL, places=out), open('out/tf_cat.json', 'w'))

def human():
    import exp_abr2 as A2, exp_abr5 as A5
    ck = json.load(open('abr5_ckpt.json')); p = dict(ck['params']); S0 = ck['S0']
    H = A2.Human(**p); gc = A5.gc_exact_dense(H)
    H.gam0 = A5.capped_gamma(H, p['mu0'], p['alpha_mu'], gc)
    def lineF(f, s, passive=False):
        if passive:
            Y, Yr, Ho = E.terms(H.coch.sec, f, H.a, H.r, 0*H.a)
            P = T._fd(H.coch, f, Y, N, 1.0); Xb = Y*P
            return dict(P=P, X_b=Xb, X_r=Yr*P, shear=Ho*Xb)
        return H.line2(f, s)
    Cc = {}
    def Cf(f):
        if f not in Cc: Cc[f] = float(np.max(np.abs(H.line2(f, np.zeros(N+1))['shear'])))
        return Cc[f]
    prof = json.load(open('out/profiles.json'))['human']
    def cf_of(xp):
        o = np.argsort(prof['pm_x'])
        return float(np.exp(np.interp(xp, np.asarray(prof['pm_x'])[o], np.log(np.asarray(prof['pm_f_khz'])[o]))))
    def freqs_of(xp):
        k = cf_of(xp)*np.geomspace(0.1, 1.6, 90)
        return dict(khz=k, dim=k/20.7)
    # places nearest 4, 2, 1, 0.5 kHz on the state's own place map (the hs708 version used fixed x = 0.172, 0.266, 0.348, 0.407,
    # whose CFs were 3.82, 1.88, 0.87, 0.46 kHz at hs708); grid points of the N-point line
    o = np.argsort(prof['pm_f_khz']); PF = np.log(np.asarray(prof['pm_f_khz'])[o]); PX = np.asarray(prof['pm_x'])[o]
    XP = tuple(float(round(np.interp(np.log(fk), PF, PX)*N)/N) for fk in (4.0, 2.0, 1.0, 0.5))
    print('places', XP, [round(cf_of(v), 3) for v in XP], flush=True)
    out = sweep(lineF, H.x, XP, freqs_of,
                (20, 40, 60, 80, 100), lambda L: 10**((L-S0)/20), Cf, H.p_sat)
    for rec in out: rec['cf_khz'] = cf_of(rec['x'])
    json.dump(dict(S0=S0, places=out), open('out/tf_human.json', 'w'))

if __name__ == '__main__':
    {'cat': cat, 'human': human}[sys.argv[1]]()
