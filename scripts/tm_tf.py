"""tm_tf.py (run 3): BM, RL and TM displacement re stapes pressure, magnitude and phase, at one place,
versus frequency and level on the self-consistent saturated line, plus the passive (gamma = 0) line.
Copy of step3/tf.py (same frequency grid, levels, calibration, saturation, Anderson solve) with the
TM displacement added: X_t = X_r - H_o X_b / G (share_s2.py and geom.power convention), and the TM/BM
ratio at the BM peak (maximum of |X_b| for f >= 0.5 CF) for each level.
Cat (from model/):   python3 ../scripts/run_cat.py ../scripts/tm_tf.py cat      (x = 0.465)
Human (from model/): python3 ../scripts/run_human.py ../scripts/tm_tf.py human  (the place nearest 1 kHz, x = 0.2827)
Writes out/tm/tm_<species>.json"""
import sys, json, math, os
import numpy as np
sys.path.insert(0, '.')
import exp_accumulation as E
import tl_model as T
N = int(os.environ.get("NPTS", "1024"))
OUTD = 'out/tm/'; os.makedirs(OUTD, exist_ok=True)

def sweep(lineF, G, places, freqs_of, levels, Afun, Cfun, psat, cf_of):
    out = []
    for xp in places:
        j = int(round(xp*N)); fr = freqs_of(xp)
        rec = dict(x=xp, j=j, N=N, f_khz=fr['khz'].tolist(), levels=list(levels), bm=[], rl=[], tm=[], peak=[])
        for kind in ('passive',) + tuple(levels):
            M = {k: ([], []) for k in ('bm', 'rl', 'tm')}; cx = {k: [] for k in ('bm', 'tm')}
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
                xt = (R['X_r'][j] - R['H_o'][j]*R['X_b'][j]/G)/P0
                for k, v in (('bm', xb), ('rl', xr), ('tm', xt)):
                    M[k][0].append(20*math.log10(abs(v))); M[k][1].append(float(np.angle(v)))
                cx['bm'].append(xb); cx['tm'].append(xt)
            for k in ('bm', 'rl', 'tm'):
                rec[k].append(dict(level=kind, mag_db=M[k][0], phase_cyc=(np.unwrap(M[k][1])/(2*np.pi)).tolist()))
            mb = np.array(M['bm'][0]); mt = np.array(M['tm'][0]); fk = np.asarray(fr['khz'])
            w = np.where(fk >= 0.5*cf_of(xp))[0]          # peak region: f >= 0.5 CF (excludes the low-frequency tail)
            i = int(w[np.argmax(mb[w])]); it = int(w[np.argmax(mt[w])])
            ratio = cx['tm'][i]/cx['bm'][i]
            rec['peak'].append(dict(level=kind, f_bm_peak_khz=float(fr['khz'][i]), bm_peak_db=float(mb[i]),
                                    tm_at_bm_peak_db=float(mt[i]), tm_bm_ratio_db=float(20*math.log10(abs(ratio))),
                                    tm_bm_ratio=float(abs(ratio)), tm_bm_phase_cyc=float(np.angle(ratio)/(2*np.pi)),
                                    f_tm_peak_khz=float(fr['khz'][it]), tm_peak_db=float(mt[it]),
                                    tm_peak_re_bm_peak_db=float(mt[it] - mb[i])))
            print(' place %.4f level %s: BM peak %.2f dB at %.3f kHz, TM/BM at BM peak %.2f dB (%.3f cyc), TM peak %.2f dB at %.3f kHz' % (
                xp, kind, mb[i], fr['khz'][i], 20*math.log10(abs(ratio)), np.angle(ratio)/(2*np.pi), mt[it], fr['khz'][it]), flush=True)
        out.append(rec)
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
        return dict(P=P, X_b=Xb, X_r=Yr*P, shear=Ho*Xb, H_o=Ho)
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
    if os.path.exists('base_battery.json'):
        CAL = 65.0 - float(np.median(json.load(open('base_battery.json'))['locus']['L']))
    print('cat calibration C0_supp %.2f dB, N %d, G %.6g' % (CAL, N, SB.COCH.sec.G), flush=True)
    out = sweep(lineF, SB.COCH.sec.G, (0.465,), freqs_of, (20, 40, 60, 80, 100),
                lambda L: 10**((L-CAL)/20), lambda f: Cp, 4.0, cf_of)
    for rec in out: rec['cf_khz'] = cf_of(rec['x'])
    json.dump(dict(state='c856 (run_cat.py)', calibration_db=CAL, G=SB.COCH.sec.G, places=out), open(OUTD + 'tm_cat.json', 'w'))

def human():
    import exp_abr2 as A2, exp_abr5 as A5
    ck = json.load(open('abr5_ckpt.json')); p = dict(ck['params']); S0 = ck['S0']
    H = A2.Human(**p); gc = A5.gc_exact_dense(H)
    H.gam0 = A5.capped_gamma(H, p['mu0'], p['alpha_mu'], gc)
    def lineF(f, s, passive=False):
        if passive:
            Y, Yr, Ho = E.terms(H.coch.sec, f, H.a, H.r, 0*H.a)
            P = T._fd(H.coch, f, Y, N, 1.0); Xb = Y*P
            return dict(P=P, X_b=Xb, X_r=Yr*P, shear=Ho*Xb, H_o=Ho)
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
    print('human S0 %.2f dB, N %d, G %.6g' % (S0, N, H.coch.sec.G), flush=True)
    # run 9: the place nearest 1 kHz on the state's own place map (run 3 used the fixed x = 0.34765625, CF 0.87 kHz at hs708)
    o = np.argsort(prof['pm_f_khz']); XP = float(round(np.interp(0.0, np.log(np.asarray(prof['pm_f_khz'])[o]), np.asarray(prof['pm_x'])[o])*N)/N)
    print('run 9 TM place', XP, round(cf_of(XP), 3), flush=True)
    out = sweep(lineF, H.coch.sec.G, (XP,), freqs_of,
                (20, 40, 60, 80, 100), lambda L: 10**((L-S0)/20), Cf, H.p_sat, cf_of)
    for rec in out: rec['cf_khz'] = cf_of(rec['x'])
    json.dump(dict(state=open('abr5_ckpt.json').name+' (run_human.py)', S0=S0, G=H.coch.sec.G, places=out), open(OUTD + 'tm_human.json', 'w'))

if __name__ == '__main__':
    {'cat': cat, 'human': human}[sys.argv[1]]()
