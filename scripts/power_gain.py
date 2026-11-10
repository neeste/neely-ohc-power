"""Run 6 item 2: amplifier power gain from the model's power accounting (Sec. II.H; p_f + p_ohc = D).
Cat c708/c85: run from model/ under run_cat.py (graded profile SB.profiles, N 4096,
lambda energy accounting). Human hs708: from model/ under run_human.py (N 8192).
Levels: quiet, 20 and 40 dB SPL. Saturation as in the paper's level read-outs: s = u^n/(1+u^n), u = A|shear(x; s)|/C,
solved by Anderson iteration. Cat (run-2 b_crit.py, flat convention): A = 10^((L - CAL)/20), CAL = 65 - median(battery
locus L) of the state's own base_battery.json, C = peak |shear| of the unsaturated line at the probe frequency, n = 4.
Human (cert_h_lam.py): A = 10^((L - S0)/20), C = peak |shear| of the unsaturated line at the tone frequency, n = p_sat.
Quantities (x in [0,1], trapezoid integrals; octaves o = log2(a(j)/a(x)) re the shear peak j, APICAL POSITIVE):
  G = int D / int p_f (10 log10), = 1/(1 - int p_ohc / int D); whole line, peak region |o| <= 0.8, region basal of the
  share > 1 band (x from 0 to the band's basal edge).
  Wave power flux Pi(x) = int_x^1 p_f (power crossing x toward the apex); cross-checked against
  (1/2) Re{P conj(U)}, U = -P'/(j w F) (ratio constant along the line).
  Flux at the shear peak, active over passive (same x), at equal stapes volume velocity (U(0)) and at equal input power (Pi(0)).
  OHC fraction of dissipation int p_ohc / int D: whole line, apical half (0 <= o <= 0.8), basal half (-0.8 <= o < 0).
  Prediction 3: Pi(apical edge of band) - Pi(basal edge) = -int_band p_f, relative to Pi(x_j).
Supplementary levels 60 and 80 dB SPL (the cat lines do not saturate at 20 and 40 dB SPL). Passive Q10: passive_q10.py.
Usage: power_gain.py cat|hum out.json"""
import sys, os, json, math, numpy as np
sys.path.insert(0, '.')
import exp_accumulation as E, tl_model as T, geom
kind, OUT = sys.argv[1], sys.argv[2]
if kind == 'cat':
    import score_base as SB
    N = 4096; x = np.linspace(0, 1, N+1); a = SB.COCH.cf(x); r, g0 = SB.profiles(N); gam0 = g0*SB.COCH.taper(x)
    SEC, COCH, BASE, FREQS = SB.COCH.sec, SB.COCH, SB.BASE, (0.5, 1.0, 2.0, 4.0, 9.0)
    CAL = 65.0 - float(np.median(json.load(open('base_battery.json'))['locus']['L']))
    NSAT = 4.0
else:
    import exp_abr2 as A2, exp_abr5 as A5
    A2.N = 8192
    c = json.load(open('abr5_ckpt.json')); pd = dict(c['params'])
    H = A2.Human(**pd); gc = A5.gc_exact_dense(H); gam0 = A5.capped_gamma(H, pd['mu0'], pd['alpha_mu'], gc)
    x, a, r = H.x, H.a, H.r; N = len(x) - 1; SEC, COCH, BASE, FREQS = H.coch.sec, H.coch, 20.7, (0.5, 1.0, 2.0, 4.0, 8.0)
    CAL = float(c['S0']); NSAT = float(H.p_sat)
h = 1.0/N
def solve(g, w):
    Y, Yr, Ho = E.terms(SEC, w, a, r, g); P = T._fd(COCH, w, Y, N, 1.0)
    Xb = Y*P; Xr = Yr*P; Xt = Xr - Ho*Xb/SEC.G
    D, po = geom.power(SEC, w, a, r, g, Xb, Xr, Xt)
    pf = 0.5*np.real(1/(1j*w*Y))*np.abs(1j*w*Xb)**2
    return dict(Y=Y, P=P, Xb=Xb, sh=Ho*Xb, D=D, po=po, pf=pf)
def cumtrap_from_apex(y):
    seg = 0.5*(y[1:] + y[:-1])*h; c = np.concatenate([np.cumsum(seg[::-1])[::-1], [0.0]]); return c
def integ(y, m):
    yy = np.where(m, y, 0.0); return float(np.trapezoid(yy, x))
def saturate(w, L):
    if L is None: return np.zeros(N+1)
    A = 10**((L - CAL)/20)
    if kind == 'cat':
        fp = float(COCH.cf(E.X_PROBE)); C = float(np.abs(solve(gam0, fp)['sh']).max())
    else:
        C = float(np.abs(solve(gam0, w)['sh']).max())
    def FF(sv):
        u = (A*np.abs(solve(gam0*(1 - sv), w)['sh'])/C)**NSAT; return u/(1 + u)
    s, _, _ = E.anderson(FF, np.zeros(N+1)); return s
out = dict(kind=kind, lam=geom.LAM, N=N, cwd=os.getcwd(), cal_db=CAL, n_sat=NSAT,
           sign='octaves re shear peak, apical positive', rows=[])
for fk in FREQS:
    w = fk/BASE
    Fp = solve(0*gam0, w)
    for L in (None, 20.0, 40.0, 60.0, 80.0):
        s = saturate(w, L); g = gam0*(1 - s); F = solve(g, w)
        j = int(np.argmax(np.abs(F['sh']))); jb = int(np.argmax(np.abs(F['Xb']))); o = np.log2(a[j]/a)
        D, po, pf = F['D'], F['po'], F['pf']
        err = float(np.max(np.abs(D - pf - po))/D.max())
        ok = D > 1e-9*D.max(); share = np.where(ok, po/np.where(ok, D, 1.0), 0.0)
        near = np.abs(o) < 2.0
        k0 = int(np.argmax(np.where(near, share, -1)))
        band = None
        if share[k0] > 1:
            i0 = i1 = k0
            while i0 > 0 and share[i0-1] > 1: i0 -= 1
            while i1 < N and share[i1+1] > 1: i1 += 1
            band = (i0, i1)
        allm = np.ones(N+1, bool); pk = np.abs(o) <= 0.8; ap = (o >= 0) & (o <= 0.8); bs = (o < 0) & (o >= -0.8)
        def G(m):
            d, f_ = integ(D, m), integ(pf, m)
            return dict(int_D=d, int_pf=f_, G=d/f_ if f_ != 0 else float('nan'),
                        G_db=10*math.log10(d/f_) if f_ > 0 else float('nan'), ohc_frac=integ(po, m)/d)
        row = dict(f_khz=fk, level=('quiet' if L is None else '%g dB SPL' % L), smax=float(s.max()),
                   x_shear_peak=float(x[j]), oct_bm_peak=float(o[jb]), identity_err=err,
                   whole=G(allm), peak=G(pk), apical_half=G(ap), basal_half=G(bs),
                   share_max=float(share[k0]), share_max_oct=float(o[k0]),
                   band_oct=[float(o[band[1]]), float(o[band[0]])] if band else None)
        if band:
            row['basal_of_band'] = G(x <= x[band[0]])
        else:
            row['basal_of_band'] = None
        Pi = cumtrap_from_apex(pf); Pip = cumtrap_from_apex(Fp['pf'])
        # direct flux cross-check: (1/2) Re{P conj(U)}, U = -P'/(j w F)
        dP = np.gradient(F['P'], h); Pi_d = 0.5*np.real(F['P']*np.conj(-dP/(1j*w*COCH.F)))
        sel = (x > 0.02) & (Pi > 1e-3*Pi[0])
        rat = Pi_d[sel]/Pi[sel]; row['flux_check'] = dict(ratio_median=float(np.median(rat)), ratio_spread=float(np.ptp(rat)/abs(np.median(rat))))
        U0a = abs(-(-3*F['P'][0] + 4*F['P'][1] - F['P'][2])/(2*h)/(1j*w*COCH.F))
        U0p = abs(-(-3*Fp['P'][0] + 4*Fp['P'][1] - Fp['P'][2])/(2*h)/(1j*w*COCH.F))
        row['flux_at_peak'] = dict(active_over_passive_equal_U_db=10*math.log10((Pi[j]/U0a**2)/(Pip[j]/U0p**2)),
                                   active_over_passive_equal_power_db=10*math.log10((Pi[j]/Pi[0])/(Pip[j]/Pip[0])),
                                   frac_input_reaching_peak_active=float(Pi[j]/Pi[0]), frac_input_reaching_peak_passive=float(Pip[j]/Pip[0]),
                                   input_power_active_over_passive_equal_U_db=10*math.log10((Pi[0]/U0a**2)/(Pip[0]/U0p**2)))
        if band:
            dPi = float(Pi[band[0]] - Pi[band[1]])       # Pi at apical edge minus basal edge: Pi decreases with x, so sign
            # Pi(x) = int_x^1 pf; across the band (x from band[0] to band[1]) Pi(band[1]) - Pi(band[0]) = -int_band pf > 0
            inc = float(Pi[band[1]] - Pi[band[0]])
            row['band_flux'] = dict(increase=inc, increase_re_flux_at_peak=inc/Pi[j], flux_at_peak=float(Pi[j]),
                                    max_flux_in_band_re_peak=float(Pi[band[0]:band[1]+1].max()/Pi[j]))
        out['rows'].append(row)
        print('%4.1f kHz %-10s smax %.3f | G whole %.2f dB (OHC %.3f) peak %.2f dB (OHC %.3f; ap %.3f bas %.3f) basal-of-band %s | flux@pk act/pas U %+.2f dB P %+.2f dB | share %.3f at %+.3f, band %s %s | err %.1e chk %.3f+-%.1e' % (
            fk, row['level'], row['smax'], row['whole']['G_db'], row['whole']['ohc_frac'], row['peak']['G_db'], row['peak']['ohc_frac'],
            row['apical_half']['ohc_frac'], row['basal_half']['ohc_frac'],
            ('%.2f dB' % row['basal_of_band']['G_db']) if row['basal_of_band'] else '-',
            row['flux_at_peak']['active_over_passive_equal_U_db'], row['flux_at_peak']['active_over_passive_equal_power_db'],
            row['share_max'], row['share_max_oct'], row['band_oct'],
            ('incr %.3f of Pi(peak)' % row['band_flux']['increase_re_flux_at_peak']) if band else '', err,
            row['flux_check']['ratio_median'], row['flux_check']['ratio_spread']), flush=True)
        json.dump(out, open(OUT, 'w'), indent=1)
json.dump(out, open(OUT, 'w'), indent=1)
print('POWER DONE', flush=True)
