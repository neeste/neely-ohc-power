"""
score_base.py -- held-out N=4096 sheet for the base-campaign
incumbent (54 kHz map).  Derived from score_grad2.py by checkpoint swap: the measured
observation battery at the converged gradient state, with the champion's
scoring machinery patched so every model evaluation carries the gradient
profiles r(x) = c_R a^(1+alpha_R), gamma(x) = frac gc_exact(rho(x)) taper.

Provenance: exp_best_final.py battery() (whose champion output was
reproduced identically in this workspace before any patch), with three
substitutions:
  E.line       -> line_grad      (suppression, locus, growth, ladders)
  T.response   -> response_grad  (tuning curve, phase, AF primaries)
  AF2.fields   -> fields_grad    (AF distributed-source solve)
and the stability section replaced by the campaign's own argument-principle
winding (exp_ftc1.certify) at the quiet gradient state plus uniformly
saturated samples, since growing_modes_mod certifies the uniform-gamma
line, not this one.

Champion reference values are printed beside each metric.
"""
import sys, math, time, json
from dataclasses import replace
import numpy as np
sys.path.insert(0, '.')
import tl_model as T
import exp_accumulation as E
import exp_twotone as TT
import exp_best1 as B
from exp_best1 import Gated3
from exp_encoded1 import build_rep
import exp_af_source2 as AF2
from exp_af_source2 import af_run
import obs_metrics as MS
import deltaT_3dof as d3

CK = json.load(open("best_d3.json"))["params"]
SECK = ("K_t", "M_t", "R_t", "R_c", "R_b", "K_c", "K_p", "M_r")
SEC = {k: CK[k] for k in SECK}
G, Q, M_G, UT, SG = (CK["gamma"], CK["q"], CK["m"], CK["u_thr"], CK["s_g"])
COCH = replace(T.WORLDS["B"], sec=replace(T.WORLDS["B"].sec, **SEC))

# Read-out constants z[6..10] (f_hp, f_lp, n_lp, B, C0) belong to the threshold
# criterion: base_ckpt.json (velocity, default) or base_ckpt_disp.json
# (displacement, FTC_CRIT=disp). The mechanics z[0..5], z[11] are the same.
import os as _os
CKPT_FILE = "base_ckpt_disp.json" if _os.environ.get("FTC_CRIT", "vel") == "disp" else "base_ckpt.json"
Z = json.load(open(CKPT_FILE))["z"]
ALPHA_A, ALPHA_B, C_R = Z[0], Z[1], math.exp(Z[2])
F_T, WID = math.exp(Z[3]), Z[4]
NU0, ALPHA_NU = math.exp(Z[5]), Z[11]
BASE = math.exp(Z[9])
A_T = F_T / BASE                      # transition CF, dimensionless

# ---- BM stiffness gradient (campaign E): patch E.terms module-wide ----
if len(Z) >= 18:
    _BETA_K = Z[16]
    _A_K = math.exp(Z[17]) / BASE
    _W_K = WID

    def _sb_bmk(a):
        u = np.log10(np.asarray(a, float) / _A_K)
        bend = _W_K * np.logaddexp(0.0, u * math.log(10.0) / _W_K) \
            / math.log(10.0)
        return 10.0 ** (-_BETA_K * bend)

    def _terms_bmk(sec, w, a, r=None, gamma=None, mp=1.0, mc=1.0,
                   mt=1.0, mrp=1.0, mrc=1.0):
        w = np.asarray(w); a = np.asarray(a, float)
        r = a if r is None else np.asarray(r, float)
        g = sec.gamma if gamma is None else np.asarray(gamma, float)
        jw = 1j * w; a2 = a * a
        sb = _sb_bmk(a)
        act = g * sec.K_a * sec.G * a2
        Rp, Rc = sec.R_p * mrp, sec.R_c * mrc
        Kp, Kc, Kt = sec.K_p * mp, sec.K_c * mc, sec.K_t * mt
        pil = jw * Rp * r + Kp * a2
        bun = jw * Rc * r + Kc * a2
        A00 = -w ** 2 * sec.M_b + jw * (sec.R_b + Rp) * r \
            + (sec.K_b * sb + Kp) * a2
        A01 = -pil + act
        A02 = -act + 0.0 * A00
        A10 = -pil + 0.0 * A00
        A11 = (-w ** 2 * sec.M_r + jw * (sec.R_r + Rp + Rc) * r
               + (Kp + Kc) * a2 - act)
        A12 = -bun + act
        A21 = -bun + 0.0 * A00
        A22 = -w ** 2 * sec.M_t + jw * (sec.R_t + Rc) * r + (Kt + Kc) * a2
        m00 = A11 * A22 - A12 * A21
        det = A00 * m00 - A01 * (A10 * A22) + A02 * (A10 * A21)
        Y = m00 / det
        Y_r = -A10 * A22 / det
        Y_t = A10 * A21 / det
        return Y, Y_r, sec.G * (Y_r - Y_t) / Y

    E.terms = _terms_bmk
    print('[bmk] E.terms patched: beta_k %.3f, break %.2f kHz'
          % (_BETA_K, math.exp(Z[17])))


# 16-param apex extension: third damping segment + apical activity
# exponent (exp_apex.profiles3 parameterization); reduces to the
# 12-param law when the extension is absent or zero.
if len(Z) >= 16:
    A_P = math.exp(Z[12]) / BASE
    W_P, ALPHA_P, P_NU = Z[13], Z[14], Z[15]
else:
    A_P, W_P, ALPHA_P, P_NU = 1.0, 0.3, 0.0, 0.0


def _bend_ap(a):
    u = np.log10(A_P / a)
    return W_P * np.logaddexp(0.0, u * math.log(10.0) / W_P) / math.log(10.0)


def _damping(a):
    """Broken power law: slope 1+ALPHA_A apical, 1+ALPHA_B basal,
    plus the localized apical third segment (zero when ALPHA_P=0)."""
    u = np.log10(a / A_T)
    bend = WID * np.logaddexp(0.0, u * math.log(10.0) / WID) / math.log(10.0)
    return 10.0 ** (math.log10(C_R) + (1.0 + ALPHA_A) * np.log10(a)
                    + (ALPHA_B - ALPHA_A) * bend
                    + ALPHA_P * _bend_ap(a))


def _nu(a):
    return np.clip(NU0 * a ** ALPHA_NU * 10.0 ** (P_NU * _bend_ap(a)),
                   0.02, 0.60)

# ---- exact gamma_c on a log-rho grid (exp_ftc1.gc_dense, verbatim) ----
def _gc_table():
    x = np.linspace(0.0, 1.0, 2049)
    a = COCH.cf(x)
    rho = _damping(a) / a
    rg = np.geomspace(max(rho.min() * 0.98, 1e-3), rho.max() * 1.02, 80)
    gg = []
    s0 = COCH.sec
    for r_ in rg:
        s2 = replace(s0, gamma=0.0, R_b=s0.R_b * r_, R_r=s0.R_r * r_,
                     R_t=s0.R_t * r_, R_p=s0.R_p * r_, R_c=s0.R_c * r_)
        gg.append(d3.gamma_critical(s2))
    return np.log(rg), np.asarray(gg)

_LX, _GY = _gc_table()
_prof_cache = {}

def profiles(N):
    """(r, gam0) on the N+1 grid; gam0 is BEFORE the taper."""
    if N not in _prof_cache:
        x = np.linspace(0.0, 1.0, N + 1)
        a = COCH.cf(x)
        r = _damping(a)
        gam0 = (1.0 - _nu(a)) * np.interp(np.log(r / a), _LX, _GY)
        _prof_cache[N] = (r, gam0)
    return _prof_cache[N]

# ---- patch 1: E.line ----
def line_grad(coch, freq, g0, s, beta, depth, N, amp=1.0):
    x = np.linspace(0.0, 1.0, N + 1)
    a = coch.cf(x)
    r, gam0 = profiles(N)
    s = np.asarray(s, float)
    bd = min(beta * depth, E.DEPTH_CAP)
    prof = gam0 * (1.0 - (1.0 - beta) * s)
    mp = 1.0 - bd * s
    gam = coch.gamma_of_x(x, None, prof)
    Y, Y_r, H_o = E.terms(coch.sec, freq, a, r, gam, mp=mp)
    P = T._fd(coch, freq, Y, N, amp)
    X_b = Y * P
    return dict(x=x, P=P, X_b=X_b, X_r=Y_r * P, shear=H_o * X_b,
                H_o=H_o, Y=Y, gamma_x=gam, mp=mp)

E.line = line_grad

# ---- patch 2: T.response ----
_resp_orig = T.response

def response_grad(coch, freq, level_db=0.0, N=None, gamma=None,
                  profile=None, engine="fd"):
    Ne = coch.N if N is None else N
    x = np.linspace(0.0, 1.0, Ne + 1)
    a = coch.cf(x)
    r, gam0 = profiles(Ne)
    if profile is not None:
        p = profile(x) if callable(profile) else np.asarray(profile, float)
        prof = p * (gam0 / G)     # proportional local rescaling of the
    elif gamma is not None and float(gamma) == 0.0:
        prof = np.zeros(Ne + 1)   # suppressor's uniform-basis profile
    else:
        prof = gam0
    gam = coch.gamma_of_x(x, None, prof)
    Y, Y_r, H_o = E.terms(coch.sec, freq, a, r, gam)
    amp = 10.0 ** (level_db / 20.0)
    P = T._fd(coch, freq, Y, Ne, amp)
    X_b = Y * P
    with np.errstate(divide="ignore"):
        lg_P = np.log10(np.abs(P))
    return dict(x=x, cf=a, P=P, X_b=X_b, X_r=Y_r * P, shear=H_o * X_b,
                H_o=H_o, Y=Y, Z_p=1.0 / (1j * 2 * np.pi * freq * Y),
                gamma_x=gam, lg_P=lg_P)

T.response = response_grad

# ---- patch 3: AF2.fields ----
def fields_grad(coch, freq, gamma, N):
    x = np.linspace(0.0, 1.0, N + 1)
    a = coch.cf(x)
    r, gam0 = profiles(N)
    gam = coch.gamma_of_x(x, None, gam0)
    Y, Y_r, H_o = E.terms(coch.sec, freq, a, r, gam)
    return x, Y, H_o

AF2.fields = fields_grad

# ---- winding certification (exp_ftc1.certify, verbatim geometry) ----
def certify(scale=1.0):
    N = 1024
    x = np.linspace(0.0, 1.0, N + 1)
    a = COCH.cf(x)
    r, gam0 = profiles(N)
    gam = COCH.gamma_of_x(x, None, gam0 * scale)
    F = COCH.F
    h = 1.0 / N
    edges = np.logspace(-2.9, 0.35, 11)
    for lo, hi in zip(edges[:-1], edges[1:]):
        depth = 0.35 * hi
        corners = [complex(hi, -depth), complex(hi, 1e-6),
                   complex(lo, 1e-6), complex(lo, -depth),
                   complex(hi, -depth)]
        pts = []
        for c0, c1 in zip(corners[:-1], corners[1:]):
            tt_ = np.linspace(0, 1, 1000, endpoint=False)
            pts.extend(c0 + (c1 - c0) * tt_)
        tot, ok, prev = 0.0, True, None
        for w in pts:
            Y, _, _ = E.terms(COCH.sec, w, a, r, gam)
            q = -(w * w) * F * Y
            AA = -(2.0 + h * h * q)
            d0, d1 = 1.0 + 0j, AA[1]
            for jj in range(2, N):
                d2 = AA[jj] * d1 - d0
                m_ = abs(d2) or 1e-300
                d0, d1 = d1 / m_, d2 / m_
            cur = np.angle(d1)
            if prev is not None:
                dph = cur - prev
                while dph > math.pi: dph -= 2 * math.pi
                while dph < -math.pi: dph += 2 * math.pi
                if abs(dph) > math.pi / 2: ok = False
                tot += dph
            prev = cur
        if (not ok) or abs(tot / (2 * math.pi)) > 0.1:
            return False
    return True


def battery():
    t0 = time.time()
    f_p = float(COCH.cf(E.X_PROBE))
    rq = line_grad(COCH, f_p, G, np.zeros(2049), 0.0, 0.0, 2048)
    E.X_READ = T._peak(rq["x"], rq["X_b"])[0]
    orig = T.gamma_for_gain
    T.gamma_for_gain = lambda *a, **k: (G, float("nan"))
    TT.P_ORD = 4.0
    coch = COCH
    dump = dict(champion=CK, gradient=dict(alpha_a=ALPHA_A,
                                           alpha_b=ALPHA_B, c_R=C_R,
                                           f_t_khz=F_T, width=WID,
                                           nu0=NU0, alpha_nu=ALPHA_NU,
                                           base_khz=BASE))
    try:
        print("=" * 78)
        print("GRADIENT-STATE BATTERY (FTC2 joint campaign, phase 2)")
        print("=" * 78)
        print(f"  broken power law: slope 1{ALPHA_A:+.4f} apical, "
              f"1{ALPHA_B:+.4f} basal, transition {F_T:.2f} kHz "
              f"(width {WID:.2f} dec, base {BASE:.1f} kHz)")
        print(f"  activity: nu = {NU0:.4f} a^({ALPHA_NU:+.3f}); "
              f"gate m {M_G} q {Q:.4f} floor {SG:.5f}; P_ORD 4")
        print(f"  probe f {f_p:.5f}, read at x {E.X_READ:.4f}")

        print("\n[a] measured locus (champion +6.3; target 0 +/- 10):")
        for N in (1024, 2048):
            row = []
            for oc in (1.0, 1.5, 2.0, 2.5, 3.0):
                TT.OCT_BELOW = oc
                tt = Gated3(coch, 30.0, UT, Q, m=M_G, s_g=SG, N=N)
                io = TT.probe_io(tt, -10, 14, 2.0)
                c_thr = float(np.interp(0.0, io[:, 0], io[:, 1]))
                L_n, _ = TT.bisect_supp(tt, tt.P0 + TT.SL_NEURAL, c_thr)
                row.append((oc, L_n))
            x = np.asarray([math.log10(2.0 ** (-o)) for o, _ in row])
            y = np.asarray([v for _, v in row])
            A = np.vstack([x, np.ones_like(x)]).T
            sl = float(np.linalg.lstsq(A, y, rcond=None)[0][0])
            print(f"    N={N}: " + " ".join(f"{o}:{v:6.2f}" for o, v in row)
                  + f"  slope {sl:+.2f} dB/decade")
            if N == 2048:
                dump["locus"] = dict(oct=[o for o, _ in row],
                                     L=[v for _, v in row], slope=sl)

        print("\n[b] growth and Delta-T (champion gr33 1.563 rms 0.57,"
              " dT 22.25/21.93):")
        for N in (1024, 2048):
            TT.OCT_BELOW = 1.8
            TT.SL_COOPER = (40.0, 50.0)
            TT.GROWTH_SPAN = (3.0, 33.0, 3.0)
            TT.N_RUN = N
            tt = Gated3(coch, 30.0, UT, Q, m=M_G, s_g=SG, N=N)
            m = TT.measure(tt)
            d40 = (m["bm"][40.0] - m["L_n"]
                   if not math.isnan(m["bm"][40.0]) else float("inf"))
            d50 = (m["bm"][50.0] - m["L_n"]
                   if not math.isnan(m["bm"][50.0]) else float("inf"))
            pts = m["growth"]
            A = np.vstack([[a_ for a_, _ in pts],
                           np.ones(len(pts))]).T
            co, res = np.linalg.lstsq(
                A, np.asarray([b_ for _, b_ in pts]), rcond=None)[:2]
            rms = float(np.sqrt(res[0] / len(pts))) if len(res) else 0.0
            print(f"    N={N}: gr33 {m['slope']:.3f} (rms {rms:.2f} dB)  "
                  f"dT@40 {d40:.2f}  dT@50 "
                  f"{d50 if math.isinf(d50) else round(d50, 2)}")
            if N == 2048:
                dump["growth"] = dict(pts=pts, slope=m["slope"], rms=rms,
                                      L_n=m["L_n"], dT40=d40, dT50=d50)

        print("\n[b2] vertical ladders (champion dT 33.1-33.2):")
        vt = []
        for oc in (1.8, 2.3, 2.9, 3.5):
            TT.OCT_BELOW = oc
            tt = Gated3(coch, 30.0, UT, Q, m=M_G, s_g=SG, N=1024)
            c0, b0, s_run, _ = tt.response(None, tt.P0 - 30.0)
            Lv = np.arange(-20.0, 44.0, 3.0)
            sn, sb = [], []
            for L in Lv:
                c, b, s_run, _ = tt.response(L, tt.P0 - 30.0, s_run)
                sn.append(c0 - c)
                sb.append(b0 - b)

            def vcross(yv):
                yv = np.asarray(yv)
                w = np.where(yv >= 1.0)[0]
                if w.size == 0 or w[0] == 0:
                    return None
                i = w[0]
                return float(Lv[i - 1] + (1.0 - yv[i - 1]) * 3.0
                             / (yv[i] - yv[i - 1]))
            tn, tb = vcross(sn), vcross(sb)
            vt.append(dict(oct=oc, thr_n=tn, thr_b=tb))
            print(f"    oct {oc}: thr_n {tn if tn is None else round(tn,1)}"
                  f"  thr_b {tb if tb is None else round(tb,1)}"
                  f"  dT {None if (tn is None or tb is None) else round(tb-tn,1)}")
        dump["vertical"] = vt

        tc, rep = build_rep(coch, G)
        o4 = MS.obs4(rep); o5 = MS.obs5(rep)
        dump["tuning"] = dict(tt=o4.metrics[0].value, sl=o4.metrics[1].value,
                              mp=o5.metrics[0].value, ph=o5.metrics[1].value)
        print(f"\n[b3] tuning (champion tt 32.65 sl 3.33 map -0.71 "
              f"phase 3.79):\n    tt {o4.metrics[0].value:.2f}, "
              f"sl {o4.metrics[1].value:.2f}, map {o5.metrics[0].value:.2f},"
              f" phase {o5.metrics[1].value:.2f}")

        for N in (2048, 4096):
            rows = af_run(coch, G, N)
            flat = [v for rr, v in rows if rr >= 1.25]
            print(f"[c] AF-dist excursion N={N}: "
                  f"{max(flat) - min(flat):.2f} dB (champion 6.88/6.87)")
            if N == 4096:
                dump["af"] = dict(r=[rr for rr, _ in rows],
                                  v=[v for _, v in rows])

        print("\n[d] stability (argument-principle winding, N=1024):")
        for sc, lbl in ((1.0, "quiet"), (0.7, "30 percent saturated"),
                        (0.4, "60 percent saturated")):
            print(f"    {lbl}: clean zero = {certify(sc)}")
        dump["stability"] = "winding certified; see log"
    finally:
        T.gamma_for_gain = orig
    json.dump(dump, open("base_battery.json", "w"))
    print(f"\n  -> base_battery.json; total {time.time() - t0:.1f} s")


if __name__ == "__main__":
    battery()
