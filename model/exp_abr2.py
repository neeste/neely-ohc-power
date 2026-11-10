"""
exp_abr2.py -- stage 2 of the human-scaling campaign: minimal symmetry
breaking. (Reconstructed in final form after container rollbacks; the
authoritative copy is committed to E:\\MOH2027\\hybrid\\tl\\.)

Levers, all reducing to the rigid-scaled line when neutral:

  alpha_R  damping-scale tilt: the damping argument of terms() becomes
           r = c_R a^(1 + alpha_R); locally the section is the
           reference section with every damping times
           rho(x) = c_R a^alpha_R.
  mu0, alpha_mu   activity as DISTANCE FROM LOCAL CRITICAL:
           1 - mu(x) = mu0 a^alpha_mu; gamma(x) = mu(x) gc(rho(x))
           taper(x). Cycles of active delay go as 1/(1 - mu), so
           alpha_mu is the direct d lever and mu0 the c lever.
  k        Greenwood curvature: a(x) = 10^(-D x) - k, clip 1e-4.
  F        fluid group (overall latency magnitude).
  p_sat    OHC saturation order: s = u^p/(1 + u^p). p < 2 spreads the
           gain erosion over more dB (the c-uniformity lever).

Latency: multi-point unwrapped phase slope of shear at its peak (the
two-point difference aliases near critical).

Stability proxy: mu <= 0.998 with gc(rho) from a cached
d3.gamma_critical grid; full FD mode check reserved for finalists.

Usage:
  python3 exp_abr2.py --scanA | --scanB
  python3 exp_abr2.py --fit --maxfev N [--resume]
Env: DECADES (3.0), NPTS (1024); z0 seeds AR, CR, NU, AMU, PSAT.
"""
import sys, os, math, time, json
from dataclasses import replace
import numpy as np
sys.path.insert(0, '.')
import tl_model as T
import deltaT_3dof as d3
import exp_accumulation as E
from exp_abr1 import F_BASE_HZ, fit_bcd

CK = json.load(open("best_d3.json"))["params"]
SEC = {k: CK[k] for k in ("K_t", "M_t", "R_t", "R_c", "R_b",
                          "K_c", "K_p", "M_r")}
DECADES = float(os.environ.get("DECADES", "3.0"))
N = int(os.environ.get("NPTS", "1024"))
FREQS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
LEVELS = (-15.0, 0.0, 15.0, 30.0, 45.0, 60.0, 75.0)

_sec_ref = None
_gc_grid = None


def sec_ref():
    global _sec_ref
    if _sec_ref is None:
        _sec_ref = replace(T.WORLDS["B"].sec, **SEC)
    return _sec_ref


def gc_of_rho():
    global _gc_grid
    if _gc_grid is None:
        rhos = np.geomspace(0.05, 20.0, 25)
        gcs = []
        for rho in rhos:
            s = sec_ref()
            s2 = replace(s, gamma=0.0,
                         R_b=s.R_b * rho, R_r=s.R_r * rho,
                         R_t=s.R_t * rho, R_p=s.R_p * rho,
                         R_c=s.R_c * rho)
            gcs.append(d3.gamma_critical(s2))
        _gc_grid = (np.log(rhos), np.asarray(gcs))
    return _gc_grid


_gc_sec_cache = {}


def gc_interp_sec(sec, rho):
    key = (round(sec.K_t, 6), round(sec.M_t, 5), round(sec.R_t, 6),
           round(sec.R_c, 6), round(sec.R_b, 6), round(sec.K_c, 6),
           round(sec.K_p, 6), round(sec.M_r, 5))
    if key not in _gc_sec_cache:
        rhos = np.geomspace(0.05, 20.0, 9)
        gcs = []
        for r_ in rhos:
            s2 = replace(sec, gamma=0.0,
                         R_b=sec.R_b * r_, R_r=sec.R_r * r_,
                         R_t=sec.R_t * r_, R_p=sec.R_p * r_,
                         R_c=sec.R_c * r_)
            gcs.append(d3.gamma_critical(s2))
        _gc_sec_cache[key] = (np.log(rhos), np.asarray(gcs))
        if len(_gc_sec_cache) > 4000:
            _gc_sec_cache.clear()
    lx, gy = _gc_sec_cache[key]
    return np.interp(np.log(np.clip(rho, 0.05, 20.0)), lx, gy)


def gc_interp(rho):
    lx, gy = gc_of_rho()
    return np.interp(np.log(np.clip(rho, 0.05, 20.0)), lx, gy)


class Human:
    """The symmetry-broken line and its latency machinery."""

    def __init__(self, alpha_R=0.0, c_R=1.0, gamma=CK["gamma"],
                 alpha_g=0.0, k=0.0, F=20000.0, mu0=None, alpha_mu=0.0,
                 p_sat=2.0, sec_over=None):
        self.p = dict(alpha_R=alpha_R, c_R=c_R, gamma=gamma,
                      alpha_g=alpha_g, k=k, F=F, mu0=mu0,
                      alpha_mu=alpha_mu, p_sat=p_sat)
        self.p_sat = p_sat
        base_sec = sec_ref() if not sec_over else replace(
            sec_ref(), **sec_over)
        self.coch = replace(T.WORLDS["B"], sec=base_sec,
                            decades=DECADES, F=F)
        self._sec_over = sec_over
        x = np.linspace(0.0, 1.0, N + 1)
        self.x = x
        self.a = np.clip(10.0 ** (-DECADES * x) - k, 1e-4, None)
        self.r = c_R * self.a ** (1.0 + alpha_R)
        rho = c_R * self.a ** alpha_R
        if sec_over:
            # section changed: local critical gamma from a per-instance
            # grid (cheap: 9 points, cached per section signature)
            self.gc_x = gc_interp_sec(base_sec, rho)
        else:
            self.gc_x = gc_interp(rho)
        if mu0 is not None:
            nu = np.clip(mu0 * self.a ** alpha_mu, 1e-3, 1.0)
            mu = 1.0 - nu
            self.gam0 = mu * self.gc_x * self.coch.taper(x)
        else:
            self.gam0 = gamma * self.a ** alpha_g * self.coch.taper(x)
        self.stable = bool(np.all(self.gam0 <= 0.998 * self.gc_x))
        self.margin = float(np.max(self.gam0 / self.gc_x))

    def line2(self, f_dim, s, amp=1.0):
        gam = self.gam0 * (1.0 - s)
        Y, Y_r, H_o = E.terms(self.coch.sec, f_dim, self.a, self.r, gam)
        P = T._fd(self.coch, f_dim, Y, N, amp)
        X_b = Y * P
        return dict(x=self.x, X_b=X_b, shear=H_o * X_b,
                    X_r=Y_r * P, Y=Y, H_o=H_o, P=P)

    def latency_ms(self, f_khz, L_db, s0=None, eps=0.01):
        f_dim = f_khz * 1000.0 / F_BASE_HZ
        amp = 10.0 ** (L_db / 20.0)
        r0 = self.line2(f_dim, np.zeros(N + 1))
        C = float(np.max(np.abs(r0["shear"])))
        p_s = self.p_sat

        def FF(sv):
            r = self.line2(f_dim, sv)
            u = (amp * np.abs(r["shear"]) / C) ** p_s
            return u / (1.0 + u)

        s, dres, it = E.anderson(FF, np.zeros(N + 1) if s0 is None
                                 else s0)
        rm = self.line2(f_dim, s)
        j = int(np.argmax(np.abs(rm["shear"])))
        for span in (eps, eps / 4.0, eps / 16.0):
            fr = f_dim * (1.0 + span * np.linspace(-1, 1, 7))
            ph = np.asarray([np.angle(self.line2(ff, s)["shear"][j])
                             for ff in fr])
            ph = np.unwrap(ph)
            if np.max(np.abs(np.diff(ph))) < 2.0:
                break
        A = np.vstack([fr, np.ones_like(fr)]).T
        slope = float(np.linalg.lstsq(A, ph, rcond=None)[0][0])
        return 1000.0 * (-slope) / (2 * math.pi * F_BASE_HZ), s

    def surface(self):
        rows = []
        for f in FREQS:
            s0 = None
            for L in LEVELS:
                tau, s0 = self.latency_ms(f, L, s0)
                rows.append((f, L, tau))
        return rows


def bcd(hum):
    if not hum.stable:
        return None
    rows = hum.surface()
    b, c, d, rms = fit_bcd(rows)
    return dict(b=b, c=c, d=d, rms=rms, rows=rows)




# ---------------------------------------------------------------------
# grid-faithful fit: the paper's own grid (0.25-8 kHz x 20-100 dB SPL),
# knee offset S0 a free parameter; b, c, d fit with SPL/100 directly.
GRID_SPLS = tuple(range(20, 101, 10))


def bcd_grid(hum, S0):
    if not hum.stable:
        return None
    rows = []
    for f in FREQS:
        s0 = None
        for spl in GRID_SPLS:
            tau, s0 = hum.latency_ms(f, spl - S0, s0)
            rows.append((f, spl, tau))
    A, y = [], []
    for f, spl, tau in rows:
        if tau and np.isfinite(tau) and tau > 0:
            A.append([1.0, -spl / 100.0, -math.log(f)])
            y.append(math.log(tau))
    if len(y) < 30:
        return None
    A = np.asarray(A); y = np.asarray(y)
    co, res = np.linalg.lstsq(A, y, rcond=None)[:2]
    b, c, d = math.exp(co[0]), math.exp(co[1]), co[2]
    rms = math.sqrt(res[0] / len(y)) if len(res) else 0.0
    return dict(b=b, c=c, d=d, rms=rms, rows=rows, S0=S0)


def fitgrid(argv):
    from scipy.optimize import minimize
    CKPT = ("abr2g2_ckpt.json"
            if os.environ.get("FREESEC", "0") == "1"
            else "abr2g_ckpt.json")

    FREESEC = os.environ.get("FREESEC", "0") == "1"
    SECN = ("K_t", "M_t", "R_t", "R_c", "R_b", "K_c", "K_p", "M_r")

    def unpack(z):
        pd = dict(alpha_R=z[0], c_R=math.exp(z[1]),
                  mu0=math.exp(-z[2]), alpha_mu=z[3],
                  k=z[4], F=math.exp(z[5]), p_sat=math.exp(z[6]))
        if FREESEC:
            pd["sec_over"] = {kk: math.exp(z[8 + i])
                              for i, kk in enumerate(SECN)}
        return pd, z[7]   # S0

    best = {"r": 1e9}

    def f(z):
        pd, S0 = unpack(z)
        so = pd.get("sec_over")
        if so:
            for kk, vv in so.items():
                ref = SEC[kk]
                if not (ref / 30.0 <= vv <= ref * 30.0):
                    return 30.0
        if not (-1.2 <= pd["alpha_R"] <= 0.3 and
                0.02 <= pd["c_R"] <= 3.0 and
                0.001 <= pd["mu0"] <= 0.7 and
                -1.2 <= pd["alpha_mu"] <= 0.3 and
                0.0 <= pd["k"] <= 0.03 and
                2000 <= pd["F"] <= 200000 and
                0.4 <= pd["p_sat"] <= 2.5 and
                10.0 <= S0 <= 90.0):
            return 30.0
        try:
            h = Human(**pd)
            r = bcd_grid(h, S0)
        except Exception:
            return 40.0
        if r is None:
            return 20.0 + h.margin
        res = (abs(math.log(r["c"] / 5.0)) / 0.2
               + abs(r["d"] - 0.413) / 0.1
               + abs(math.log(r["b"] / 12.9)) / 0.23
               + r["rms"] / 0.15)
        if res < best["r"]:
            best["r"] = res
            json.dump(dict(res=res, params=pd, S0=S0, fit=dict(
                b=r["b"], c=r["c"], d=r["d"], rms=r["rms"])),
                open(CKPT, "w"), indent=1)
            print(f"    best {res:.3f}: b={r['b']:.2f} c={r['c']:.2f} "
                  f"d={r['d']:.3f} rms={r['rms']:.3f} S0={S0:.1f} "
                  + " ".join(f"{k2}={v:.4g}" for k2, v in pd.items()
                             if not isinstance(v, dict)),
                  flush=True)
        return res

    maxfev = int(argv[argv.index("--maxfev") + 1])         if "--maxfev" in argv else 120
    if "--resume" in argv and os.path.exists(CKPT):
        ck = json.load(open(CKPT))
        pd = ck["params"]
        z0 = [pd["alpha_R"], math.log(pd["c_R"]), -math.log(pd["mu0"]),
              pd["alpha_mu"], pd["k"], math.log(pd["F"]),
              math.log(pd["p_sat"]), ck["S0"]]
        if FREESEC:
            so = pd.get("sec_over") or dict(SEC)
            z0 += [math.log(so[kk]) for kk in SECN]
            if not pd.get("sec_over"):
                best["r"] = 1e9      # objective changed dimension
            else:
                best["r"] = ck["res"]
        else:
            best["r"] = ck["res"]
        print(f"[grid] resuming from {ck.get('res', 1e9):.3f}"
              f" (FREESEC={FREESEC})")
    else:
        ck = json.load(open("abr2_ckpt.json"))
        pd = ck["params"]
        z0 = [pd["alpha_R"], math.log(pd["c_R"]), -math.log(pd["mu0"]),
              pd["alpha_mu"], pd["k"], math.log(pd["F"]),
              math.log(pd["p_sat"]),
              float(os.environ.get("S0", "35"))]
    res = minimize(f, np.asarray(z0), method="Nelder-Mead",
                   options=dict(maxfev=maxfev, xatol=1e-3, fatol=1e-3))
    print(f"[grid] chunk done: {res.fun:.3f} after {res.nfev} evals")


def main(argv):
    t0 = time.time()
    if "--fitgrid" in argv:
        fitgrid(argv)
    if "--fit" in argv:
        from scipy.optimize import minimize
        CKPT = "abr2_ckpt.json"

        def unpack(z):
            return dict(alpha_R=z[0], c_R=math.exp(z[1]),
                        mu0=math.exp(-z[2]), alpha_mu=z[3],
                        k=z[4], F=math.exp(z[5]),
                        p_sat=math.exp(z[6]))

        best = {"r": 1e9}

        def f(z):
            pd = unpack(z)
            if not (-1.2 <= pd["alpha_R"] <= 0.3 and
                    0.02 <= pd["c_R"] <= 3.0 and
                    0.002 <= pd["mu0"] <= 0.7 and
                    -1.2 <= pd["alpha_mu"] <= 0.3 and
                    0.0 <= pd["k"] <= 0.03 and
                    2000 <= pd["F"] <= 200000 and
                    0.4 <= pd["p_sat"] <= 2.5):
                return 30.0
            try:
                h = Human(**pd)
                r = bcd(h)
            except Exception:
                return 40.0
            if r is None:
                return 20.0 + h.margin
            S0lo, S0hi = 20.0, 80.0
            blo = r["b"] * r["c"] ** (S0lo / 100.0)
            bhi = r["b"] * r["c"] ** (S0hi / 100.0)
            berr = 0.0 if blo <= 12.9 <= bhi else \
                min(abs(math.log(12.9 / blo)), abs(math.log(12.9 / bhi)))
            res = (abs(math.log(r["c"] / 5.0)) / 0.2
                   + abs(r["d"] - 0.413) / 0.1 + berr / 0.23
                   + r["rms"] / 0.15)
            if res < best["r"]:
                best["r"] = res
                json.dump(dict(res=res, params=pd, fit=dict(
                    b=r["b"], c=r["c"], d=r["d"], rms=r["rms"])),
                    open(CKPT, "w"), indent=1)
                print(f"    best {res:.3f}: b'={r['b']:.2f} "
                      f"c={r['c']:.2f} d={r['d']:.3f} rms={r['rms']:.3f} "
                      + " ".join(f"{k2}={v:.4g}"
                                 for k2, v in pd.items()
                                 if k2 not in ("gamma", "alpha_g",
                                               "mu0") or k2 == "mu0"),
                      flush=True)
            return res

        maxfev = int(argv[argv.index("--maxfev") + 1]) \
            if "--maxfev" in argv else 150
        if "--resume" in argv and os.path.exists(CKPT):
            ck = json.load(open(CKPT))
            pd = ck["params"]
            z0 = [pd["alpha_R"], math.log(pd["c_R"]),
                  -math.log(pd["mu0"]), pd["alpha_mu"], pd["k"],
                  math.log(pd["F"]), math.log(pd.get("p_sat", 2.0))]
            best["r"] = ck["res"]
            print(f"resuming from {ck['res']:.3f}")
        else:
            nu_i = float(os.environ.get("NU", "0.004"))
            z0 = [float(os.environ.get("AR", "-0.3")),
                  math.log(float(os.environ.get("CR", "1.0"))),
                  -math.log(nu_i),
                  float(os.environ.get("AMU", "-0.5")), 0.007,
                  math.log(20000.0),
                  math.log(float(os.environ.get("PSAT", "1.0")))]
        res = minimize(f, np.asarray(z0), method="Nelder-Mead",
                       options=dict(maxfev=maxfev, xatol=1e-3,
                                    fatol=1e-3))
        print(f"chunk done: {res.fun:.3f} after {res.nfev} evals")
    print(f"total {time.time() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
