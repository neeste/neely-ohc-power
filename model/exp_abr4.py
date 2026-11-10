"""
exp_abr4.py -- H4: onset response with synaptic adaptation and an
absolute threshold, per STN's physiological steer (ABR is an onset
response; latency shaped by synaptic adaptation).

Read-out model, few global constants:
  drive        v(t) = amp(L) * envelope of the linear burst response at
               the frozen saturation state, in OHC-knee units (the same
               normalization as the saturator drive)
  rate         r(t) = v^pr / (theta^pr + v^pr)          (saturating)
  adaptation   A(t) = exp(-(1/tau_a) * integral_0^t r dt')  (depletion)
  onset rate   w(t) = r(t) * A(t)
  latency      t at the peak of w(t); wave V = a + latency.

Fit (theta, tau_a, pr, a) globally against the data's wave V surface
tau_V = 5 + 12.9 * 5^-(SPL/100) * f^-0.413 on the 6 x 9 grid.

Usage:
  python3 exp_abr4.py --bank      compute and save the envelope bank
  python3 exp_abr4.py --scan      grid-search the read-out constants
"""
import sys, os, math, time, json
sys.path.insert(0, '.')
import math
import numpy as np
from exp_abr2 import Human, F_BASE_HZ, GRID_SPLS, FREQS
from exp_abr3 import burst, NT, FS, NFRQ
import exp_accumulation as E

BANK = "abr4_bank.npz"


def load_model():
    ck = json.load(open("abr2g2_ckpt.json"))
    return Human(**dict(ck["params"])), ck["S0"]


def make_bank():
    h, S0 = load_model()
    fr_hz = np.fft.rfftfreq(NT, 1.0 / FS)
    envs, keys = [], []
    for f_khz in FREQS:
        f_dim0 = f_khz * 1000.0 / F_BASE_HZ
        X = np.fft.rfft(burst(f_khz))
        mag = np.abs(X)
        keep = mag > mag.max() * 10 ** (-55 / 20.0)
        keep[0] = False
        idx = np.where(keep)[0]
        sub = idx[np.linspace(0, len(idx) - 1,
                              min(NFRQ, len(idx))).astype(int)]
        sp = None
        for spl in GRID_SPLS:
            L = spl - S0
            amp = 10.0 ** (L / 20.0)
            r0 = h.line2(f_dim0, np.zeros(len(h.x)))
            C = float(np.max(np.abs(r0["shear"])))
            p_s = h.p_sat

            def FF(sv):
                r = h.line2(f_dim0, sv)
                u = (amp * np.abs(r["shear"]) / C) ** p_s
                return u / (1.0 + u)

            sp, _, _ = E.anderson(FF, np.zeros(len(h.x))
                                  if sp is None else sp)
            rm = h.line2(f_dim0, sp)
            j = int(np.argmax(np.abs(rm["shear"])))
            Hs = np.array([h.line2(fr_hz[i] / F_BASE_HZ, sp)["shear"][j]
                           for i in sub])
            Hf = np.zeros(len(fr_hz), complex)
            Hf[idx] = (np.interp(fr_hz[idx], fr_hz[sub], Hs.real)
                       + 1j * np.interp(fr_hz[idx], fr_hz[sub], Hs.imag))
            y = np.fft.irfft(Hf * X, NT)
            env = np.abs(np.fft.ifft(
                np.fft.fft(y) * (np.arange(NT) < NT // 2) * 2))[:NT // 2]
            # absolute drive in knee units: amp * |shear|/C
            envs.append(amp * env / C)
            keys.append((f_khz, float(spl)))
            print(f"  bank {f_khz:5.2f} kHz {spl:3.0f} dB", flush=True)
    np.savez_compressed(BANK, envs=np.asarray(envs),
                        keys=np.asarray(keys))
    print("bank saved:", BANK)


def scan():
    Z = np.load(BANK)
    envs, keys = Z["envs"], Z["keys"]
    t_ms = np.arange(envs.shape[1]) / FS * 1000.0
    dt = 1.0 / FS
    tV_data = np.array([5.0 + 12.9 * 5.0 ** (-spl / 100.0)
                        * f ** (-0.413) for f, spl in keys])
    best = None
    for pr in (1.0, 2.0):
        for th_log in np.linspace(-3.5, 1.0, 19):
            theta = 10.0 ** th_log
            r = envs ** pr / (theta ** pr + envs ** pr)
            for tau_a in (1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 1e9):
                A = np.exp(-np.cumsum(r, axis=1) * dt * 1000.0 / tau_a)
                w = r * A
                lat = t_ms[np.argmax(w, axis=1)]
                a = float(np.mean(tV_data - lat))
                rms = float(np.sqrt(np.mean((a + lat - tV_data) ** 2)))
                if best is None or rms < best["rms"]:
                    best = dict(pr=pr, theta=theta, tau_a=tau_a, a=a,
                                rms=rms, lat=lat.tolist())
                    print(f"  best rms {rms:6.3f} ms: pr {pr}, theta "
                          f"{theta:.4g}, tau_a {tau_a}, a {a:.2f}",
                          flush=True)
    # report the winning surface's 1988-style decomposition
    lat = np.asarray(best["lat"])
    A2, y2 = [], []
    for (f, spl), tb in zip(keys, lat):
        mech = tb + best["a"] - 5.0
        if mech > 0:
            A2.append([1.0, -spl / 100.0, -math.log(f)])
            y2.append(math.log(mech))
    co, res = np.linalg.lstsq(np.asarray(A2), np.asarray(y2),
                              rcond=None)[:2]
    b, c, d = math.exp(co[0]), math.exp(co[1]), co[2]
    print(f"\nWINNER: rms {best['rms']:.3f} ms, a {best['a']:.2f}, "
          f"theta {best['theta']:.4g}, tau_a {best['tau_a']}, "
          f"pr {best['pr']}")
    print(f"1988-style decomposition of the model's wave V (subtract "
          f"5 ms): b {b:.2f} (12.9), c {c:.2f} (5.0), d {d:.3f} (0.413)")
    # latency-intensity slope at 1 and 2 kHz, ms per 10 dB, 40-80 SPL
    for f0 in (1.0, 2.0):
        pick = [(spl, tb) for (f, spl), tb in zip(keys, lat)
                if f == f0 and 40 <= spl <= 80]
        if len(pick) >= 2:
            sl = (pick[0][1] - pick[-1][1]) / ((pick[-1][0]
                                                - pick[0][0]) / 10.0)
            print(f"latency-intensity slope at {f0} kHz: {sl:.2f} ms "
                  f"per 10 dB (classic ABR ~0.3-0.5)")
    json.dump({k: v for k, v in best.items() if k != "lat"},
              open("abr4_best.json", "w"), indent=1)




TAU_BOUNDS = (1.0, 20.0)   # ms; adaptation time constants outside this range are not used


def joint_rms(h, S0, th_grid, tau_grid, pr_grid, reject_onset=False, refine=False):
    """Best read-out rms for a given Human model (inline bank).
    reject_onset: skip read-out constants for which any latency falls on
    the first sample (a peak of r*A at t = 0 is an onset artifact).
    refine: after the grid, refine theta and tau_a continuously (Nelder-Mead
    in their logarithms) at the best pr, so no constant is held at a grid edge."""
    fr_hz = np.fft.rfftfreq(NT, 1.0 / FS)
    envs, keys = [], []
    for f_khz in FREQS:
        f_dim0 = f_khz * 1000.0 / F_BASE_HZ
        X = np.fft.rfft(burst(f_khz))
        mag = np.abs(X)
        keep = mag > mag.max() * 10 ** (-55 / 20.0)
        keep[0] = False
        idx = np.where(keep)[0]
        sub = idx[np.linspace(0, len(idx) - 1,
                              min(96, len(idx))).astype(int)]
        sp = None
        for spl in GRID_SPLS:
            amp = 10.0 ** ((spl - S0) / 20.0)
            r0 = h.line2(f_dim0, np.zeros(len(h.x)))
            C = float(np.max(np.abs(r0["shear"])))
            p_s = h.p_sat

            def FF(sv):
                r = h.line2(f_dim0, sv)
                u = (amp * np.abs(r["shear"]) / C) ** p_s
                return u / (1.0 + u)

            sp, _, _ = E.anderson(FF, np.zeros(len(h.x))
                                  if sp is None else sp)
            rm = h.line2(f_dim0, sp)
            j = int(np.argmax(np.abs(rm["shear"])))
            Hs = np.array([h.line2(fr_hz[i] / F_BASE_HZ, sp)["shear"][j]
                           for i in sub])
            Hf = np.zeros(len(fr_hz), complex)
            Hf[idx] = (np.interp(fr_hz[idx], fr_hz[sub], Hs.real)
                       + 1j * np.interp(fr_hz[idx], fr_hz[sub],
                                        Hs.imag))
            y = np.fft.irfft(Hf * X, NT)
            env = np.abs(np.fft.ifft(np.fft.fft(y)
                         * (np.arange(NT) < NT // 2) * 2))[:NT // 2]
            envs.append(amp * env / C)
            keys.append((f_khz, float(spl)))
    envs = np.asarray(envs)
    t_ms = np.arange(envs.shape[1]) / FS * 1000.0
    dt = 1.0 / FS
    tV = np.array([5.0 + 12.9 * 5.0 ** (-s_ / 100.0) * f_ ** (-0.413)
                   for f_, s_ in keys])
    best = (1e9, None)
    for pr in pr_grid:
        for theta in th_grid:
            r = envs ** pr / (theta ** pr + envs ** pr)
            for tau_a in tau_grid:
                A = np.exp(-np.cumsum(r, axis=1) * dt * 1000.0 / tau_a)
                im = np.argmax(r * A, axis=1)
                if reject_onset and np.any(im == 0):
                    continue
                lat = t_ms[im]
                a = float(np.mean(tV - lat))
                rms = float(np.sqrt(np.mean((a + lat - tV) ** 2)))
                if rms < best[0]:
                    best = (rms, dict(pr=pr, theta=theta, tau_a=tau_a,
                                      a=a, lat=lat.tolist(),
                                      keys=[list(k) for k in keys]))
    if refine and best[1] is not None:
        from scipy.optimize import minimize as _min
        pr = best[1]['pr']

        def ev(z):
            theta, tau_a = math.exp(z[0]), math.exp(z[1])
            if not (TAU_BOUNDS[0] <= tau_a <= TAU_BOUNDS[1] and 1e-3 <= theta <= 10.0):
                return 1e9, None
            r = envs ** pr / (theta ** pr + envs ** pr)
            A = np.exp(-np.cumsum(r, axis=1) * dt * 1000.0 / tau_a)
            im = np.argmax(r * A, axis=1)
            if reject_onset and np.any(im == 0):
                return 1e9, None
            lat = t_ms[im]; a = float(np.mean(tV - lat))
            return float(np.sqrt(np.mean((a + lat - tV) ** 2))), (theta, tau_a, a, lat)
        z0 = [math.log(best[1]['theta']), math.log(best[1]['tau_a'])]
        res = _min(lambda z: ev(z)[0], z0, method='Nelder-Mead',
                   options=dict(xatol=1e-3, fatol=1e-5, maxfev=200,
                                initial_simplex=[z0, [z0[0] + 0.3, z0[1]], [z0[0], z0[1] + 0.3]]))
        rms, info = ev(res.x)
        if info is not None and rms < best[0]:
            theta, tau_a, a, lat = info
            best = (rms, dict(pr=pr, theta=theta, tau_a=tau_a, a=a,
                              lat=lat.tolist(), keys=[list(k) for k in keys]))
    return best


def fit_joint(argv):
    from scipy.optimize import minimize
    CKPT = "abr4_ckpt.json"
    ck0 = json.load(open("abr2g2_ckpt.json"))
    SECN = ("K_t", "M_t", "R_t", "R_c", "R_b", "K_c", "K_p", "M_r")
    NM = ["alpha_R", "log_cR", "nlog_mu0", "alpha_mu", "k", "logF",
          "log_ps", "S0"] + ["log_" + k for k in SECN]
    TH = [10.0 ** e for e in np.linspace(-1.5, 0.5, 7)]
    TAU = (1.0, 2.0, 3.0, 5.0, 8.0)
    PRG = (1.5, 2.0, 3.0)
    best = {"r": 1e9}

    def unpack(z):
        pd = dict(alpha_R=z[0], c_R=math.exp(z[1]),
                  mu0=math.exp(-z[2]), alpha_mu=z[3], k=z[4],
                  F=math.exp(z[5]), p_sat=math.exp(z[6]),
                  sec_over={k: math.exp(z[8 + i])
                            for i, k in enumerate(SECN)})
        return pd, z[7]

    def f(z):
        pd, S0 = unpack(z)
        if not (-1.6 <= pd["alpha_R"] <= 0.3 and
                0.01 <= pd["c_R"] <= 3.0 and
                0.001 <= pd["mu0"] <= 0.7 and
                -1.2 <= pd["alpha_mu"] <= 0.3 and
                0.0 <= pd["k"] <= 0.03 and
                2000 <= pd["F"] <= 200000 and
                0.4 <= pd["p_sat"] <= 2.5 and
                10.0 <= S0 <= 90.0):
            return 30.0
        try:
            h = Human(**pd)
            if not h.stable:
                return 20.0
            rms, ro = joint_rms(h, S0, TH, TAU, PRG)
        except Exception:
            return 40.0
        if not (2.0 <= ro["a"] <= 8.0):
            rms += abs(ro["a"] - 5.0) * 0.2
        if rms < best["r"]:
            best["r"] = rms
            json.dump(dict(rms=rms, params=pd, S0=S0, readout={
                kk: vv for kk, vv in ro.items() if kk != "lat"
                and kk != "keys"}, lat=ro["lat"], keys=ro["keys"]),
                open(CKPT, "w"), indent=1)
            print(f"    best rms {rms:.3f} ms: a {ro['a']:.2f} theta "
                  f"{ro['theta']:.3g} tau {ro['tau_a']} pr {ro['pr']}",
                  flush=True)
        return rms

    maxfev = int(argv[argv.index("--maxfev") + 1]) \
        if "--maxfev" in argv else 100
    if "--resume" in argv and os.path.exists(CKPT):
        ck = json.load(open(CKPT))
        pd = ck["params"]
        z0 = [pd["alpha_R"], math.log(pd["c_R"]), -math.log(pd["mu0"]),
              pd["alpha_mu"], pd["k"], math.log(pd["F"]),
              math.log(pd["p_sat"]), ck["S0"]] + \
             [math.log(pd["sec_over"][k]) for k in SECN]
        best["r"] = ck["rms"]
        print(f"[joint] resuming from {ck['rms']:.3f}")
    else:
        pd = dict(ck0["params"])
        z0 = [pd["alpha_R"], math.log(pd["c_R"]), -math.log(pd["mu0"]),
              pd["alpha_mu"], pd["k"], math.log(pd["F"]),
              math.log(pd["p_sat"]), ck0["S0"]] + \
             [math.log(pd["sec_over"][k]) for k in SECN]
    res = minimize(f, np.asarray(z0), method="Nelder-Mead",
                   options=dict(maxfev=maxfev, xatol=1e-3, fatol=1e-3))
    print(f"[joint] chunk done: {res.fun:.3f} after {res.nfev} evals")


def main(argv):
    t0 = time.time()
    if "--bank" in argv:
        make_bank()
    if "--scan" in argv:
        scan()
    if "--joint" in argv:
        fit_joint(argv)
    print(f"total {time.time() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
