"""
exp_abr3.py -- resolving the 2x latency discrepancy (phase 2 of the
human-scaling campaign).

Hypothesis stack, in test order:
  H1 TONE-BURST LATENCY: the 1988 latencies are onset-to-envelope-peak
     times for specific tone bursts (their Table I: total duration in
     three equal parts, rise / plateau / fall, cosine-squared ramps).
     For a sharp filter the envelope peak lags the steady-state group
     delay. Compute the model's burst-response envelope-peak latency
     by linear synthesis at the frozen saturation state.
  H2 FREE NEURAL CONSTANT: the 1988 decomposition fixed a = 5.0 ms.
     Fit tau_V = a + b c^-i f^-d to the MODEL's wave-V-like latencies
     (burst envelope peak + a) with a free, and ask whether
     (b, c, d) reach (12.9, 5.0, 0.413) at a plausible a.
  H3 POPULATION READ-OUT: wave V pools places; an excitation-weighted
     latency across the peak region instead of the single peak place.

Machinery: the model is linear once the saturation state s is frozen
(s solved per level by the usual Anderson iteration, CW approximation).
For each (f0, SPL): freeze s; sweep the transfer function H(f) of the
IHC drive at the read place over the burst bandwidth; synthesize
y(t) = IFFT(H(f) X(f)) for the Table I burst x(t); latency = time of
the analytic-envelope peak, onset reference (the paper's reference).

Usage:
  python3 exp_abr3.py --check       three-point validation + timing
  python3 exp_abr3.py --surface     full grid, save abr3_surface.json
  python3 exp_abr3.py --fit         free-a and fixed-a fits of the
                                    saved surface
Env: NPTS (1024), NFRQ (192 transfer-function points).
"""
import sys, os, math, time, json
sys.path.insert(0, '.')
import numpy as np
from exp_abr2 import Human, F_BASE_HZ, GRID_SPLS, FREQS

NFRQ = int(os.environ.get("NFRQ", "192"))
FS = 64000.0                      # synthesis rate, Hz
NT = 1 << 12                      # 4096 samples = 64 ms window

# Table I of Neely et al. (1988), ABR column: total burst duration (ms)
DUR_MS = {0.25: 12.0, 0.5: 12.0, 0.75: 8.4, 1.0: 6.0, 1.5: 6.0,
          2.0: 6.0, 3.0: 4.2, 4.0: 3.0, 6.0: 2.04, 8.0: 1.5}


def burst(f_khz):
    """Tone burst per the paper: three equal parts, cos^2 ramps."""
    dur = DUR_MS[f_khz] * 1e-3
    t = np.arange(NT) / FS
    third = dur / 3.0
    env = np.zeros(NT)
    ramp = (t < third)
    env[ramp] = np.sin(0.5 * math.pi * t[ramp] / third) ** 2
    plat = (t >= third) & (t < 2 * third)
    env[plat] = 1.0
    fall = (t >= 2 * third) & (t < dur)
    env[fall] = np.cos(0.5 * math.pi * (t[fall] - 2 * third) / third) ** 2
    x = env * np.sin(2 * math.pi * f_khz * 1000.0 * t)
    return x


def burst_latency_ms(h, f_khz, L_model, s_prev=None):
    """Envelope-peak latency of the model's IHC-drive response to the
    Table I burst at f_khz, saturation state frozen at level L_model.
    Returns (latency_ms, s_state, peak_index j)."""
    f_dim0 = f_khz * 1000.0 / F_BASE_HZ
    # freeze the saturation state (CW solve at this level)
    amp = 10.0 ** (L_model / 20.0)
    r0 = h.line2(f_dim0, np.zeros(len(h.x)))
    C = float(np.max(np.abs(r0["shear"])))
    p_s = h.p_sat
    import exp_accumulation as E

    def FF(sv):
        r = h.line2(f_dim0, sv)
        u = (amp * np.abs(r["shear"]) / C) ** p_s
        return u / (1.0 + u)

    s, _, _ = E.anderson(FF, np.zeros(len(h.x)) if s_prev is None
                         else s_prev)
    rm = h.line2(f_dim0, s)
    j = int(np.argmax(np.abs(rm["shear"])))
    # transfer function over the burst band at the frozen state
    fr_hz = np.fft.rfftfreq(NT, 1.0 / FS)
    X = np.fft.rfft(burst(f_khz))
    # frequencies that matter: within 55 dB of the spectral peak
    mag = np.abs(X)
    keep = mag > mag.max() * 10 ** (-55 / 20.0)
    keep[0] = False
    idx = np.where(keep)[0]
    # subsample the kept band to NFRQ points, interpolate H between
    sub = idx[np.linspace(0, len(idx) - 1, min(NFRQ, len(idx))
                          ).astype(int)]
    Hs = np.empty(len(sub), complex)
    for m, i in enumerate(sub):
        fd = fr_hz[i] / F_BASE_HZ
        Hs[m] = h.line2(fd, s)["shear"][j]
    # interpolate real/imag onto the kept band
    Hfull = np.zeros(len(fr_hz), complex)
    Hfull[idx] = (np.interp(fr_hz[idx], fr_hz[sub], Hs.real)
                  + 1j * np.interp(fr_hz[idx], fr_hz[sub], Hs.imag))
    y = np.fft.irfft(Hfull * X, NT)
    env = np.abs(np.fft.ifft(
        np.fft.fft(y) * (np.arange(NT) < NT // 2) * 2))  # analytic env
    pk = int(np.argmax(env[:NT // 2]))
    return 1000.0 * pk / FS, s, j


def load_model():
    ck = json.load(open("abr2g2_ckpt.json"))
    pd = dict(ck["params"])
    return Human(**pd), ck["S0"]


def main(argv):
    t0 = time.time()
    if "--check" in argv:
        h, S0 = load_model()
        for f_khz, spl in ((1.0, 50.0), (1.0, 90.0), (4.0, 50.0),
                           (0.25, 50.0)):
            t1 = time.time()
            tb, s, j = burst_latency_ms(h, f_khz, spl - S0)
            tg, _ = h.latency_ms(f_khz, spl - S0)
            print(f"f {f_khz} kHz, {spl:.0f} dB SPL: burst peak "
                  f"{tb:6.2f} ms | group delay {tg:6.2f} ms | ratio "
                  f"{tb / tg if tg > 0 else float('nan'):.2f} "
                  f"({time.time() - t1:.1f} s)")
    if "--surface" in argv:
        h, S0 = load_model()
        rows = []
        for f_khz in FREQS:
            sp = None
            for spl in GRID_SPLS:
                tb, sp, _ = burst_latency_ms(h, f_khz, spl - S0, sp)
                rows.append((f_khz, float(spl), tb))
                print(f"  {f_khz:5.2f} kHz {spl:3d} dB: {tb:6.2f} ms",
                      flush=True)
        json.dump(dict(rows=rows, S0=S0), open("abr3_surface.json", "w"))
        print("saved abr3_surface.json")
    if "--fit" in argv:
        S = json.load(open("abr3_surface.json"))
        rows = S["rows"]
        # tau_V = a + tau_burst(model); fit a + b c^-i f^-d with a free:
        # for each trial a, the mechanical part is tau_V - a and we fit
        # b, c, d log-linearly to (tau_burst + a_model_neural - a)...
        # Direct approach: the model's wave-V analog is
        # tau_V = a_true + tau_burst. The 1988 analysis subtracts a
        # fixed 5 ms. We ask: with a free, does the model's tau_V match
        # the DATA's tau_V = 5 + 12.9 c^-i f^-d?
        print("comparing model wave-V (a + burst latency) to the data's "
              "wave-V (5 + 12.9 * 5^-i f^-0.413):")
        best = None
        for a_try in np.arange(0.0, 8.01, 0.25):
            err = []
            for f, spl, tb in rows:
                tV_data = 5.0 + 12.9 * 5.0 ** (-spl / 100.0) * f ** (-0.413)
                err.append((a_try + tb) - tV_data)
            err = np.asarray(err)
            rms = float(np.sqrt(np.mean(err ** 2)))
            if best is None or rms < best[1]:
                best = (a_try, rms, float(np.mean(err)))
        print(f"  best neural constant a = {best[0]:.2f} ms: rms "
              f"{best[1]:.3f} ms (mean err {best[2]:+.3f})")
        # and the 1988-style decomposition of the model's tau_V with
        # a fixed at 5: what (b, c, d) does the model then show?
        for a_fix in (5.0, best[0]):
            A, y = [], []
            for f, spl, tb in rows:
                mech = tb + a_fix - 5.0     # what 1988 would call tau_f
                if mech > 0:
                    A.append([1.0, -spl / 100.0, -math.log(f)])
                    y.append(math.log(mech))
            co, res = np.linalg.lstsq(np.asarray(A), np.asarray(y),
                                      rcond=None)[:2]
            b, c, d = math.exp(co[0]), math.exp(co[1]), co[2]
            rms = math.sqrt(res[0] / len(y)) if len(res) else 0.0
            print(f"  1988-style fit with a={a_fix:.2f} subtracted at 5: "
                  f"b={b:.2f} c={c:.2f} d={d:.3f} log-rms {rms:.3f}")
    print(f"total {time.time() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
