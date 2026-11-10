"""
exp_abr1.py -- ABR forward-latency machinery (human-scaling campaign,
stage 1). Reconstructed after container rollback; stage-1 result is
recorded in claude/abr-campaign.md (b' 3.60, c 2.17, d 1.141) and
reproduces exactly.

Target: Neely, Norton, Gorga, Jesteadt (1988): tau = a + b c^-i f^-d,
a = 5.0 ms neural, b = 12.9 ms, c = 5.0, d = 0.413; i = dB SPL/100,
f in kHz. Forward latency tau_f = b c^-i f^-d; EOAE about 2 tau_f.
"""
import sys, os, math, time, json
from dataclasses import replace
import numpy as np
sys.path.insert(0, '.')
import tl_model as T
import exp_accumulation as E

CK = json.load(open("best_d3.json"))["params"]
SEC = {k: CK[k] for k in ("K_t", "M_t", "R_t", "R_c", "R_b",
                          "K_c", "K_p", "M_r")}
GAMMA = CK["gamma"]

F_BASE_HZ = 20700.0
DECADES = float(os.environ.get("DECADES", "3.0"))
FGROUP = float(os.environ.get("FGROUP", "20000"))
N = int(os.environ.get("NPTS", "1024"))


def human_coch():
    return replace(T.WORLDS["B"],
                   sec=replace(T.WORLDS["B"].sec, **SEC),
                   decades=DECADES, F=FGROUP)


def solve_tone(coch, f_dim, L_db, s0=None):
    amp = 10.0 ** (L_db / 20.0)
    r0 = E.line(coch, f_dim, GAMMA, np.zeros(N + 1), 0.0, 0.0, N)
    C = float(np.max(np.abs(r0["shear"])))

    def F(sv):
        r = E.line(coch, f_dim, GAMMA, sv, 0.0, 0.0, N)
        u = amp * np.abs(r["shear"]) / C
        return u * u / (1.0 + u * u)

    s0 = np.zeros(N + 1) if s0 is None else s0
    s, d, it = E.anderson(F, s0)
    return s, d, it


def latency_ms(coch, f_khz, L_db, s0=None, eps=0.01):
    f_dim = (f_khz * 1000.0) / F_BASE_HZ
    s, d, it = solve_tone(coch, f_dim, L_db, s0)
    r1 = E.line(coch, f_dim * (1 - eps), GAMMA, s, 0.0, 0.0, N)
    r2 = E.line(coch, f_dim * (1 + eps), GAMMA, s, 0.0, 0.0, N)
    rm = E.line(coch, f_dim, GAMMA, s, 0.0, 0.0, N)
    j = int(np.argmax(np.abs(rm["shear"])))
    dphi = (np.angle(r2["shear"][j]) - np.angle(r1["shear"][j]))
    while dphi > math.pi:
        dphi -= 2 * math.pi
    while dphi < -math.pi:
        dphi += 2 * math.pi
    tau_dim = -dphi / (2 * eps * f_dim)
    tau_ms = 1000.0 * tau_dim / (2 * math.pi * F_BASE_HZ)
    return tau_ms, s, dict(res=d, it=it, j=j, x_pk=j / N,
                           s_max=float(s.max()))


def fit_bcd(rows):
    """Fit tau = b' c^-(L/100) f^-d; b' at model level 0 (SPL offset
    free), c and d offset-free."""
    A, y = [], []
    for f, L, tau in rows:
        if tau is None or not np.isfinite(tau) or tau <= 0:
            continue
        A.append([1.0, -L / 100.0, -math.log(f)])
        y.append(math.log(tau))
    A = np.asarray(A); y = np.asarray(y)
    co, res = np.linalg.lstsq(A, y, rcond=None)[:2]
    b = math.exp(co[0]); c = math.exp(co[1]); d = co[2]
    rms = math.sqrt(res[0] / len(y)) if len(res) else 0.0
    return b, c, d, rms


def main(argv):
    t0 = time.time()
    coch = human_coch()
    if "--stab" in argv:
        gc = coch.gamma_critical()
        nm, cl = T.growing_modes(coch, GAMMA, 1024)
        print(f"D={DECADES} F={FGROUP}: gamma/gc {GAMMA/gc:.3f}, growing "
              f"modes {nm} (clean={cl})")
        return 0
    print(f"rigid scaling: D={DECADES}, F={FGROUP}, N={N}")
    rows = []
    for f_khz in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
        s0 = None
        for L in (0.0, 20.0, 40.0, 60.0):
            tau, s0, info = latency_ms(coch, f_khz, L, s0)
            rows.append((f_khz, L, tau))
            print(f"{f_khz:6.2f} {L:5.0f} {tau:8.3f} {info['x_pk']:6.3f} "
                  f"{info['s_max']:6.3f} {info['it']:4d}")
    b, c, d, rms = fit_bcd(rows)
    print(f"\nfit: b' {b:.2f} ms, c {c:.2f}, d {d:.3f} (rms {rms:.3f}); "
          f"target b 12.9, c 5.0, d 0.413")
    json.dump(dict(rows=rows, b=b, c=c, d=d, rms=rms, D=DECADES,
                   F=FGROUP), open("abr1_scan.json", "w"))
    print(f"total {time.time() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
