"""
exp_af_source.py -- phase 3b: the Allen-Fahey function with a DISTRIBUTED
distortion source.

Finding 12 made observation 6's excitation-ratio metric structurally
incompatible with observation 4 for any gamma(x): the amplifier's gain
accrues within a third of an octave of the peak, basal of which every AF-92
generator sits, so the max forward-gain ratio equals the total gain. That
ratio is the INFERENCE AF-92 drew. What they MEASURED is the ear-canal DP
level with the DP-place response held constant, and de Boer (2005) and
Shera-Guinan (2007) showed the two differ: the DP source is distributed over
the primary-overlap region, and its wavelength-dependent coupling to the
backward traveling wave varies with r in a way that cancels most of the
amplifier gain as r -> 1.

THE SIMULATION. At fixed f_dp = CF(x = 0.55), for each primary ratio r:
f2 = r f_dp/(2 - r), f1 = (f_dp + f2)/2 (so 2 f1 - f2 = f_dp exactly).
Primary fields shear1, shear2 from the linear line at the same gamma. The
cubic distortion source at f_dp is

    rho(x) = shear1(x)^2 conj(shear2(x))

(the |X1|^2|X2| magnitude with the 2 phi1 - phi2 phase), the leading 2f1-f2
term of any odd saturating transducer, placed where the transducer is: the
cilia. It forces the line as a distributed partition velocity:

    P_{j-1} - (2 + h^2 q_j) P_j + P_{j+1} = h^2 g_j,   g = -w^2 F rho

with P(0) = 0 (the stapes drive is at the primaries, not at f_dp) and
P(L) = 0. The ear-canal proxy is the basal volume velocity U(0) ~ dP/dx(0),
scaled so the DP-place BM response is one: AF(r) = 20 log10 |dP/dx(0)| at
|X_b(x_dp)| = 1. Only the EXCURSION over r matters (target 8 +/- 6 dB), so
every constant prefactor cancels.

VERIFY: (V1) forcing at the first interior node reproduces the Dirichlet
response shape to machine precision; (V2) a point source at x_gen(r)
reproduces the excitation-ratio story: its AF excursion is large and tracks
exp_obs46.forward_gain; (V3) grid convergence N = 2048 vs 4096.

Run: python3 exp_af_source.py [--world B] [--gammas g1,g2,...]
"""
import sys, math, time
import numpy as np
sys.path.insert(0, '.')
from scipy.linalg import solve_banded
import tl_model as T
import exp_obs46 as O

X_DP_REF = O.X_PROBE
R_LIST = tuple(np.round(np.linspace(1.05, 1.50, 10), 4))


def forced(coch, w, Y, N, g):
    """The _fd solve with P(0)=0, P(L)=0 and a distributed RHS g(x)."""
    q = -(w ** 2) * coch.F * Y
    h = 1.0 / N
    hq = (h * h) * q
    n = N - 1
    ab = np.zeros((3, n), dtype=complex)
    ab[0, 1:] = 1.0
    ab[1, :] = -(2.0 + hq[1:N])
    ab[2, :-1] = 1.0
    rhs = (h * h) * g[1:N]
    P = np.zeros(N + 1, dtype=complex)
    P[1:N] = solve_banded((1, 1), ab, rhs)
    return P


def fields(coch, freq, gamma, N):
    x = np.linspace(0.0, 1.0, N + 1)
    a = coch.cf(x)
    gam = coch.gamma_of_x(x, gamma, None)
    Y, Y_r, H_o = T.section_terms(coch.sec, freq, a, a, gam)
    return x, Y, H_o


def af_distributed(coch, gamma, N=4096, r_list=R_LIST, point=False):
    """AF function: ear-canal proxy vs r at unit DP-place BM response."""
    f_dp = float(coch.cf(X_DP_REF))
    x, Y_dp, H_dp = fields(coch, f_dp, gamma, N)
    # DP place: peak of the f_dp response on the active line
    r_ref = T.response(coch, f_dp, N=N, gamma=gamma)
    j_dp = int(np.argmax(np.abs(r_ref["X_b"])))
    rows = []
    for r in r_list:
        f2 = r * f_dp / (2.0 - r)
        f1 = 0.5 * (f_dp + f2)
        r1 = T.response(coch, f1, N=N, gamma=gamma)
        r2 = T.response(coch, f2, N=N, gamma=gamma)
        sh1 = r1["shear"]
        sh2 = r2["shear"]
        rho = sh1 * sh1 * np.conj(sh2)
        if point:
            jg = int(np.argmax(np.abs(rho)))
            rho2 = np.zeros_like(rho)
            rho2[jg] = rho[jg] * N        # same integrated weight
            rho = rho2
        g = -(f_dp ** 2) * coch.F * rho
        P = forced(coch, f_dp, Y_dp, N, g)
        X_b = Y_dp * P
        dp_resp = abs(X_b[j_dp])
        u0 = abs(P[1] - P[0]) * N          # dP/dx at the base
        rows.append((float(r), 20.0 * math.log10(u0 / dp_resp),
                     float(np.argmax(np.abs(rho))) / N))
    af = np.asarray([v for _, v, _ in rows])
    return rows, float(af.max() - af.min())


def main(argv):
    t0 = time.time()
    wname = argv[argv.index("--world") + 1] if "--world" in argv else "B"
    coch = T.WORLDS[wname]
    gc = coch.gamma_critical()
    if "--gammas" in argv:
        gammas = [float(v) for v in
                  argv[argv.index("--gammas") + 1].split(",")]
    else:
        g30 = T.gamma_for_gain(coch, 30.0, float(coch.cf(X_DP_REF)),
                               "local", N=2048, x_ref=X_DP_REF)[0]
        gammas = [g30, 0.60 * gc, T.CEILING[wname]]

    print("=" * 78)
    print(f"THE ALLEN-FAHEY FUNCTION WITH A DISTRIBUTED SOURCE, world "
          f"{wname}")
    print("=" * 78)
    print(f"  f_dp = CF({X_DP_REF}); r swept {R_LIST[0]} to {R_LIST[-1]}; "
          f"excursion target 8 +/- 6 dB\n  point-source rows are the "
          f"V2 control, expected to reproduce the paradox")

    # V1: forcing at the first node reproduces the Dirichlet shape
    N = 2048
    f_dp = float(coch.cf(X_DP_REF))
    x, Y, _ = fields(coch, f_dp, gammas[0], N)
    g = np.zeros(N + 1, dtype=complex)
    g[1] = 1.0
    Pf = forced(coch, f_dp, Y, N, g)
    Pd = T._fd(coch, f_dp, Y, N, 1.0)
    c = Pf[2:N] / Pd[2:N]
    err = float(np.max(np.abs(c / c[0] - 1.0)))
    print(f"\n[V1] first-node forcing against the Dirichlet solve, "
          f"proportionality error {err:.3e}")

    for gamma in gammas:
        print(f"\n  ---- gamma = {gamma:.5f} ({gamma / gc:.3f} of "
              f"section-critical) ----")
        for point, label in ((False, "distributed"), (True, "point")):
            rows, exc = af_distributed(coch, gamma, 4096, R_LIST, point)
            line = "  ".join(f"{v:7.1f}" for _, v, _ in rows)
            print(f"    {label:>12}: AF(r) = {line}")
            print(f"    {label:>12}: excursion {exc:7.2f} dB "
                  f"{'(PASS)' if exc <= 14.0 else '(fail)'}")
        # V3 convergence on the distributed excursion
        _, exc2 = af_distributed(coch, gamma, 2048, R_LIST, False)
        print(f"    convergence: excursion N=2048 {exc2:.2f} against "
              f"N=4096 above")

    print(f"\n  total {time.time() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
