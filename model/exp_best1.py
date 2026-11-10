"""
exp_best1.py -- the best-model campaign's composite score.

Ranking (user directive 2026-08-21): hard requirement is passing 6/6 under
both scorings; rank by the worst-case normalized margin (interior beats
corner point); physicality as secondary score: bounded gate depth, gate
exponent near the Cheatham-Dallos first order, straight growth curves,
stability margin.

For each metric with window [lo, hi], the normalized margin is
    m = min(x - lo, hi - x) / ((hi - lo) / 2)
so m = 1 at the window center, 0 at an edge, negative outside. The primary
score is the minimum margin over all metrics of both scorings. The
secondary score is the mean of four physicality components in [0, 1]:
  p_floor    gate depth cap -20 log10(s) at most 60 dB, best at 20 dB
  p_m        gate exponent within 0.5 of 1
  p_straight RMS residual of the linear fit to the growth ladder
  p_stab     gamma/gamma_critical comfortably inside 0.95

Gate (amplitude convention, canonical): the neural amplitude is multiplied
by   G = s + (1 - s) * (1 + (u_s/q)^2)^(-m)
with floor s bounding the suppression depth at -20 log10(s) dB.

Parameter vector (13): K_t M_t R_t R_c R_b K_c K_p M_r gamma q m u_thr s_g

Usage:
  python3 exp_best1.py --baseline     evaluate Findings 18/19 points
"""
import sys, math, time, json
from dataclasses import replace
import numpy as np
sys.path.insert(0, '.')
sys.path.insert(0, '../../scoring')
import tl_model as T
import exp_accumulation as E
import exp_twotone as TT
from exp_twotone5 import Gated
from exp_encoded1 import score_linear
from exp_af_source2 import af_run

NAMES = ["K_t", "M_t", "R_t", "R_c", "R_b", "K_c", "K_p", "M_r",
         "gamma", "q", "m", "u_thr", "s_g"]

# metric windows [lo, hi]
WIN = dict(
    tt=(32.0, 48.0), sl=(3.0, 9.0), mp=(-0.75, -0.25),
    ph=(math.pi - 0.7, math.pi + 0.7), fwd=(0.0, 13.0), exc=(2.0, 14.0),
    vsl=(1.4, 2.6), vdt=(14.0, 36.0),
    gr33=(1.4, 2.6), mdt=(18.0, 32.0), afd=(2.0, 14.0),
)

P_ORD = 2.0        # saturator drive-summation order (d3 lever)

P19 = [0.011953911543652536, 0.14444647050251735, 0.0269890954292853,
       0.06668441571442195, 0.16276494445217984, 0.05131255122810848,
       0.11529210100373503, 0.15937084662213039, 0.09283172164064372,
       0.021859586666877852, 1.1512913340000535, 0.0026541879319179926,
       1e-4]
P18 = [0.01181, 0.16004, 0.02898, 0.06268, 0.22733, 0.03308, 0.14171,
       0.17718, 0.0999, 0.05, 1.0, 0.02, 1e-4]


class Gated3(Gated):
    """IHC gate with floor s_g and exponent m, both instance attributes."""

    def __init__(self, coch, gain_db, u_thr, q, m=1.0, s_g=0.0, **kw):
        super().__init__(coch, gain_db, u_thr, q, **kw)
        self.m = m
        self.s_g = s_g

    def readout(self, s, P_p):
        r = E.line(self.coch, self.f_p, self.g0, s, self.beta,
                   self.depth, self.N)
        Ap = 10.0 ** (P_p / 20.0)
        c_lin = Ap * abs(r["shear"][self.j])
        b = Ap * abs(r["X_b"][self.j])
        gate = 1.0
        if self._Ls is not None:
            As = 10.0 ** (self._Ls / 20.0)
            rs = E.line(self.coch, self.f_s, self.g0, s, self.beta,
                        self.depth, self.N)
            us = As * abs(rs["shear"][self.j]) / self.C
            gate = (self.s_g + (1.0 - self.s_g)
                    * (1.0 + (us / self.q) ** 2) ** (-self.m))
        return (20.0 * math.log10(gate * c_lin), 20.0 * math.log10(b))


def margin(x, key):
    lo, hi = WIN[key]
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return -5.0
    if isinstance(x, float) and math.isinf(x):
        return -4.0
    return min(x - lo, hi - x) / ((hi - lo) / 2.0)


def evaluate(p, N=1024, verbose=False):
    """Composite evaluation. Returns dict with margins, primary, secondary,
    raw values; primary is -inf on stability rejection."""
    (kt, mt, rt, rc, rb, kc, kp, mr, g, q, m_g, u, s_g) = p
    coch = replace(T.WORLDS["B"], sec=replace(
        T.WORLDS["B"].sec, K_t=kt, M_t=mt, R_t=rt, R_c=rc, R_b=rb,
        K_c=kc, K_p=kp, M_r=mr))
    gc = coch.gamma_critical()
    ratio = g / gc
    if ratio >= 0.97:
        return dict(primary=-9.0, secondary=0.0, ratio=ratio,
                    margins={}, values={})

    vals = {}
    # linear six, exactly the official scorer's numbers
    tt_, sl_, mp_, ph_, fwd_, exc_ = score_linear(coch, g)
    vals.update(tt=tt_, sl=sl_, mp=mp_, ph=ph_, fwd=fwd_, exc=exc_)

    f_p = float(coch.cf(E.X_PROBE))
    r = T.response(coch, f_p, N=2048, gamma=g)
    E.X_READ = T._peak(r["x"], r["X_b"])[0]
    orig = T.gamma_for_gain
    T.gamma_for_gain = lambda *a, **k: (g, float("nan"))
    try:
        TT.P_ORD = P_ORD
        TT.OCT_BELOW = 1.8
        TT.SL_COOPER = (40.0, 50.0)
        TT.GROWTH_SPAN = (3.0, 33.0, 3.0)
        TT.N_RUN = N
        gt = Gated3(coch, 30.0, u, q, m=m_g, s_g=s_g, N=N)
        meas = TT.measure(gt)
        vals["gr33"] = meas["slope"]
        dts = []
        for sl in TT.SL_COOPER:
            b = meas["bm"][sl]
            if math.isnan(meas["L_n"]) or math.isnan(b):
                dts.append(float("inf"))
            else:
                dts.append(b - meas["L_n"])
        # mdt: the better-margined of the two Cooper SLs
        vals["mdt"] = (dts[0] if margin(dts[0], "mdt")
                       >= margin(dts[1], "mdt") else dts[1])
        vals["dts"] = [round(d, 1) if not math.isinf(d) else d for d in dts]
        # growth straightness (RMS residual of the linear fit, dB)
        pts = meas["growth"]
        if len(pts) >= 4:
            x = np.asarray([a for a, _ in pts])
            y = np.asarray([b for _, b in pts])
            A = np.vstack([x, np.ones_like(x)]).T
            co, res = np.linalg.lstsq(A, y, rcond=None)[:2]
            rms = float(np.sqrt(res[0] / len(pts))) if len(res) else 0.0
        else:
            rms = 5.0
        vals["rms"] = rms

        # vertical proxy (encoded obs 2/3): fixed small probe, swept
        # suppressor, gated neural vs mechanical BM read-outs
        if math.isnan(meas["L_n"]):
            vals["vsl"] = vals["vdt"] = float("nan")
        else:
            Lv = np.arange(meas["L_n"] - 12.0, meas["L_n"] + 36.0, 3.0)
            c0, b0, s_run, _ = gt.response(None, gt.P0 - 30.0)
            sn, sb = [], []
            for L in Lv:
                c, b, s_run, _ = gt.response(L, gt.P0 - 30.0, s_run)
                sn.append(c0 - c)
                sb.append(b0 - b)
            sn = np.asarray(sn); sb = np.asarray(sb)

            def vcross(yv):
                w = np.where(yv >= 1.0)[0]
                if w.size == 0 or w[0] == 0:
                    return None
                i = w[0]
                return float(Lv[i - 1] + (1.0 - yv[i - 1]) * 3.0
                             / (yv[i] - yv[i - 1]))
            tn, tb = vcross(sn), vcross(sb)
            if tn is None:
                vals["vsl"] = vals["vdt"] = float("nan")
            else:
                sel = [(a, b_) for a, b_ in zip(Lv, sn) if a >= tn]
                if len(sel) >= 3:
                    A = np.vstack([[a for a, _ in sel],
                                   np.ones(len(sel))]).T
                    vals["vsl"] = float(np.linalg.lstsq(
                        A, np.asarray([b_ for _, b_ in sel]),
                        rcond=None)[0][0])
                else:
                    vals["vsl"] = float("nan")
                vals["vdt"] = (tb - tn) if tb is not None else float("inf")

        rows = af_run(coch, g, 2048)
        flat = [v for rr_, v in rows if rr_ >= 1.25]
        vals["afd"] = max(flat) - min(flat)
    finally:
        T.gamma_for_gain = orig

    margins = {k: margin(vals.get(k), k) for k in WIN}
    primary = min(margins.values())

    depth_cap = -20.0 * math.log10(max(s_g, 1e-12))
    p_floor = max(0.0, min(1.0, (60.0 - depth_cap) / 40.0))
    p_m = max(0.0, min(1.0, 1.0 - abs(m_g - 1.0) / 0.5))
    p_straight = max(0.0, min(1.0, 1.0 - vals.get("rms", 5.0) / 1.5))
    p_stab = max(0.0, min(1.0, (0.95 - ratio) / 0.30))
    secondary = (p_floor + p_m + p_straight + p_stab) / 4.0

    out = dict(primary=primary, secondary=secondary, ratio=ratio,
               margins=margins, values=vals,
               phys=dict(p_floor=round(p_floor, 3), p_m=round(p_m, 3),
                         p_straight=round(p_straight, 3),
                         p_stab=round(p_stab, 3)))
    if verbose:
        show(out)
    return out


def show(out):
    v, mg = out["values"], out["margins"]
    print(f"  primary {out['primary']:+.3f}  secondary "
          f"{out['secondary']:.3f}  gamma/gc {out['ratio']:.3f}")
    order = sorted(mg, key=lambda k: mg[k])
    for k in order:
        x = v.get(k)
        xs = ("inf" if isinstance(x, float) and math.isinf(x)
              else "nan" if x is None or (isinstance(x, float)
                                          and math.isnan(x))
              else f"{x:.3f}")
        print(f"    {k:>5} = {xs:>8}   window {WIN[k]}   "
              f"margin {mg[k]:+.3f}")
    print(f"    phys {out['phys']}  rms {v.get('rms', float('nan')):.3f}  "
          f"dts {v.get('dts')}")


def main(argv):
    t0 = time.time()
    if "--baseline" in argv:
        for name, p in (("Finding 19 (unified)", P19),
                        ("Finding 18 (encoded winner, m=1.0)", P18)):
            print("=" * 78)
            print(f"BASELINE: {name}")
            print("=" * 78)
            out = evaluate(p, N=1024, verbose=True)
            json.dump(dict(name=name, primary=out["primary"],
                           secondary=out["secondary"]),
                      open(f"best_base_{name[8:10]}.json", "w"))
    print(f"\ntotal {time.time() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
