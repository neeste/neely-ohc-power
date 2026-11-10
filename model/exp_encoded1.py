"""
exp_encoded1.py -- the ENCODED test set, linear front.

The campaign that satisfied the six observations did so under protocols
matched to what each experiment measured. This companion campaign asks the
complementary question: is the test set satisfiable AS ENCODED in
obs_metrics, and the two closure theorems name the class where the answer
could be yes: a nearly passive line (wave gain small, so obs 6's
excitation-ratio metric and AF excursion are passable) whose tip contrast
is carried by a SHARP tectorial resonance in H_o (obs 4, 5), with the
vertical-protocol suppression observables (obs 1, 2, 3) to be carried by
the IHC transducer gate in phase 2. That class is essentially the 1980
resonant-TM model with an IHC nonlinearity.

This file is the linear front: scan (R_t, R_c, K_t, gamma) from world A
and score obs 4, 5, 6 through obs_metrics's own arithmetic (obs4, obs5 on
the report dict; forward gain and AF excursion via exp_obs46's metrics,
which are what the encoded obs 6 judges).

Run: python3 exp_encoded1.py [--part 1|2|3]
"""
import sys, math, time
from dataclasses import replace
import numpy as np
sys.path.insert(0, '.')
sys.path.insert(0, '../../scoring')
import tl_model as T
import exp_obs46 as O
import obs_metrics as MS

BASE = T.WORLDS["B"]


def build_rep(coch, gamma):
    tc = O.tuning_curve(coch, O.X_PROBE, gamma)
    thr = O.threshold_of(tc)
    f_hz = O.hz(tc["w"])
    return tc, dict(
        tuning_curve=dict(place_cf_hz=float(O.hz(tc["bf"])),
                          frequency_hz=list(map(float, f_hz)),
                          threshold_db_spl=list(map(float, thr))),
        transduction_phase=dict(place_cf_hz=float(O.hz(tc["bf"])),
                                frequency_hz=list(map(float, f_hz)),
                                phase_rad=list(map(float,
                                    np.unwrap(np.angle(tc["H_o"]))))))


def score_linear(coch, gamma):
    tc, rep = build_rep(coch, gamma)
    o4 = MS.obs4(rep); o5 = MS.obs5(rep)
    tt = o4.metrics[0].value; sl = o4.metrics[1].value
    mp = o5.metrics[0].value; ph = o5.metrics[1].value
    fg = O.forward_gain(coch, gamma)
    fwd = O.fwd_primary(fg)
    af = O.af_function(coch, gamma)
    return tt, sl, mp, ph, fwd, af["excursion"]


def ok_linear(tt, sl, mp, ph, fwd, exc):
    return (tt is not None and 32 <= tt <= 48
            and sl is not None and 3 <= sl <= 9
            and mp is not None and -0.75 <= mp <= -0.25
            and ph is not None and abs(ph - math.pi) <= 0.7
            and fwd <= 13.0 and 2.0 <= exc <= 14.0)


def main(argv):
    part = int(argv[argv.index("--part") + 1]) if "--part" in argv else 1
    t0 = time.time()
    # scan grid, chunked by part
    RTS = {1: (0.02, 0.01), 2: (0.005, 0.002), 3: (0.001,)}[part]
    RCS = (0.02, 0.005, 0.002)
    GAMS = (0.0, 0.115, 0.145)
    print("=" * 78)
    print(f"ENCODED TEST SET, LINEAR FRONT (part {part}: R_t {RTS})")
    print("=" * 78)
    print("  targets: tt 32-48, tail sl 3-9, map -0.75..-0.25, phase step "
          "pi +/- 0.7,\n  fwd <= 13, AF exc 2-14; gamma small so the wave "
          "gain is not the tip")
    print(f"\n  {'R_t':>7} {'R_c':>7} {'gamma':>6} {'tt':>7} {'sl':>6} "
          f"{'map':>7} {'phase':>6} {'fwd':>6} {'AFexc':>6} {'OK':>3}")
    for rt in RTS:
        for rc in RCS:
            coch = replace(BASE, sec=replace(BASE.sec, R_t=rt, R_c=rc))
            for g in GAMS:
                try:
                    tt, sl, mp, ph, fwd, exc = score_linear(coch, g)
                    ok = ok_linear(tt, sl, mp, ph, fwd, exc)
                except Exception as e:
                    print(f"  {rt:7.3f} {rc:7.3f} {g:6.2f} failed: {e}")
                    continue
                print(f"  {rt:7.3f} {rc:7.3f} {g:6.2f} "
                      f"{(float('nan') if tt is None else tt):7.2f} "
                      f"{(float('nan') if sl is None else sl):6.2f} "
                      f"{(float('nan') if mp is None else mp):7.3f} "
                      f"{(float('nan') if ph is None else ph):6.2f} "
                      f"{fwd:6.2f} {exc:6.2f} {'YES' if ok else 'no':>3}")
    print(f"\n  total {time.time() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
