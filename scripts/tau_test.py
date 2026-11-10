"""Step 6(a) test at the adopted human state: latency rms with the read-out tau_a lower bound
relaxed (1 ms -> 0.5, 0.25 ms) and with the read-out held at tau_a = 2, 3 ms (refit of theta, pr only)."""
import sys, json, numpy as np
sys.path.insert(0, '.')
import exp_abr2 as A2, exp_abr4 as A4, exp_abr5 as A5
TH = tuple(np.round(np.geomspace(0.02, 5.0, 15), 4)); TAU = (1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 18.0); PR = (1.0, 1.5, 2.0, 3.0)
ck = json.load(open('abr5_ckpt.json')); p = dict(ck['params']); S0 = ck['S0']
H = A2.Human(**p); gc = A5.gc_exact_dense(H); H.gam0 = A5.capped_gamma(H, p['mu0'], p['alpha_mu'], gc); H.stable = True
out = {}
for lo in (1.0, 0.5, 0.25):
    A4.TAU_BOUNDS = (lo, 20.0) if hasattr(A4, 'TAU_BOUNDS') else None
    A5.TAU_BOUNDS = (lo, 20.0)
    rms, info = A4.joint_rms(H, S0, TH, (0.25, 0.5) + TAU if lo < 1 else TAU, PR, reject_onset=True, refine=True)
    out['tau_lo_%g' % lo] = dict(rms=rms, **{k: info[k] for k in ('theta', 'tau_a', 'pr', 'a')})
    print('tau_a lower bound %.2f ms: rms %.3f' % (lo, rms), {k: round(info[k], 3) for k in ('theta', 'tau_a', 'pr', 'a')}, flush=True)
for t in (2.0, 3.0, 5.0):
    A5.TAU_BOUNDS = (t, t + 1e-9)
    if hasattr(A4, 'TAU_BOUNDS'): A4.TAU_BOUNDS = (t, t + 1e-9)
    rms, info = A4.joint_rms(H, S0, TH, (t,), PR, reject_onset=True, refine=True)
    out['tau_fixed_%g' % t] = dict(rms=rms, **{k: info[k] for k in ('theta', 'tau_a', 'pr', 'a')})
    print('tau_a held at %.1f ms: rms %.3f' % (t, rms), {k: round(info[k], 3) for k in ('theta', 'tau_a', 'pr', 'a')}, flush=True)
json.dump(out, open('out/tau_test.json', 'w'), indent=1)
