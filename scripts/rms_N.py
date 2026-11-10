"""Wave-V latency rms versus grid N at a joint_h state, with the joint_h read-out search
(widened grids, continuous refinement, onset rejection), as used in the fit."""
import sys, json, time, numpy as np
sys.path.insert(0, '.')
import exp_abr2 as A2, exp_abr4 as A4, exp_abr5 as A5
TH = tuple(np.round(np.geomspace(0.02, 5.0, 15), 4)); TAU = (1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 18.0); PR = (1.0, 1.5, 2.0, 3.0)
ck = json.load(open('abr5_ckpt.json')); p = dict(ck['params']); S0 = ck['S0']
for N in [int(v) for v in sys.argv[1:]]:
    A2.N = N; t0 = time.time()
    H = A2.Human(**p); gc = A5.gc_exact_dense(H); H.gam0 = A5.capped_gamma(H, p['mu0'], p['alpha_mu'], gc); H.stable = True
    rms, info = A4.joint_rms(H, S0, TH, TAU, PR, reject_onset=True, refine=True)
    print(N, 'rms %.3f' % rms, {k: info[k] for k in ('theta', 'tau_a', 'pr', 'a')}, '%.0fs' % (time.time()-t0), flush=True)
