"""Run 4 item 3, independent check of the winding test: direct count of growing modes of the discretized quiet
human line (same equations the winding test uses: (P[j+1] - 2P[j] + P[j-1])/h^2 = F s^2 X_b[j], j = 1..N-1,
P[0] = P[N] = 0, and at each place the 3-dof section (s^2 M + s r_j C + a_j^2 K(gamma_j)) X = (P_j, 0, 0), with
M, C, K from geom.matrices (force split lambda installed by run_human.py); s = i omega in the model's units).
Quadratic eigenproblem linearized and solved densely (scipy.linalg.eig); growing = finite eigenvalue with Re s > 1e-9.
Also counts, for each profile, the places whose isolated section is unstable (geom.margin > 0).
Usage (from model/): python3 ../scripts/run_human.py ../scripts/eig_h_ctrl.py joint_hs_hs856.json out/eig_h_ctrl.json N [N ...]"""
import sys, os, json, time
sys.path.insert(1, __import__('os').path.dirname(__import__('os').path.abspath(__file__))); import cat_state
import numpy as np, scipy.linalg as sl
from dataclasses import replace
sys.path.insert(0, '.')
import exp_abr2 as A2, exp_abr5 as A5, geom
c = json.load(open(sys.argv[1])); pd = dict(c['params']); sec_over = pd.pop('sec_over'); nfl = c['nu_floor']
H = A2.Human(**dict(pd, sec_over=sec_over)); gc = A5.gc_exact_dense(H); tap = H.coch.taper(H.x); sec = H.coch.sec; F = H.coch.F
nu = np.clip(pd['mu0']*H.a**pd['alpha_mu'], nfl, 1.0); mu_nc = 1 - nu
t = np.clip((H.x-A5.X0)/(A5.X1-A5.X0), 0, 1); w = 1-(3*t*t-2*t**3); mu_c = (1-w)*mu_nc + w*np.minimum(mu_nc, A5.CAP)
prof = {'hs708': mu_c*gc*tap, 'no basal cap': mu_nc*gc*tap, 'uniform 0.99': 0.99*gc*tap, 'uniform 1.01': 1.01*gc*tap,
        'passive': 0*gc}
OUT = sys.argv[2]; res = json.load(open(OUT)) if os.path.exists(OUT) else dict(source=sys.argv[1], lam=geom.LAM, rows={})
for N in map(int, sys.argv[3:]):
    x = np.linspace(0, 1, N+1); a = np.interp(x, H.x, H.a); r = np.interp(x, H.x, H.r); h = 1.0/N
    J = np.arange(1, N); n = 4*(N-1)
    for name, gam in prof.items():
        key = '%s | N=%d' % (name, N)
        if key in res['rows']: continue
        t0 = time.time(); g = np.interp(x, H.x, gam)
        Mg = np.zeros((n, n)); Cg = np.zeros((n, n)); Kg = np.zeros((n, n)); nloc = 0
        for i, j in enumerate(J):
            M, C, K = geom.matrices(replace(sec, gamma=float(g[j])), 1.0)
            q = r[j]/a[j]          # isolated section at this place: s = a sigma gives damping x r/a (as gamma_c(rho))
            if geom.margin(replace(sec, gamma=float(g[j]), R_b=sec.R_b*q, R_r=sec.R_r*q, R_t=sec.R_t*q, R_p=sec.R_p*q, R_c=sec.R_c*q), 1.0) > 0: nloc += 1
            b = 3*i; Mg[b:b+3, b:b+3] = M; Cg[b:b+3, b:b+3] = r[j]*C; Kg[b:b+3, b:b+3] = a[j]**2*K
            ip = 3*(N-1) + i
            Kg[b, ip] = -1.0                                   # section row 0: ... = P_j
            Kg[ip, ip] = -2.0                                  # fluid row (times h^2)
            if i > 0: Kg[ip, ip-1] = 1.0
            if i < N-2: Kg[ip, ip+1] = 1.0
            Mg[ip, b] = -F*h*h
        I = np.eye(n); Z = np.zeros((n, n))
        A = np.block([[Z, I], [-Kg, -Cg]]); B = np.block([[I, Z], [Z, Mg]])
        ev = sl.eig(A, B, right=False)
        ev = ev[np.isfinite(ev)]
        gr = ev[ev.real > 1e-9]
        row = dict(N=N, n_finite=int(ev.size), growing=int(gr.size), growing_conj_pairs=int(np.sum(gr.imag > 0)),
                   max_growth=float(ev.real.max()), unstable_sections=nloc,
                   growing_freqs=sorted(float(abs(v.imag)) for v in gr if v.imag >= 0)[:60], secs=time.time()-t0)
        res['rows'][key] = row; json.dump(res, open(OUT, 'w'), indent=1)
        print('%-14s N=%4d growing eigenvalues %4d (%d pairs), max Re s %.3e, unstable isolated sections %d (%.0f s)' %
              (name, N, row['growing'], row['growing_conj_pairs'], row['max_growth'], nloc, row['secs']), flush=True)
print('EIG DONE', flush=True)
