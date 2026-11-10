"""run2 queue D: limited kH coth kH fluid correction at fixed parameters (WKB-local).
Long-wave line P'' = -w^2 F Y P is re-solved with F_eff(x) = F * kH/tanh(kH) (equivalently H_eff = H tanh(kH)/(kH)),
|k| from the complex wavenumber of the previous iterate's pressure, k = -dphi/dx + i dln|P|/dx (per cochlear length),
kH = |k| * (H/L). Iterate until the BM and shear magnitudes (within 30 dB of their peaks) change by < 0.1 dB;
cap 10 iterations. Plain fixed-point iteration (under-relaxation was tested and converges more slowly).
This is not a 2-D solution: no near-field pressure focusing, no WKB amplitude factor change, stability not certified."""
import numpy as np
from scipy.linalg import solve_banded
import exp_accumulation as E
def fd_var(F, w, Y, N, amp=1.0):
    q = -(w**2)*F*Y; h = 1.0/N; hq = h*h*q; n = N - 1
    ab = np.zeros((3, n), dtype=complex); ab[0, 1:] = 1.0; ab[1, :] = -(2.0 + hq[1:N]); ab[2, :-1] = 1.0
    rhs = np.zeros(n, dtype=complex); rhs[0] = -amp; P = np.zeros(N + 1, dtype=complex); P[0] = amp
    P[1:N] = solve_banded((1, 1), ab, rhs); return P
def kfac(P, x, HL, cap=400.0):
    ph = np.unwrap(np.angle(P)); lm = np.log(np.abs(P) + 1e-300)
    k = np.hypot(np.gradient(ph, x), np.gradient(lm, x)); kh = np.clip(k*HL, 1e-6, cap)
    return kh/np.tanh(kh), k
def db_change(A, B):
    out = 0.0
    for a, b in zip(A, B):
        ma, mb = np.abs(a), np.abs(b); m = ma >= ma.max()*10**(-30/20)
        out = max(out, float(np.max(np.abs(20*np.log10(mb[m]/ma[m])))))
    return out
class KH:
    """Wrap a run2 State S (r2cat or r2hum) so that S.fields uses the corrected fluid; stats in self.stats."""
    def __init__(self, S, HL, tol=0.1, itmax=10):
        self.S, self.HL, self.tol, self.itmax = S, HL, tol, itmax; self.orig = S.fields
        self.stats = dict(calls=0, plain=0, failed=0, iters=[]); self.last = None
    def solve(self, f, gam=None):
        S = self.S; S._set(); g = S.gam if gam is None else gam
        Y, Yr, Ho = E.terms(S.sec, f, S.a, S.r, g)
        F0 = S.coch.F
        P0 = fd_var(F0*np.ones_like(S.x), f, Y, S.N)
        P = P0; prev = (Y*P, Ho*Y*P); ok = False; d = float('nan')
        for it in range(1, self.itmax + 1):
            fac, k = kfac(P, S.x, self.HL)
            P = fd_var(F0*fac, f, Y, S.N); cur = (Y*P, Ho*Y*P); d = db_change(prev, cur); prev = cur
            if d < self.tol: ok = True; break
        self.stats['calls'] += 1; self.stats['iters'].append(it); self.stats['plain' if ok else 'failed'] += 1
        self.last = dict(converged=ok, iters=it, last_change_db=d, fac=fac, k=k)
        return Y, Yr, Ho, P
    def fields(self, f, gam=None):
        Y, Yr, Ho, P = self.solve(f, gam); sec = self.S.sec
        Xb = Y*P; Xr = Yr*P; Xt = Xr - Ho*Xb/sec.G
        return dict(Y=Y, P=P, Xb=Xb, Xr=Xr, Xt=Xt, sh=Ho*Xb, Ho=Ho)
    def install(self): self.S.fields = self.fields
    def remove(self): self.S.fields = self.orig
