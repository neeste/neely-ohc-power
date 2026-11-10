"""geom.py: OHC force geometry switch for the scoping computation.
GEOM = 'pair' (BM-RL force pair, the paper's model) or 'bm' (the same
force, -Gamma*(X_r - X_t), applied to the BM only, reacting on ground).
install(sb=None) replaces exp_accumulation.terms module-wide; sb is an
optional BM-stiffness factor function S_b(a) (cat record)."""
import numpy as np
from dataclasses import replace
import exp_accumulation as E
import deltaT_3dof as d3

GEOM = 'pair'
LAM = None      # if set, fraction of the OHC force applied to the RL (1 = pair, 0 = BM only)
_SB = None

def terms(sec, w, a, r=None, gamma=None, mp=1.0, mc=1.0, mt=1.0,
          mrp=1.0, mrc=1.0):
    w = np.asarray(w); a = np.asarray(a, float)
    r = a if r is None else np.asarray(r, float)
    g = sec.gamma if gamma is None else np.asarray(gamma, float)
    jw = 1j * w; a2 = a * a
    sb = 1.0 if _SB is None else _SB(a)
    act = g * sec.K_a * sec.G * a2
    Rp, Rc = sec.R_p * mrp, sec.R_c * mrc
    Kp, Kc, Kt = sec.K_p * mp, sec.K_c * mc, sec.K_t * mt
    pil = jw * Rp * r + Kp * a2
    bun = jw * Rc * r + Kc * a2
    rl = (1.0 if GEOM == 'pair' else 0.0) if LAM is None else LAM
    A00 = -w**2*sec.M_b + jw*(sec.R_b + Rp)*r + (sec.K_b*sb + Kp)*a2
    A01 = -pil + act
    A02 = -act + 0.0*A00
    A10 = -pil + 0.0*A00
    A11 = -w**2*sec.M_r + jw*(sec.R_r + Rp + Rc)*r + (Kp + Kc)*a2 - rl*act
    A12 = -bun + rl*act
    A21 = -bun + 0.0*A00
    A22 = -w**2*sec.M_t + jw*(sec.R_t + Rc)*r + (Kt + Kc)*a2
    # general 3x3 with A20 = 0
    det = (A00*(A11*A22 - A12*A21) - A01*(A10*A22) + A02*(A10*A21))
    Y = (A11*A22 - A12*A21)/det
    Y_r = -(A10*A22)/det
    Y_t = (A10*A21)/det
    return Y, Y_r, sec.G*(Y_r - Y_t)/Y

def matrices(sec, sb=1.0):
    act = sec.gamma*sec.K_a*sec.G
    rl = (1.0 if GEOM == 'pair' else 0.0) if LAM is None else LAM
    M = np.diag([sec.M_b, sec.M_r, sec.M_t]).astype(float)
    C = np.array([[sec.R_b+sec.R_p, -sec.R_p, 0.0],
                  [-sec.R_p, sec.R_r+sec.R_p+sec.R_c, -sec.R_c],
                  [0.0, -sec.R_c, sec.R_t+sec.R_c]])
    K = np.array([[sec.K_b*sb+sec.K_p, -sec.K_p+act, -act],
                  [-sec.K_p, sec.K_p+sec.K_c-rl*act, -sec.K_c+rl*act],
                  [0.0, -sec.K_c, sec.K_t+sec.K_c]])
    return M, C, K

def margin(sec, sb=1.0):
    M, C, K = matrices(sec, sb)
    Mi = np.linalg.inv(M); Z = np.zeros((3, 3))
    A = np.block([[Z, np.eye(3)], [-Mi@K, -Mi@C]])
    return float(np.max(np.linalg.eigvals(A).real))

def gamma_critical(sec, sb=1.0, hi=200.0, tol=1e-6):
    if margin(replace(sec, gamma=0.0), sb) >= 0: return 0.0
    lo = 0.0
    if margin(replace(sec, gamma=hi), sb) < 0: return hi
    while hi - lo > tol:
        mid = 0.5*(lo+hi)
        if margin(replace(sec, gamma=mid), sb) < 0: lo = mid
        else: hi = mid
    return lo

def gc_of_rho(sec, rho, n=80, sb=1.0):
    rg = np.geomspace(rho.min()*0.98, rho.max()*1.02, n)
    gg = [gamma_critical(replace(sec, gamma=0.0, R_b=sec.R_b*q, R_r=sec.R_r*q,
          R_t=sec.R_t*q, R_p=sec.R_p*q, R_c=sec.R_c*q), sb) for q in rg]
    return np.interp(np.log(rho), np.log(rg), np.asarray(gg))

def power(sec, w, a, r, gam, X_b, X_r, X_t):
    """(D, p_fluid-equivalent, p_ohc) per unit length for geometry GEOM."""
    vb, vr, vt = 1j*w*X_b, 1j*w*X_r, 1j*w*X_t
    D = 0.5*r*(sec.R_b*abs(vb)**2 + sec.R_p*abs(vr-vb)**2 + sec.R_r*abs(vr)**2
               + sec.R_c*abs(vr-vt)**2 + sec.R_t*abs(vt)**2)
    f = gam*sec.K_a*sec.G*a*a*(X_r - X_t)     # force magnitude on RL (pair)
    rl = (1.0 if GEOM == 'pair' else 0.0) if LAM is None else LAM
    p = 0.5*np.real(f*np.conj(rl*vr - vb))
    return D, p

def install(sb=None):
    global _SB
    _SB = sb
    E.terms = terms
