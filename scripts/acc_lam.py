"""exp_place6.accounting with the force split lambda (geom.power); identical to the original at lambda = 1."""
import numpy as np
import geom
def accounting(sec, a, r, gam, w, Xb, Xr, Ho, Zp):
    Xt = Xr - Ho*Xb/sec.G
    D, p = geom.power(sec, w, a, r, gam, Xb, Xr, Xt)
    pf = 0.5*np.real(Zp)*np.abs(1j*w*Xb)**2
    return D, pf, p
def install():
    import exp_place6
    exp_place6.accounting = accounting
