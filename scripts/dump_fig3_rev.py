"""Fig. 3 data at the revision states: cat record (score_base) and human
abr5. Same format as dump_fig3.py. x = -log2(a/a_peak): positive = apical."""
import sys, json, numpy as np
sys.path.insert(0, '.')
import os
import exp_accumulation as E, tl_model as T
from exp_place6 import accounting
PART = sys.argv[1] if len(sys.argv) > 1 else "both"
out = json.load(open("out/fig3_curves.json")) if os.path.exists("out/fig3_curves.json") else {}
N = 2048
if PART in ('cat', 'both'):
  import score_base as SB
  x = np.linspace(0, 1, N+1); a = SB.COCH.cf(x); r, g0 = SB.profiles(N); gam = g0*SB.COCH.taper(x)
  def cat_curves(fk):
      w = fk/SB.BASE
      Y, Yr, Ho = E.terms(SB.COCH.sec, w, a, r, gam)
      P = T._fd(SB.COCH, w, Y, N, 1.0); Xb = Y*P; Xr = Yr*P
      Zp = 1/(1j*w*Y)
      D, pf, po = accounting(SB.COCH.sec, a, r, gam, w, Xb, Xr, Ho, Zp)
      ipk = int(np.argmax(np.abs(Xb[int(0.02*N):]))) + int(0.02*N)
      # BM displacement is weakly tuned; take the peak of the shear instead
      ish = int(np.argmax(np.abs(Ho*Xb)))
      oc = np.log2(a/a[ish])
      return oc, D, po, pf, ish, ipk
  oc, D, po, pf, ish, ipk = cat_curves(1.0)
  m = (oc > -0.8) & (oc < 0.8)
  out['left'] = dict(x=(-oc[m]).tolist(), D=D[m].tolist(), po=po[m].tolist(), peak='shear')
  out['cat_share'] = {}
  for fk in (1.0, 4.0):
      oc, D, po, pf, ish, ipk = cat_curves(fk)
      k = (oc > -0.8) & (oc < 0.8); sh = (po/D)[k]
      out['cat_share']['%.1f' % fk] = dict(x=(-oc[k]).tolist(), share=sh.tolist(), peak=float(sh.max()),
                                            x_shear_peak=float(x[ish]), x_bm_peak=float(x[ipk]))
      print('cat %.0f kHz peak share %.4f  shear peak x %.3f  BM peak x %.3f' % (fk, sh.max(), x[ish], x[ipk]), flush=True)
if PART in ('human', 'both'):
  import exp_abr2 as A2, exp_abr5 as A5
  ck = json.load(open('abr5_ckpt.json')); pd = dict(ck['params'])
  H = A2.Human(**pd); gc = A5.gc_exact_dense(H)
  H.gam0 = A5.capped_gamma(H, pd['mu0'], pd['alpha_mu'], gc)
  out['human_share'] = {}
  for fk in (1.0, 4.0):
      w = fk*1000.0/A2.F_BASE_HZ
      rr = H.line2(w, np.zeros(len(H.x)))
      Zp2 = 1.0/(1j*w*rr['Y'])
      ish = int(np.argmax(np.abs(rr['shear'])))
      oc2 = np.log2(H.a/H.a[ish])
      D2, pf2, po2 = accounting(H.coch.sec, H.a, H.r, H.gam0, w, rr['X_b'], rr['X_r'], rr['H_o'], Zp2)
      k = (oc2 > -0.8) & (oc2 < 0.8); sh = (po2/D2)[k]
      out['human_share']['%.1f' % fk] = dict(x=(-oc2[k]).tolist(), share=sh.tolist(), peak=float(sh.max()))
      print('human %.0f kHz peak share %.4f' % (fk, sh.max()), flush=True)
json.dump(out, open('out/fig3_curves.json', 'w'))
