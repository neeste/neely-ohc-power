# Parameter files

The parameter files are kept beside the code in `../model` so that the scripts find them when run from
that directory.

## Adopted states

Cat, state c856 (force split lambda = 0.8562, gamma/gamma_c = 0.950 at rho = 1):

- `best_d3.json`: section parameters (K_t, M_t, R_t, R_c, R_b, K_c, K_p, M_r), stage-1 uniform gamma,
  saturation and neural-gate parameters (q, m, u_thr, s_g), and lambda. Built from `combo_r7b.json`.
- `base_ckpt.json`: the place-profile and read-out vector z (12 entries: alpha_a, alpha_b, c_R,
  log transition frequency, transition width, log nu0, log f_hp, log f_lp, n_lp, log B, C0, alpha_nu)
  with the shear-velocity read-out (C0 79.47 dB, f_hp 0.050 kHz, f_lp 40.0 kHz, n_lp 0.30). Default.
- `base_ckpt_disp.json`: the same z with the displacement read-out of the fit (C0 73.19 dB, f_hp 0.0825 kHz,
  f_lp 28.64 kHz, n_lp 1.771). Selected by `FTC_CRIT=disp`.
- `combo_r7b.json`: the record of the combined fit (section p, lambda, z with the displacement read-out,
  margins, values). `combo_c856v.json`: the same record with z[6], z[7], z[8], z[10] replaced by the
  velocity read-out. `vel_fit_c856.json`: the read-out refit that produced it (`scripts/vel_fit.py`).

Human, state hs856 (cat section and lambda shared):

- `abr5_ckpt.json` (identical to `joint_hs_hs856.json`): human place map, damping and activity
  profiles, F, saturation exponent p_sat, S0, nu floor, the wave-V read-out (theta, tau_a, p_r, t_n),
  the fitted Q_ERB and latencies, and the section override `sec_over` (the c856 section).
- `init/init_hs856a.json`, `init/init_hs856b.json`: the two starts of the human refit (start B adopted).

## Earlier states (kept for reference; not used by the default scripts)

- `*_lam1.json`: the submitted states with lambda = 1 (cat `best_d3_lam1.json`, `base_ckpt_lam1.json`;
  human `abr5_ckpt_lam1.json`).
- `earlier/combo_c708.json`: the lambda = 0.7085 cat state (Supplement Table SVIII and the rescoring of the
  profiled-line excursion). `earlier/combo_c85.json`: the start of the c856 refit.
  `earlier/combo_r7a.json`: the refit from c708 (Supplement).
- `earlier/combo_b0.json`, `earlier/combo_b1.json`: the fixed-lambda fits (lambda = 0 and 0.10, Table VII),
  with their velocity read-outs `earlier/vel_fit_r4_b0.json`, `earlier/vel_fit_r4_b1.json`
  (`earlier/vel_fit_r7a.json` for r7a).
- `earlier/joint_hs_s708.json`: the human state hs708 (Supplement Table SXI). `earlier/joint_h_wl7.json`,
  `earlier/human_n8192_ckpt.json`: inputs of the earlier human fits (`joint_h.py`).
- `earlier/ftc2_lam_ckpt.json`: the stage-2 incumbent that starts `combo.py`.
