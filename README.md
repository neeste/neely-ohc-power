# Model code for "Where outer hair cells deliver net power in a cochlear model of cat and human"

Stephen T. Neely, Boys Town National Research Hospital.

This repository holds the model code, the fitted parameter files, and the scripts that produce the
numbers, tables and figures of the article (submitted to J. Acoust. Soc. Am.) and of its
supplementary material. The model is a one-dimensional long-wave transmission line whose sections have
three degrees of freedom (basilar membrane, reticular lamina, tectorial membrane) and an outer-hair-cell
force that saturates with hair-bundle shear and is split between the reticular lamina and the basilar
membrane by the force split lambda. The equations are in Sec. II of the article and in the supplement.

## Layout

| Folder | Contents |
|---|---|
| `model/` | Model and read-out code (Python 3, NumPy, SciPy) and the parameter files it loads; `model/earlier/` earlier states kept for reference; `model/init/` starts of the human refit |
| `params/` | Description of the parameter files |
| `scripts/` | One script per computed result; each writes JSON to `model/out/` |
| `figures/` | Generators that convert the JSON to PLT plot files |
| `data/` | `tc_bins.json`: binned medians of the cat tuning-curve data used in fitting |
| `supplement/` | Placeholder; the supplement PDF is added at submission |

## Installation

Python 3.10 or later (tested with 3.13) and the two packages in `requirements.txt`:

```
python3 -m venv venv
venv/bin/pip install -r requirements.txt
```

## States

The adopted states are in `model/` (details in `params/README.md`):

- Cat c856: `best_d3.json` and `base_ckpt.json` (also `combo_r7b.json` and `combo_c856v.json`, the
  records of the combined fit), lambda = 0.8562.
- Human hs856: `abr5_ckpt.json` (also `joint_hs_hs856.json`), which shares the cat section and lambda.

Files with the suffix `_lam1` hold the submitted states with lambda = 1, and `model/earlier/` holds the
other earlier states that some supplementary tables use. None of these is read by default.

**Cat population read-out.** The cat tuning-curve thresholds use a constant hair-bundle shear velocity
criterion by default, with the read-out constants fitted for it (C0 79.47 dB, f_hp 0.050 kHz,
f_lp 40.0 kHz, n_lp 0.30; `base_ckpt.json`, `combo_c856v.json`). The displacement criterion of the fit
and its read-out constants (`base_ckpt_disp.json`, `combo_r7b.json`) are selected by the environment
variable `FTC_CRIT=disp`. Only the tuning-curve read-out depends on the switch (Table IV, Fig. 3, the
tuning-curve objective J_ftc and the BM velocity at threshold); the mechanics and all other metrics are
identical.

## Running

All scripts run from `model/`. Scripts that need the fitted state run through the wrappers
`scripts/run_cat.py` (cat: installs lambda, the fitted 24-point gamma_c table and the lambda energy
accounting) and `scripts/run_human.py` (human: lambda, N = 8192 and the state's nu floor). Scripts
that build the state from a fit record (`c6_state.py`, `obs13.py si`, `shoulder.py`, `vel_fit.py`,
`combo7.py`, the Fig. 9 section scripts) run with plain `python3`.

```
cd model
mkdir -p out
python3 ../scripts/run_cat.py ../scripts/battery.py
python3 ../scripts/run_cat.py ../scripts/profiles.py
python3 ../scripts/run_human.py ../scripts/profiles_h.py
python3 ../scripts/run_cat.py ../scripts/<script>.py [arguments]
python3 ../scripts/run_human.py ../scripts/<script>.py [arguments]
```

`battery.py`, `profiles.py` and `profiles_h.py` are run first: later scripts read the calibration of
the suppression read-out from `base_battery.json` and their place maps from `out/profiles.json`.

`./run_all.sh` runs the whole default sequence in two lanes (cat and human) and writes the outputs to
`model/out/`, logs and wall times to `model/out/logs/`. It takes about 30 minutes on two cores
(timings below are for one core per lane, NumPy 2.5.3, SciPy 1.18.1). `./run_all.sh --long` adds the
long jobs listed at the end of this file. `PYTHON=venv/bin/python ./run_all.sh` selects the interpreter.

## Reproducing the figures and tables

Script names are in `scripts/` unless a folder is given; C = through `run_cat.py`, H = through
`run_human.py`, P = plain `python3`; all from `model/`. Times are wall times of `run_all.sh`.

### Article

| Item | Scripts (from model/; C run_cat.py, H run_human.py, P plain python3) | Time | Note |
|---|---|---|---|
| Fig. 1 (model structure) | figures/mkfig1.py (P, from figures/) |  | schematic |
| Fig. 2 (place profiles, cat and human) | C profiles.py; H profiles_h.py; figures/make_figs_cat.py, figures/make_rev_figs.py | 1 s |  |
| Table I (six observations, both cat lines) | P c6_state.py 0.9501 (18 metrics, BM-gain metric, worst margin); P obs13.py si out/obs13_si.json and C obs13.py prof out/obs13_prof.json (observations 1 to 3, C_supp); C battery.py (calibration, distributed-source excursion) | 107 s |  |
| Table II (cat parameters) | C table2.py out/table2.json; files best_d3.json, base_ckpt.json; fit: P combo7.py earlier/combo_c85.json r7b (FTC_CRIT=disp, as fitted), read-out P vel_fit.py out/vel_fit.json | 2 s | fits are long jobs |
| Table III (human parameters) | files abr5_ckpt.json = joint_hs_hs856.json; H map_h.py out/map_h.json, H place_map_h.py out/place_map_h.json; fit: P joint_hs_cnt.py 1 1500 hs856b with INIT=init/init_hs856b.json (stop rule jhs_watch.py) | 4 s | fit is a long job |
| Fig. 3 (cat tuning curves) and Sec. III.A per-curve numbers | C fig2_rev.py; P tc_numbers.py out/tc_numbers.json; figures/make_fig03.py ../model/out/fig2_curves.json ../model/out/table2.json plt |  | needs the individual curves (model/tc.plt), not redistributed |
| Table IV (cat tuning-curve metrics, binned) | C bmdef_shoulder_s2.py base_ckpt.json out/bmdef_shoulder.json; P ftcbins_model.py out/bmdef_shoulder.json ../data/tc_bins.json out/ftcbins.json (displacement: FTC_CRIT=disp, _disp files) | 1 s | binned medians suffice |
| J_ftc (tuning-curve objective) | P c6_state.py 0.9501 (velocity); FTC_CRIT=disp C6OUT=out/c6_state_disp.json P c6_state.py 0.9501 (displacement) | 48 s |  |
| Fig. 4 (human wave-V latency) | H rms_N.py 8192 (and 16384); P latency_detail.py joint_hs_hs856.json out/latency_detail.json; figures/make_rev_figs.py | 8 s |  |
| Fig. 5 (human tuning curves), Q_ERB | H ftc_h.py; H shape_h.py joint_hs_hs856.json out/shape_human.json; H qerb_h.py abr5_ckpt.json out/qerb_h.json | 4 s |  |
| Fig. 6 (cat transfer functions, 2.2-kHz place) | C tf.py cat; figures/make_figs_cat.py | 6 s |  |
| Fig. 7 (human transfer functions, 1-kHz place) | H tf.py human; figures/make_rev_figs.py | 40 s |  |
| Table V (low-level transfer functions at CF) | C tf.py cat and P extract_cat.py out out/extract_cat.json (cat rows); H tf.py human and P tfx.py out/tf_human.json out/tfx_human.json (human rows) | 46 s |  |
| Fig. 8 (energy accounting) | C dump_fig3_rev.py cat; H dump_fig3_rev.py human; C share_scan.py; C share_s2.py base_ckpt.json out/share_s2.json; figures/make_figs_cat.py, figures/make_rev_figs.py | 2 s |  |
| Fig. 9 (the OHC share as a property of the section) | P fig09_data.py 0.950; P share_local.py 0.950; P crit_mode.py 0.950; P line_peak.py 0.950; figures/make_figs_cat.py | 1 s |  |
| Sec. III.C (local inactivation) | C removal.py cat out/removal_cat.json; H removal.py hum out/removal_hum.json (force kept at the read-out place); C fisher_analog.py | 2 s |  |
| Sec. III.D (power gain G_P, power flux) | C power_gain.py cat out/power_cat.json; H power_gain.py hum out/power_hum.json; C flux_profile.py out/flux_profile.json | 4 s |  |
| Table VI (share maximum against lambda) | C c_robust.py cat; H c_robust.py human joint_hs_hs856.json; C c_scan_cert.py; H c_scan_cert_h.py joint_hs_hs856.json |  | long jobs |
| Table VII (fixed-lambda cat fits) | P combo_bm.py 0 b0, P combo_bm.py 0.1 b1 (records earlier/combo_b0.json, combo_b1.json); rescoring C6CK=earlier/combo_b0.json P c6_state.py 0.9501; own read-outs C6CK=earlier/combo_b0.json P vel_fit.py out/vel_fit_b0.json (records earlier/vel_fit_r4_b0.json, _b1); P c6_state.py rescoring in run_all.sh step rescore | 105 s | combo_bm.py fits are long jobs |
| Fig. 10 (human, distance from criticality and level) | H fig6_rev.py; figures/make_rev_figs.py | 14 min |  |
| Fig. 11 (cat SFOAE delay against activity) | C b_crit.py; figures/make_figs_cat.py |  | long job |
| Fig. 12 (SFOAE delays), tuning ratio | C sfoae_cat.py; C tuning_ratio.py; C sfoae_cat_lv.py; H sfoae_h.py; figures/make_figs_cat.py, figures/make_rev_figs.py | 6 s |  |
| Table VIII (long-wave test) | C d_kh.py cat; H d_kh.py human joint_hs_hs856.json; P cd_tables.py; C wavenumber.py | 0 s | d_kh is a long job |
| Line stability margin (Sec. II.E, III.D) | C line_margin.py cat_prof out/lm_cat_prof.json; C line_margin.py cat_si out/lm_cat_si.json; H line_margin.py hum out/lm_hum.json | 6 min |  |
| Other Sec. III numbers | C passive_q10.py out/passive_q10.json (passive Q10); P shoulder.py ref out/shoulder_ref.json (shoulder index); P si_readplace.py combo_c856v.json; C shoulder_tm.py base_ckpt.json out/shoulder_tm.json | 6 s |  |

### Supplementary material

| Item | Scripts (from model/; C run_cat.py, H run_human.py, P plain python3) | Time | Note |
|---|---|---|---|
| Figs. S1 and S2, Table SI (TM transfer functions) | C tm_tf.py cat; H tm_tf.py human; P tm_post.py; figures/make_tm_figs.py | 13 s |  |
| Table SII (model-to-physical scales) | C units.py out/bmdef_shoulder.json; C table2.py out/table2.json | 1 s |  |
| Table SIII (cat BM motion at neural threshold) | C bmdef_shoulder_s2.py base_ckpt.json out/bmdef_shoulder.json; C units.py | 1 s |  |
| Table SIV (cat, activity scaled) | C b_crit.py |  | long job |
| Tables SV and SVI (one-factor variations) | C c_robust.py cat; H c_robust.py human joint_hs_hs856.json; P cd_tables.py |  | long jobs |
| Table SVII (share and p_ohc maxima against lambda) | C c_robust.py cat (lambda scan); C c_scan_cert.py; H c_scan_cert_h.py joint_hs_hs856.json |  | long jobs |
| Table SVIII (share at a fixed distance from the line limit) | P mkstate.py earlier/combo_c708.json ../state_c708; then in ../state_c708: C lam_fixed.py cat out/lam_fixed_cat.json |  | long job; made at the lambda = 0.7085 section |
| Tables SIX and SX (long-wave test, iteration) | C d_kh.py cat; H d_kh.py human joint_hs_hs856.json; P cd_tables.py |  | long jobs |
| Table SXI (human fits) | P joint_hs_cnt.py 1 1500 hs856a / hs856b (INIT=init/init_hs856a.json, init_hs856b.json; jhs_watch.py); earlier hs708: SEC_JSON=earlier/combo_c708.json P joint_hs.py; P latency_detail.py; H qerb_h.py | 1 s | fits are long jobs |
| Tables SXII and SXIII (18 metrics along the activity scan, zero crossings) | C6OUT=out/c6_scan.json P c6_state.py 0.9501 0.93 0.90 0.88 0.85; P c6_table.py | 118 s |  |
| Table SXIV (grid convergence) | C conv_c.py; H conv_h3.py 8192 16384; H rms_N.py 8192 16384 | 24 s |  |
| Table SXV (stability tests) | C cert_s2.py base_ckpt.json out/cert_s2.json (40); P cert_h_lam.py joint_hs_hs856.json out/cert_h.json (18); controls C cert2_ctrl.py, H cert_h_ctrl.py, H eig_h_ctrl.py | 16 min | controls are long jobs |
| Table SXVI (fixed-lambda fits, every metric) | C6CK=earlier/combo_b0.json (combo_b1.json, combo_c856v.json) P c6_state.py 0.9501 | 100 s |  |
| Table SXVII (ablations) | P abl.py VARIANT for ref, Kt4_g, Kt4_r, Kt025_g, Kt025_r, clamp_g, clamp_r, pass, g098, g102 |  | long jobs |
| Table SXVIII (box bounds) | bounds in combo7.py, combo.py, combo_bm.py, model/exp_ftc2.py |  | no computation |
| Supplement, profiled-line excursion correction | C6CK=earlier/combo_c708.json (earlier/combo_c85.json) P c6_state.py 0.9501; P check_fix.py; P ps_bc.py | 100 s |  |
| Table SXIX (scripts) | this table |  |  |

## Data

The cat auditory-nerve tuning curves were provided by M. C. Liberman and B. Delgutte and are not
redistributed. `data/tc_bins.json` holds only their binned medians (tip frequency, tip threshold, Q10,
tip-to-tail difference of the eight CF bins with at least three fibers), which suffice to compute every
number in Table IV and the tuning-curve objective J_ftc (the code uses them when the individual curves
are absent). Fig. 3, which plots the individual curves, and the per-curve numbers of its caption and of
Sec. III.A (`fig2_rev.py`, `tc_numbers.py`, `figures/make_fig03.py`) need the original curves in
`model/tc.plt`; `run_all.sh` runs those steps only when that file is present.

The human wave-V latency target is the published formula of Neely et al. (1988), evaluated in code
(`model/exp_abr4.py`, `scripts/latency_detail.py`):

    tau_V(f, L) = 5 + 12.9 * 5^(-L/100) * f^(-0.413)  ms,  f in kHz, L in dB SPL.

## Figures

`figures/` holds the generators of the PLT files of the figures: `mkfig1.py` (Fig. 1),
`make_figs_cat.py` (cat panels of Figs. 2, 6, 8, 9, 11, 12a; Fig. 3 through `make_fig03.py`),
`make_rev_figs.py` (human panels of Figs. 2, 4, 5, 7, 8b, 10, 12b; it also holds the shared plotting
helpers), `make_tm_figs.py` (Figs. S1 and S2) and `overlap.py` (a check that no lettering overlaps).
Run them from `figures/` after the scripts; they read `../model/out/`. The two-species panels
(Figs. 2a, 2b, 8b) combine the cat series of `make_figs_cat.py` with the human series of
`make_rev_figs.py`. The PLT files are rendered to EPS and SVG with the author's plotting program tplt,
which is not part of this repository; the JSON in `model/out/` holds every plotted value.

## Long jobs (documented, not run by default)

| Step | Command (from model/) | Result | Time |
|---|---|---|---|
| b_crit | C b_crit.py | Fig. 11, Table SIV | 34 min |
| c_robust_cat | C c_robust.py cat | Tables VI, SV, SVII | 47 min |
| c_scan_cert | C c_scan_cert.py | Tables VI, SVII | 15 min |
| d_kh_cat | C d_kh.py cat | Tables VIII, SIX, SX | 23 s |
| abl_ref | P abl.py VARIANT (10 variants; ref first) | Table SXVII | 3 min |
| lam_conv | C lam_conv.py out/lam_conv.json | Sec. III.C, grid check of the lambda scan | 40 min |
| lam_fixed | P mkstate.py earlier/combo_c708.json ../state_c708; then in ../state_c708: C lam_fixed.py cat out/lam_fixed_cat.json | Table SVIII (made at the lambda = 0.7085 section) | 43 min |
| cert_h_ctrl | H cert_h_ctrl.py joint_hs_hs856.json out/cert_h_ctrl.json | Table SXV (positive control) | 13 min |
| eig_h_ctrl | H eig_h_ctrl.py joint_hs_hs856.json out/eig_h_ctrl.json 128 256 512 | Table SXV (eigenvalue counts) | 34 min |
| c_scan_cert_h | H c_scan_cert_h.py joint_hs_hs856.json | Tables VI, SVII | 26 min |
| c_robust_hum | H c_robust.py human joint_hs_hs856.json | Tables VI, SVI | 59 min |
| d_kh_hum | H d_kh.py human joint_hs_hs856.json | Tables VIII, SIX, SX | 13 s |
| tau_test | H tau_test.py | Sec. III.A (read-out tau_a bound) | 40 s |
| nu_scan | H nu_scan.py | Sec. III.A (activity-profile sensitivity) | 72 s |
| cd_tables | P cd_tables.py | tables from the c_robust and d_kh outputs | 0 s |

Fits (not run by `run_all.sh`; times from the original runs on one core):

| Command (from model/) | Result | Time |
|---|---|---|
| P combo7.py earlier/combo_c85.json r7b, with FTC_CRIT=disp | cat combined fit (the c856 state, 1500 evaluations); the fit used the displacement read-out | about 75 min |
| P combo7.py earlier/combo_c708.json r7a, with FTC_CRIT=disp | cat refit from c708 (Supplement) | about 75 min |
| P combo_bm.py 0 b0 ; P combo_bm.py 0.1 b1 | fixed-lambda cat fits (Table VII, SXVI) | hours |
| INIT=init/init_hs856b.json P joint_hs_cnt.py 1 1500 hs856b (and hs856a); P jhs_watch.py TAG PID stops at a stall of 150 evaluations | human refit at the c856 section (Table III, SXI) | about 1 h per start |
| P combo.py, P joint_hs.py, P joint_h.py, P lam_refit.py, P run_s2.py, P refit_*.py, P bmgain_refit.py | earlier fit stages (Sec. II.I; Tables II and III of the submitted version) | hours |

## License

The source code (`*.py`, `*.sh`) is licensed under the BSD 3-Clause License (`LICENSE`). The parameter
files, data, documentation and figure files are licensed under the Creative Commons Attribution 4.0
International license (CC BY 4.0; `LICENSE-CC-BY-4.0.txt`). `data/tc_bins.json` holds binned medians of
cat tuning curves provided by M. C. Liberman and B. Delgutte; please credit them when using these values.
An archived copy with a DOI is deposited on Zenodo. Please cite the article (`CITATION.cff`).

## Acknowledgment

Claude (Anthropic) assisted with coding, writing, and preparation of figures.
