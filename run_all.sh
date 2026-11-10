#!/bin/bash
# run_all.sh : reproduce the numbers of the paper that finish in reasonable time (about 1 to 2 hours on
# two cores). Runs from any directory; works in model/ and writes JSON to model/out/ and logs to
# model/out/logs/. Two lanes run in parallel (cat, human). Each step is skipped when its done-marker
# exists, so the script can be restarted. ./run_all.sh --long adds the long jobs listed in README.md
# (fits, scans and certifications that take hours). PYTHON selects the interpreter (default python3).
# The read-out switch FTC_CRIT is not set here: velocity (default) everywhere, plus the explicit
# displacement steps marked _disp.
set -u
cd "$(dirname "$0")/model" || exit 1
PY=${PYTHON:-python3}
LONG=0; [ "${1:-}" = "--long" ] && LONG=1
mkdir -p out/logs
unset NPTS LAM CAT_COMBO HUMAN_STATE FTC_CRIT
C="$PY ../scripts/run_cat.py"; H="$PY ../scripts/run_human.py"; S=../scripts
step() {   # step TAG command... : run once, log, record wall time
    local tag=$1; shift
    [ -f out/logs/.done_$tag ] && return 0
    local t0=$(date +%s)
    if "$@" > out/logs/$tag.log 2>&1; then touch out/logs/.done_$tag; st=ok; else st=FAILED; fi
    printf '%-18s %-6s %6d s\n' "$tag" "$st" $(( $(date +%s) - t0 )) | tee -a out/logs/times.txt
}

lane_cat() {
    step battery      $C $S/battery.py
    step profiles     $C $S/profiles.py
    step c6_state     env C6OUT=out/c6_state.json $PY $S/c6_state.py 0.9501
    step c6_disp      env FTC_CRIT=disp C6OUT=out/c6_state_disp.json $PY $S/c6_state.py 0.9501
    step obs13_si     $PY $S/obs13.py si out/obs13_si.json
    step obs13_prof   $C $S/obs13.py prof out/obs13_prof.json
    step table2       $C $S/table2.py out/table2.json
    step bmdef        $C $S/bmdef_shoulder_s2.py base_ckpt.json out/bmdef_shoulder.json
    step units        $C $S/units.py out/bmdef_shoulder.json
    step ftcbins      $PY $S/ftcbins_model.py out/bmdef_shoulder.json ../data/tc_bins.json out/ftcbins.json
    step bmdef_disp   env FTC_CRIT=disp $C $S/bmdef_shoulder_s2.py base_ckpt.json out/bmdef_shoulder_disp.json
    step ftcbins_disp $PY $S/ftcbins_model.py out/bmdef_shoulder_disp.json ../data/tc_bins.json out/ftcbins_disp.json
    step shoulder     $PY $S/shoulder.py ref out/shoulder_ref.json
    step shoulder_disp env FTC_CRIT=disp $PY $S/shoulder.py ref out/shoulder_ref_disp.json
    step share_s2     $C $S/share_s2.py base_ckpt.json out/share_s2.json
    step dump_fig3_cat $C $S/dump_fig3_rev.py cat
    step share_scan   $C $S/share_scan.py
    step share_local  $PY $S/share_local.py 0.950
    step fig09_data   $PY $S/fig09_data.py 0.950
    step crit_mode    $PY $S/crit_mode.py 0.950
    step line_peak    $PY $S/line_peak.py 0.950
    step power_cat    $C $S/power_gain.py cat out/power_cat.json
    step removal_cat  $C $S/removal.py cat out/removal_cat.json
    step passive_q10  $C $S/passive_q10.py out/passive_q10.json
    step flux_profile $C $S/flux_profile.py out/flux_profile.json
    step si_readplace $PY $S/si_readplace.py combo_c856v.json
    step tf_cat       $C $S/tf.py cat
    step tm_cat       $C $S/tm_tf.py cat
    step sfoae_cat    $C $S/sfoae_cat.py
    step tuning_ratio $C $S/tuning_ratio.py
    step sfoae_cat_lv $C $S/sfoae_cat_lv.py
    step wavenumber   $C $S/wavenumber.py
    step fisher       $C $S/fisher_analog.py
    step shoulder_tm  $C $S/shoulder_tm.py base_ckpt.json out/shoulder_tm.json
    step conv_c       $C $S/conv_c.py
    step lm_prof      $C $S/line_margin.py cat_prof out/lm_cat_prof.json
    step lm_si        $C $S/line_margin.py cat_si out/lm_cat_si.json
    step c6_scan      env C6OUT=out/c6_scan.json $PY $S/c6_state.py 0.9501 0.93 0.90 0.88 0.85
    step c6_table     $PY $S/c6_table.py
    step rescore      bash -c 'for s in c708 c85 b0 b1; do C6CK=earlier/combo_$s.json C6OUT=out/c6_$s.json $0 ../scripts/c6_state.py 0.9501 || exit 1; done' $PY
    step vel_fit      $PY $S/vel_fit.py out/vel_fit.json
    step vel_fit_b0   env C6CK=earlier/combo_b0.json $PY $S/vel_fit.py out/vel_fit_b0.json
    step vel_fit_b1   env C6CK=earlier/combo_b1.json $PY $S/vel_fit.py out/vel_fit_b1.json
    step check_fix    $PY $S/check_fix.py earlier/combo_c708.json
    step ps_bc        $PY $S/ps_bc.py earlier/combo_c708.json out/ps_bc_c708.json
    step cert_cat     $C $S/cert_s2.py base_ckpt.json out/cert_s2.json
    if [ -e tc.plt ]; then     # Fig. 3: needs the individual curves (tc.plt, not redistributed)
        step fig2_rev      $C $S/fig2_rev.py
        step fig2_rev_disp bash -c "cp out/fig2_curves.json out/fig2_curves_vel.json && FTC_CRIT=disp $C $S/fig2_rev.py && mv out/fig2_curves.json out/fig2_curves_disp.json && mv out/fig2_curves_vel.json out/fig2_curves.json"
        step tc_numbers    $PY $S/tc_numbers.py out/tc_numbers.json
    fi
    step extract_cat  $PY $S/extract_cat.py out out/extract_cat.json
    if [ $LONG = 1 ]; then
        step b_crit       $C $S/b_crit.py
        step c_robust_cat $C $S/c_robust.py cat
        step c_scan_cert  $C $S/c_scan_cert.py
        step d_kh_cat     $C $S/d_kh.py cat
        for v in ref Kt4_g Kt4_r Kt025_g Kt025_r clamp_g clamp_r pass g098 g102; do step abl_$v $PY $S/abl.py $v; done
        step lam_conv     $C $S/lam_conv.py out/lam_conv.json
        step lam_fixed    bash -c '$0 ../scripts/mkstate.py earlier/combo_c708.json ../state_c708 && cd ../state_c708 && $0 ../scripts/run_cat.py ../scripts/lam_fixed.py cat out/lam_fixed_cat.json' $PY
    fi
}

lane_human() {
    while [ ! -f out/logs/.done_profiles ]; do sleep 5; done    # profiles_h.py adds the human key to out/profiles.json
    step profiles_h   $H $S/profiles_h.py
    step rms_N        $H $S/rms_N.py 8192
    step latency      $PY $S/latency_detail.py joint_hs_hs856.json out/latency_detail.json
    step ftc_h        $H $S/ftc_h.py
    step qerb_h       $H $S/qerb_h.py abr5_ckpt.json out/qerb_h.json
    step shape_h      $H $S/shape_h.py joint_hs_hs856.json out/shape_human.json
    step sfoae_h      $H $S/sfoae_h.py
    step tf_hum       $H $S/tf.py human
    step tfx          $PY $S/tfx.py out/tf_human.json out/tfx_human.json
    step tm_hum       $H $S/tm_tf.py human
    step dump_fig3_h  $H $S/dump_fig3_rev.py human
    step fig6_rev     $H $S/fig6_rev.py
    step map_h        $H $S/map_h.py out/map_h.json
    step place_map_h  $H $S/place_map_h.py out/place_map_h.json
    step power_hum    $H $S/power_gain.py hum out/power_hum.json
    step removal_hum  $H $S/removal.py hum out/removal_hum.json
    step lm_hum       $H $S/line_margin.py hum out/lm_hum.json
    step conv_h3      $H $S/conv_h3.py 8192
    step rms_N16384   $H $S/rms_N.py 16384
    step conv_h16384  $H $S/conv_h3.py 16384
    step cert_hum     $PY $S/cert_h_lam.py joint_hs_hs856.json out/cert_h.json
    if [ $LONG = 1 ]; then
        step cert_h_ctrl  $H $S/cert_h_ctrl.py joint_hs_hs856.json out/cert_h_ctrl.json
        step eig_h_ctrl   $H $S/eig_h_ctrl.py joint_hs_hs856.json out/eig_h_ctrl.json 128 256 512
        step c_scan_cert_h $H $S/c_scan_cert_h.py joint_hs_hs856.json
        step c_robust_hum $H $S/c_robust.py human joint_hs_hs856.json
        step d_kh_hum     $H $S/d_kh.py human joint_hs_hs856.json
        step tau_test     $H $S/tau_test.py
        step nu_scan      $H $S/nu_scan.py
    fi
}

echo "run_all.sh start $(date -u +%FT%TZ) python $($PY -c 'import sys, numpy, scipy; print(sys.version.split()[0], numpy.__version__, scipy.__version__)')" | tee -a out/logs/times.txt
lane_cat & PA=$!
lane_human & PB=$!
wait $PA $PB
step tm_post $PY $S/tm_post.py
step figures bash -c 'cd ../figures && for f in mkfig1 make_rev_figs make_figs_cat make_tm_figs; do $0 $f.py || exit 1; done' $PY
if [ $LONG = 1 ]; then step cd_tables $PY $S/cd_tables.py; fi
echo "run_all.sh end $(date -u +%FT%TZ)" | tee -a out/logs/times.txt
grep -c FAILED out/logs/times.txt | sed 's/^/failed steps: /'
