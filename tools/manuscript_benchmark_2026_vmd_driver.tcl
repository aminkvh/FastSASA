# Figure 2 (VMD workflow performance) driver for the FastSASA manuscript
# benchmark. Compares `measure sasa` against the FastSASA VMD extension
# (fastsasa.tcl) on the same Cx46 trajectory used for Figure 1, over two
# panels (complete-system, selected-atom-with-full-system-occlusion) and
# five configurations: measure sasa, FastSASA VMD CPU 1-thread, CPU
# 15-thread, Vulkan FP64, CUDA FP64.
#
# Usage: vmd -dispdev text -e tools/manuscript_benchmark_2026_vmd_driver.tcl
#
# Timing methodology: the trajectory is loaded once (untimed). Each
# configuration then runs a warm-up pass over all frames plus 3 timed
# passes; the timer covers only the analysis loop (VMD startup and
# trajectory loading are excluded, matching the spec). `measure sasa` and
# the FastSASA extension both use -radii vmd (VMD's own per-atom radii),
# since measure sasa has no radii override -- this is the one part of the
# benchmark where "matched radii" means "both tools read VMD's radii table",
# not the element-guessed table used for Figure 1's Python-side comparison.

set repo_root [file dirname [file dirname [file normalize [info script]]]]
source [file join $repo_root integration/vmd/fastsasa/fastsasa.tcl]

set topology [file join $repo_root benchmark_corpus/trajectories/cx46_hemichannel/Cx46_Ace_ProtIon.psf]
set trajectory [file join $repo_root benchmark_corpus/trajectories/cx46_hemichannel/Cx46_Ace_ProtIon_01.dcd]
set out_dir [file join $repo_root benchmark_corpus/results/manuscript_2026/cx46]
set out_csv [file join $out_dir figure2_raw_timings.csv]

set probe 1.4
set samples 100
set repeats 3

puts "loading $topology / $trajectory ..."
mol new $topology
mol addfile $trajectory waitfor all
set m [molinfo top]
set n_frames_available [molinfo $m get numframes]
set all_sel [atomselect $m "all"]
set restrict_sel [atomselect $m "segid AP1"]

# Figure 1 times the full 250-frame trajectory because its measurements are
# in-process (one call covers all frames). Figure 2's VMD-extension configs
# each launch one fastsasa subprocess per frame (see the module-level note
# in manuscript_benchmark_2026.py on why the extension has no batched
# -restrict path); at ~0.2-0.8s/frame that makes a 250-frame x 4-pass x
# 5-config sweep impractical to run repeatedly. This uses a smaller, fixed
# frame count instead -- applied identically to measure_sasa and every
# FastSASA VMD configuration in both panels, so the comparison stays
# apples-to-apples. It is a real methodology difference from Figure 1's
# frame range and is reported as such, not hidden in the CSV.
set n_frames 20
if {$n_frames > $n_frames_available} { set n_frames $n_frames_available }
puts "loaded: [$all_sel num] atoms x $n_frames_available frames available; using $n_frames frames for timing; selection segid AP1 = [$restrict_sel num] atoms"

set csv [open $out_csv w]
puts $csv "dataset,tool,threads,precision,panel,repeat_index,n_frames,elapsed_seconds,frames_per_second,total_sasa_last_repeat"

# body scripts are evaluated at #0 (absolute global scope), not relative to
# the caller: time_pass is called from run_config, which is itself called
# from global scope, so a relative `uplevel 1` would land in run_config's
# proc-local frame -- where none of probe/all_sel/samples/restrict_sel/
# backend/threads_opt (all set at the top-level script scope, i.e. global)
# are visible. #0 always means "the global namespace" regardless of call depth.
proc time_pass {n_frames body_var} {
    upvar 1 $body_var body
    set t0 [clock microseconds]
    set total 0.0
    for {set f 0} {$f < $n_frames} {incr f} {
        animate goto $f
        set total [expr {$total + [uplevel #0 $body]}]
    }
    set t1 [clock microseconds]
    return [list [expr {($t1 - $t0) / 1.0e6}] $total]
}

proc run_config {csv tool threads precision panel n_frames repeats body_var} {
    upvar 1 $body_var body
    # warm-up (not recorded)
    time_pass $n_frames body
    for {set r 0} {$r < $repeats} {incr r} {
        set result [time_pass $n_frames body]
        set elapsed [lindex $result 0]
        set total [lindex $result 1]
        set fps [expr {$n_frames / $elapsed}]
        puts $csv "cx46,$tool,$threads,$precision,$panel,$r,$n_frames,$elapsed,$fps,$total"
        puts [format "%s threads=%s precision=%s panel=%s repeat=%d: %.3f frames/s (total=%.6f)" \
            $tool $threads $precision $panel $r $fps $total]
        flush $csv
    }
}

# --- measure sasa: complete-system panel ---
set body {measure sasa $probe $all_sel -samples $samples}
run_config $csv "measure_sasa" "n/a" "fp32(fixed)" "complete" $n_frames $repeats body

# --- measure sasa: selected-atom panel (full-system occlusion via -restrict) ---
set body {measure sasa $probe $all_sel -samples $samples -restrict $restrict_sel}
run_config $csv "measure_sasa" "n/a" "fp32(fixed)" "selected" $n_frames $repeats body

# --- FastSASA VMD extension configurations ---
set configs {
    {fastsasa_cpu 1 fp64}
    {fastsasa_cpu 15 fp64}
    {fastsasa_vulkan 0 fp64}
    {fastsasa_cuda 0 fp64}
}
foreach cfg $configs {
    set tool [lindex $cfg 0]
    set threads [lindex $cfg 1]
    set precision [lindex $cfg 2]
    set backend [string range $tool 9 end]
    set threads_opt {}
    if {$backend eq "cpu"} {
        set threads_opt [list -threads $threads]
    }

    # complete-system panel
    set body [list fastsasa -sel $all_sel -probe $probe -samples $samples \
        -radii vmd -backend $backend -precision $precision {*}$threads_opt]
    run_config $csv $tool $threads $precision "complete" $n_frames $repeats body

    # selected-atom panel: full system provides occlusion, -restrict reports the subset
    set body [list fastsasa -sel $all_sel -probe $probe -samples $samples \
        -radii vmd -backend $backend -precision $precision -restrict $restrict_sel {*}$threads_opt]
    run_config $csv $tool $threads $precision "selected" $n_frames $repeats body
}

close $csv
puts "\nwritten: $out_csv"

# --- numerical cross-check: max per-frame |measure_sasa - fastsasa| difference ---
puts "\n--- numerical cross-check (complete-system panel, per frame) ---"
set max_diff 0.0
set max_diff_frame -1
for {set f 0} {$f < $n_frames} {incr f} {
    animate goto $f
    set vmd_val [measure sasa $probe $all_sel -samples $samples]
    set fs_val [fastsasa -sel $all_sel -probe $probe -samples $samples -radii vmd -backend cpu -precision fp64]
    set diff [expr {abs($vmd_val - $fs_val)}]
    if {$diff > $max_diff} {
        set max_diff $diff
        set max_diff_frame $f
    }
}
puts [format "max |measure_sasa - fastsasa_cpu| over %d frames = %.6f at frame %d" $n_frames $max_diff $max_diff_frame]
set check_out [open [file join $out_dir figure2_numerical_check.txt] w]
puts $check_out "max_abs_diff_measure_sasa_vs_fastsasa_cpu=$max_diff"
puts $check_out "max_diff_frame=$max_diff_frame"
puts $check_out "n_frames=$n_frames"
close $check_out

quit
