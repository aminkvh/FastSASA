# fastsasa.tcl - GPU-accelerated SASA for VMD through the FastSASA CLI.
#
# Usage inside VMD:
#   lappend auto_path /path/to/fastsasa/integration/vmd
#   package require fastsasa
#   set sel [atomselect top protein]
#   set area [fastsasa -sel $sel]                      ;# current frame total, numbers only
#   set areas [fastsasa -sel $sel -frames all]         ;# per-frame list, numbers only
#   set part [fastsasa -sel $all -restrict $selA]      ;# measure sasa -restrict
#   fastsasa -sel $sel -points pts                     ;# accessible surface points
#   fastsasa -sel $sel -visualize 1 -color yellow      ;# draw the surface + on-screen counter
#   fastsasa -sel $sel -frames all -visualize 1        ;# synced surface track + counter
#   fastsasa -sel $sel -clear 1                        ;# remove any FastSASA visuals
#
# Nothing is drawn unless asked: -visualize 1 turns on the visuals -- the
# static point cloud for -frames now, the animated surface track for
# -frames all, and the on-screen counter in both cases. -track and -hud
# remain available to pick those pieces individually. Once something is
# drawn it stays until replaced by another -visualize 1 call or removed
# with -clear 1 -- a plain "fastsasa -sel $sel" call does not touch it.
#
# The command mirrors "measure sasa" semantics: every atom in -sel occludes
# and is measured, using the molecule's current radius values by default
# (-radii vmd). Both FastSASA (Fibonacci sphere) and VMD's measure sasa use
# deterministic point placement -- neither re-randomizes call to call -- but
# they use different point sets, so totals agree to within a fraction of a
# percent at matched sample counts, not digit for digit. Requires the
# fastsasa binary on PATH or FASTSASA_EXE.
#
# Tcl 8.5 compatible except -hud, which needs Tk (present in the standard
# VMD build; falls back to a console warning if no display is available).

package provide fastsasa 0.1

namespace eval ::FastSASA {
    variable version 0.1
    # graphics primitive ids drawn by -visualize, per molecule id, so a new
    # call can replace its previous point cloud instead of stacking clouds
    # from different frames on top of each other.
    variable drawn
    array set drawn {}
    # molecule id of the "FastSASA surface" track created by -track, per
    # source molecule id, so a new call replaces the previous track.
    variable track
    array set track {}
    # on-screen SASA counter: a small always-on-top Tk window pinned to the
    # top-right of the screen, plus the per-frame area list and vmd_frame
    # trace that keep it in sync while a trajectory animates.
    variable hud_window ".fastsasa_hud"
    variable hud_label ""
    variable hud_areas
    array set hud_areas {}
}

proc ::FastSASA::find_executable {override} {
    if {$override ne ""} {
        if {![file executable $override]} {
            error "fastsasa executable not found at '$override'"
        }
        return $override
    }
    if {[info exists ::env(FASTSASA_EXE)] && [file executable $::env(FASTSASA_EXE)]} {
        return $::env(FASTSASA_EXE)
    }
    set found [auto_execok fastsasa]
    if {$found eq ""} {
        error "fastsasa executable not found; add it to PATH or set FASTSASA_EXE"
    }
    return [lindex $found 0]
}

proc ::FastSASA::temp_stem {} {
    # Prefer RAM-backed storage for the temporary topology/trajectory
    # exports so large selections never round-trip through the disk;
    # an explicit TMPDIR still wins. Windows has no /tmp or /dev/shm, so
    # fall back to its own TEMP/TMP conventions there.
    if {$::tcl_platform(platform) eq "windows"} {
        set dir [pwd]
        foreach var {TEMP TMP} {
            if {[info exists ::env($var)] && [file isdirectory $::env($var)]} {
                set dir $::env($var)
                break
            }
        }
    } else {
        set dir "/tmp"
        if {[file isdirectory "/dev/shm"] && [file writable "/dev/shm"]} {
            set dir "/dev/shm"
        }
    }
    if {[info exists ::env(TMPDIR)] && [file isdirectory $::env(TMPDIR)]} {
        set dir $::env(TMPDIR)
    }
    return [file join $dir "fastsasa_vmd_[pid]_[clock clicks]"]
}

# Writes a FastSASA radius config that reproduces the molecule's current
# per-atom radius values, keyed by residue and atom name.
proc ::FastSASA::write_vmd_radius_config {sel path} {
    set resnames [$sel get resname]
    set names [$sel get name]
    set radii [$sel get radius]
    array set radius_of {}
    set pairs {}
    foreach resname $resnames name $names radius $radii {
        set key "$resname $name"
        if {[info exists radius_of($key)]} {
            if {abs($radius_of($key) - $radius) > 1.0e-6} {
                puts "fastsasa: warning: '$key' has multiple VMD radii; using [format %.3f $radius_of($key)]"
            }
            continue
        }
        set radius_of($key) $radius
        lappend pairs $key
    }
    set out [open $path w]
    puts $out "types:"
    array set type_of {}
    foreach key $pairs {
        set radius $radius_of($key)
        set type "VMD_[string map {. _ - m} [format %.3f $radius]]"
        if {![info exists type_of($type)]} {
            set type_of($type) 1
            puts $out "$type [format %.3f $radius] apolar"
        }
    }
    puts $out ""
    puts $out "atoms:"
    foreach key $pairs {
        set radius $radius_of($key)
        set type "VMD_[string map {. _ - m} [format %.3f $radius]]"
        puts $out "$key $type"
    }
    close $out
}

# Number of points per frame in a surface-track DCD written by
# "fastsasa trajectory --surface-points FILE.dcd" (CHARMM DCD header:
# 84-byte control block, a title block, then the atom count).
proc ::FastSASA::dcd_point_count {path} {
    set in [open $path r]
    fconfigure $in -translation binary
    set head [read $in 96]
    if {[string length $head] < 96 || ![binary scan $head @92n title_bytes]} {
        close $in
        error "fastsasa: surface track '$path' is not a DCD file"
    }
    seek $in [expr {104 + $title_bytes}] start
    if {![binary scan [read $in 4] n count]} {
        close $in
        error "fastsasa: surface track '$path' has no atom count"
    }
    close $in
    return $count
}

# Loads the per-frame surface point cloud written by "fastsasa trajectory
# --surface-points FILE.dcd" as its own molecule with one frame per source
# frame, so VMD animates the cloud in step with the trajectory. The track
# can be hidden and shown with "mol off/on" or the molecule list's D
# column without recomputing anything. The DCD goes onto an empty
# "mol new atoms N" molecule through VMD's compiled DCD reader, which
# loads a 100-frame track in a fraction of the time text XYZ takes.
proc ::FastSASA::load_surface_track {molid dcd color} {
    global vmd_frame
    if {[info exists ::FastSASA::track($molid)]} {
        catch {mol delete $::FastSASA::track($molid)}
        unset ::FastSASA::track($molid)
        catch {trace remove variable vmd_frame($molid) write \
            [list ::FastSASA::track_frame_callback $molid]}
    }
    set color_index [lsearch -exact [colorinfo colors] $color]
    if {$color_index < 0} {
        error "fastsasa: unknown color '$color'"
    }
    set trackmol [mol new atoms [::FastSASA::dcd_point_count $dcd]]
    mol addfile $dcd type dcd waitfor all molid $trackmol
    mol rename $trackmol "FastSASA surface ([molinfo $molid get name])"
    while {[molinfo $trackmol get numreps] > 0} {
        mol delrep 0 $trackmol
    }
    mol representation Points 2.0
    mol color ColorID $color_index
    mol selection all
    mol material Opaque
    mol addrep $trackmol
    set frames [molinfo $molid get numframes]
    if {[molinfo $trackmol get numframes] != $frames} {
        mol delete $trackmol
        error "fastsasa: surface track frame count does not match molecule $molid"
    }
    mol top $molid
    set ::FastSASA::track($molid) $trackmol
    # VMD leaves a freshly loaded molecule on its last frame and then
    # animates every active molecule from its own current frame, so the
    # cloud would play back offset from the protein. Take the track out of
    # VMD's own animation and drive its frame from the source molecule's
    # vmd_frame callback instead, so the two can never drift apart.
    mol inactive $trackmol
    molinfo $trackmol set frame [molinfo $molid get frame]
    if {[catch {
        trace add variable vmd_frame($molid) write \
            [list ::FastSASA::track_frame_callback $molid]
    } message]} {
        puts "fastsasa: warning: could not keep the surface track in sync with frame changes ($message)"
    }
    return $trackmol
}

proc ::FastSASA::track_frame_callback {molid name1 name2 op} {
    global vmd_frame
    if {![info exists ::FastSASA::track($molid)] || ![info exists vmd_frame($molid)]} {
        return
    }
    set trackmol $::FastSASA::track($molid)
    if {[lsearch -exact [molinfo list] $trackmol] < 0} {
        # The user deleted the track molecule; stop following.
        unset ::FastSASA::track($molid)
        catch {trace remove variable vmd_frame($molid) write \
            [list ::FastSASA::track_frame_callback $molid]}
        return
    }
    set frame $vmd_frame($molid)
    if {$frame < 0 || $frame >= [molinfo $trackmol get numframes]} {
        return
    }
    if {[molinfo $trackmol get frame] != $frame} {
        molinfo $trackmol set frame $frame
    }
}

# Creates (once) the on-screen SASA counter as a small, borderless,
# always-on-top Tk window anchored to the top-right corner of the screen.
# Returns 0 and prints a one-time warning instead of erroring when no
# display/Tk is available (e.g. a headless `vmd -dispdev text` run), so
# -hud is always safe to pass.
proc ::FastSASA::hud_ensure {} {
    variable hud_unavailable
    if {[info commands winfo] eq ""} {
        # Tk is not loaded at all (e.g. `vmd -dispdev text` without a
        # display) -- "winfo" itself would be an undefined-command error,
        # not something a Tcl catch around it can intercept.
        if {![info exists hud_unavailable]} {
            set hud_unavailable 1
            puts "fastsasa: warning: on-screen SASA counter unavailable (no Tk display)"
        }
        return 0
    }
    if {[winfo exists $::FastSASA::hud_window]} {
        return 1
    }
    if {[catch {
        toplevel $::FastSASA::hud_window
        wm title $::FastSASA::hud_window "FastSASA"
        wm overrideredirect $::FastSASA::hud_window 1
        wm attributes $::FastSASA::hud_window -topmost 1
        label $::FastSASA::hud_window.l -text "SASA: -- Å²" \
            -font {Helvetica 14 bold} -background black -foreground yellow \
            -padx 10 -pady 6
        pack $::FastSASA::hud_window.l
        update idletasks
        set width [winfo reqwidth $::FastSASA::hud_window]
        set x [expr {[winfo screenwidth .] - $width - 10}]
        wm geometry $::FastSASA::hud_window "+${x}+10"
        set ::FastSASA::hud_label $::FastSASA::hud_window.l
    } message]} {
        puts "fastsasa: warning: could not create on-screen SASA counter ($message)"
        return 0
    }
    return 1
}

proc ::FastSASA::hud_set {text} {
    if {![::FastSASA::hud_ensure]} {
        return
    }
    catch {$::FastSASA::hud_label configure -text $text}
    catch {raise $::FastSASA::hud_window}
}

proc ::FastSASA::hud_frame_callback {molid name1 name2 op} {
    global vmd_frame
    if {![info exists ::FastSASA::hud_areas($molid)] || ![info exists vmd_frame($molid)]} {
        return
    }
    set frame $vmd_frame($molid)
    set areas $::FastSASA::hud_areas($molid)
    if {$frame < 0 || $frame >= [llength $areas]} {
        return
    }
    ::FastSASA::hud_set [format "SASA: %.1f Å²  (frame %d)" [lindex $areas $frame] $frame]
}

# Wires the on-screen counter to a per-frame area list so it tracks
# whatever frame $molid (and any molecule animating in step with it, such
# as a -track surface) is currently showing. A later call for the same
# molid replaces the previous binding.
proc ::FastSASA::hud_attach_track {molid areas} {
    global vmd_frame
    if {[info exists ::FastSASA::hud_areas($molid)]} {
        catch {trace remove variable vmd_frame($molid) write \
            [list ::FastSASA::hud_frame_callback $molid]}
    }
    set ::FastSASA::hud_areas($molid) $areas
    if {[catch {
        trace add variable vmd_frame($molid) write \
            [list ::FastSASA::hud_frame_callback $molid]
    } message]} {
        puts "fastsasa: warning: could not track frame changes for the SASA counter ($message)"
    }
    ::FastSASA::hud_frame_callback $molid "" "" write
}

# Removes molid's FastSASA drawn cloud (-visualize on -frames now) and
# surface track (-frames all), and stops the on-screen counter from
# following it. "fastsasa -sel $sel -clear 1" is the supported way to call
# this: a plain "fastsasa -sel $sel" call leaves any earlier -visualize
# output alone, so once something is drawn it stays until replaced by
# another -visualize 1 call or removed with -clear.
proc ::FastSASA::clear_visuals {molid} {
    global vmd_frame
    if {[info exists ::FastSASA::drawn($molid)]} {
        foreach graphic_id $::FastSASA::drawn($molid) {
            catch {graphics $molid delete $graphic_id}
        }
        unset ::FastSASA::drawn($molid)
    }
    if {[info exists ::FastSASA::track($molid)]} {
        catch {mol delete $::FastSASA::track($molid)}
        unset ::FastSASA::track($molid)
        catch {trace remove variable vmd_frame($molid) write \
            [list ::FastSASA::track_frame_callback $molid]}
    }
    if {[info exists ::FastSASA::hud_areas($molid)]} {
        catch {trace remove variable vmd_frame($molid) write \
            [list ::FastSASA::hud_frame_callback $molid]}
        unset ::FastSASA::hud_areas($molid)
    }
}

proc ::FastSASA::run_cli {arguments} {
    if {[catch {eval exec $arguments 2>@1} output]} {
        error "fastsasa failed: $output"
    }
    return $output
}

proc ::FastSASA::parse_pdb_sasa {path} {
    set values {}
    set in [open $path r]
    while {[gets $in line] >= 0} {
        set record [string range $line 0 5]
        if {$record ne "ATOM  " && $record ne "HETATM"} continue
        lappend values [string trim [string range $line 60 65]]
    }
    close $in
    return $values
}

# fastsasa ?options?
#   -sel $atomselect     selection to measure (required unless -topfile)
#   -probe R             probe radius, default 1.4
#   -samples N           SR sphere points, default 500 (measure sasa's
#                        default; FastSASA outside VMD defaults to 100)
#   -frames now|all      current frame or the whole loaded trajectory
#   -restrict $sel2      report only these atoms; -sel still occludes
#   -peratom varname     store the per-atom SASA list (frames now only)
#   -points varname      store accessible surface points as {x y z} lists
#   -visualize 0|1       show the accessible surface, default 0 (off). With
#                        -frames now: draw the current frame's points as
#                        static graphics (each call replaces the previous
#                        FastSASA cloud for that molecule). With -frames
#                        all: load the animated surface track (see -track).
#                        Either way the on-screen counter comes with it.
#   -track 0|1           with -frames all: load the per-frame accessible
#                        surface as a separate "FastSASA surface" molecule
#                        (Points representation) that animates in sync
#                        with the trajectory; toggle it with mol on/off.
#                        Each call replaces the previous track. The
#                        track's molecule id is stored in
#                        ::FastSASA::track(<source molid>). Default: on
#                        when -visualize 1 and -frames all, otherwise off.
#   -track-samples N     surface sample density for -track, independent of
#                        -samples (which still sets the reported SASA
#                        accuracy). Default: -samples/10 (50 for the 500
#                        default). A full-density track is several hundred
#                        thousand points per frame, takes far longer to
#                        export and load than the SASA itself, and slows
#                        rotation; pass -track-samples 500 for a dense cloud.
#   -hud 0|1             show an on-screen "SASA: ... Å²" counter pinned to
#                        the top-right of the screen. With -frames now it
#                        shows that call's total; with -frames all it
#                        tracks the currently displayed frame automatically
#                        as the trajectory animates (via -track or plain
#                        playback), no recomputation per frame. Requires Tk
#                        (bundled with VMD); silently falls back to a
#                        console warning if no display is available.
#                        Default: on when -visualize 1 or -track 1,
#                        otherwise off.
#   -color name          drawing color for -visualize/-track, default yellow
#   -radii vmd|protor    radius source, default vmd (molecule radii)
#   -backend auto|vulkan|cuda|cpu
#   -precision fp32|fp64  default fp32 (measure sasa computes in single
#                         precision; FastSASA outside VMD defaults to fp64)
#   -threads N           CPU thread count passthrough to the fastsasa CLI's
#                        --threads (ignored on GPU backends); default "" lets
#                        the CLI pick its own default thread count
#   -exe /path/to/fastsasa
#   -topfile F -trajfile F -filter EXPR   file passthrough without VMD memory
#   -clear 0|1            remove -sel's molecule's FastSASA cloud/track and
#                        stop; does not run fastsasa or touch any other
#                        option. "fastsasa -sel $sel -clear 1" is the way
#                        to get rid of a visualization once it's drawn.
proc fastsasa {args} {
    array set option {
        -sel "" -probe 1.4 -samples 500 -frames now -restrict "" -peratom ""
        -points "" -visualize 0 -track "" -track-samples "" -hud "" -color yellow
        -radii vmd -backend auto -precision fp32 -exe "" -threads ""
        -topfile "" -trajfile "" -filter protein -clear 0
    }
    foreach {key value} $args {
        if {![info exists option($key)]} {
            error "fastsasa: unknown option '$key'"
        }
        set option($key) $value
    }
    set threads_args {}
    if {$option(-threads) ne ""} {
        set threads_args [list --threads $option(-threads)]
    }
    if {$option(-clear)} {
        if {$option(-sel) eq ""} {
            error "fastsasa: -clear requires -sel"
        }
        ::FastSASA::clear_visuals [$option(-sel) molid]
        return
    }
    # Nothing visual happens by default. -visualize 1 is the one switch
    # that turns it on: with -frames all that means the animated surface
    # track, and in both modes the on-screen counter. -track and -hud can
    # still be set explicitly to pick those pieces individually.
    if {$option(-track) eq ""} {
        set option(-track) [expr {$option(-visualize) && $option(-frames) eq "all"}]
    }
    if {$option(-hud) eq ""} {
        set option(-hud) [expr {$option(-visualize) || $option(-track)}]
    }
    set exe [::FastSASA::find_executable $option(-exe)]
    set stem [::FastSASA::temp_stem]
    set cleanup {}

    if {$option(-topfile) ne ""} {
        if {$option(-trajfile) eq ""} {
            error "fastsasa: -topfile requires -trajfile"
        }
        set output [::FastSASA::run_cli [concat [list $exe trajectory \
            --topology $option(-topfile) --trajectory $option(-trajfile) \
            --frames : --filter $option(-filter) \
            --probe-radius $option(-probe) --resolution $option(-samples) \
            --backend $option(-backend) --precision $option(-precision)] $threads_args]]
        set totals {}
        foreach line [split $output "\n"] {
            if {[regexp {^\d+,([-0-9.eE+]+),} $line -> total]} {
                lappend totals $total
            }
        }
        return $totals
    }

    if {$option(-sel) eq ""} {
        error "fastsasa: -sel is required"
    }
    set sel $option(-sel)
    if {[$sel num] == 0} {
        return 0.0
    }

    set topology "$stem.pdb"
    $sel writepdb $topology
    lappend cleanup $topology
    set common [concat [list --probe-radius $option(-probe) \
                     --resolution $option(-samples) \
                     --backend $option(-backend) \
                     --precision $option(-precision) \
                     --hydrogen --hetatm] $threads_args]
    if {$option(-radii) eq "vmd"} {
        set config "$stem.config"
        ::FastSASA::write_vmd_radius_config $sel $config
        lappend cleanup $config
        lappend common --config-file $config
    } elseif {$option(-radii) ne "protor"} {
        error "fastsasa: -radii must be vmd or protor"
    }

    if {[catch {
        if {$option(-frames) eq "all"} {
            if {$option(-restrict) ne "" || $option(-peratom) ne "" ||
                $option(-points) ne ""} {
                error "fastsasa: -restrict, -peratom, and -points support -frames now only"
            }
            if {$option(-track-samples) ne "" && !$option(-track)} {
                error "fastsasa: -track-samples requires -track 1"
            }
            set trajectory "$stem.dcd"
            animate write dcd $trajectory waitfor all sel $sel
            lappend cleanup $trajectory
            set trajectory_args [list $exe trajectory \
                --topology $topology --trajectory $trajectory --frames :]
            set track_file ""
            set track_samples $option(-samples)
            if {$option(-track)} {
                if {$option(-track-samples) ne ""} {
                    set track_samples $option(-track-samples)
                } else {
                    # The reported SASA keeps the full -samples density; the
                    # drawn cloud only needs a tenth of it. A full-density
                    # track is hundreds of thousands of points per frame
                    # (555k points per frame on a 25k-atom system) and
                    # dominates the whole call.
                    set track_samples [expr {max(10, $option(-samples) / 10)}]
                }
                set track_file "$stem.surface.dcd"
                lappend cleanup $track_file
                # --surface-resolution decouples the exported point density
                # from --resolution (used for the reported SASA), so both are
                # produced in this one pass instead of a second full read of
                # the topology and trajectory.
                lappend trajectory_args --surface-points $track_file \
                    --surface-resolution $track_samples
            }
            set output [::FastSASA::run_cli [concat $trajectory_args $common]]
            set result {}
            foreach line [split $output "\n"] {
                if {[regexp {^\d+,([-0-9.eE+]+),} $line -> total]} {
                    lappend result $total
                }
            }
            if {$track_file ne ""} {
                set trackmol [::FastSASA::load_surface_track [$sel molid] \
                    $track_file $option(-color)]
                puts "fastsasa: surface track loaded as molecule $trackmol ([molinfo $trackmol get numatoms] points/frame); mol off $trackmol hides it; omit -visualize 1 for numbers only"
            }
            if {$option(-hud)} {
                ::FastSASA::hud_attach_track [$sel molid] $result
            }
        } elseif {$option(-frames) eq "now"} {
            if {$option(-track)} {
                error "fastsasa: -track requires -frames all; use -visualize for the current frame"
            }
            set sasa_pdb "$stem.sasa.pdb"
            lappend cleanup $sasa_pdb
            set structure_args [list $exe --format pdb \
                --output $sasa_pdb --unknown guess]
            set surface_file ""
            if {$option(-points) ne "" || $option(-visualize)} {
                set surface_file "$stem.surface.txt"
                lappend cleanup $surface_file
                lappend structure_args --surface-points $surface_file
            }
            ::FastSASA::run_cli [concat $structure_args $common [list $topology]]
            if {$surface_file ne ""} {
                set surface_points {}
                set in [open $surface_file r]
                while {[gets $in line] >= 0} {
                    lappend surface_points [lrange $line 0 2]
                }
                close $in
                if {$option(-points) ne ""} {
                    upvar 1 $option(-points) points_out
                    set points_out $surface_points
                }
                if {$option(-visualize)} {
                    set molid [$sel molid]
                    # Replace this molecule's previous FastSASA point cloud;
                    # the points belong to the frame that was current when
                    # this call ran and do not follow animation.
                    if {[info exists ::FastSASA::drawn($molid)]} {
                        foreach graphic_id $::FastSASA::drawn($molid) {
                            catch {graphics $molid delete $graphic_id}
                        }
                    }
                    set drawn_ids [list [graphics $molid color $option(-color)]]
                    foreach surface_point $surface_points {
                        lappend drawn_ids [graphics $molid point $surface_point]
                    }
                    set ::FastSASA::drawn($molid) $drawn_ids
                }
            }
            set atom_sasa [::FastSASA::parse_pdb_sasa $sasa_pdb]
            if {[llength $atom_sasa] != [$sel num]} {
                error "fastsasa: atom count mismatch between VMD and FastSASA output"
            }
            if {$option(-peratom) ne ""} {
                upvar 1 $option(-peratom) peratom
                set peratom $atom_sasa
            }
            if {$option(-restrict) ne ""} {
                array set wanted {}
                foreach index [$option(-restrict) get index] {
                    set wanted($index) 1
                }
                set result 0.0
                foreach index [$sel get index] area $atom_sasa {
                    if {[info exists wanted($index)]} {
                        set result [expr {$result + $area}]
                    }
                }
            } else {
                set result 0.0
                foreach area $atom_sasa {
                    set result [expr {$result + $area}]
                }
            }
            if {$option(-hud)} {
                ::FastSASA::hud_set [format "SASA: %.1f Å²" $result]
            }
        } else {
            error "fastsasa: -frames must be now or all"
        }
    } message]} {
        foreach path $cleanup {catch {file delete $path}}
        error $message
    }
    foreach path $cleanup {catch {file delete $path}}
    return $result
}
