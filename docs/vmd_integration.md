# VMD Integration

FastSASA ships a `fastsasa` Tcl command for VMD that behaves like a
GPU-accelerated alternative to VMD's built-in `measure sasa`: same selection
semantics, same radii by default, plus whole-trajectory calculation in one
call, an animated accessible-surface visualization, and an on-screen SASA
counter.

## Setup

Add the package to VMD's plugin path in `~/.vmdrc`:

```tcl
lappend auto_path /path/to/fastsasa/integration/vmd
```

and make sure the `fastsasa` binary is on `PATH`, or point at it explicitly
with the `FASTSASA_EXE` environment variable or the `-exe` option.

## Usage

In any VMD session (GUI or `vmd -dispdev text`):

```tcl
package require fastsasa

set sel [atomselect top protein]

# Total SASA of the current frame (VMD radii, like measure sasa):
set area [fastsasa -sel $sel]

# Every frame of the loaded trajectory, streamed through the GPU engine:
set areas [fastsasa -sel $sel -frames all]

# measure sasa -restrict equivalent (selA reported, everything occludes):
set all [atomselect top all]
set chainA_area [fastsasa -sel $all -restrict [atomselect top "chain A"]]

# Per-atom SASA into a Tcl list, and stored on the atoms if you like:
fastsasa -sel $sel -peratom areas
$sel set user $areas

# Accessible surface points (measure sasa -points equivalent):
fastsasa -sel $sel -points pts

# Draw the accessible surface and show the on-screen counter:
fastsasa -sel $sel -visualize 1 -color yellow

# Whole trajectory with a synced animated surface and counter:
set areas [fastsasa -sel $sel -frames all -visualize 1]

# Huge trajectory on disk without loading it into VMD:
set areas [fastsasa -topfile sys.psf -trajfile run.dcd -filter protein]
```

## Options

- `-probe R` — probe radius, default 1.4.
- `-samples N` — sphere points per atom, default 500 (matching
  `measure sasa`; FastSASA outside VMD defaults to 100).
- `-frames now|all` — current frame, or every loaded frame as a per-frame
  list.
- `-restrict $sel2` — report only these atoms; everything in `-sel` still
  occludes. `-frames now` only.
- `-peratom var` — store the per-atom SASA list. `-frames now` only.
- `-points var` — store accessible surface points as `{x y z}` lists.
  `-frames now` only.
- `-visualize 0|1` — default off; the one switch for all visuals. With
  `-frames now` it draws the current frame's point cloud, with
  `-frames all` it loads the animated surface track; both show the counter.
- `-track 0|1` — the animated surface as its own molecule (default: on with
  `-visualize 1 -frames all`).
- `-track-samples N` — surface density for the track, independent of
  `-samples` (default: `-samples`/10).
- `-hud 0|1` — on-screen SASA counter (default: on with `-visualize 1` or
  `-track 1`).
- `-color name` — drawing color, default yellow.
- `-radii vmd|protor` — radius source, default `vmd` (the molecule's own
  per-atom radii).
- `-backend auto|vulkan|cuda|cpu` and `-precision fp32|fp64` — default
  `fp32` inside VMD, matching VMD's single-precision arithmetic.
- `-threads N` — CPU thread count passed through to the CLI's `--threads`
  (ignored on GPU backends).
- `-exe path` — explicit path to the `fastsasa` binary.
- `-topfile F -trajfile F -filter EXPR` — analyze files on disk directly,
  without loading them into VMD.
- `-clear 1` — remove the molecule's FastSASA cloud/track and counter;
  runs no calculation.

Mapping to `measure sasa` and the CLI:

| VMD `measure sasa` | `fastsasa` wrapper | FastSASA CLI |
| --- | --- | --- |
| `<srad>` | `-probe R` | `--probe-radius R` |
| `-samples N` (default 500) | `-samples N` (default 500) | `--resolution N` (default 100) |
| `-restrict $sel2` | `-restrict $sel2` | `--select` |
| molecule radii | `-radii vmd` (default) | generated `--config-file` |
| n/a | `-radii protor` | bundled ProtOr table |
| n/a | `-frames now\|all` | `--frames` |
| n/a | `-backend`, `-precision` | `--backend`, `--precision` |

## Defaults and comparability with `measure sasa`

Inside VMD the wrapper defaults to **`-radii vmd`, `-samples 500`, and
`-precision fp32`**, matching `measure sasa`'s radii, sampling density, and
precision class, so bare `fastsasa` and bare `measure sasa` calls are
directly comparable. (VMD stores coordinates and accumulates SASA in single
precision; FastSASA's fp32 mode still counts exposures exactly and reduces
in FP64, so it is the more accurate of the two.) Outside VMD, FastSASA
defaults to ProtOr radii, 100 samples, and FP64; pass
`-radii protor -samples 100 -precision fp64` in VMD to reproduce the native
CLI/Python defaults exactly.

**Values differ slightly from `measure sasa` by design.** Both tools use
deterministic sphere sampling, but different point sets (FastSASA uses a
Fibonacci sphere), so at a finite sample count the two quadratures carry
different discretization bias. With matched radii and sample counts, totals
typically agree within a fraction of a percent, and the gap shrinks as
`-samples` grows. Use the same radii, probe radius, atom set, and sample count
when comparing results from different programs.

The surface points are generated by the same exposure test as the SASA
calculation, so the point cloud is exactly the reported surface: per atom,
`4 * pi * r^2 * points / samples` equals the atom's SASA.

## Frames and indexing

With `-frames all`, element *i* of the returned list is VMD frame *i* of
the source molecule — the same index the animation slider, the surface
track, and the on-screen counter use. Note that when a topology file
contributes a coordinate frame (e.g. `mol new top.pdb` followed by
`mol addfile traj.dcd`), that PDB frame is VMD frame 0, so the list has one
more element than the DCD has frames. Run the CLI on the trajectory file
directly (or use `-topfile`/`-trajfile`) if you want DCD-only indexing.

## Visualization

Nothing is drawn unless you ask: bare calls return numbers only.
`-visualize 1` is the single switch — with `-frames now` it draws the
current frame's points, with `-frames all` it loads the animated surface
track, and in both cases it shows the on-screen counter. Once something is
drawn it stays until you replace it with another `-visualize 1` call or
remove it:

```tcl
fastsasa -sel $sel -clear 1
```

Three things to know when visualizing:

- **A `-frames now -visualize 1` cloud belongs to one frame** — whatever
  frame was current when the call ran. Drawn points are static graphics and
  do not follow animation; re-run after moving to a new frame, or use
  `-frames all -visualize 1` for an animated surface.
- **Everything in `-sel` occludes.** With `-sel [atomselect top all]` on a
  solvated or membrane system, water and lipids bury most of the protein
  surface and the points there correctly disappear. Select the molecule set
  that should count, e.g. `protein`.
- **Buried surface has no points by definition** — interfaces and the core
  show gaps because their accessible area is zero. The cloud sits on the
  probe-center surface, about a probe radius above the van der Waals
  surface.

### Surface track: animated, toggleable

`fastsasa -sel $sel -frames all -visualize 1` (or `-track 1`) computes the
accessible surface of every frame alongside the per-frame SASA values and
loads it as a separate molecule named `FastSASA surface (<source>)`, one
frame per trajectory frame, drawn with the Points representation. It is an
ordinary molecule, so the usual controls apply:

```tcl
set areas [fastsasa -sel $sel -frames all -visualize 1 -color yellow]
set track $::FastSASA::track([$sel molid])   ;# molecule id of the track
mol off $track                              ;# hide (or click D in the molecule list)
mol on $track                               ;# show again
mol modstyle 0 $track Points 4.0            ;# bigger dots
mol modcolor 0 $track ColorID 1             ;# recolor
mol delete $track                           ;# remove
```

Each call replaces the previous track for that source molecule. The track
molecule is kept out of VMD's own animation and its frame is driven from
the source molecule's `vmd_frame` callback, so the cloud always shows
exactly the frame the protein is showing, whether the trajectory is playing
or being scrubbed.

**Track density and cost.** The reported areas always use the full
`-samples` density; the drawn cloud defaults to a tenth of it
(`-track-samples`), because surface-point extraction and loading dominate
the call at full density. The default track adds a few seconds to a
`-frames all` call; a full-density track can take an order of magnitude
longer and produce hundreds of thousands of points per frame. Pass
`-track-samples 500` if you want the dense cloud anyway. The reported
totals (`-samples`) and the track (`-track-samples`) are extracted in one
CLI pass. If rotating the scene stutters, a smaller
`mol modstyle ... Points 1.0` point size and disabling depth cueing help.

`-track` surface extraction uses the Vulkan kernel when available. CPU and
CUDA backends use the threaded CPU surface-point path. Single-frame
`-points`/`-visualize` also uses the CPU path.

### On-screen SASA counter

`-hud` comes on with `-visualize 1` (or `-track 1`) — a small
"SASA: 1234.5 Å²" label pinned to the top-right corner of the screen. Pass
`-hud 1` on its own for the counter without any surface drawing, or
`-hud 0` to suppress it while visualizing.

With `-frames now` the counter shows that one call's total. With
`-frames all` it follows the currently displayed frame as the trajectory
plays or is scrubbed — no recomputation, just a lookup into the per-frame
list the call already returned. `-hud` needs Tk, which ships with VMD;
under `vmd -dispdev text` with no display it prints one warning and is
otherwise a no-op, so it is safe in scripts that might run headless.

## Performance notes

- Prefer one `-frames all` call over a per-frame Tcl loop: each separate
  call pays process startup and (on GPU backends) context creation.
- For unavoidable single-frame calls, `-backend cpu` skips GPU context
  creation and is usually the fastest end-to-end option.
- Temporary topology/trajectory exports go to RAM-backed storage when
  available (`/dev/shm`), honor `TMPDIR`, and are removed on completion and
  on error. They can be large for big systems.

## Caveats

- The wrapper is Tcl 8.5-compatible, matching the Tcl bundled with VMD
  1.9.3/1.9.4.
- FastSASA builds natively on Windows (`fastsasa.exe`, CPU and Vulkan), and
  the wrapper can point `FASTSASA_EXE` at it, but the combination inside
  Windows VMD is untested.
