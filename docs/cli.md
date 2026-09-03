# CLI Reference

The native CLI is `fastsasa`.

```sh
fastsasa [options] structure.pdb|structure.cif
fastsasa trajectory --topology FILE --trajectory FILE [trajectory-options]
```

The examples in these docs assume `fastsasa` is on your `PATH`. If you built
from source, the binary is at `./build/fastsasa`; either add the build directory
to `PATH` or prefix the commands accordingly.

## Choosing a Backend and Precision

FastSASA has three compute backends. By default it picks the first one that is
compiled in and available at runtime, in this order:

1. Vulkan (NVIDIA, AMD, and Intel GPUs)
2. CUDA (NVIDIA GPUs) — for pinned NVIDIA/HPC deployments.
3. Native threaded CPU

You can pin a backend explicitly:

| Option | Meaning |
| --- | --- |
| `--backend auto\|vulkan\|cuda\|cpu` | select the compute backend; `auto` is the default order above |
| `--cpu` | shorthand for `--backend cpu` |
| `--no-cpu-fallback` | fail instead of falling back to CPU (structure command) |

The same choice is available to the C and Python APIs through the environment
variable `FASTSASA_BACKEND=auto|vulkan|cuda|cpu`. The flag and the environment
variable are equivalent; the flag is the preferred spelling on the command
line.

Arithmetic precision is controlled with `--precision fp64|fp32` on both the
structure and trajectory commands. FP64 is the default everywhere: Vulkan and
CUDA FP64 Shrake-Rupley and Lee-Richards results are bit-identical to the
CPU reference, atom for atom (both use non-contracted arithmetic in the
CPU's operation order and the same acos/atan2 implementation, and totals,
residue sums, and selection sums are formed on the host in a fixed order on
every backend). `--precision fp32` trades that guarantee for speed. Observed
maximum relative error against the FP64 reference is around 1e-5 for
structure totals; on trajectory per-frame totals it is typically below 4e-5
for whole-system sums and up to a few times 1e-4 for small selection sums
(smaller sums leave less room for error cancellation across atoms).

Vulkan FP64 requires the device feature `shaderFloat64`. On a Vulkan device
without it, requesting FP64 fails with a clear error; either add
`--precision fp32` or let `--backend auto` fall back to the CPU path.

`--backend cpu --precision fp32` is also supported, for Shrake-Rupley only;
CPU Lee-Richards is fp64-only and prints a warning (computing at fp64) if
`--precision fp32 --lee-richards` is combined with `--backend cpu`.

Matching backends use the same algorithm, radii, probe radius, resolution,
and reduction order. See [Radius Configuration](classifier_config.md) when
comparing FastSASA with another SASA implementation.

## Core Options

| Option | Meaning |
| --- | --- |
| `--help`, `-h` | print command help |
| `--version` | print the FastSASA version |
| `--shrake-rupley`, `-S` | use Shrake-Rupley, default structure algorithm |
| `--lee-richards`, `-L` | use Lee-Richards |
| `--probe-radius N`, `-p N` | solvent probe radius, default `1.4` |
| `--resolution N`, `-n N` | SR points or LR slices; defaults are SR `100`, LR `20` |
| `--backend auto\|vulkan\|cuda\|cpu` | select the compute backend, default `auto` |
| `--cpu` | shorthand for `--backend cpu` |
| `--threads N` | CPU thread count; default is `max(1, detected CPU threads - 1)` |
| `--no-cpu-fallback` | fail instead of falling back to CPU |
| `--precision fp64\|fp32` | arithmetic precision, default `fp64` |
| `--config-file FILE`, `-c FILE` | load a radius configuration |
| `--hetatm`, `-H` | include HETATM records |
| `--hydrogen`, `-Y` | include hydrogen atoms |
| `--unknown guess\|skip\|halt` | policy for unknown radii, default `guess` |
| `--cif` | force the mmCIF reader even if the file suffix is not `.cif` |
| `--classes` | also report polar, apolar, and unknown SASA classes |
| `--select 'expression'` | calculate selected SASA; output name is generated |
| `--select 'name, expression'` | calculate selected SASA with an explicit output name |
| `--format log\|res\|seq\|rsa\|pdb\|cif\|json\|xml`, `-f ...` | output format |
| `--output FILE`, `-o FILE` | write output to file |
| `--surface-points FILE` | also write the accessible surface points as `x y z atom_index` lines |

## Radius Assignment

FastSASA assigns atom radii from a user-provided `--config-file`, then
`FASTSASA_DEFAULT_CONFIG` when set, then the bundled `share/protor.config`
table, with an element-radius fallback (and a warning) for unknown
residue/atom pairs. See
[Radius Configuration](classifier_config.md) for the full assignment order,
the warning format, the `--unknown skip|halt` policies, and the config file
format.

When `--classes` is used, FastSASA reports SASA split into `polar`, `apolar`,
and `unknown` atom classes. The split uses atom classes from the loaded radius
config when available. Atoms without a class in the loaded config are reported
as unknown. If no config is loaded at all, FastSASA uses a simple element fallback:
carbon is apolar, non-hydrogen heteroatoms are polar, and unmatched atoms are
unknown. This does not change the SASA calculation; it only changes reporting.

FastSASA's bundled defaults use ProtOr radii. Other tools may use different
atom radii from topology or display fields. For example, VMD
`measure sasa` uses the molecule's current `radius` values. If you need values
closer to a VMD workflow, export or define those radii explicitly and pass them
with `--config-file`; do not expect default FastSASA and default VMD radii to
produce identical absolute values. The bundled [VMD integration](vmd_integration.md)
automates exactly this radius matching from inside VMD.

## Output Formats

- `log`: human-readable structure summary.
- `res`: totals grouped by amino-acid type.
- `seq`: one absolute SASA value per residue.
- `rsa`: per-residue absolute and relative total, side-chain, main-chain,
  apolar, and polar SASA, followed by chain and structure sums.
- `pdb`: coordinates with atom SASA in the B-factor-like field.
- `cif`, `json`, `xml`: structured result exports. JSON includes
  `parameters`, `total_sasa`, `atoms`, `residues`, and any requested
  selections or classes.

Relative RSA values use the bundled ProtOr residue references, reported
whenever the loaded config declares `name: ProtOr` (the bundled table,
however it was loaded). With any other radius configuration, including the
glycan extension, FastSASA writes `N/A` for relative fields because the
built-in references no longer describe that radius model. MD residue
variants such as HIE or CYX resolve to HIS and CYS; see
[Radius Configuration](classifier_config.md#md-residue-names).

```sh
fastsasa --format log structure.pdb
fastsasa --format res structure.pdb
fastsasa --format seq structure.pdb
fastsasa --format rsa structure.pdb
fastsasa --format pdb structure.pdb
fastsasa --format cif structure.cif
fastsasa --format json structure.pdb
fastsasa --format xml structure.pdb
```

CIF output keeps the source mmCIF text and appends FastSASA-owned
`_FastSASA_*` result categories.

## Advanced Structure Input

`--cif` forces the mmCIF reader when the filename does not end in `.cif`.
`--join-models`, or `-m`, treats every model in a multi-model PDB/mmCIF file
as one combined atom set. Without it, FastSASA calculates the first model.
Joining models is uncommon and is not appropriate when models represent an
NMR ensemble or separate trajectory frames.

## Selection Language

See [Selection Syntax](selection.md) for the selector table and the full
selection reference, including output naming rules, negative residue
numbers, PSF segment naming, and warning behavior. In short: expressions
combine selectors like `name`, `resn`, `resi`, `chain`, `segid`, and
`protein` with `and`, `or`, `not`, parentheses, `+` lists, and ranges. A
`--select` command is either a plain expression or an explicit
`output_name, expression` pair.

Examples:

```sh
fastsasa --select 'chain A' tests/data/2isk.pdb
fastsasa --select 'domain, chain A and resi 125-300' tests/data/2isk.pdb
fastsasa --select 'ligand, resn ABU' structure.pdb
fastsasa trajectory --topology topology.psf --trajectory trajectory.dcd \
  --frames 1 --summary --filter protein --select 'res677_ap, segid AP and resi 677'
```

`--select` and `--filter` are scientifically different: `--filter` removes
atoms before calculating (changes the physical system), `--select` reports
a subset after calculating with everything else still present (so
interfaces are not double-counted). See
[Filter And Select](selection.md#filter-and-select) for the full
explanation and examples.

Full MDAnalysis or VMD selection language is intentionally handled upstream.
For complex selections, build masks or coordinate subsets in your MD tool and
pass arrays to the Python API.

## Trajectory CLI

```sh
fastsasa trajectory --topology topology.pdb --trajectory trajectory.xtc \
  --frames : --summary --filter protein

fastsasa trajectory --topology topology.pdb --trajectory trajectory.xtc \
  --summary --filter 'complex, protein or resn ABU'

fastsasa trajectory --topology topology.pdb --trajectory trajectory.xtc \
  --summary --select 'chain_a, chain A'

fastsasa trajectory --topology topology.psf --trajectory trajectory.dcd \
  --frames :100 --residue --filter protein
```

Options:

| Option | Meaning |
| --- | --- |
| `--topology FILE` | `.pdb`, `.cif`, or `.psf` topology |
| `--trajectory FILE` | `.dcd` or `.xtc` trajectory |
| `--frames SPEC` | Python-style frame selection; `:` means all, `10` means frame 10, `-1` means last frame |
| `--batch-size N` | frames per GPU or CPU trajectory batch; omitted means conservative automatic choice |
| `--probe-radius N`, `-p N` | solvent probe radius, default `1.4` |
| `--resolution N`, `-n N` | SR points or LR slices; defaults are SR `100`, LR `20` |
| `--precision fp64\|fp32` | arithmetic precision, default `fp64` |
| `--output FILE`, `-o FILE` | write trajectory CSV output to a file instead of stdout |
| `--summary` | output total SASA summary |
| `--residue` | output residue SASA per frame |
| `--config-file FILE`, `-c FILE` | load a radius configuration |
| `--hydrogen`, `-Y` | include hydrogen atoms in the calculation |
| `--hetatm`, `-H` | include PDB/mmCIF HETATM records in the calculation |
| `--threads N` | CPU trajectory worker budget |
| `--backend auto\|vulkan\|cuda\|cpu` | select the compute backend, default `auto` |
| `--cpu` | shorthand for `--backend cpu` |
| `--shrake-rupley`, `-S` | use Shrake-Rupley, the default |
| `--lee-richards`, `-L` | use Lee-Richards |
| `--classes` | report polar, apolar, and unknown SASA classes |
| `--filter EXPR` | define the calculation universe |
| `--select 'expression\|name, expression'` | selected SASA |
| `--surface-points FILE` | write accessible points as text, multi-frame XYZ, or DCD for visualization |
| `--surface-resolution N` | point density for `--surface-points`, independent of `--resolution` (default: same as `--resolution`; 100 for Lee-Richards) |
| `--help`, `-h` | print trajectory command help |
| `--version` | print the FastSASA version |

`trajectory --surface-points` uses the Vulkan kernel when available. CPU and
CUDA trajectory backends use the threaded CPU surface-point path; the CLI
prints a note when it falls back. Both paths produce the same exposed-point
set. Single-structure surface-point export uses the CPU path.

Trajectory topology readers preserve the complete atom order so DCD/XTC
coordinates remain aligned with their topology. Hydrogen atoms and PDB/mmCIF
HETATM records are excluded from the SASA calculation by default, matching the
structure CLI. Use `--hydrogen` or `--hetatm` to include them. PSF has no
ATOM/HETATM record distinction, so use `--filter` to control ligands, lipids,
water, and ions in PSF-based trajectories.

Apart from those default hydrogen/HETATM rules, the trajectory CLI does not
silently remove water, lipids, or ions. If no `--filter` is supplied, all
eligible topology atoms participate and FastSASA emits a warning.

FastSASA does not wrap, unwrap, reimage, or center trajectories. Prepare
coordinates upstream with MDAnalysis, MDTraj, GROMACS, VMD, or your MD engine
workflow before passing them to FastSASA.

Use `fastsasa trajectory` for scripts, benchmarks, and interactive use. See
[Trajectory Analysis](trajectory.md) for examples.
