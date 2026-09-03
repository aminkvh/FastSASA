# Selection Syntax

FastSASA has a compact selection language for common structure and trajectory
workflows. It is intentionally smaller than MDAnalysis or VMD — no `around`,
`byres`, `same...as`, bonded, or distance-based selections. For those, select
in MDAnalysis or MDTraj first and hand FastSASA the result: build a Python
selection with `mdanalysis_selection_arrays()` (see
[Adapter API](api.md#adapter-api)) or pass an MDAnalysis `AtomGroup` directly
to `sasa_mdanalysis()`/`SASAAnalysis`.

The two most important options are:

| Option | Meaning |
| --- | --- |
| `--filter EXPR` | Choose atoms that exist in the calculation. |
| `--select EXPR` | Report atoms after calculating in the current context. |

For most explicit-solvent trajectories, start with:

```sh
--filter protein
```

For one residue in protein context:

```sh
--filter protein --select 'segid AP and resi 677'
```

## Expressions

`--select` commands can be a plain expression or a named expression:

```text
expression
output_name, expression
```

If no output name is provided, FastSASA derives one from the expression by
replacing non-alphanumeric runs with `_`. For example, `segid AP and resi 677`
is reported as `segid_AP_and_resi_677`. Selection names are truncated to 50
characters.

Examples:

```text
chain A
chain_a, chain A
domain, chain A and resi 125-300
ligand, resn ABU
protein, protein
```

The native selectors are:

| Selector | Meaning | Example |
| --- | --- | --- |
| `name` | atom name | `name CA` |
| `symbol` | element symbol | `symbol C+N+O` |
| `resn` | residue name | `resn ALA+VAL+LEU` |
| `resi` | residue number or range | `resi 125-300` |
| `chain` | chain label or range | `chain A+C-E` |
| `segid`, `segname`, `segment` | segment label | `segid AP` |
| `protein` | recognized protein residues | `protein` |

Expressions support `and`, `or`, `not`, parentheses, `+` lists, and simple
residue or chain ranges.

The native selector set covers common structure and trajectory use cases.
Full VMD or MDAnalysis selections are still best handled upstream in Python
and passed to FastSASA as coordinates or masks.

`--filter` accepts plain expressions. It does not need an output name:

```sh
--filter protein
```

Invalid selector terms are ignored with a warning when the rest of the
expression can still be evaluated. Syntax errors still fail. For example,
`name ABCDE+CA` warns about the too-long atom name and still selects `CA`.
Selection warnings identify the term that was ignored:

```text
FastSASA: warning: selection: ignoring invalid atom name 'ABCDE'; atom names are limited to 4 characters
```

Residue ranges use dash notation. `resi 10-20` selects residues 10
through 20, `resi -10` selects residues up to 10, and `resi 10-` selects
residues from 10 onward. For actual negative residue numbers, escape the minus
sign:

```text
negative_residues, resi \-20-\-15+\-10-5
```

For PSF topologies, use the PSF segment name with `segid`, `segname`, or
`segment`. For CHARMM-GUI systems this often looks like `AP`, `BP`, `CP`, and
`DP`, not chains `A`, `B`, `C`, and `D`:

```text
target, segid AP and resi 677
```

Do not put multiple comma-separated selectors in one command. The comma only
separates the output name from the expression:

```text
target, chain AP and resi 677
```

not:

```text
chain AP, resi 677
```

## Filter And Select

These operations answer different scientific questions:

| Operation | Geometry | Typical question |
| --- | --- | --- |
| `--filter` | Remove non-selected atoms before calculation. | What is protein SASA after excluding water, lipids, and ions? |
| `--select` | Calculate with all included atoms, then sum the selected atoms. | What is chain A SASA inside this complex? |

Example:

```sh
--filter 'segid AP and resi 677'
```

This calculates residue 677 by itself. Other protein atoms do not exist in the
calculation and cannot bury the residue.

```sh
--filter protein --select 'segid AP and resi 677'
```

This calculates residue 677 in the protein context. Other protein atoms can
bury the residue, but only residue 677 is reported.

`--select` does not double-count an interface. If chains A and B touch,
selecting chain A reports the exposed area of chain A while chain B remains
present and buries the interface.

`--filter` changes the physical system. Filtering to chain A calculates
isolated chain A SASA.

## CLI Examples

Structure selection:

```sh
fastsasa --select 'domain, chain A and resi 125-300' structure.pdb
fastsasa --select 'chain A and resi 125-300' structure.pdb
```

Protein-only explicit-solvent trajectory:

```sh
fastsasa trajectory \
  --topology topology.pdb \
  --trajectory trajectory.xtc \
  --filter protein
```

Residue SASA inside the protein context:

```sh
fastsasa trajectory \
  --topology topology.psf \
  --trajectory trajectory.dcd \
  --filter protein \
  --select 'target, segid AP and resi 677'
```

Isolated residue SASA:

```sh
fastsasa trajectory \
  --topology topology.psf \
  --trajectory trajectory.dcd \
  --filter 'segid AP and resi 677'
```
