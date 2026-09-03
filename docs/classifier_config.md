# Radius Configuration

FastSASA assigns a radius to every included atom before SASA calculation.

## Assignment Order

1. User-provided `--config-file FILE`.
2. `FASTSASA_DEFAULT_CONFIG`, when set.
3. Bundled or installed `share/protor.config`.
4. Element-radius fallback for unknown residue/atom pairs.

When running from a source tree or an installed prefix, FastSASA searches common
`share/protor.config` locations automatically. If you run the binary from an
unusual directory layout, pass `--config-file` or set `FASTSASA_DEFAULT_CONFIG`
so the intended radius table is unambiguous.

When FastSASA uses an element fallback, it prints a warning:

```text
FastSASA: warning: atom 'NAG C1' unknown, guessing element is 'C', and radius 1.700 A
```

The atom remains included. Use `--unknown skip` or `--unknown halt` when guessed
radii are not acceptable.

## Configuration Format

A radius file contains a `types:` section and an `atoms:` section:

```text
types:
C_ALIPHATIC 1.70 apolar
N_POLAR     1.55 polar
O_POLAR     1.52 polar

atoms:
ANY N  N_POLAR
ANY CA C_ALIPHATIC
ANY O  O_POLAR
```

The first section defines named radius classes. The second maps residue and
atom names to those classes.

Use a custom file:

```sh
fastsasa --config-file my_radii.config structure.pdb
```

or for a trajectory:

```sh
fastsasa trajectory \
  --topology topology.pdb \
  --trajectory trajectory.xtc \
  --config-file my_radii.config \
  --filter protein
```

## Nonstandard Residues

The bundled table covers standard residues and common structural entries.
Unknown ligands, glycans, lipids, and modified residues fall back to element
radii with warnings unless the user provides a reviewed configuration.

Do not treat element fallback as a force-field-specific parameterization. For
publication work involving nonstandard chemistry, document the radius source
and provide the configuration file.

## Bundled ProtOr Table

FastSASA's default configuration uses the ProtOr atomic-group radii reported
by Tsai, Taylor, Chothia, and Gerstein (1999). The table maps standard protein,
nucleic-acid, and common cap atom names onto those published types. The
original data attribution and redistribution notice are recorded in
the repository [NOTICE](https://github.com/aminkvh/fastsasa/blob/main/NOTICE).

The element fallback is an approximation for atoms not covered by that table.
It uses van der Waals radii from Bondi (1964), with the common extensions
for elements Bondi did not tabulate (H 1.10 Å after Rowland & Taylor 1996,
Ca 2.31 Å after Mantina et al. 2009, Fe 1.80 Å as a conventional
approximation), and is intended to keep exploratory calculations usable, not
to replace a reviewed parameter set for nonstandard chemistry. FastSASA
covers 16 elements (`src/fastsasa_radius.c`); any other element has no
fallback and is rejected.

| Element | Radius (Å) | Element | Radius (Å) |
| --- | --- | --- | --- |
| H / D | 1.10 | Br | 1.85 |
| C | 1.70 | I | 1.98 |
| N | 1.55 | Na | 2.27 |
| O | 1.52 | Mg | 1.73 |
| F | 1.47 | K | 2.75 |
| P | 1.80 | Ca | 2.31 |
| S | 1.80 | Fe | 1.80 |
| Cl | 1.75 | Zn | 1.39 |

## Glycans And Lipids

ProtOr covers amino acids and nucleic acids. Sugar and lipid atoms fall back
to element radii with a warning.

For glycans, FastSASA ships an opt-in extension:

```sh
fastsasa --hetatm --config-file share/protor_glycans.config glycoprotein.pdb
```

`protor_glycans.config` is the full ProtOr table plus common pyranose
residues (GLC, BGC, MAN, BMA, GAL, GLA, NAG, NDG, FUC, FUL, XYS, XYP) mapped
onto the existing ProtOr chemical types. It introduces no new radius values,
so any atom that was already covered produces exactly the same result as the
default table; only previously-guessed sugar atoms change (typically from
the element carbon radius 1.70 to the ProtOr aliphatic 1.88). That is the
compatibility contract for any future table extension: new entries may only
map additional residue/atom pairs onto existing published types, never alter
an existing assignment.

For lipids there is deliberately no bundled table: membrane force fields use
their own atom naming and their own radii, so the right radii come from the
simulation force field, not from a structure-database table. Generate a
config from your force field (or, inside VMD, use the VMD integration's
`-radii vmd` mode, which exports the loaded molecule's radii automatically),
or rely on element guessing when approximate lipid radii are acceptable.

Relative-SASA (RSA) output uses the bundled ProtOr residue references. They
are reported whenever the loaded config declares `name: ProtOr`, which the
bundled `share/protor.config` does, whether it was auto-discovered, named by
`FASTSASA_DEFAULT_CONFIG`, or passed with `--config-file`. Any other table,
including the glycan extension (`name: ProtOr-Glycans`), reports `N/A` for
the relative fields, because the references describe ProtOr radii only.

## MD Residue Names

Force-field topologies name protonation states, tautomers, and
disulfide-bonded cysteine differently from the PDB. FastSASA resolves the
common variants to the standard residue for radius lookup, polar/apolar
classification, and RSA references, in the CLI, the trajectory command, the
Python `RadiusConfig`, and the feature extraction's `per_residue_rsa`:

| Variant | Standard | Variant | Standard |
| --- | --- | --- | --- |
| HID, HIE, HIP | HIS | CYX, CYM | CYS |
| HSD, HSE, HSP | HIS | ASH | ASP |
| LYN | LYS | GLH | GLU |
| ARN | ARG | | |

An exact `residue atom` entry in the config always wins, so a custom table
can still give a variant its own radii. Other nonstandard residues get the
element fallback for radii and `N/A`/NaN for relative SASA; the Python API
warns once per unknown residue name.

## Reference

J. Tsai, R. Taylor, C. Chothia, and M. Gerstein, “The packing density in
proteins: standard radii and volumes,” *Journal of Molecular Biology* 290,
253–266 (1999), [doi:10.1006/jmbi.1999.2829](https://doi.org/10.1006/jmbi.1999.2829).

A. Bondi, “van der Waals Volumes and Radii,” *Journal of Physical Chemistry*
68(3), 441–451 (1964), [doi:10.1021/j100785a001](https://doi.org/10.1021/j100785a001).

R. S. Rowland and R. Taylor, “Intermolecular Nonbonded Contact Distances in
Organic Crystal Structures,” *Journal of Physical Chemistry* 100(18),
7384–7391 (1996), [doi:10.1021/jp953141+](https://doi.org/10.1021/jp953141+).

M. Mantina, A. C. Chamberlin, R. Valero, C. J. Cramer, and D. G. Truhlar,
“Consistent van der Waals Radii for the Whole Main Group,” *Journal of
Physical Chemistry A* 113(19), 5806–5812 (2009),
[doi:10.1021/jp8111556](https://doi.org/10.1021/jp8111556).
