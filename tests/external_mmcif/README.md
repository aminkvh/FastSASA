# External mmCIF Fixtures

These files are external PDBx/mmCIF fixtures used for parser and release
hardening. They are not synthetic unit tests.

Sources:

- `1EN2.cif`: downloaded from `https://files.rcsb.org/download/1EN2.cif`.
  This fixture contains many alternate-location rows and exercises altloc-heavy
  atom-site handling. FastSASA keeps the first alternate conformer at each
  residue position, yielding 614 non-hydrogen ATOM records.
- `2K39.cif`: downloaded from `https://files.rcsb.org/download/2K39.cif`.
  This fixture contains many model numbers and exercises multi-model mmCIF
  handling.

Validation:

```sh
./build/fastsasa_mmcif_corpus_validate \
  tests/external_mmcif/1EN2.cif \
  tests/external_mmcif/2K39.cif
```

When configured with these fixture files present, CMake also exposes:

```sh
cmake --build build --target fastsasa_external_mmcif_validate
```

These fixtures test parser behavior rather than output formatting. External
files are especially useful for alternate conformers, model handling, and
metadata combinations that are difficult to represent with small synthetic
fixtures.
