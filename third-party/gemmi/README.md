# Notes for FastSASA

FastSASA vendors a compact header-only snapshot of GEMMI version 0.5.8 for native
mmCIF parsing and writing. Only the header files used by FastSASA are included
here. Version 0.5.8 is used because it is header-only and keeps the build
simple.

Go to the [GEMMI GitHub pages](https://github.com/project-gemmi/gemmi) for
the full source code and newer versions.

# Original README

GEMMI can help if you work with:

* macromolecular models (from mmCIF, PDB and mmJSON files),
* refinement restraints (CIF files),
* crystallographic reflections (from MTZ and SF-mmCIF files),
* electron density maps (MRC/CCP4 files),
* crystallographic symmetries,
* or if you just read and write CIF/STAR files (where C=Crystallographic).

GEMMI is a header-only C++11 library accompanied by:

* command-line [tools](https://gemmi.readthedocs.io/en/latest/utils.html),
* Python bindings (supporting CPython and PyPy),
* Fortran 2003+ interface (in progress),
* WebAssembly ports (see [here](https://project-gemmi.github.io/wasm/) and
  [here](https://www.npmjs.com/package/mtz)),
* and little data viz [projects](https://project-gemmi.github.io/pdb-stats/).

Documentation: http://gemmi.readthedocs.io/en/latest/

GEMMI is an open-source project of [CCP4](https://www.ccp4.ac.uk/)
and [Global Phasing Ltd](https://www.globalphasing.com/),
two major providers of software for macromolecular crystallography.

Citing: [JOSS paper](https://doi.org/10.21105/joss.04200).

License: MPLv2, or (at your option) LGPLv3.
© 2017-2022 Global Phasing Ltd.

`fileutil.hpp` and `utf.hpp` are included for the Windows build (`cif.hpp`
includes `fileutil.hpp` only under `_WIN32`).
