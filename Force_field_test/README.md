# Force-Field Validation

This directory contains a test program for comparing the PyTorch and OpenMM implementations of the TTD water model.

`Validation.py` reads the water-dimer configurations from `dimer_test.xyz` and the force-field parameters from `../TTD_properties/drude_par.h`. The real O and H coordinates are kept fixed, while the Drude particles are relaxed self-consistently using the PyTorch implementation. The same resulting `OHHDDD` coordinates are then used to calculate energies and forces with both implementations.

The OpenMM calculations are performed both without and with periodic boundary conditions (PBC).

For the nonperiodic calculations, the periodic nonbonded methods defined in `drude_ff.py` are replaced by `NoCutoff`.

For the periodic calculations, the original periodic settings in `drude_ff.py` are retained, including PME for electrostatic interactions and `CutoffPeriodic` for the short-range nonbonded interactions. A cubic box with a side length of 300 Å is used to minimize interactions between the dimer and its periodic images.

For the energy comparison, each implementation uses the energy of configuration 1 as its own reference. Force differences are calculated using the PyTorch results as the common reference.

Run the test using:

```bash
python Validation.py
```

The program outputs:

- `torch_openmm_energy.dat`: energy comparison in `kJ/mol`
- `torch_openmm_force.dat`: force comparison in `kJ/(mol nm)`