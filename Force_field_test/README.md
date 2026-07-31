# Force-Field Validation

This directory contains a test program for comparing the PyTorch and OpenMM implementations of the TTD water model.

`Validation.py` reads the water-dimer configurations from `dimer_test.xyz`. 
The real O and H coordinates are kept fixed, while the Drude particles are relaxed self-consistently using the PyTorch implementation. 
The same resulting `OHHDDD` coordinates are then used to calculate energies and forces with both implementations.

For the dimer calculations, the program disables the periodic nonbonded methods defined in drude_ff.py.

Run the test using:

```bash
python Validation.py
```

The program outputs:

- `torch_openmm_energy.dat`: energy comparison in `kJ/mol`
- `torch_openmm_force.dat`: force comparison in `kJ/(mol nm)`