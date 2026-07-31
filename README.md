# TTD Water Model

This repository contains the calculation programs developed for the Thole-type Drude (TTD) polarizable water model.

The repository consists of three modules:

- [`TTD_properties/`](./TTD_properties/)  
  Calculation of molecular properties of the water monomer and clusters.

- [`TTD_MD/`](./TTD_MD/)  
  Molecular dynamics simulations of liquid water using the TTD model.

- [`Force_field_test/`](./Force_field_test/)  
  Validation of the consistency of energies and forces between the PyTorch and OpenMM implementations.

Detailed requirements, input files, calculation procedures, and output files are described in the `README.md` file of each module.


## Requirements

The molecular-property calculations are implemented mainly with PyTorch, while the molecular dynamics simulations are performed with OpenMM.

The programs has been tested with:
- Python 3.11
- PyTorch 2.13.0 with CUDA 13.0 support
- OpenMM 8.4
- NumPy 2.4.6
- SciPy 1.17.1

