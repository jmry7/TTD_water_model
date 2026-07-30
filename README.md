# TTD Water Model

This repository contains the calculation programs developed for the Thole-type Drude (TTD) polarizable water model.

The repository consists of two modules:

- [`TTD_properties/`](./TTD_properties/)  
  Calculation of molecular properties of water monomer and clusters.

- [`TTD_MD/`](./TTD_MD/)  
  Molecular dynamics simulations of liquid water using the TTD model.

Detailed requirements, input files, calculation procedures, and output files are described in the `README.md` file of each module.


## Requirements

The molecular-property calculations are implemented mainly with PyTorch, while the molecular dynamics simulations are performed with OpenMM.

The program has been tested with:
- Python 3.11
- PyTorch 2.13.0 with CUDA 13.0 support
- OpenMM 8.4
- NumPy 2.4.6
- Scipy 1.17.1

