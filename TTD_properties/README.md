# TTD Water Model Property Calculation and Cluster Optimization

This code package provides two computing modules for the TTD (Thole-type Drude) polarizable water model:

1. calculation of equilibrium properties for an isolated water monomer;
2. geometry optimization of water clusters.

### `Directory Structure`
``` 

    project_directory/
    ├── cal_monomer_properties.py
    ├── cluster_opt.py
    ├── drude_par.h
    ├── monomer_config.xyz
    ├── cluster.xyz
    └── Drude/
        ├── drude_wat.py
        ├── bend.py
        ├── stret.py
        ├── coul.py
        └── other internal calculation modules
```
### `cal_monomer_properties.py`

The main program for monomer-property calculations. It reads the force-field parameters and monomer configuration, calls the TTD calculation routines, optimizes the monomer geometry, evaluates the selected properties, and writes the comparison file.

### `cluster_opt.py`

Optimizes water-cluster geometries with the TTD force field.


### `drude_par.h`

Contains the fitted force-field parameters.

The parameter file contains 42 numerical entries. The two parameters associated with H–H coupling are retained but are not used in the actual calculations. The charge of the H-attached Drude particle is determined from the molecular charge-neutrality condition rather than treated as an independent fitting parameter.

Consequently, the TTD model contains 41 active parameter values in total, of which 40 are independent parameters.

### `monomer_config.xyz`

Contains the initial water-monomer geometry in XYZ format.

### `cluster.xyz`

Contains the initial water-cluster geometries in multi-frame XYZ format.


Only the real O and H atoms should be included in the input file. Drude particles are generated during the TTD optimization and are included in the output file.

### `Drude/`

Contains the internal energy, force, geometry-optimization, vibrational, multipole, and (hyper)polarizability routines.


The `drude_wat()` function is the core calculation program of the `Drude/` module in `drude_wat.py`. It accepts either a single water configuration or a batch of configurations containing the real atoms in the order O H H, 
initializes the corresponding core and Drude particles, relaxes the Drude coordinates, evaluates the potential energy and forces, and performs the additional calculations selected by its logical arguments.

The calculation options are controlled by logical variables: 

- `lgrad` returns forces on the real atoms; lfc enables force-constant evaluation; \
- `ldip` calculates the dipole moment; lpol calculates the dipole polarizability; \
- `lpol2` calculates the first and second hyperpolarizabilities together with the A, B, C, and D response tensors; \
- `lquad` calculates the quadrupole, octupole, and hexadecapole moments; \
- `lopt` performs geometry optimization; \
- `lmode` performs normal-mode analysis; \
- `lg` calculates the O–H bond length and H–O–H angle of the first water molecule; and\
- `loo` calculates all unique O–O distances.

The function returns a dictionary containing the SCF-relaxed coordinates, potential energies, and forces on all core and Drude particles, together with any optional properties requested through these logical arguments.

## Requirements

The program has been tested with:

- Python 3.11
- PyTorch 2.13.0 with CUDA 13.0 support

CUDA is used automatically when a compatible GPU and CUDA-enabled PyTorch installation are available. Otherwise, the calculations run on the CPU.

## Running the Program

Run the monomer-property calculation from the project directory:
```bash
python cal_monomer_properties.py
```

Run the cluster geometry optimization from the project directory:
```bash
python cluster_opt.py
```


## Output Files

### `monomer_equilibrium_properties_au.dat`

Contains the optimized monomer geometry and the calculated monomer properties together with the corresponding *ab initio* reference values.

The monomer-property calculation includes:

- geometry optimization;
- harmonic vibrational frequencies;
- dipole, traceless quadrupole, traceless octupole, and traceless hexadecapole moments;
- dipole polarizability;
- first and second hyperpolarizabilities;
- traceless dipole–quadrupole polarizability A;
- traceless dipole–dipole–quadrupole polarizability B;
- traceless quadrupole–quadrupole polarizability C;
- traceless dipole–octupole polarizability D.

### `cluster_optimized.xyz`

Contains the optimized cluster structures in multi-frame XYZ format. Each water molecule contains six interaction sites in the order `O H H D D D`.

## Reference Data

The harmonic vibrational frequency reference data are from:

> D. P. Tew, W. Klopper, M. Heckert, and J. Gauss, “Basis Set Limit CCSD(T) Harmonic Vibrational Frequencies,” *J. Phys. Chem. A* **111** (2007) 11242–11248.

The multipole and (hyper)polarizability reference data are from:

> H. Chen, M. Liu, and T. Yan, “Molecular multipoles and (hyper)polarizabilities of water by *ab initio* calculations,” *Chemical Physics Letters* **752** (2020) 137555.