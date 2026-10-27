# TTD Polarizable Water Model: Molecular Dynamics

This package implements molecular dynamics simulations for the TTD
polarizable water model using OpenMM. The current version has been
tested with **OpenMM 8.4**.

The package contains separate subdirectories for **NVT** and **NPT**
simulations. The thermostat and barostat implementations are based on
the methods described in *Molecular Physics*, **1996**, 87(5),
1117--1157.

## Files and Entry Points

-   **NVT simulation:** `NVT/nvt_run.py`
-   **NPT simulation:** `NPT/npt_run.py`

## Requirements

The program has been tested with:

-   Python 3.11
-   OpenMM 8.4
-   NumPy 2.4.6
-   SciPy 1.17.1

An NVIDIA GPU with CUDA support is recommended. If CUDA is unavailable,
the program can attempt to use OpenCL for GPU acceleration.

## Input Structure

The main program reads an input structure in PDB format.

Each water molecule must contain six interaction sites:

-   `O`
-   `H1`
-   `H2`
-   `DO`
-   `DH1`
-   `DH2`

The residue name must be `HOH`.

In the initial PDB structure, each Drude particle should overlap with
its corresponding core site.

**Recommended initial structure:** Unbalanced configurations may trigger
a polarization catastrophe or cause the SCF procedure to fail to
converge. We therefore recommend using the provided `water_6p.pdb` file
as the initial structure. To run multiple trajectories, use
`extract_last_xyz_to_pdb.py` to generate a new initial PDB file from the
final frame of a previously generated trajectory.\
For NPT simulations, because the current force field yields a relatively high 
equilibrium density, we recommend using a larger initial simulation box and a 
larger number of water molecules (e.g., the 512 water molecules used in the current setup). 
Otherwise, the program will raise an error if the box length is less than twice the short-range cutoff distance.

## Running a Simulation

``` bash
python NVT/nvt_run.py
```

or

``` bash
python NPT/npt_run.py
```

The input PDB filename and the main simulation parameters can be
configured in `nvt_run.py` or `npt_run.py`.

## Checkpoint and Restart

**NVT simulations support checkpoint/restart.**

Restart information is stored in the following files:

-   `restart.chk`
-   `restart_meta.json`

If both files are present, the program automatically loads the
checkpoint and resumes the production simulation from the saved step.

To start a completely new simulation, delete both checkpoint files
before running the program again.

## Output Files

Depending on the settings in `nvt_run.py` or `npt_run.py`, the program
can generate the following files:

-   `config.xyz` --- Coordinate trajectory for all particles, with
    coordinates in ångströms (Å).
-   `vel.xyz` --- Velocity trajectory for all particles, with velocities
    in ångströms per picosecond (Å/ps).
-   `.csv` files --- Thermodynamic quantities and SCF iteration counts.

## Performance

On a local Windows computer equipped with an NVIDIA GeForce RTX 4070
GPU, the current simulation setup runs at approximately **1.2 ns/day**.
