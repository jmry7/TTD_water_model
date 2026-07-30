# TTD Polarizable Water Model Molecular Dynamics Program

This code package implements molecular dynamics simulations for the TTD  water model using OpenMM. The current version has been tested with OpenMM 8.4.

The program uses a self-consistent-field (SCF) procedure to optimize the Drude-particle positions and a Nosé–Hoover chain thermostat to create an NVT ensemble.

## Files

- `main_program.py`
  Main program for reading the input structure, constructing the simulation system, performing equilibration and production simulations, writing output files, and managing checkpoint/restart operations.

- `drude_ff.py`
  Force-field module defining the interactions of the TTD water model.

- `nhc_integrators.py`
  Integrator module implementing the SCF Drude relaxation, velocity-Verlet propagation, and Nosé–Hoover chain thermostat.

- `record.py`
  Reporter module for writing coordinate and velocity trajectories and monitoring the system temperature.

## Requirements

The program has been tested with:

- Python 3.10
- OpenMM 8.4

A CUDA-compatible NVIDIA GPU is recommended. The program can also attempt to use OpenCL if CUDA is unavailable.

## Input Structure

The main program reads an input structure in PDB format.

Each water molecule must contain six interaction sites:

- `O`
- `H1`
- `H2`
- `DO`
- `DH1`
- `DH2`

The residue name must be `HOH`.

In the initial PDB structure, each Drude particle should overlap with its corresponding core site.

## Running the Program

Place the Python modules and the input PDB file in the same directory, and run:

```bash
python main_program.py
```

The input PDB filename and the main simulation parameters can be modified in `main_program.py`.

## Checkpoint and Restart

The program supports restarting from checkpoints.

The restart information is stored in:

- `restart.chk`
- `restart_meta.json`

If both files are present, the program automatically loads the checkpoint and continues the production simulation from the saved step.

To start a completely new simulation, must delete both checkpoint files before running the program again.

## Output Files

Depending on the settings in `main_program.py`, the program produces files including:

- `config.xyz`: coordinate trajectory of all particles, with coordinates recorded in angstroms (Å)
- `vel.xyz`: velocity trajectory of all particles, with velocities recorded in angstroms per picosecond (Å/ps)
- `thermo_state.csv`: thermodynamic quantities and SCF iteration counts
- `md.log`: production simulation log
- `equil_md.log`: equilibration log
- `equilibrated_216.pdb`: structure saved after equilibration
- `restart.chk`: checkpoint
- `restart_meta.json`: restart metadata

## Performance

On a local Windows computer equipped with an NVIDIA GeForce RTX 4070 GPU, the present simulation setup runs at approximately 1.2 ns/day.
