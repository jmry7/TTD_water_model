import openmm
from openmm.app import *
from openmm import *
import openmm.unit as unit
import sys
import numpy as np
import os
from record import TrajectoryReporter, ConservedThermoReporter, ClockReporter
from drude_ff import apply_polarizable_water
from nhc_integrators import setup_scf_drude_env
from tool_functions import (
    remove_com_velocity,
    set_core_velocities_to_temperature,
    load_restart_step,
    prepare_output_files_for_restart,
    run_production_with_checkpoints,
)


input_pdb = 'water_6p.pdb'
log_file = 'md.log'

checkpoint_file = 'restart.chk'
restart_meta_file = 'restart_meta.json'
checkpoint_interval = 10000

dt = 0.4 * unit.femtoseconds # Time step: 0.4 fs
equil_time = 1000.0 * unit.picoseconds # Equilibration time (1 ns)
equil_steps = int(equil_time / dt)   # Number of equilibration steps (2500000 steps)
total_time = 3000.0 * unit.picoseconds # Production time (3 ns)
total_steps = int(total_time / dt) # Number of production steps (7500000 steps)
report_steps = 10

if checkpoint_interval % report_steps != 0:
    raise ValueError(
        "checkpoint_interval must be an integer multiple of report_steps. "
    )

pdb = PDBFile(input_pdb)
topology = pdb.topology
system = System()
box_vectors = topology.getPeriodicBoxVectors()
system.setDefaultPeriodicBoxVectors(*box_vectors)
topology.setPeriodicBoxVectors(box_vectors)

mass_O = 15.99943 * unit.amu
mass_H = 1.007947 * unit.amu
# The masses assigned to Drude particles are used only for topology construction
# and are not used by the integrator, as the Drude particle positions are determined by SCF.
mass_Drude = 0.4 * unit.amu

for atom in topology.atoms():
    if atom.name == 'O':
        system.addParticle(mass_O)
    elif atom.name in ['H1', 'H2']:
        system.addParticle(mass_H)
    elif atom.name in ['DO', 'DH1', 'DH2']:
        system.addParticle(mass_Drude)

system = apply_polarizable_water(system, topology)
target_temp = 298.15
temperature = target_temp * unit.kelvin

# Define the SCF convergence criterion
# 1(kJ/mol/nm)^2=1e-6(A*a.u./(10fs)^2)^2
tolerance_per_drude = 1.0e-6 # (kJ/mol/nm)^2

system, integrator, mask_drude, mask_core = setup_scf_drude_env(
    system=system,
    temperature=temperature,
    scf_tol=tolerance_per_drude,
    dt=dt,
    ghost_mass=0.4 * unit.dalton,
    scf_alpha=5.0e-6,
    scf_max_disp=1.0e-3,
    max_iters=500,
)

try:
    platform = Platform.getPlatformByName('CUDA')
except Exception as e:
    print("CUDA was not detected. Falling back to OpenCL...")
    platform = Platform.getPlatformByName('OpenCL')

properties = {'Precision': 'double'}
print(f"Using the {platform.getName()} platform for acceleration with {properties['Precision']} precision")

simulation = Simulation(topology, system, integrator, platform, properties)
simulation.context.setPositions(pdb.positions)

integrator.setPerDofVariableByName("is_drude", mask_drude)
integrator.setPerDofVariableByName("is_core", mask_core)

restart = os.path.exists(checkpoint_file) and os.path.exists(restart_meta_file)
if os.path.exists(checkpoint_file) != os.path.exists(restart_meta_file):
    raise RuntimeError(
        "restart.chk and restart_meta.json are inconsistent. "
        "Please keep both files or delete both files."
    )

if restart:
    restart_step = load_restart_step(restart_meta_file)
    print(f"Restart files detected. Continuing the production run from step {restart_step}...")

    with open(checkpoint_file, "rb") as f:
        simulation.context.loadCheckpoint(f.read())

    simulation.currentStep = restart_step
    integrator.setPerDofVariableByName("is_drude", mask_drude)
    integrator.setPerDofVariableByName("is_core", mask_core)

    prepare_output_files_for_restart(restart_step)
    simulation.reporters.clear()

else:
    simulation.minimizeEnergy(maxIterations=5000)
    print("Energy minimization completed")

    print("Initializing Maxwell-Boltzmann velocities for the core particles and removing Drude kinetic energy...")

    v_com_init, temp_before_rescale = set_core_velocities_to_temperature(
        simulation,
        system,
        mask_drude,
        temperature,
        seed=12345
    )

    print("All Drude particle velocities have been set to zero")
    print(f"The initial core center-of-mass drift has been removed; velocity correction: {np.round(v_com_init, 6)} nm/ps")
    print(f"Instantaneous temperature after center-of-mass removal and before rescaling: {temp_before_rescale:.6f} K")

    # Reset the Drude particles to provide an initial SCF guess
    drude_core_map = {}
    for force in system.getForces():
        if isinstance(force, openmm.DrudeForce):
            for i in range(force.getNumParticles()):
                params = force.getParticleParameters(i)
                drude_core_map[params[0]] = params[1]

    state = simulation.context.getState(getPositions=True)
    pos = state.getPositions(asNumpy=True).value_in_unit(unit.nanometers)
    for drude_idx, core_idx in drude_core_map.items():
        pos[drude_idx] = pos[core_idx]
    simulation.context.setPositions(pos * unit.nanometer)

    # ==========================================
    # Equilibration steps
    # ==========================================
    print(f"Starting system equilibration: {equil_steps} steps, duration: {equil_time}")
    # During equilibration, report progress to the terminal without writing to the production log or trajectory files
    simulation.reporters.append(StateDataReporter(sys.stdout, 1000, step=True, potentialEnergy=True, temperature=True, speed=True))
    simulation.reporters.append(StateDataReporter('equil_md.log', 1000, step=True, potentialEnergy=True, temperature=True))
    simulation.reporters.append(ConservedThermoReporter('equil_thermo.csv', 100, target_temp))
    simulation.reporters.append(ClockReporter(1000))


    simulation.step(equil_steps)
    print("Equilibration completed")
    print("Exporting the equilibrated system configuration...")
    current_state = simulation.context.getState(getPositions=True)
    current_positions = current_state.getPositions()
    # Write the topology and latest coordinates to a PDB file
    with open('equilibrated_water.pdb', 'w') as f:
        PDBFile.writeFile(topology, current_positions, f)
    print("The equilibrated structure has been saved as equilibrated_water.pdb")

    # ==========================================
    # Production steps
    # ==========================================
    print("Performing a second momentum removal before the production run...")
    v_com_eq = remove_com_velocity(simulation, system, mask_drude)
    print(f"The center-of-mass drift accumulated during equilibration has been removed (velocity correction: {np.round(v_com_eq, 6)} nm/ps)")

    simulation.currentStep = 0
    simulation.context.setTime(0.0)
    simulation.reporters.clear()

simulation.reporters.append(StateDataReporter(sys.stdout, 1000, step=True, potentialEnergy=True, temperature=True, speed=True))
simulation.reporters.append(StateDataReporter(log_file, 1000, step=True, potentialEnergy=True, temperature=True, append=restart))
simulation.reporters.append(TrajectoryReporter('config.xyz', 'vel.xyz', report_steps, topology, append=restart))
simulation.reporters.append(ConservedThermoReporter('thermo_state.csv', 100, target_temp, append=restart))
simulation.reporters.append(ClockReporter(1000))

remaining_steps = total_steps - simulation.currentStep
print(f"Starting the production MD run: {remaining_steps} steps remaining, {total_steps} total target steps, {remaining_steps * dt.value_in_unit(unit.picoseconds)} ps remaining")
run_production_with_checkpoints(
    simulation=simulation,
    total_steps=total_steps,
    checkpoint_interval=checkpoint_interval,
    checkpoint_file=checkpoint_file,
    restart_meta_file=restart_meta_file,
    report_steps=report_steps
)
print("🎉🎉🎉 Simulation completed successfully! 🎉🎉🎉")
