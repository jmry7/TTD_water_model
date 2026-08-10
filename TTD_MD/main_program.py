import openmm
from openmm.app import *
from openmm import *
import openmm.unit as unit
import sys
import numpy as np
import os
import json
import re
from record import TrajectoryReporter, ConservedThermoReporter, ClockReporter
from drude_ff import apply_polarizable_water
from nhc_integrators import setup_scf_drude_env


def remove_com_velocity(simulation, system, mask_drude):
    state = simulation.context.getState(getVelocities=True)
    vels = state.getVelocities(asNumpy=True)
    v_np = vels.value_in_unit(unit.nanometers / unit.picoseconds)
    masses = np.array([system.getParticleMass(i).value_in_unit(unit.amu) for i in range(system.getNumParticles())])

    total_momentum = np.zeros(3)
    core_mass = 0.0

    # Set all Drude velocities to zero and accumulate the total momentum of the core particles
    for i, mask in enumerate(mask_drude):
        if mask[0] > 0.5:
            v_np[i] = np.array([0.0, 0.0, 0.0])
        else:
            total_momentum += masses[i] * v_np[i]
            core_mass += masses[i]

    # Calculate and remove the center-of-mass velocity
    v_com = total_momentum / core_mass
    for i, mask in enumerate(mask_drude):
        if mask[0] < 0.5:
            v_np[i] -= v_com

    simulation.context.setVelocities(v_np * (unit.nanometers / unit.picoseconds))
    return v_com

# OpenMM provides Context.setVelocitiesToTemperature() for assigning initial velocities.
# However, a Windows pip build previously produced identical x, y, and z velocity
# components for each particle; see https://github.com/openmm/openmm/issues/5335.
# The bug has now been fixed.
def set_core_velocities_to_temperature(simulation, system, mask_drude, temperature, seed=123):
    """
    Assign Maxwell-Boltzmann initial velocities only to the core particles;
    set all Drude particle velocities to zero;
    then remove the core center-of-mass velocity and rescale to the target temperature.
    """
    rng = np.random.default_rng(seed)

    R = unit.MOLAR_GAS_CONSTANT_R.value_in_unit(
        unit.kilojoule_per_mole / unit.kelvin
    )
    T = temperature.value_in_unit(unit.kelvin)

    n_particles = system.getNumParticles()

    masses = np.array([
        system.getParticleMass(i).value_in_unit(unit.dalton)
        for i in range(n_particles)
    ])

    is_drude = np.array([mask[0] > 0.5 for mask in mask_drude])
    is_core = ~is_drude

    v_np = np.zeros((n_particles, 3), dtype=np.float64)

    # 1 dalton * (nm/ps)^2 = 1 kJ/mol
    for i in range(n_particles):
        if is_core[i]:
            sigma = np.sqrt(R * T / masses[i])
            v_np[i, :] = rng.normal(0.0, sigma, size=3)
        else:
            v_np[i, :] = 0.0

    # Remove the center-of-mass velocity of the core particles
    core_masses = masses[is_core]
    total_core_mass = np.sum(core_masses)

    v_com = np.sum(core_masses[:, None] * v_np[is_core], axis=0) / total_core_mass
    v_np[is_core] -= v_com

    # Rescale to the target temperature after removing the center-of-mass velocity
    dof_core = 3 * np.sum(is_core) - 3

    kinetic = 0.5 * np.sum(core_masses[:, None] * v_np[is_core] ** 2)
    current_temp = 2.0 * kinetic / (dof_core * R)

    scale = np.sqrt(T / current_temp)
    v_np[is_core] *= scale

    # Remove any residual center-of-mass velocity caused by numerical roundoff after rescaling
    v_com_after = np.sum(core_masses[:, None] * v_np[is_core], axis=0) / total_core_mass
    v_np[is_core] -= v_com_after

    simulation.context.setVelocities(v_np * unit.nanometer / unit.picosecond)

    return v_com, current_temp

def save_checkpoint_with_meta(simulation, checkpoint_file, restart_meta_file, total_steps, report_steps, checkpoint_interval):
    checkpoint_tmp = checkpoint_file + ".tmp"
    meta_tmp = restart_meta_file + ".tmp"

    with open(checkpoint_tmp, "wb") as f:
        f.write(simulation.context.createCheckpoint())
    os.replace(checkpoint_tmp, checkpoint_file)

    state = simulation.context.getState()
    time_ps = state.getTime().value_in_unit(unit.picosecond)
    meta = {
        "step": int(simulation.currentStep),
        "time_ps": float(time_ps),
        "total_steps": int(total_steps),
        "report_steps": int(report_steps),
        "checkpoint_interval": int(checkpoint_interval)
    }
    with open(meta_tmp, "w") as f:
        json.dump(meta, f, indent=2)
    os.replace(meta_tmp, restart_meta_file)


def load_restart_step(restart_meta_file):
    with open(restart_meta_file, "r") as f:
        meta = json.load(f)
    return int(meta["step"])


def truncate_xyz_to_step(filename, max_step):
    if not os.path.exists(filename):
        return

    kept_lines = []
    with open(filename, "r", errors="ignore") as f:
        while True:
            header = f.readline()
            if not header:
                break

            try:
                n_atoms = int(header.strip())
            except ValueError:
                break

            comment = f.readline()
            if not comment:
                break

            atom_lines = []
            complete = True
            for _ in range(n_atoms):
                line = f.readline()
                if not line:
                    complete = False
                    break
                atom_lines.append(line)
            if not complete:
                break

            match = re.search(r"Step:\s*(\d+)|Velocity Step:\s*(\d+)", comment)
            if match:
                step = int(match.group(1) or match.group(2))
                if step <= max_step:
                    kept_lines.append(header)
                    kept_lines.append(comment)
                    kept_lines.extend(atom_lines)
            else:
                kept_lines.append(header)
                kept_lines.append(comment)
                kept_lines.extend(atom_lines)

    with open(filename, "w") as f:
        f.writelines(kept_lines)


def truncate_table_to_step(filename, max_step):
    if not os.path.exists(filename):
        return

    kept_lines = []
    with open(filename, "r", errors="ignore") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                kept_lines.append(line)
                continue

            first_field = stripped.split(",", 1)[0].strip().strip('"')
            if first_field.startswith("#"):
                kept_lines.append(line)
                continue

            try:
                step = int(float(first_field))
            except ValueError:
                kept_lines.append(line)
                continue

            if step <= max_step:
                kept_lines.append(line)

    with open(filename, "w") as f:
        f.writelines(kept_lines)


def prepare_output_files_for_restart(restart_step):
    truncate_xyz_to_step("config.xyz", restart_step)
    truncate_xyz_to_step("vel.xyz", restart_step)
    truncate_table_to_step("thermo_state.csv", restart_step)
    truncate_table_to_step("md.log", restart_step)


def run_production_with_checkpoints(simulation, total_steps, checkpoint_interval, checkpoint_file, restart_meta_file, report_steps):
    while simulation.currentStep < total_steps:
        steps_to_run = min(checkpoint_interval, total_steps - simulation.currentStep)
        simulation.step(steps_to_run)
        save_checkpoint_with_meta(
            simulation=simulation,
            checkpoint_file=checkpoint_file,
            restart_meta_file=restart_meta_file,
            total_steps=total_steps,
            report_steps=report_steps,
            checkpoint_interval=checkpoint_interval
        )
        print(f"Checkpoint saved at production step {simulation.currentStep}")

# ==========================================
# Main program
# ==========================================
input_pdb = 'water_6p.pdb'
log_file = 'md.log'

checkpoint_file = 'restart.chk'
restart_meta_file = 'restart_meta.json'
checkpoint_interval = 10000

dt = 0.4 * unit.femtoseconds # Time step: 0.4 fs
total_steps = 3* 2500000  # Number of production steps (3 ns)
equil_time = 100.0 * unit.picoseconds # Equilibration time (100.0 ps)
equil_steps = int(equil_time / dt)   # Number of equilibration steps (250000 steps)
report_steps = 10

if checkpoint_interval % report_steps != 0:
    raise ValueError(
        "checkpoint_interval must be an integer multiple of report_steps; "
        "otherwise, the output files may not remain aligned after restarting."
    )

pdb = PDBFile(input_pdb)
topology = pdb.topology
system = System()
box_vectors = [
                  Vec3(1.864, 0, 0),
                  Vec3(0, 1.864, 0),
                  Vec3(0, 0, 1.864)
              ] * unit.nanometers
system.setDefaultPeriodicBoxVectors(*box_vectors)
topology.setPeriodicBoxVectors(box_vectors)

mass_O = 15.99943 * unit.amu
mass_H = 1.007947 * unit.amu
mass_Drude = 0.4 * unit.amu

for atom in topology.atoms():
    if atom.name == 'O':
        system.addParticle(mass_O)
    elif atom.name in ['H1', 'H2']:
        system.addParticle(mass_H)
    elif atom.name in ['DO', 'DH1', 'DH2']:
        system.addParticle(mass_Drude)

system = apply_polarizable_water(system, topology)


print("Configuring the SCF integrator environment...")
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
    # ==========================================
    # Stage 1: Energy minimization and initialization
    # ==========================================
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
    # Stage 2: System equilibration (100 ps)
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
    # 1. Retrieve the latest state from the current simulation context (getPositions=True must be specified)
    current_state = simulation.context.getState(getPositions=True)
    current_positions = current_state.getPositions()

    # 2. Write the topology and latest coordinates to a PDB file
    with open('equilibrated_216.pdb', 'w') as f:
        PDBFile.writeFile(topology, current_positions, f)
    print("The equilibrated structure has been saved as equilibrated_216.pdb")

    # ==========================================
    # Stage 3: Second momentum removal and production-run preparation
    # ==========================================
    print("Performing a second momentum removal before the production run...")
    v_com_eq = remove_com_velocity(simulation, system, mask_drude)
    print(f"The center-of-mass drift accumulated during equilibration has been removed (velocity correction: {np.round(v_com_eq, 6)} nm/ps)")

    # Reset the step count and time so that the production trajectory starts at step 0 and 0 ps
    simulation.currentStep = 0
    simulation.context.setTime(0.0)
    simulation.reporters.clear()

print("Attaching production log, trajectory, and data reporters...")
simulation.reporters.append(StateDataReporter(sys.stdout, 1000, step=True, potentialEnergy=True, temperature=True, speed=True))
simulation.reporters.append(StateDataReporter(log_file, 1000, step=True, potentialEnergy=True, temperature=True, append=restart))
simulation.reporters.append(TrajectoryReporter('config.xyz', 'vel.xyz', report_steps, topology, append=restart))
simulation.reporters.append(ConservedThermoReporter('thermo_state.csv', 100, target_temp, append=restart))
simulation.reporters.append(ClockReporter(1000))

# ==========================================
# Stage 4: Production run
# ==========================================
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
