import os
import json
import re
import numpy as np
import openmm.unit as unit


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
