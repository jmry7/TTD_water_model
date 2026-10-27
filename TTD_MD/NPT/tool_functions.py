import math
import os

import numpy as np
import openmm as mm
import openmm.unit as unit


R_KJ_PER_MOL_K = 8.31446261815324e-3
BAR_TO_KJ_PER_MOL_NM3 = 0.0602214076


def set_core_velocities(context, system, is_drude, temperature, seed=33):
    """Initialize only core velocities, remove COM motion, and rescale."""
    rng = np.random.default_rng(seed)
    masses = np.array([
        system.getParticleMass(i).value_in_unit(unit.dalton)
        for i in range(system.getNumParticles())
    ])
    core = ~is_drude
    target_temperature = temperature.value_in_unit(unit.kelvin)

    velocities = np.zeros((system.getNumParticles(), 3), dtype=np.float64)
    velocities[core] = rng.normal(size=(np.count_nonzero(core), 3))
    velocities[core] *= np.sqrt(
        R_KJ_PER_MOL_K * target_temperature / masses[core]
    )[:, None]

    total_core_mass = np.sum(masses[core])
    com_velocity = np.sum(
        masses[core, None] * velocities[core], axis=0
    ) / total_core_mass
    velocities[core] -= com_velocity

    dof_core = 3 * np.count_nonzero(core) - 3
    kinetic = 0.5 * np.sum(masses[core, None] * velocities[core] ** 2)
    current_temperature = 2.0 * kinetic / (dof_core * R_KJ_PER_MOL_K)
    velocities[core] *= math.sqrt(target_temperature / current_temperature)
    context.setVelocities(velocities * unit.nanometer / unit.picosecond)


def reset_drudes_to_cores(context, system):
    """Place every Drude site on its parent before the initial SCF solve."""
    drude_to_core = {}
    for force in system.getForces():
        if isinstance(force, mm.DrudeForce):
            for index in range(force.getNumParticles()):
                params = force.getParticleParameters(index)
                drude_to_core[int(params[0])] = int(params[1])

    positions = context.getState(getPositions=True).getPositions(asNumpy=True)
    positions_nm = positions.value_in_unit(unit.nanometer)
    for drude_index, core_index in drude_to_core.items():
        positions_nm[drude_index] = positions_nm[core_index]
    context.setPositions(positions_nm * unit.nanometer)


def select_and_step(compound, index):
    compound.setCurrentIntegrator(index)
    compound.step(1)


def transfer_nhc_state(source, target, chain_length):

    """Transfer the NVT thermostat chain to the NPT thermostat.

    Q_1 changes slightly because the NPT thermostat also controls the
    barostat momentum.  Preserve its conjugate momentum Q_1*v_eta_1 across
    the hand-off; the remaining thermostat masses are unchanged.
    """
    for index in range(1, chain_length + 1):
        eta_name = f"eta_{index}"
        velocity_name = f"v_eta_{index}"
        thermostat_velocity = source.getGlobalVariableByName(velocity_name)

        source_mass = source.getGlobalVariableByName(f"Q_{index}")
        target_mass = target.getGlobalVariableByName(f"Q_{index}")
        thermostat_velocity *= source_mass / target_mass

        target.setGlobalVariableByName(
            eta_name, source.getGlobalVariableByName(eta_name)
        )
        target.setGlobalVariableByName(velocity_name, thermostat_velocity)


def get_box_nm(context):
    state = context.getState()
    return state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer)


def get_volume_nm3(context):
    return abs(float(np.linalg.det(get_box_nm(context))))


def get_core_kinetic_energy(context, masses, core):
    velocities = context.getState(getVelocities=True).getVelocities(asNumpy=True)
    velocities_nm_ps = velocities.value_in_unit(
        unit.nanometer / unit.picosecond
    )
    return 0.5 * float(
        np.sum(masses[core, None] * velocities_nm_ps[core] ** 2)
    )


def get_thermo_observables(context, masses, core, dof_core):
    """Return box potential energy, core temperature, and box volume."""
    state = context.getState(getEnergy=True, getVelocities=True)
    potential = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    velocities = state.getVelocities(asNumpy=True).value_in_unit(
        unit.nanometer / unit.picosecond
    )
    kinetic = 0.5 * float(np.sum(masses[core, None] * velocities[core] ** 2))
    temperature = 2.0 * kinetic / (dof_core * R_KJ_PER_MOL_K)
    box = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer)
    volume = abs(float(np.linalg.det(box)))
    return potential, temperature, volume


def pressure_in_bar(pressure_force, context):
    pressure = pressure_force.computeCurrentPressure(context)
    if unit.is_quantity(pressure):
        return pressure.value_in_unit(unit.bar)
    return float(pressure)


def get_config_pressure_work(
        pressure_force,
        context,
        masses,
        core,
        target_pressure_bar,
):
    """Return ``3*V*(P_config-P_external)`` in kJ/mol.

    OpenMM's instantaneous atomic pressure contains the kinetic contribution
    ``2*K/(3*V)``.  Appendix B keeps only the position-dependent pressure term
    fixed during each NHCPISOINT half step and updates the kinetic term after
    every particle-velocity scaling.  Subtracting ``2*K`` from ``3*V*P``
    supplies exactly that fixed configuration contribution.
    """
    pressure_bar = pressure_in_bar(pressure_force, context)
    volume_nm3 = get_volume_nm3(context)
    kinetic_kj_mol = get_core_kinetic_energy(context, masses, core)
    total_pressure_work = (
        3.0
        * volume_nm3
        * (pressure_bar - target_pressure_bar)
        * BAR_TO_KJ_PER_MOL_NM3
    )
    return total_pressure_work - 2.0 * kinetic_kj_mol


def coupled_coordinate_box_drift(context, core, barostat_velocity, dt_ps):
    """Exact central drift for fixed isotropic barostat velocity."""
    state = context.getState(getPositions=True, getVelocities=True)
    positions = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    velocities = state.getVelocities(asNumpy=True).value_in_unit(
        unit.nanometer / unit.picosecond
    )
    box = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer)

    x = barostat_velocity * dt_ps
    growth = math.exp(x)
    if abs(x) < 1.0e-8:
        exprel = 1.0 + 0.5 * x + x * x / 6.0
    else:
        exprel = math.expm1(x) / x

    # All sites follow the homogeneous box dilation.  Only physical core sites
    # have a velocity drift; Drude sites are reoptimized immediately by SCF.
    new_positions = growth * positions
    new_positions[core] += dt_ps * exprel * velocities[core]
    new_box = growth * box

    context.setPeriodicBoxVectors(
        mm.Vec3(*new_box[0]) * unit.nanometer,
        mm.Vec3(*new_box[1]) * unit.nanometer,
        mm.Vec3(*new_box[2]) * unit.nanometer,
    )
    context.setPositions(new_positions * unit.nanometer)


def set_physical_clock(context, step, dt_ps):
    """Hide the bookkeeping steps taken by the split zero-time operators."""
    context.setTime(step * dt_ps * unit.picosecond)
    if hasattr(context, "setStepCount"):
        context.setStepCount(step)


def create_platform():
    requested = os.environ.get("TTD_NPT_PLATFORM")
    names = (requested,) if requested else ("CUDA", "OpenCL", "CPU")
    for name in names:
        try:
            platform = mm.Platform.getPlatformByName(name)
            properties = {"Precision": "double"} if name != "CPU" else {}
            return platform, properties
        except Exception:
            continue
    raise RuntimeError("No usable OpenMM platform was found")


