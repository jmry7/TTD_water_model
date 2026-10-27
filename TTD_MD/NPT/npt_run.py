import csv
import os
from pathlib import Path

import numpy as np
import openmm as mm
import openmm.unit as unit
from openmm.app import PDBFile, Simulation

import nhc_integrators
from drude_ff import apply_polarizable_water
from drude_scf import setup_drude_scf_env
from half_nhcp import create_half_nhcp_integrator
from velocity_position_scf import create_momentum_half_integrator
from record import TrajectoryWriter
from tool_functions import (
    R_KJ_PER_MOL_K,
    set_core_velocities,
    reset_drudes_to_cores,
    select_and_step,
    transfer_nhc_state,
    get_thermo_observables,
    pressure_in_bar,
    get_config_pressure_work,
    coupled_coordinate_box_drift,
    set_physical_clock,
    create_platform,
)


input_pdb = 'water_512.pdb'
nvt_output_csv = "nvt_warmup.csv"
output_csv = "npt_run.csv"
coord_file = "config.xyz"
vel_file = "vel.xyz"

temperature = 298.15 * unit.kelvin
target_pressure = 1.01325 * unit.bar
thermostat_tau = 0.5 * unit.picoseconds
barostat_tau = 1.0 * unit.picoseconds
scf_tolerance_per_drude = 1.0e-6
report_interval = 50

dt = 0.4 * unit.femtoseconds # Time step: 0.4 fs
nvt_time = 100.0 * unit.picoseconds
nvt_steps = int(nvt_time / dt)
npt_time = 3000.0 * unit.picoseconds
total_steps = int(npt_time / dt)
report_steps = 10

dt_ps = dt.value_in_unit(unit.picosecond)
temperature_k = temperature.value_in_unit(unit.kelvin)
target_pressure_bar = target_pressure.value_in_unit(unit.bar)
barostat_tau_ps = barostat_tau.value_in_unit(unit.picosecond)


pdb = PDBFile(str(input_pdb))
topology = pdb.topology
box_vectors = topology.getPeriodicBoxVectors()
system = mm.System()
system.setDefaultPeriodicBoxVectors(*box_vectors)

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

pressure_force = mm.MonteCarloBarostat(target_pressure, temperature, 0)
if not (
        hasattr(pressure_force, "setScaleMoleculesAsRigid")
        and hasattr(pressure_force, "computeCurrentPressure")
):
    raise RuntimeError(
        "Requires an OpenMM version that "
        "supports atomic-virial pressure evaluation."
    )
pressure_force.setScaleMoleculesAsRigid(False)
system.addForce(pressure_force)

system, scf_integrator, mask_drude, mask_core = setup_drude_scf_env(
        system=system,
        temperature=temperature,
        scf_tol=scf_tolerance_per_drude,
        dt=dt,
        ghost_mass=0.4 * unit.dalton,
        scf_alpha=5.0e-6,
        scf_max_disp=1.0e-3,
        max_iters=500,
    )
scf_integrator.setStepSize(0.0)

is_drude = np.array([mask[0] > 0.5 for mask in mask_drude])
core = ~is_drude
num_core = int(np.count_nonzero(core))
dof_core = 3 * num_core - 3
num_drudes = int(np.count_nonzero(is_drude))
total_scf_tolerance = num_drudes * scf_tolerance_per_drude
kT = R_KJ_PER_MOL_K * temperature_k
barostat_mass = (dof_core + 3) * kT * barostat_tau_ps ** 2

nvt_integrator = nhc_integrators.create_fortran_strict_nhc_scf_integrator(
        temperature=temperature,
        dof_core=dof_core,
        scf_tol=total_scf_tolerance,
        timestep=dt,
        nh_tau=thermostat_tau,
        scf_alpha=5.0e-6,
        scf_max_disp=1.0e-3,
        max_iters=500,
        nnos=5,
        nys=5,
    )
npt_nhc_integrator = create_half_nhcp_integrator(
        temperature=temperature,
        dof_core=dof_core,
        barostat_mass=barostat_mass,
        timestep=dt,
        nh_tau=thermostat_tau,
        nnos=5,
        nys=5,
    )
momentum_integrator = create_momentum_half_integrator(
        dof_core=dof_core,
        timestep=dt,
    )

compound = mm.CompoundIntegrator()
scf_index = compound.addIntegrator(scf_integrator)
nvt_index = compound.addIntegrator(nvt_integrator)
npt_nhc_index = compound.addIntegrator(npt_nhc_integrator)
momentum_index = compound.addIntegrator(momentum_integrator)

platform, properties = create_platform()
print(f"Platform: {platform.getName()} {properties}")
simulation = Simulation(topology, system, compound, platform, properties)
context = simulation.context
context.setPositions(pdb.positions)
masses = np.array([
    system.getParticleMass(i).value_in_unit(unit.dalton)
    for i in range(system.getNumParticles())
])

simulation.minimizeEnergy(maxIterations=5000)
print("Energy minimization completed")

scf_integrator.setPerDofVariableByName("is_drude", mask_drude)
scf_integrator.setPerDofVariableByName("is_core", mask_core)
nvt_integrator.setPerDofVariableByName("is_drude", mask_drude)
nvt_integrator.setPerDofVariableByName("is_core", mask_core)

npt_nhc_integrator.setPerDofVariableByName("is_drude", mask_drude)
npt_nhc_integrator.setPerDofVariableByName("is_core", mask_core)
momentum_integrator.setPerDofVariableByName("is_core", mask_core)

set_core_velocities(context, system, is_drude, temperature)
reset_drudes_to_cores(context, system)

select_and_step(compound, scf_index)
set_physical_clock(context, 0, dt_ps)
initial_bad = int(round(
    scf_integrator.getGlobalVariableByName("n_bad_components_step")
))
print(
    "Initial SCF:",
    f"iterations={int(scf_integrator.getGlobalVariableByName('iter_step'))}",
    f"converged={initial_bad == 0}",
)
if initial_bad:
    initial_ssq = scf_integrator.getGlobalVariableByName("ssq_step")
    raise RuntimeError(
        "Initial Drude SCF did not converge; refusing to start NVT "
        f"(force-squared sum={initial_ssq:.6g}, "
        f"threshold={total_scf_tolerance:.6g})."
    )

nvt_integrator.setGlobalVariableByName("init_done", 1.0)

nvt_initial_potential, nvt_initial_temperature, _ = get_thermo_observables(
    context, masses, core, dof_core
)

print(f"Starting {nvt_steps} NVT warm-up steps")

with open(nvt_output_csv, "w", newline="") as nvt_handle:
    nvt_writer = csv.writer(nvt_handle)
    nvt_writer.writerow([
        "Step", "Time(ps)", "PE(kJ/mol)", "T_Core(K)", "SCF_Iters",
    ])
    nvt_writer.writerow([
        0,
        0.0,
        nvt_initial_potential,
        nvt_initial_temperature,
        int(scf_integrator.getGlobalVariableByName("iter_step")),
    ])

    for warm_step in range(1, nvt_steps + 1):
        # One call performs a complete NHC/Verlet/SCF NVT step using the
        # established implementation in nhc_integrators.py.
        select_and_step(compound, nvt_index)
        scf_bad = int(round(
            nvt_integrator.getGlobalVariableByName(
                "n_bad_components_step"
            )
        ))
        scf_iterations = int(
            nvt_integrator.getGlobalVariableByName("iter_step")
        )

        if scf_bad:
            scf_ssq = nvt_integrator.getGlobalVariableByName("ssq_step")
            raise RuntimeError(
                f"NVT Drude SCF did not converge at warm-up step "
                f"{warm_step} after {scf_iterations} iterations "
                f"(force-squared sum={scf_ssq:.6g}, "
                f"threshold={total_scf_tolerance:.6g}); aborting "
                "before the state can become NaN."
            )

        if warm_step == 1 or warm_step % report_interval == 0:
            potential_warm, temperature_warm, _ = get_thermo_observables(
                context, masses, core, dof_core
            )
            nvt_writer.writerow([
                warm_step,
                warm_step * dt_ps,
                potential_warm,
                temperature_warm,
                scf_iterations,
            ])
            nvt_handle.flush()
            print(
                f"NVT step={warm_step:7d}  "
                f"T={temperature_warm:9.3f} K  "
                f"SCF={scf_iterations:3d}  "
                f"PE={potential_warm:13.3f} kJ/mol"
            )

print("NVT warm-up finished")
print(f"NVT thermodynamic output: {nvt_output_csv}")

transfer_nhc_state(nvt_integrator, npt_nhc_integrator, chain_length=5)
print("Transferred the equilibrated NVT state to the NPT state")

set_physical_clock(context, 0, dt_ps)

barostat_velocity = 0.0
with open(output_csv, "w", newline="") as handle, TrajectoryWriter(
    topology, coord_file, vel_file
) as trajectory:
    writer = csv.writer(handle)
    writer.writerow([
        "Step", "Time(ps)", "PE(kJ/mol)", "T_Core(K)",
        "Pressure(bar)", "Volume(nm^3)", "Box_Length(nm)", "SCF_Iters",
    ])
    handle.flush()
    for step in range(1, total_steps + 1):
        config_pressure_work = get_config_pressure_work(
            pressure_force,
            context,
            masses,
            core,
            target_pressure_bar,
        )
        npt_nhc_integrator.setGlobalVariableByName(
            "barostat_velocity", barostat_velocity
        )
        npt_nhc_integrator.setGlobalVariableByName(
            "config_pressure_work", config_pressure_work
        )
        select_and_step(compound, npt_nhc_index)
        barostat_velocity = npt_nhc_integrator.getGlobalVariableByName(
            "barostat_velocity"
        )
        select_and_step(compound, momentum_index)

        coupled_coordinate_box_drift(
            context, core, barostat_velocity, dt_ps
        )
        select_and_step(compound, scf_index)
        scf_bad = int(round(
            scf_integrator.getGlobalVariableByName(
                "n_bad_components_step"
            )
        ))
        scf_iterations = int(
            scf_integrator.getGlobalVariableByName("iter_step")
        )
        if scf_bad:
            scf_ssq = scf_integrator.getGlobalVariableByName("ssq_step")
            raise RuntimeError(
                f"NPT Drude SCF did not converge at step {step} "
                f"after {scf_iterations} iterations "
                f"(force-squared sum={scf_ssq:.6g}, "
                f"threshold={total_scf_tolerance:.6g}); aborting "
                "because Appendix-B propagation requires the new force."
            )

        select_and_step(compound, momentum_index)

        config_pressure_work = get_config_pressure_work(
            pressure_force,
            context,
            masses,
            core,
            target_pressure_bar,
        )
        npt_nhc_integrator.setGlobalVariableByName(
            "barostat_velocity", barostat_velocity
        )
        npt_nhc_integrator.setGlobalVariableByName(
            "config_pressure_work", config_pressure_work
        )
        select_and_step(compound, npt_nhc_index)
        barostat_velocity = npt_nhc_integrator.getGlobalVariableByName(
            "barostat_velocity"
        )
        set_physical_clock(context, step, dt_ps)

        if step == 1 or step % report_interval == 0:
            potential_report, temperature_report, volume_report = (
                get_thermo_observables(context, masses, core, dof_core)
            )
            pressure_report = pressure_in_bar(pressure_force, context)
            box_length = volume_report ** (1.0 / 3.0)
            writer.writerow([
                step,
                step * dt_ps,
                potential_report,
                temperature_report,
                pressure_report,
                volume_report,
                box_length,
                scf_iterations,
            ])
            handle.flush()
            trajectory.write(context, step)
            print(
                f"step={step:7d}  T={temperature_report:9.3f} K  "
                f"P={pressure_report:11.3f} bar  "
                f"V={volume_report:9.5f} nm^3  "
                f"SCF={scf_iterations:3d}  "
                f"PE={potential_report:13.3f} kJ/mol"
            )

final_state = context.getState(getPositions=True)
final_positions = final_state.getPositions()
topology.setPeriodicBoxVectors(final_state.getPeriodicBoxVectors())
print("🎉🎉🎉 Simulation completed successfully! 🎉🎉🎉")
