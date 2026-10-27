import openmm as mm
from openmm import CustomIntegrator
import openmm.unit as unit


def drude_scf_integrator(
        temperature,
        scf_tol,
        scf_alpha=5.0e-6,
        scf_max_disp=1.0e-3,
        max_iters=500,
        timestep=0.4 * unit.femtoseconds,
):
    dt_ps = timestep.value_in_unit(unit.picoseconds)
    T_k = temperature.value_in_unit(unit.kelvin)

    integrator = CustomIntegrator(dt_ps)

    # ==== Register system variables ====
    integrator.addPerDofVariable("is_drude", 0)
    integrator.addPerDofVariable("is_core", 0)

    # SCF variables
    # Scf_tol is the total tolerance for all Drude particles:
    # sum_i |F_i^D|^2 < num_drudes * tolerance_per_drude
    # The conversion from per-Drude tolerance to total tolerance is done in setup_scf_drude_env().
    integrator.addGlobalVariable("scf_tol", scf_tol)
    integrator.addGlobalVariable("scf_alpha", scf_alpha)
    integrator.addGlobalVariable("scf_max_disp", scf_max_disp)
    integrator.addGlobalVariable("max_iters", max_iters)

    # Current SCF diagnostics
    integrator.addGlobalVariable("iter", 0)
    integrator.addGlobalVariable("drude_force_sq", 1000.0)
    integrator.addGlobalVariable("n_bad_components", 1)

    # Initialization SCF diagnostics
    integrator.addGlobalVariable("init_done", 0)
    integrator.addGlobalVariable("iter_init", 0)
    integrator.addGlobalVariable("ssq_init", 1000.0)
    integrator.addGlobalVariable("n_bad_components_init", 0)

    # Production-step SCF diagnostics
    integrator.addGlobalVariable("iter_step", 0)
    integrator.addGlobalVariable("ssq_step", 1000.0)
    integrator.addGlobalVariable("n_bad_components_step", 0)

    # Optional old position, useful if constraints are later used
    integrator.addPerDofVariable("x_old", 0)

    def add_scf_block(intg, iter_out, ssq_out, nbad_out):
        """
        Relax Drude particles by steepest descent until the average squared
        Drude force per Drude particle satisfies

            (sum_i |F_i^D|^2) / N_D < tolerance_per_drude

        Actually, the stopping condition used here is

            drude_force_sq = sum_i |F_i^D|^2 < scf_tol

        in which scf_tol = N_D * tolerance_per_drude.
        """
        intg.addUpdateContextState()
        intg.addComputeGlobal("iter", "0")

        # Total sum of Drude force squares over all Drude coordinates:
        # drude_force_sq = sum_i (Fx_i^2 + Fy_i^2 + Fz_i^2)
        intg.addComputeSum("drude_force_sq", "f*f*is_drude")

        # Kept for diagnostic compatibility.
        # It is 1 if the total Drude force-squared sum is still above tolerance and 0 if converged.
        intg.addComputeGlobal("n_bad_components", "step(drude_force_sq - scf_tol)")

        intg.beginWhileBlock(
            "step(drude_force_sq - scf_tol) * step(max_iters - iter - 1) > 0"
        )
        intg.addComputePerDof(
            "x",
            "x + is_drude*max(-scf_max_disp, min(scf_max_disp, scf_alpha*f))"
        )
        intg.addUpdateContextState()

        intg.addComputeSum("drude_force_sq", "f*f*is_drude")
        intg.addComputeGlobal("n_bad_components", "step(drude_force_sq - scf_tol)")

        intg.addComputeGlobal("iter", "iter + 1")
        intg.endBlock()

        intg.addComputeGlobal(iter_out, "iter")
        intg.addComputeGlobal(ssq_out, "drude_force_sq")
        intg.addComputeGlobal(nbad_out, "n_bad_components")

    add_scf_block(integrator, "iter_step", "ssq_step", "n_bad_components_step")
    return integrator


def setup_drude_scf_env(system, temperature, scf_tol, dt, ghost_mass=0.4 * unit.dalton,
                        scf_alpha=5.0e-6,
                        scf_max_disp=1.0e-3,
                        max_iters=500):
    ghost_mass_val = ghost_mass.value_in_unit(unit.dalton)
    total_atoms = system.getNumParticles()

    is_drude_vals = []
    is_core_vals = []
    num_drudes = 0

    # Identify Drude particles from the DrudeForce definitions
    for force in system.getForces():
        if isinstance(force, mm.DrudeForce):
            num_drudes = force.getNumParticles()
            for i in range(num_drudes):
                params = force.getParticleParameters(i)
                drude_idx = params[0]

                system.setParticleMass(drude_idx, ghost_mass)

    drude_indices = set()
    for force in system.getForces():
        if isinstance(force, mm.DrudeForce):
            for i in range(force.getNumParticles()):
                drude_indices.add(force.getParticleParameters(i)[0])

    for i in range(total_atoms):
        if i in drude_indices:
            is_drude_vals.append(mm.Vec3(1.0, 1.0, 1.0))
            is_core_vals.append(mm.Vec3(0.0, 0.0, 0.0))
        else:
            is_drude_vals.append(mm.Vec3(0.0, 0.0, 0.0))
            is_core_vals.append(mm.Vec3(1.0, 1.0, 1.0))

    # The input scf_tol is interpreted as the per-Drude tolerance:
    #     <|F_D|^2> = (sum_i |F_i^D|^2) / N_D < scf_tol
    # The integrator itself uses the equivalent total tolerance:
    #     sum_i |F_i^D|^2 < N_D * scf_tol
    if num_drudes <= 0:
        raise ValueError("No Drude particles were found in the system.")
    total_scf_tol = num_drudes * scf_tol

    integrator = drude_scf_integrator(
        temperature=temperature,
        scf_tol=total_scf_tol,
        timestep=dt,
        scf_alpha=scf_alpha,
        scf_max_disp=scf_max_disp,
        max_iters=max_iters
    )

    return system, integrator, is_drude_vals, is_core_vals