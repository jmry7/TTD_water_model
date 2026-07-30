import openmm as mm
from openmm import CustomIntegrator
import openmm.unit as unit


def create_fortran_strict_nhc_scf_integrator(
        temperature,
        dof_core,
        scf_tol,
        timestep=0.4 * unit.femtoseconds,
        nh_tau=0.5 * unit.picoseconds,
        scf_alpha=5.0e-6,
        scf_max_disp=1.0e-3,
        max_iters=500,
        nnos=5,
        nys=5
):
    dt_ps = timestep.value_in_unit(unit.picoseconds)
    T_k = temperature.value_in_unit(unit.kelvin)
    tau_ps = nh_tau.value_in_unit(unit.picoseconds)

    R = 8.31446261815324e-3  # kJ/(mol*K)
    gkt = R * T_k
    sigma = dof_core * gkt
    tau2 = tau_ps ** 2

    nresn = int(1000.0 * dt_ps / tau_ps)
    if nresn == 0:
        nresn = 1

    wys = [0.0] * nys
    if nys == 1:
        wys[0] = 1.0
    elif nys == 3:
        wys[0] = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
        wys[1] = 1.0 - 2.0 * wys[0]
        wys[2] = wys[0]
    elif nys == 5:
        wys[0] = 1.0 / (4.0 - 4.0 ** (1.0 / 3.0))
        wys[1] = wys[0]
        wys[2] = 1.0 - 4.0 * wys[0]
        wys[3] = wys[0]
        wys[4] = wys[0]
    else:
        raise ValueError("nys can only be 1, 3, or 5!")

    integrator = CustomIntegrator(dt_ps)

    # ==== Register system variables ====
    integrator.addPerDofVariable("is_drude", 0)
    integrator.addPerDofVariable("is_core", 0)

    integrator.addGlobalVariable("gkt", gkt)
    integrator.addGlobalVariable("sigma", sigma)
    integrator.addGlobalVariable("KE_core", 0.0)
    integrator.addGlobalVariable("scale", 1.0)

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

    for i in range(1, nnos + 1):
        Q_val = sigma * tau2 if i == 1 else gkt * tau2
        integrator.addGlobalVariable(f"Q_{i}", Q_val)
        integrator.addGlobalVariable(f"v_eta_{i}", 0.0)
        integrator.addGlobalVariable(f"eta_{i}", 0.0)
        integrator.addGlobalVariable(f"G_{i}", 0.0)
        integrator.addGlobalVariable(f"aa_{i}", 0.0)

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


    def add_nhc_half_step(intg):
        intg.addComputeSum("KE_core", "0.5 * m * v * v * is_core")
        intg.addComputeGlobal("G_1", "(2 * KE_core - sigma) / Q_1")

        for i in range(2, nnos + 1):
            intg.addComputeGlobal(f"G_{i}", f"(Q_{i - 1} * v_eta_{i - 1} * v_eta_{i - 1} - gkt) / Q_{i}")

        intg.addComputeGlobal("scale", "1.0")

        for _ in range(nresn):
            for iyosh in range(nys):
                c2 = wys[iyosh] / (nresn * 2.0)
                c4 = wys[iyosh] / (nresn * 4.0)
                c8 = wys[iyosh] / (nresn * 8.0)
                wdt2 = f"({c2} * dt)"
                wdt4 = f"({c4} * dt)"
                wdt8 = f"({c8} * dt)"

                intg.addComputeGlobal(f"v_eta_{nnos}", f"v_eta_{nnos} + G_{nnos} * {wdt4}")

                for inos in range(1, nnos):
                    k = nnos - inos
                    intg.addComputeGlobal(f"aa_{k}", f"exp(-{wdt8} * v_eta_{k + 1})")
                    intg.addComputeGlobal(f"v_eta_{k}", f"v_eta_{k} * aa_{k} * aa_{k} + {wdt4} * G_{k} * aa_{k}")

                intg.addComputeGlobal("scale", f"scale * exp(-{wdt2} * v_eta_1)")

                intg.addComputeGlobal("G_1", f"(scale * scale * 2 * KE_core - sigma) / Q_1")

                for inos in range(1, nnos + 1):
                    intg.addComputeGlobal(f"eta_{inos}", f"eta_{inos} + v_eta_{inos} * {wdt2}")

                for inos in range(1, nnos):
                    intg.addComputeGlobal(f"aa_{inos}", f"exp(-{wdt8} * v_eta_{inos + 1})")
                    intg.addComputeGlobal(f"v_eta_{inos}",
                                          f"v_eta_{inos} * aa_{inos} * aa_{inos} + {wdt4} * G_{inos} * aa_{inos}")
                    intg.addComputeGlobal(f"G_{inos + 1}",
                                          f"(Q_{inos} * v_eta_{inos} * v_eta_{inos} - gkt) / Q_{inos + 1}")

                intg.addComputeGlobal(f"v_eta_{nnos}", f"v_eta_{nnos} + G_{nnos} * {wdt4}")

        intg.addComputePerDof("v", "v * scale * is_core")

    # =======================================================
    # Assemble the complete integration sequence (combined Verlet-SCF scheme)
    # =======================================================

    # ==========================================================
    # 0. Only once: initial SCF before the first velocity kick
    # ==========================================================
    integrator.beginIfBlock("init_done < 0.5")
    add_scf_block(integrator, "iter_init", "ssq_init", "n_bad_components_init")
    integrator.addComputeGlobal("init_done", "1")
    integrator.endBlock()

    # 1. First thermostat half-step
    add_nhc_half_step(integrator)

    # 2. First half-step momentum update
    integrator.addComputePerDof("v", "v + 0.5 * dt * f / m * is_core")

    # 3. Full-step position update
    integrator.addComputePerDof("x_old", "x")
    integrator.addComputePerDof("x", "x + v * dt * is_core")

    # 4. SCF relaxation of the Drude particles
    add_scf_block(integrator, "iter_step", "ssq_step", "n_bad_components_step")

    # 5. Second half-step momentum update
    integrator.addComputePerDof("v", "v + 0.5 * dt * f / m * is_core")

    # 6. Second thermostat half-step
    add_nhc_half_step(integrator)

    integrator.addComputePerDof("v", "v*is_core")

    return integrator


def setup_scf_drude_env(system, temperature, scf_tol, dt, ghost_mass=0.4 * unit.dalton,
                        scf_alpha=5.0e-6,
                        scf_max_disp=1.0e-4,
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

    dof_core = 3 * (total_atoms - num_drudes) - system.getNumConstraints()
    # The input scf_tol is interpreted as the per-Drude tolerance:
    #     <|F_D|^2> = (sum_i |F_i^D|^2) / N_D < scf_tol
    # The integrator itself uses the equivalent total tolerance:
    #     sum_i |F_i^D|^2 < N_D * scf_tol
    if num_drudes <= 0:
        raise ValueError("No Drude particles were found in the system.")
    total_scf_tol = num_drudes * scf_tol

    integrator = create_fortran_strict_nhc_scf_integrator(
        temperature=temperature,
        dof_core=dof_core,
        scf_tol=total_scf_tol,
        timestep=dt,
        scf_alpha=scf_alpha,
        scf_max_disp=scf_max_disp,
        max_iters=max_iters
    )

    return system, integrator, is_drude_vals, is_core_vals