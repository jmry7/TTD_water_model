import openmm as mm
from openmm import CustomIntegrator
import openmm.unit as unit


def create_half_nhcp_integrator(
        temperature,
        dof_core,
        barostat_mass=None,
        timestep=0.4 * unit.femtoseconds,
        nh_tau=0.5 * unit.picoseconds,
        nnos=5,
        nys=5,
):
    """Create the isotropic ``NHCPISOINT`` half propagator.

    One call advances the coupled Nose-Hoover-chain/barostat operator through
    ``timestep/2``.  ``config_pressure_work`` must be set before every call to

        3*V*(P_config-P_external)

    in kJ/mol.
    """
    dt_ps = timestep.value_in_unit(unit.picoseconds)
    T_k = temperature.value_in_unit(unit.kelvin)
    tau_ps = nh_tau.value_in_unit(unit.picoseconds)

    if dof_core <= 0:
        raise ValueError("dof_core must be positive")
    if barostat_mass is None or float(barostat_mass) <= 0.0:
        raise ValueError("barostat_mass must be positive for NHCPISOINT")

    R = 8.31446261815324e-3  # kJ/(mol*K)
    gkt = R * T_k
    thermostat_dof = dof_core + 1
    sigma = thermostat_dof * gkt
    mtk_alpha = 1.0 + 3.0 / dof_core
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

    integrator = CustomIntegrator(0.0)

    integrator.addPerDofVariable("is_drude", 0)
    integrator.addPerDofVariable("is_core", 0)

    integrator.addGlobalVariable("gkt", gkt)
    integrator.addGlobalVariable("h", dt_ps)
    integrator.addGlobalVariable("sigma", sigma)
    integrator.addGlobalVariable("AKIN", 0.0)
    integrator.addGlobalVariable("scale", 1.0)
    integrator.addGlobalVariable("barostat_mass", float(barostat_mass))
    integrator.addGlobalVariable("barostat_velocity", 0.0)
    integrator.addGlobalVariable("config_pressure_work", 0.0)
    integrator.addGlobalVariable("mtk_alpha", mtk_alpha)
    integrator.addGlobalVariable("G_barostat", 0.0)
    integrator.addGlobalVariable("aa_barostat", 0.0)
    integrator.addGlobalVariable("aa_particle", 0.0)

    for i in range(1, nnos + 1):
        Q_val = sigma * tau2 if i == 1 else gkt * tau2
        integrator.addGlobalVariable(f"Q_{i}", Q_val)
        integrator.addGlobalVariable(f"v_eta_{i}", 0.0)
        integrator.addGlobalVariable(f"eta_{i}", 0.0)
        integrator.addGlobalVariable(f"G_{i}", 0.0)
        integrator.addGlobalVariable(f"aa_{i}", 0.0)

    def add_nhc_half_step(intg):
        intg.addComputeSum("AKIN", "m * v * v * is_core")
        intg.addComputeGlobal(
            "G_1",
            "(AKIN + barostat_mass*barostat_velocity*barostat_velocity - sigma) / Q_1",
        )
        intg.addComputeGlobal(
            "G_barostat",
            "(mtk_alpha*AKIN + config_pressure_work) / barostat_mass",
        )

        for i in range(2, nnos + 1):
            intg.addComputeGlobal(f"G_{i}", f"(Q_{i - 1} * v_eta_{i - 1} * v_eta_{i - 1} - gkt) / Q_{i}")

        intg.addComputeGlobal("scale", "1.0")

        for _ in range(nresn):
            for iyosh in range(nys):
                c2 = wys[iyosh] / (nresn * 2.0)
                c4 = wys[iyosh] / (nresn * 4.0)
                c8 = wys[iyosh] / (nresn * 8.0)
                wdt2 = f"({c2} * h)"
                wdt4 = f"({c4} * h)"
                wdt8 = f"({c8} * h)"

                intg.addComputeGlobal(f"v_eta_{nnos}", f"v_eta_{nnos} + G_{nnos} * {wdt4}")

                for inos in range(1, nnos):
                    k = nnos - inos
                    intg.addComputeGlobal(f"aa_{k}", f"exp(-{wdt8} * v_eta_{k + 1})")
                    intg.addComputeGlobal(f"v_eta_{k}", f"v_eta_{k} * aa_{k} * aa_{k} + {wdt4} * G_{k} * aa_{k}")

                intg.addComputeGlobal(
                    "aa_barostat", f"exp(-{wdt8} * v_eta_1)"
                )
                intg.addComputeGlobal(
                    "barostat_velocity",
                    f"barostat_velocity*aa_barostat*aa_barostat + {wdt4}*G_barostat*aa_barostat",
                )

                intg.addComputeGlobal(
                    "aa_particle",
                    f"exp(-{wdt2} * (v_eta_1 + mtk_alpha*barostat_velocity))",
                )
                intg.addComputeGlobal("scale", "scale * aa_particle")
                intg.addComputeGlobal("AKIN", "AKIN * aa_particle * aa_particle")
                intg.addComputeGlobal(
                    "G_barostat",
                    "(mtk_alpha*AKIN + config_pressure_work) / barostat_mass",
                )

                for inos in range(1, nnos + 1):
                    intg.addComputeGlobal(f"eta_{inos}", f"eta_{inos} + v_eta_{inos} * {wdt2}")

                intg.addComputeGlobal(
                    "aa_barostat", f"exp(-{wdt8} * v_eta_1)"
                )
                intg.addComputeGlobal(
                    "barostat_velocity",
                    f"barostat_velocity*aa_barostat*aa_barostat + {wdt4}*G_barostat*aa_barostat",
                )
                intg.addComputeGlobal(
                    "G_1",
                    "(AKIN + barostat_mass*barostat_velocity*barostat_velocity - sigma) / Q_1",
                )

                for inos in range(1, nnos):
                    intg.addComputeGlobal(f"aa_{inos}", f"exp(-{wdt8} * v_eta_{inos + 1})")
                    intg.addComputeGlobal(f"v_eta_{inos}",
                                          f"v_eta_{inos} * aa_{inos} * aa_{inos} + {wdt4} * G_{inos} * aa_{inos}")
                    intg.addComputeGlobal(f"G_{inos + 1}",
                                          f"(Q_{inos} * v_eta_{inos} * v_eta_{inos} - gkt) / Q_{inos + 1}")

                intg.addComputeGlobal(f"v_eta_{nnos}", f"v_eta_{nnos} + G_{nnos} * {wdt4}")

        intg.addComputePerDof("v", "v * scale * is_core")

    add_nhc_half_step(integrator)

    return integrator
