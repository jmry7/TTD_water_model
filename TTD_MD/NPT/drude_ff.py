from openmm import *
import openmm.unit as unit
import scipy.special as sp
import numpy as np

def apply_polarizable_water(system, topology):

    # Intermolecular Buckingham–Tang–Toennies interactions
    # between the atomic core sites O, H1, and H2
    buckingham = """
            term1 - term2 * term3;

            term1 = fnA(type1, type2) * exp(-fnB(type1, type2) * r);
            term2 = fnC(type1, type2) / (r^6);
            term3 = 1 - exp(-u_var) * (1 + u_var * (1 + u_var * (1.0/2.0 + u_var * (1.0/6.0 + u_var * (1.0/24.0 + u_var * (1.0/120.0 + u_var/720.0))))));
            u_var = fnD(type1, type2) * r;
        """
    bk = CustomNonbondedForce(buckingham)
    bk.addPerParticleParameter("type")

    A_values = [
                   26137.662132, 375.887831, 0.0,
                   375.887831, 278.494814, 0.0,
                   0.0, 0.0, 0.0
               ] * unit.kilocalories_per_mole

    B_values = [
                   3.305236, 2.606274, 1.0,
                   2.606274, 4.621006, 1.0,
                   1.0, 1.0, 1.0
               ] / unit.angstrom

    C_values = [
                   1690.296468, 56.093632, 0.0,
                   56.093632, 30.775496, 0.0,
                   0.0, 0.0, 0.0
               ] * unit.kilocalories_per_mole * (unit.angstroms ** 6)

    D_values = [
                   3.200131, 0.266506, 1.0,
                   0.266506, 0.495342, 1.0,
                   1.0, 1.0, 1.0
               ] / unit.angstrom

    A_md = A_values.value_in_unit_system(unit.md_unit_system)
    B_md = B_values.value_in_unit_system(unit.md_unit_system)
    C_md = C_values.value_in_unit_system(unit.md_unit_system)
    D_md = D_values.value_in_unit_system(unit.md_unit_system)

    bk.addTabulatedFunction("fnA", Discrete2DFunction(3, 3, A_md))
    bk.addTabulatedFunction("fnB", Discrete2DFunction(3, 3, B_md))
    bk.addTabulatedFunction("fnC", Discrete2DFunction(3, 3, C_md))
    bk.addTabulatedFunction("fnD", Discrete2DFunction(3, 3, D_md))

    real_atoms = set()

    for atom in topology.atoms():
        if atom.name == 'O':
            bk.addParticle([0])
            real_atoms.add(atom.index)
        elif atom.name in ['H1', 'H2']:
            bk.addParticle([1])
            real_atoms.add(atom.index)
        else:
            bk.addParticle([2])

    # Only interactions within the core system are calculated
    bk.addInteractionGroup(real_atoms, real_atoms)

    bk.setNonbondedMethod(CustomNonbondedForce.CutoffPeriodic)
    bk.setCutoffDistance(0.9 * unit.nanometer)


    # Intramolecular H–O–H bending potential
    angle_bend = """
        dt^2 * (k2 + dt * (k3 + dt * (k4 + dt * (k5 + dt * k6))));
        dt = theta - theta0;
    """
    bend = CustomAngleForce(angle_bend)

    bend.addPerAngleParameter("theta0")
    bend.addPerAngleParameter("k2")
    bend.addPerAngleParameter("k3")
    bend.addPerAngleParameter("k4")
    bend.addPerAngleParameter("k5")
    bend.addPerAngleParameter("k6")

    theta0b = 130.110000 * unit.degrees
    K2b = 16.221000 * unit.kilocalories_per_mole / (unit.radians ** 2)
    K3b = -33.143900 * unit.kilocalories_per_mole / (unit.radians ** 3)
    K4b = 0.009075 * unit.kilocalories_per_mole / (unit.radians ** 4)
    K5b = -0.537930 * unit.kilocalories_per_mole / (unit.radians ** 5)
    K6b = 0.012957 * unit.kilocalories_per_mole / (unit.radians ** 6)

    theta0 = theta0b.value_in_unit_system(unit.md_unit_system)
    K2_b = K2b.value_in_unit_system(unit.md_unit_system)
    K3_b = K3b.value_in_unit_system(unit.md_unit_system)
    K4_b = K4b.value_in_unit_system(unit.md_unit_system)
    K5_b = K5b.value_in_unit_system(unit.md_unit_system)
    K6_b = K6b.value_in_unit_system(unit.md_unit_system)

    angle_params = [theta0, K2_b, K3_b, K4_b, K5_b, K6_b]

    for residue in topology.residues():
        if residue.name == 'HOH':
            idx_O = None
            idx_H1 = None
            idx_H2 = None

            for atom in residue.atoms():
                if atom.element is not None:
                    if atom.element.symbol == 'O':
                        idx_O = atom.index
                    elif atom.element.symbol == 'H':
                        if idx_H1 is None:
                            idx_H1 = atom.index
                        else:
                            idx_H2 = atom.index

            if idx_O is not None and idx_H1 is not None and idx_H2 is not None:
                bend.addAngle(idx_H1, idx_O, idx_H2, angle_params)

    # Intramolecular O–H stretching potential
    lenth_stret = """
        dl^2 * (k2 + dl * (k3 + dl * (k4 + dl * (k5 + dl * k6))));
        dl = r - l0;
    """
    stret = CustomBondForce(lenth_stret)

    stret.addPerBondParameter("l0")
    stret.addPerBondParameter("k2")
    stret.addPerBondParameter("k3")
    stret.addPerBondParameter("k4")
    stret.addPerBondParameter("k5")
    stret.addPerBondParameter("k6")

    l0_lit = 1.089880 * unit.angstrom
    K2s = 115.821590 * unit.kilocalories_per_mole / (unit.angstroms ** 2)
    K3s = -1178.76573 * unit.kilocalories_per_mole / (unit.angstroms ** 3)
    K4s = 48.047720 * unit.kilocalories_per_mole / (unit.angstroms ** 4)
    K5s = -12.694530 * unit.kilocalories_per_mole / (unit.angstroms ** 5)
    K6s = 11.828150 * unit.kilocalories_per_mole / (unit.angstroms ** 6)

    l0 = l0_lit.value_in_unit_system(unit.md_unit_system)
    K2_s = K2s.value_in_unit_system(unit.md_unit_system)
    K3_s = K3s.value_in_unit_system(unit.md_unit_system)
    K4_s = K4s.value_in_unit_system(unit.md_unit_system)
    K5_s = K5s.value_in_unit_system(unit.md_unit_system)
    K6_s = K6s.value_in_unit_system(unit.md_unit_system)

    bond_params = [l0, K2_s, K3_s, K4_s, K5_s, K6_s]

    for residue in topology.residues():
        if residue.name == 'HOH':
            idx_O = None
            h_indices = []
            for atom in residue.atoms():
                if atom.element is not None:
                    if atom.element.symbol == 'O':
                        idx_O = atom.index
                    elif atom.element.symbol == 'H':
                        h_indices.append(atom.index)

            if idx_O is not None:
                for idx_H in h_indices:
                    stret.addBond(idx_O, idx_H, bond_params)

    # Coulomb
    # (1) undamped Coulomb interactions evaluated with PME,
    # (2) a short-range Thole correction for intermolecular pairs, and
    # (3) selected intramolecular Thole-damped Coulomb interactions.
    q_O_core = 0.001835
    q_H = 0.824144
    q_DO = -1.226973
    q_DH = -(q_O_core + 2 * q_H + q_DO) / 2.0
    total_particles = topology.getNumAtoms()
    charges = [0.0] * total_particles
    particle_types = [0] * total_particles

    for atom in topology.atoms():
        idx = atom.index
        if atom.name == 'O':
            charges[idx] = q_O_core
            particle_types[idx] = 0
        elif 'H' in atom.name and 'D' not in atom.name:
            charges[idx] = q_H
            particle_types[idx] = 0
        elif atom.name == 'DO':
            charges[idx] = q_DO
            particle_types[idx] = 1
        elif 'DH' in atom.name:
            charges[idx] = q_DH
            particle_types[idx] = 1

    ONE_4PI_EPS0 = 138.935456
    r_min, r_max = 1e-5, 1.5  # Unit:nm
    n_r = 10000  # Number of sampling points
    n_types = 3  # 0=c-c, 1=c-d, 2=d-d
    # Thole damping parameters converted from Angstrom^-3 to nm^-3.
    b_values = {
        "CC": 0.544488 * 1000.0,
        "CD": 0.364501 * 1000.0,
        "DD": 0.338670 * 1000.0,
    }

    GAMMA_2_3 = sp.gamma(2.0 / 3.0)

    def calc_thole_shape(r, b):
        x = b * r ** 3
        coulomb_part = -np.expm1(-x) / r
        gamma_part = GAMMA_2_3 * sp.gammaincc(2.0 / 3.0, x)

        return coulomb_part + np.cbrt(b) * gamma_part

    values = np.zeros(n_r * n_types)
    r_grid = np.linspace(r_min, r_max, n_r)

    for i, r in enumerate(r_grid):
        values[i + n_r * 0] = calc_thole_shape(r, b_values['CC'])
        values[i + n_r * 1] = calc_thole_shape(r, b_values['CD'])
        values[i + n_r * 2] = calc_thole_shape(r, b_values['DD'])

    thole_table = Continuous2DFunction(n_r, n_types, values, r_min, r_max, 0, 2)
    lookup_table = Discrete2DFunction(2, 2, [0, 1, 1, 2])

    # Undamped 1/r Coulomb interactions evaluated with PME.
    pme_force = NonbondedForce()
    pme_force.setNonbondedMethod(NonbondedForce.PME)
    pme_force.setCutoffDistance(0.9 * unit.nanometer)
    pme_force.setEwaldErrorTolerance(1e-6)

    for i in range(total_particles):
        # Assign electrostatic charges only; no Lennard-Jones interactions.
        pme_force.addParticle(charges[i], 0.0, 0.0)

    # Short-range correction for intermolecular electrostatics.
    corr_func = """
         138.935456 * q1 * q2 * (F_thole(r, type_pair_idx) - 1/max(r, 1e-3));
         type_pair_idx = table_lookup(t1, t2);
    """
    thole_corr = CustomNonbondedForce(corr_func)
    thole_corr.addTabulatedFunction("F_thole", thole_table)
    thole_corr.addTabulatedFunction("table_lookup", lookup_table)
    thole_corr.addPerParticleParameter("q")
    thole_corr.addPerParticleParameter("t")

    for i in range(total_particles):
        thole_corr.addParticle([charges[i], particle_types[i]])

    thole_corr.setNonbondedMethod(CustomNonbondedForce.CutoffPeriodic)
    thole_corr.setCutoffDistance(0.9 * unit.nanometer)


    # Intramolecular Coulomb interactions
    intra_func = """
         138.935456 * q1 * q2 * F_thole(distance(p1, p2), type_pair_idx);
    """
    # 1/(4*pi*epsilon0) = 138.935456 kJ nm mol^-1 e^-2
    intra_thole = CustomCompoundBondForce(2, intra_func)
    intra_thole.addPerBondParameter("q1")
    intra_thole.addPerBondParameter("q2")
    intra_thole.addPerBondParameter("type_pair_idx")
    thole_table_intra = Continuous2DFunction(n_r, n_types, values, r_min, r_max, 0, 2)
    intra_thole.addTabulatedFunction("F_thole", thole_table_intra)

    for residue in topology.residues():
        if residue.name == 'HOH':
            idx_O = idx_H1 = idx_H2 = None
            idx_DO = idx_DH1 = idx_DH2 = None

            for atom in residue.atoms():
                if atom.name == 'O':
                    idx_O = atom.index
                elif atom.name == 'H1':
                    idx_H1 = atom.index
                elif atom.name == 'H2':
                    idx_H2 = atom.index
                elif atom.name == 'DO':
                    idx_DO = atom.index
                elif atom.name == 'DH1':
                    idx_DH1 = atom.index
                elif atom.name == 'DH2':
                    idx_DH2 = atom.index

            if None not in (idx_DO, idx_DH1, idx_DH2):
                intra_thole.addBond([idx_DO, idx_DH1], [charges[idx_DO], charges[idx_DH1], 2])
                intra_thole.addBond([idx_DO, idx_DH2], [charges[idx_DO], charges[idx_DH2], 2])
                intra_thole.addBond([idx_DH1, idx_DH2], [charges[idx_DH1], charges[idx_DH2], 2])

                if None not in (idx_O, idx_H1, idx_H2):
                    intra_thole.addBond([idx_O, idx_DH1], [charges[idx_O], charges[idx_DH1], 1])
                    intra_thole.addBond([idx_O, idx_DH2], [charges[idx_O], charges[idx_DH2], 1])
                    intra_thole.addBond([idx_H1, idx_DO], [charges[idx_H1], charges[idx_DO], 1])
                    intra_thole.addBond([idx_H2, idx_DO], [charges[idx_H2], charges[idx_DO], 1])
                    intra_thole.addBond([idx_H1, idx_DH2], [charges[idx_H1], charges[idx_DH2], 1])
                    intra_thole.addBond([idx_H2, idx_DH1], [charges[idx_H2], charges[idx_DH1], 1])

    # Drude-spring potentials
    drude_force = DrudeForce()

    # The cubic through sixth-order anharmonic terms for the O–DO displacement.
    springO_engpe = f"k3*r^3 + k4*r^4 + k5*r^5 + k6*r^6"
    springO = CustomBondForce(springO_engpe)
    springO.addPerBondParameter("k3")
    springO.addPerBondParameter("k4")
    springO.addPerBondParameter("k5")
    springO.addPerBondParameter("k6")

    K3d1 = -30.188000 * unit.kilocalories_per_mole / (unit.angstroms ** 3)
    K4d1 = 4.560110 * unit.kilocalories_per_mole / (unit.angstroms ** 4)
    K5d1 = -10.888310 * unit.kilocalories_per_mole / (unit.angstroms ** 5)
    K6d1 = 1.062370 * unit.kilocalories_per_mole / (unit.angstroms ** 6)

    springO_params = [
        K3d1.value_in_unit_system(unit.md_unit_system),
        K4d1.value_in_unit_system(unit.md_unit_system),
        K5d1.value_in_unit_system(unit.md_unit_system),
        K6d1.value_in_unit_system(unit.md_unit_system)
    ]

    #  The cubic through sixth-order anharmonic terms for the H–DH displacement.
    springH_engpe = f"k3*r^3 + k4*r^4 + k5*r^5 + k6*r^6"
    springH = CustomBondForce(springH_engpe)
    springH.addPerBondParameter("k3")
    springH.addPerBondParameter("k4")
    springH.addPerBondParameter("k5")
    springH.addPerBondParameter("k6")

    K3d2 = -389.62457 * unit.kilocalories_per_mole / (unit.angstroms ** 3)
    K4d2 = 1.774920 * unit.kilocalories_per_mole / (unit.angstroms ** 4)
    K5d2 = -48.208330 * unit.kilocalories_per_mole / (unit.angstroms ** 5)
    K6d2 = 1.122970 * unit.kilocalories_per_mole / (unit.angstroms ** 6)

    springH_params = [
        K3d2.value_in_unit_system(unit.md_unit_system),
        K4d2.value_in_unit_system(unit.md_unit_system),
        K5d2.value_in_unit_system(unit.md_unit_system),
        K6d2.value_in_unit_system(unit.md_unit_system)
    ]

    for residue in topology.residues():
        if residue.name == 'HOH':
            idx_O = idx_H1 = idx_H2 = None
            idx_DO = idx_D1 = idx_D2 = None

            for atom in residue.atoms():
                if atom.name == 'O':
                    idx_O = atom.index
                elif atom.name == 'H1':
                    idx_H1 = atom.index
                elif atom.name == 'H2':
                    idx_H2 = atom.index
                elif atom.name == 'DO':
                    idx_DO = atom.index
                elif atom.name == 'DH1':
                    idx_D1 = atom.index
                elif atom.name == 'DH2':
                    idx_D2 = atom.index
            # The harmonic term is provided by DrudeForce
            # DrudeForce expects polarizability in nm^3.
            # U_harm = k2*r^2 = (1/2)*K*r^2, the physical spring constant is K = 2*k2. Therefore:
            # alpha(Angstrom^3) = 332.06371*q_D^2/(2*k2) = 166.03185*q_D^2/k2.
            # Divide by 1000 to convert Angstrom^3 to nm^3.
            if None not in (idx_O, idx_DO):
                springO.addBond(idx_O, idx_DO, springO_params)
                alpha_O = 166.03185 * (charges[idx_DO] ** 2) / 121.300820 / 1000.0
                drude_force.addParticle(idx_DO, idx_O, -1, -1, -1, charges[idx_DO], alpha_O, 0.0, 0.0)

            if None not in (idx_H1, idx_H2, idx_D1, idx_D2):
                springH.addBond(idx_H1, idx_D1, springH_params)
                springH.addBond(idx_H2, idx_D2, springH_params)
                alpha_H = 166.03185 * (charges[idx_D1] ** 2) / 151.859140 / 1000.0
                drude_force.addParticle(idx_D1, idx_H1, -1, -1, -1, charges[idx_D1], alpha_H, 0.0, 0.0)
                drude_force.addParticle(idx_D2, idx_H2, -1, -1, -1, charges[idx_D2], alpha_H, 0.0, 0.0)

            water_atoms = [idx_O, idx_H1, idx_H2, idx_DO, idx_D1, idx_D2]
            water_atoms = [i for i in water_atoms if i is not None]

            for i in range(len(water_atoms)):
                for j in range(i + 1, len(water_atoms)):
                    # Excluding intramolecular long-range PME interactions
                    pme_force.addException(water_atoms[i], water_atoms[j], 0.0, 0.0, 0.0)
                    # Excluding intramolecular Thole-type correction
                    thole_corr.addExclusion(water_atoms[i], water_atoms[j])
                    # Excluding intramolecular Buckingham interactions
                    bk.addExclusion(water_atoms[i], water_atoms[j])


    system.addForce(bk)
    system.addForce(bend)
    system.addForce(stret)
    system.addForce(pme_force)
    system.addForce(thole_corr)
    system.addForce(intra_thole)
    system.addForce(springO)
    system.addForce(springH)
    system.addForce(drude_force)

    #The COM momentum removal function can be commented out, as no significant COM drift was observed in the simulations.
    #system.addForce(CMMotionRemover(10000))

    return system