from __future__ import annotations

from pathlib import Path
import re
import sys

import numpy as np
import torch
import openmm
import openmm.unit as unit
from openmm.app import Topology, element


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
PROPERTY_DIR = ROOT_DIR / "TTD_properties"
DRUDE_DIR = PROPERTY_DIR / "Drude"
MD_DIR = ROOT_DIR / "TTD_MD"

for path in (DRUDE_DIR, MD_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import drude_wat
import init as ttd_init
import engfor as ttd_engfor
from drude_ff import apply_polarizable_water


DTYPE = torch.float64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

INPUT_XYZ = "dimer_test.xyz"
PARAMETER_FILE = PROPERTY_DIR / "drude_par.h"
ENERGY_OUTPUT = SCRIPT_DIR / "torch_openmm_energy.dat"
FORCE_OUTPUT = SCRIPT_DIR / "torch_openmm_force.dat"

TORCH_ENERGY_TO_KJ_MOL = 100.0
TORCH_FORCE_TO_KJ_MOL_NM = 1000.0

BOX_LENGTH_ANGSTROM = 300.0
BOX_LENGTH_NM = BOX_LENGTH_ANGSTROM * 0.1
NONBONDED_CUTOFF_NM = 0.9

OPENMM_PLATFORM = "Reference"


def read_force_field_parameters(filename: str | Path) -> dict[str, torch.Tensor]:
    numbers = re.findall(r"[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?", Path(filename).read_text(encoding="utf-8"))
    values = torch.tensor([float(value) for value in numbers], dtype=DTYPE, device=DEVICE)

    if values.numel() < 42:
        raise ValueError(f"Only {values.numel()} numerical parameters were read from {filename}; at least 42 are required.")

    cd = values[0:3]
    q_dh = -(cd[0] + 2.0 * cd[1] + cd[2]) / 2.0

    return {
        "cd": torch.cat((cd, q_dh.reshape(1))),
        "b": values[3:6],
        "ds": torch.stack((values[6:11], values[11:16])),
        "poh": values[16:22],
        "phh": values[22:24],
        "phoh": values[24:30],
        "bk": torch.stack((values[30:34], values[34:38], values[38:42])),
    }


def read_xyz(filename: str | Path) -> tuple[torch.Tensor, list[str]]:
    lines = Path(filename).read_text(encoding="utf-8").splitlines()
    structures = []
    comments = []
    i = 0

    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue

        natoms = int(lines[i].strip())

        if i + natoms + 2 > len(lines):
            raise ValueError(f"Incomplete XYZ block beginning at line {i + 1}.")

        comment = lines[i + 1].strip()
        labels = []
        coordinates = []

        for line in lines[i + 2:i + natoms + 2]:
            parts = line.split()

            if len(parts) < 4:
                raise ValueError(f"Invalid XYZ coordinate line: {line}")

            labels.append(parts[0].upper())
            coordinates.append([float(parts[1]), float(parts[2]), float(parts[3])])

        if natoms % 3 != 0:
            raise ValueError(f"Each configuration must contain 3N real atoms, but {natoms} atoms were found.")

        for iw in range(natoms // 3):
            if labels[3 * iw] != "O" or not labels[3 * iw + 1].startswith("H") or not labels[3 * iw + 2].startswith("H"):
                raise ValueError(f"Configuration {len(structures) + 1} is not ordered as O H H for each water molecule.")

        structures.append(torch.tensor(coordinates, dtype=DTYPE, device=DEVICE))
        comments.append(comment)
        i += natoms + 2

    if not structures:
        raise ValueError(f"No configurations were read from {filename}.")

    atom_counts = {structure.shape[0] for structure in structures}

    if len(atom_counts) != 1:
        raise ValueError("All configurations must contain the same number of water molecules.")

    return torch.stack(structures, dim=0), comments


def optimize_with_torch(qa: torch.Tensor, params: dict[str, torch.Tensor]) -> torch.Tensor:
    results = drude_wat.drude_wat(False, False, False, False, False, False, False, False, False, False, qa, params["cd"], params["b"], params["ds"], params["poh"], params["phh"], params["phoh"], params["bk"])
    return results["dposition"]


def calculate_with_torch(qa: torch.Tensor, coordinates: torch.Tensor, params: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    lst, lb, ld, lcc, lcd, ldd, lbk, _, _, chg, pst, pb, pd, pbk, extfld, extfg = ttd_init.init(qa, params["cd"], params["ds"], params["poh"], params["phh"], params["phoh"], params["bk"])
    energy, force, _ = ttd_engfor.engfor(True, False, lst, lb, ld, lcc, lcd, ldd, lbk, coordinates, chg, params["b"], pst, pb, pd, pbk, extfld, extfg)

    return energy * TORCH_ENERGY_TO_KJ_MOL, force * TORCH_FORCE_TO_KJ_MOL_NM


def make_box_vectors() -> tuple[unit.Quantity, unit.Quantity, unit.Quantity]:
    a = openmm.Vec3(BOX_LENGTH_NM, 0.0, 0.0) * unit.nanometer
    b = openmm.Vec3(0.0, BOX_LENGTH_NM, 0.0) * unit.nanometer
    c = openmm.Vec3(0.0, 0.0, BOX_LENGTH_NM) * unit.nanometer

    return a, b, c


def turn_off_pbc(system: openmm.System) -> None:
    for index, force in enumerate(system.getForces()):
        if isinstance(force, openmm.NonbondedForce):
            force.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)
            print(f"Nonperiodic system, force {index}: NonbondedForce -> NoCutoff")

        elif isinstance(force, openmm.CustomNonbondedForce):
            force.setNonbondedMethod(openmm.CustomNonbondedForce.NoCutoff)
            print(f"Nonperiodic system, force {index}: CustomNonbondedForce -> NoCutoff")

    periodic_forces = [f"{index}:{force.__class__.__name__}" for index, force in enumerate(system.getForces()) if force.usesPeriodicBoundaryConditions()]

    if periodic_forces:
        raise RuntimeError("Periodic forces remain enabled in the nonperiodic system: " + ", ".join(periodic_forces))


def validate_periodic_forces(system: openmm.System) -> None:
    periodic_nonbonded = []

    for index, force in enumerate(system.getForces()):
        if isinstance(force, openmm.NonbondedForce):
            method = force.getNonbondedMethod()
            periodic_nonbonded.append(f"{index}:NonbondedForce(method={int(method)})")

            if method != openmm.NonbondedForce.PME:
                raise RuntimeError(f"Force {index} is expected to use PME, but method {int(method)} was found.")

        elif isinstance(force, openmm.CustomNonbondedForce):
            method = force.getNonbondedMethod()
            periodic_nonbonded.append(f"{index}:CustomNonbondedForce(method={int(method)})")

            if method != openmm.CustomNonbondedForce.CutoffPeriodic:
                raise RuntimeError(f"Force {index} is expected to use CutoffPeriodic, but method {int(method)} was found.")

    if not periodic_nonbonded:
        raise RuntimeError("No periodic nonbonded forces were found in the periodic system.")

    if not system.usesPeriodicBoundaryConditions():
        raise RuntimeError("The periodic OpenMM system does not report periodic boundary conditions.")

    print("Periodic nonbonded forces: " + ", ".join(periodic_nonbonded))


def build_openmm_topology(n_water: int, periodic: bool) -> Topology:
    topology = Topology()
    chain = topology.addChain()

    for iw in range(n_water):
        residue = topology.addResidue("HOH", chain, id=str(iw + 1))

        atom_o = topology.addAtom("O", element.oxygen, residue)
        atom_h1 = topology.addAtom("H1", element.hydrogen, residue)
        atom_h2 = topology.addAtom("H2", element.hydrogen, residue)

        topology.addAtom("DO", None, residue)
        topology.addAtom("DH1", None, residue)
        topology.addAtom("DH2", None, residue)

        topology.addBond(atom_o, atom_h1)
        topology.addBond(atom_o, atom_h2)

    if periodic:
        topology.setPeriodicBoxVectors(make_box_vectors())

    return topology


def build_openmm_system(topology: Topology, periodic: bool) -> openmm.System:
    system = openmm.System()

    for atom in topology.atoms():
        if atom.name == "O":
            system.addParticle(15.99943 * unit.amu)

        elif atom.name in ("H1", "H2"):
            system.addParticle(1.007947 * unit.amu)

        else:
            system.addParticle(0.4 * unit.amu)

    system = apply_polarizable_water(system, topology)

    if periodic:
        system.setDefaultPeriodicBoxVectors(*make_box_vectors())
        validate_periodic_forces(system)

    else:
        turn_off_pbc(system)

    return system


def select_openmm_platform() -> tuple[openmm.Platform, dict[str, str]]:
    platform = openmm.Platform.getPlatformByName(OPENMM_PLATFORM)
    properties: dict[str, str] = {}

    if OPENMM_PLATFORM in ("CUDA", "OpenCL"):
        properties["Precision"] = "double"

    return platform, properties


def center_cluster_in_box(coordinates_nm: np.ndarray) -> np.ndarray:
    centered = np.array(coordinates_nm, copy=True)

    center = np.mean(centered, axis=0)

    centered += np.array([BOX_LENGTH_NM / 2.0, BOX_LENGTH_NM / 2.0, BOX_LENGTH_NM / 2.0]) - center

    span = np.ptp(centered, axis=0)

    nearest_image_separation = BOX_LENGTH_NM - np.max(span)

    if nearest_image_separation <= NONBONDED_CUTOFF_NM:
        raise ValueError(f"The cluster is too large for the {BOX_LENGTH_ANGSTROM:.1f} A box: nearest image separation is only {nearest_image_separation:.6f} nm.")

    return centered


def calculate_with_openmm(coordinates: torch.Tensor, periodic: bool) -> tuple[np.ndarray, np.ndarray, str]:
    n_config, n_particles, _ = coordinates.shape

    if n_particles % 6 != 0:
        raise ValueError("Optimized coordinates must be ordered as O H H D D D for each water molecule.")

    topology = build_openmm_topology(n_particles // 6, periodic)
    system = build_openmm_system(topology, periodic)

    integrator = openmm.VerletIntegrator(0.001 * unit.picoseconds)

    platform, properties = select_openmm_platform()

    context = openmm.Context(system, integrator, platform, properties)

    if periodic:
        context.setPeriodicBoxVectors(*make_box_vectors())

    energies = np.zeros(n_config)
    forces = np.zeros((n_config, n_particles, 3))

    coordinates_nm = coordinates.detach().cpu().numpy() * 0.1

    for iconf in range(n_config):
        positions_nm = center_cluster_in_box(coordinates_nm[iconf]) if periodic else coordinates_nm[iconf]

        context.setPositions(positions_nm * unit.nanometer)

        state = context.getState(getEnergy=True, getForces=True)

        energies[iconf] = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

        forces[iconf] = state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)

    del context, integrator

    return energies, forces, platform.getName()


def write_energy_comparison(filename: str | Path, torch_energy: np.ndarray, openmm_no_pbc_energy: np.ndarray, openmm_pbc_energy: np.ndarray) -> None:
    # Each implementation uses its own dimer 1 as the energy reference.
    torch_relative = torch_energy - torch_energy[0]
    openmm_no_pbc_relative = openmm_no_pbc_energy - openmm_no_pbc_energy[0]
    openmm_pbc_relative = openmm_pbc_energy - openmm_pbc_energy[0]

    no_pbc_difference = openmm_no_pbc_relative - torch_relative
    pbc_difference = openmm_pbc_relative - torch_relative

    with Path(filename).open("w", encoding="utf-8") as handle:
        handle.write(f"# Energies are in kJ/mol. The periodic OpenMM calculation uses a cubic box of {BOX_LENGTH_ANGSTROM:.1f} A.\n")

        handle.write("# Relative energies for each implementation use its own configuration 1 (dimer 1) as the reference.\n")

        handle.write("# config torch_relative openmm_no_pbc_relative openmm_pbc_relative no_pbc_minus_torch pbc_minus_torch\n")

        for i in range(len(torch_energy)):
            handle.write(f"{i + 1:8d}{torch_relative[i]:20.10f}{openmm_no_pbc_relative[i]:24.10f}{openmm_pbc_relative[i]:22.10f}{no_pbc_difference[i]:22.10e}{pbc_difference[i]:22.10e}\n")


def write_force_comparison(filename: str | Path, torch_force: np.ndarray, openmm_no_pbc_force: np.ndarray, openmm_pbc_force: np.ndarray) -> None:
    names = ["O", "H1", "H2", "DO", "DH1", "DH2"] * (torch_force.shape[1] // 6)

    # PyTorch forces are the common reference.
    no_pbc_difference = openmm_no_pbc_force - torch_force
    pbc_difference = openmm_pbc_force - torch_force

    with Path(filename).open("w", encoding="utf-8") as handle:
        handle.write(f"# Forces are in kJ/(mol nm). The periodic OpenMM calculation uses a cubic box of {BOX_LENGTH_ANGSTROM:.1f} A.\n")

        handle.write("# Force differences use PyTorch as the common reference: difference = OpenMM - PyTorch.\n")

        handle.write("# config particle name torch_fx torch_fy torch_fz openmm_no_pbc_fx openmm_no_pbc_fy openmm_no_pbc_fz no_pbc_diff_fx no_pbc_diff_fy no_pbc_diff_fz openmm_pbc_fx openmm_pbc_fy openmm_pbc_fz pbc_diff_fx pbc_diff_fy pbc_diff_fz\n")

        for iconf in range(torch_force.shape[0]):
            for iparticle, name in enumerate(names):
                tf = torch_force[iconf, iparticle]

                nf = openmm_no_pbc_force[iconf, iparticle]
                ndf = no_pbc_difference[iconf, iparticle]

                pf = openmm_pbc_force[iconf, iparticle]
                pdf = pbc_difference[iconf, iparticle]

                handle.write(f"{iconf + 1:6d}{iparticle + 1:9d}{name:>7s}{tf[0]:18.10e}{tf[1]:18.10e}{tf[2]:18.10e}{nf[0]:22.10e}{nf[1]:22.10e}{nf[2]:22.10e}{ndf[0]:20.10e}{ndf[1]:20.10e}{ndf[2]:20.10e}{pf[0]:20.10e}{pf[1]:20.10e}{pf[2]:20.10e}{pdf[0]:18.10e}{pdf[1]:18.10e}{pdf[2]:18.10e}\n")


def print_force_summary(label: str, difference: np.ndarray) -> None:
    print(f"\n{label} (kJ/(mol nm))")

    for iconf in range(difference.shape[0]):
        max_abs = np.max(np.abs(difference[iconf]))
        rms = np.sqrt(np.mean(difference[iconf] ** 2))

        print(f"Configuration {iconf + 1}: max_abs_difference={max_abs:.10e}, RMS_difference={rms:.10e}")


def main() -> None:
    params = read_force_field_parameters(PARAMETER_FILE)

    qa, comments = read_xyz(INPUT_XYZ)

    print(f"PyTorch device: {DEVICE}")
    print(f"Number of configurations: {qa.shape[0]}")
    print(f"Number of water molecules per configuration: {qa.shape[1] // 3}")
    print(f"Periodic box length: {BOX_LENGTH_ANGSTROM:.1f} A")

    optimized_coordinates = optimize_with_torch(qa, params)

    torch_energy, torch_force = calculate_with_torch(qa, optimized_coordinates, params)

    print("\n--- OpenMM no-PBC calculation ---")

    openmm_no_pbc_energy, openmm_no_pbc_force, no_pbc_platform = calculate_with_openmm(optimized_coordinates, periodic=False)

    print("\n--- OpenMM PBC calculation ---")

    openmm_pbc_energy, openmm_pbc_force, pbc_platform = calculate_with_openmm(optimized_coordinates, periodic=True)

    torch_energy_np = torch_energy.detach().cpu().numpy()
    torch_force_np = torch_force.detach().cpu().numpy()

    write_energy_comparison(ENERGY_OUTPUT, torch_energy_np, openmm_no_pbc_energy, openmm_pbc_energy)

    write_force_comparison(FORCE_OUTPUT, torch_force_np, openmm_no_pbc_force, openmm_pbc_force)

    torch_relative = torch_energy_np - torch_energy_np[0]
    openmm_no_pbc_relative = openmm_no_pbc_energy - openmm_no_pbc_energy[0]
    openmm_pbc_relative = openmm_pbc_energy - openmm_pbc_energy[0]

    no_pbc_energy_difference = openmm_no_pbc_relative - torch_relative
    pbc_energy_difference = openmm_pbc_relative - torch_relative

    no_pbc_force_difference = openmm_no_pbc_force - torch_force_np
    pbc_force_difference = openmm_pbc_force - torch_force_np

    print(f"OpenMM no-PBC platform: {no_pbc_platform}")
    print(f"OpenMM PBC platform: {pbc_platform}")

    print("\nRelative-energy comparison using the respective dimer 1 as reference (kJ/mol)")

    print("config        PyTorch    OpenMM(no-PBC)      OpenMM(PBC)    no-PBC-PyTorch      PBC-PyTorch")

    for i in range(qa.shape[0]):
        print(f"{i + 1:6d}{torch_relative[i]:17.10f}{openmm_no_pbc_relative[i]:20.10f}{openmm_pbc_relative[i]:18.10f}{no_pbc_energy_difference[i]:20.10e}{pbc_energy_difference[i]:18.10e}")

    print_force_summary("OpenMM no-PBC minus PyTorch force", no_pbc_force_difference)

    print_force_summary("OpenMM PBC minus PyTorch force", pbc_force_difference)

    print(f"\nEnergy comparison: {ENERGY_OUTPUT}")

    print(f"Force comparison: {FORCE_OUTPUT}")


if __name__ == "__main__":
    main()