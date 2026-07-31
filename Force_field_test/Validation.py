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
    results = drude_wat.drude_wat(False, False, False, False, False, False, False, False, False, False,
                                  qa, params["cd"], params["b"], params["ds"], params["poh"], params["phh"], params["phoh"], params["bk"])
    return results["dposition"]

def calculate_with_torch(qa: torch.Tensor, coordinates: torch.Tensor, params: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    lst, lb, ld, lcc, lcd, ldd, lbk, _, _, chg, pst, pb, pd, pbk, extfld, extfg = ttd_init.init(qa, params["cd"], params["ds"], params["poh"], params["phh"], params["phoh"], params["bk"])
    energy, force, _ = ttd_engfor.engfor(True, False, lst, lb, ld, lcc, lcd, ldd, lbk, coordinates, chg, params["b"], pst, pb, pd, pbk, extfld, extfg)
    return energy * TORCH_ENERGY_TO_KJ_MOL, force * TORCH_FORCE_TO_KJ_MOL_NM

# Turn off the periodic nonbonded methods.
def turn_off_pbc(system: openmm.System) -> None:
    for index, force in enumerate(system.getForces()):
        if isinstance(force, openmm.NonbondedForce):
            force.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)
            print(f"Force {index}: NonbondedForce -> NoCutoff")
        elif isinstance(force, openmm.CustomNonbondedForce):
            force.setNonbondedMethod(openmm.CustomNonbondedForce.NoCutoff)
            print(f"Force {index}: CustomNonbondedForce -> NoCutoff")

    periodic_forces = [f"{index}:{force.__class__.__name__}" for index, force in enumerate(system.getForces()) if isinstance(force, (openmm.NonbondedForce, openmm.CustomNonbondedForce)) and force.usesPeriodicBoundaryConditions()]

    if periodic_forces:
        raise RuntimeError("Periodic nonbonded forces remain enabled: " + ", ".join(periodic_forces))


def build_openmm_system(topology: Topology) -> openmm.System:
    system = openmm.System()

    for atom in topology.atoms():
        if atom.name == "O":
            system.addParticle(15.99943 * unit.amu)
        elif atom.name in ("H1", "H2"):
            system.addParticle(1.007947 * unit.amu)
        else:
            system.addParticle(0.4 * unit.amu)

    system = apply_polarizable_water(system, topology)
    turn_off_pbc(system)

    return system

def build_openmm_topology(n_water: int) -> Topology:
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

    return topology
def select_openmm_platform() -> tuple[openmm.Platform, dict[str, str]]:
    return openmm.Platform.getPlatformByName("Reference"), {}


def calculate_with_openmm(coordinates: torch.Tensor) -> tuple[np.ndarray, np.ndarray, str]:
    n_config, n_particles, _ = coordinates.shape

    if n_particles % 6 != 0:
        raise ValueError("Optimized coordinates must be ordered as O H H D D D for each water molecule.")

    topology = build_openmm_topology(n_particles // 6)
    system = build_openmm_system(topology)

    integrator = openmm.VerletIntegrator(0.001 * unit.picoseconds)
    platform, properties = select_openmm_platform()
    context = openmm.Context(system, integrator, platform, properties)

    energies = np.zeros(n_config)
    forces = np.zeros((n_config, n_particles, 3))
    coordinates_nm = coordinates.detach().cpu().numpy() * 0.1

    for iconf in range(n_config):
        context.setPositions(coordinates_nm[iconf] * unit.nanometer)
        state = context.getState(getEnergy=True, getForces=True)
        energies[iconf] = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        forces[iconf] = state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)

    del context, integrator

    return energies, forces, platform.getName()


def write_energy_comparison(filename: str | Path, torch_energy: np.ndarray, openmm_energy: np.ndarray) -> None:
    torch_relative = torch_energy - torch_energy[0]
    openmm_relative = openmm_energy - openmm_energy[0]

    with Path(filename).open("w", encoding="utf-8") as handle:
        handle.write("# Energies are in kJ/mol. Relative energies use configuration 1 as the reference.\n")
        handle.write("# config  torch_absolute  openmm_absolute  torch_relative  openmm_relative  relative_difference  absolute_difference\n")

        for i in range(len(torch_energy)):
            handle.write(f"{i + 1:8d}{torch_energy[i]:18.10f}{openmm_energy[i]:18.10f}{torch_relative[i]:18.10f}{openmm_relative[i]:18.10f}{openmm_relative[i] - torch_relative[i]:22.10e}{openmm_energy[i] - torch_energy[i]:22.10e}\n")


def write_force_comparison(filename: str | Path, torch_force: np.ndarray, openmm_force: np.ndarray) -> None:
    names = ["O", "H1", "H2", "DO", "DH1", "DH2"] * (torch_force.shape[1] // 6)
    difference = openmm_force - torch_force

    with Path(filename).open("w", encoding="utf-8") as handle:
        handle.write("# Forces are in kJ/(mol nm). Difference = OpenMM - PyTorch.\n")
        handle.write("# config particle name torch_fx torch_fy torch_fz openmm_fx openmm_fy openmm_fz diff_fx diff_fy diff_fz\n")

        for iconf in range(torch_force.shape[0]):
            for iparticle, name in enumerate(names):
                tf = torch_force[iconf, iparticle]
                of = openmm_force[iconf, iparticle]
                df = difference[iconf, iparticle]
                handle.write(f"{iconf + 1:6d}{iparticle + 1:9d}{name:>7s}{tf[0]:18.10e}{tf[1]:18.10e}{tf[2]:18.10e}{of[0]:18.10e}{of[1]:18.10e}{of[2]:18.10e}{df[0]:18.10e}{df[1]:18.10e}{df[2]:18.10e}\n")


params = read_force_field_parameters(PARAMETER_FILE)
qa, comments = read_xyz(INPUT_XYZ)

print(f"PyTorch device: {DEVICE}")
print(f"Number of configurations: {qa.shape[0]}")
print(f"Number of water molecules per configuration: {qa.shape[1] // 3}")

optimized_coordinates = optimize_with_torch(qa, params)

torch_energy, torch_force = calculate_with_torch(qa, optimized_coordinates, params)
openmm_energy, openmm_force, platform_name = calculate_with_openmm(optimized_coordinates)

torch_energy_np = torch_energy.detach().cpu().numpy()
torch_force_np = torch_force.detach().cpu().numpy()

write_energy_comparison(ENERGY_OUTPUT, torch_energy_np, openmm_energy)
write_force_comparison(FORCE_OUTPUT, torch_force_np, openmm_force)

torch_relative = torch_energy_np - torch_energy_np[0]
openmm_relative = openmm_energy - openmm_energy[0]
force_difference = openmm_force - torch_force_np

print(f"OpenMM platform: {platform_name}")
print("\nEnergy comparison relative to configuration 1 (kJ/mol)")
print("config        PyTorch          OpenMM      OpenMM-PyTorch")

for i in range(qa.shape[0]):
    print(f"{i + 1:6d}{torch_relative[i]:17.10f}{openmm_relative[i]:17.10f}{openmm_relative[i] - torch_relative[i]:20.10e}")

print("\nForce comparison (kJ/(mol nm))")

for i in range(qa.shape[0]):
    max_abs = np.max(np.abs(force_difference[i]))
    rms = np.sqrt(np.mean(force_difference[i] ** 2))
    print(f"Configuration {i + 1}: max_abs_difference={max_abs:.10e}, RMS_difference={rms:.10e}")

print(f"Energy comparison: {ENERGY_OUTPUT}")
print(f"Force comparison: {FORCE_OUTPUT}")