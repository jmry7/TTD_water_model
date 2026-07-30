from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import os
import re
import sys
import torch

CURRENT_DIR = Path(__file__).resolve().parent
DRUDE_DIR = (CURRENT_DIR  / "Drude").resolve()
if str(DRUDE_DIR) not in sys.path:
    sys.path.append(str(DRUDE_DIR))

import drude_wat


DTYPE = torch.float64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ENGUNIT = 0.04184
ANG_TO_BOHR = 1.8897259886
EA_TO_DEBYE = 1.0 / 0.20819434
DIP_AU_TO_DEBYE = 2.541746
PARAMETER_FILE = "drude_par.h"


def read_force_field_parameters(filename: str = PARAMETER_FILE) -> dict[str, torch.Tensor]:
    """
    Read the force-field parameters in drude_par.h and organize them.

    Parameter order:
    1. q_O, q_H, q_dO,
    2. b_cc, b_cd, b_dd,
    3. coefficients of the O-dO anharmonic spring,
    4. coefficients of the H-dH anharmonic spring,
    5. r0, coefficients of the bond stretching potential,
    6. r_HH, k_HH, (k_HH is set to 0, so these two parameters are excluded from the calculation.)
    7. theta0, coefficients of the angle bending potential,
    8. Buckingham-Tang−Toennies parameters for O-O,
    9. Buckingham-Tang−Toennies parameters for O-H,
    10. Buckingham-Tang−Toennies parameters for H-H.

    The H-Drude charge q_dH is derived from molecular charge neutrality.
    """
    number_pattern = r"[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?"

    with open(filename, "r", encoding="utf-8") as handle:
        numbers = re.findall(number_pattern, handle.read())

    values = torch.tensor(
        [float(value) for value in numbers],
        dtype=DTYPE,
        device=DEVICE,
    )

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


def calculate(
    qa: torch.Tensor,
    params: dict[str, torch.Tensor],
    *,
    lgrad: bool = False,
    lfc: bool = False,
    ldip: bool = False,
    lpol: bool = False,
    lpol2: bool = False,
    lquad: bool = False,
    lopt: bool = False,
    lmode: bool = False,
    lg: bool = False,
    loo: bool = False,
) -> dict:
    return drude_wat.drude_wat(
        lgrad,
        lfc,
        ldip,
        lpol,
        lpol2,
        lquad,
        lopt,
        lmode,
        lg,
        loo,
        qa,
        params["cd"],
        params["b"],
        params["ds"],
        params["poh"],
        params["phh"],
        params["phoh"],
        params["bk"],
    )

def wat_pole_data(device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Ab initio reference data for multipoles and (hyper)polarizabilities
    from Chen et al., Chem. Phys. Lett. 752 (2020) 137555.

    All data are in atomic units (a.u.).
    """
    dtype = torch.float64

    dip0=torch.zeros(3,dtype=dtype,device=device)
    dip0[2]=0.7300

    quad0 = torch.zeros((3, 3), dtype=dtype, device=device)
    quad0[0, 0] = 1.9089
    quad0[1, 1] = -1.8091
    quad0[2, 2] = -0.0998

    octu0 = torch.zeros((3, 3, 3), dtype=dtype, device=device)
    octu0[0, 0, 2] = 3.2120
    octu0[1, 1, 2] = -1.3176
    octu0[2, 2, 2] = -1.8944

    hexa0 = torch.zeros((3, 3, 3, 3), dtype=dtype, device=device)
    hexa0[0, 0, 0, 0] = -1.0385
    hexa0[1, 1, 1, 1] = 4.0330
    hexa0[2, 2, 2, 2] = -3.6247
    hexa0[0, 0, 1, 1] = -3.3096
    hexa0[0, 0, 2, 2] = 4.3481
    hexa0[1, 1, 2, 2] = -0.7234

    alp0 = torch.zeros((3, 3), dtype=dtype, device=device)
    alp0[0, 0] = 9.88
    alp0[1, 1] = 9.22
    alp0[2, 2] = 9.52

    beta0 = torch.zeros((3, 3, 3), dtype=dtype, device=device)
    beta0[2, 0, 0] = -10.14
    beta0[2, 1, 1] = -5.38
    beta0[2, 2, 2] = -13.94

    gama0 = torch.zeros((3, 3, 3, 3), dtype=dtype, device=device)
    gama0[0, 0, 0, 0] = 782.0
    gama0[1, 1, 1, 1] = 2314.0
    gama0[2, 2, 2, 2] = 1361.0
    gama0[0, 0, 1, 1] = 589.0
    gama0[1, 1, 2, 2] = 642.0
    gama0[2, 2, 0, 0] = 404.0

    Aijk0 = torch.zeros((3, 3, 3), dtype=dtype, device=device)
    Aijk0[0, 2, 0] = 6.63
    Aijk0[1, 2, 1] = 2.41
    Aijk0[2, 0, 0] = 1.04
    Aijk0[2, 2, 2] = 2.73
    Aijk0[2, 1, 1] = -Aijk0[2, 0, 0] - Aijk0[2, 2, 2]

    Bijkl0 = torch.zeros((3, 3, 3, 3), dtype=dtype, device=device)
    Bijkl0[0, 0, 0, 0] = -103.71
    Bijkl0[0, 0, 1, 1] = 48.75
    Bijkl0[0, 0, 2, 2] = 54.96
    Bijkl0[1, 1, 0, 0] = 116.95
    Bijkl0[1, 1, 1, 1] = -216.31
    Bijkl0[1, 1, 2, 2] = 99.36
    Bijkl0[2, 2, 0, 0] = 85.61
    Bijkl0[2, 2, 1, 1] = 66.09
    Bijkl0[2, 2, 2, 2] = -151.70
    Bijkl0[0, 1, 0, 1] = -127.15
    Bijkl0[0, 2, 0, 2] = -115.34
    Bijkl0[1, 2, 1, 2] = -135.27

    Cijkl0 = torch.zeros((3, 3, 3, 3), dtype=dtype, device=device)
    Cijkl0[0, 0, 0, 0] = 14.44
    Cijkl0[0, 0, 1, 1] = -7.86
    Cijkl0[1, 1, 1, 1] = 15.48
    Cijkl0[1, 1, 2, 2] = -7.62
    Cijkl0[2, 2, 0, 0] = -6.58
    Cijkl0[2, 2, 2, 2] = 14.20
    Cijkl0[0, 1, 0, 1] = 11.02
    Cijkl0[0, 2, 0, 2] = 12.76
    Cijkl0[1, 2, 1, 2] = 11.13

    Dijkl0 = torch.zeros((3, 3, 3, 3), dtype=dtype, device=device)
    Dijkl0[0, 0, 0, 0] = -4.04
    Dijkl0[0, 0, 1, 1] = -2.46
    Dijkl0[0, 0, 2, 2] = 6.50
    Dijkl0[1, 1, 0, 0] = -1.94
    Dijkl0[1, 1, 1, 1] = 1.80
    Dijkl0[1, 1, 2, 2] = 0.14
    Dijkl0[2, 2, 0, 0] = 3.82
    Dijkl0[2, 2, 1, 1] = -1.45
    Dijkl0[2, 2, 2, 2] = -2.37

    return dip0, quad0, octu0, hexa0, alp0, beta0, gama0, Aijk0, Bijkl0, Cijkl0, Dijkl0

dip0, quad0, octu0, hexa0, alp0, beta0, gama0, Aijk0, Bijkl0, Cijkl0, Dijkl0=wat_pole_data()

def export_monomer_properties(opt_res: dict) -> torch.Tensor:
    roh = opt_res["rohf"][0].reshape(())
    angle = opt_res["ahohf"][0].reshape(())
    frequencies = opt_res["freq"][0][0]
    dipole = opt_res["dip"][1][0]
    quadrupole = opt_res["quad"][0]
    octupole = opt_res["octu"][0]
    hexadecapole = opt_res["hexa"][0]
    alpha = opt_res["alp"][0]
    beta = opt_res["beta"][0]
    gamma = opt_res["gama"][0]
    tensor_a = opt_res["A"][0]
    tensor_b = opt_res["B"][0]
    tensor_c = opt_res["C"][0]
    tensor_d = opt_res["D"][0]

    dipole_au = dipole * ANG_TO_BOHR
    dipole_debye = dipole_au * DIP_AU_TO_DEBYE
    quadrupole_au = quadrupole * ANG_TO_BOHR**2
    octupole_au = octupole * ANG_TO_BOHR**3
    hexadecapole_au = hexadecapole * ANG_TO_BOHR**4
    alpha_au = alpha * ANG_TO_BOHR**3
    beta_au = beta * ANG_TO_BOHR**5
    gamma_au = gamma * ANG_TO_BOHR**7
    tensor_a_au = tensor_a * ANG_TO_BOHR**4
    tensor_b_au = tensor_b * ANG_TO_BOHR**6
    tensor_c_au = tensor_c * ANG_TO_BOHR**5
    tensor_d_au = tensor_d * ANG_TO_BOHR**5

    reference_roh = 0.9587
    reference_angle = 104.305

    reference_frequencies = (1649.39,3835.89,3946.65)

    def scalar(value: float | torch.Tensor) -> float:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().reshape(()).item()
        return float(value)

    def write_section(
            handle,
            title: str,
            unit: str,
            rows: list[tuple[str, float | torch.Tensor, float | torch.Tensor]],
    ) -> None:

        handle.write(f"[{title}]\n")
        handle.write(f"# unit: {unit}\n")
        handle.write(f"{'component':<24}" f"{'ab initio':>20}" f"{'TTD':>20}\n")

        for label, reference_value, calculated_value in rows:
            handle.write(
                f"{label:<24}" f"{scalar(reference_value):>20.10f}" f"{scalar(calculated_value):>20.10f}\n")
        handle.write("\n")


    with open( "monomer_equilibrium_properties_au.dat", "w", encoding="utf-8", ) as handle:

        handle.write("# Comparison of the fitted TTD water model with " "ab initio reference data\n")
        handle.write("# Reference: H. Chen, M. Liu, and T. Yan, ""Chem. Phys. Lett. 752 (2020) 137555.\n")

        write_section(
            handle,
            "Geometry",
            "Angstrom or degree",
            [
                ("r_OH", reference_roh, roh),
                ("theta_HOH", reference_angle, angle),
            ],
        )

        write_section(
            handle,
            "Harmonic vibrational frequencies",
            "cm^-1",
            [
                ("nu_1", reference_frequencies[0], frequencies[6]),
                ("nu_2", reference_frequencies[1], frequencies[7]),
                ("nu_3", reference_frequencies[2], frequencies[8]),
            ],
        )

        write_section(
            handle,
            "Dipole",
            "e*a_0",
            [
                ("mu_z", dip0[2], dipole_au[2]),
            ],
        )

        write_section(
            handle,
            "Quadrupole",
            "e*a_0^2",
            [
                ("Theta_xx", quad0[0, 0], quadrupole_au[0, 0]),
                ("Theta_zz", quad0[2, 2], quadrupole_au[2, 2]),
            ],
        )

        write_section(
            handle,
            "Octupole",
            "e*a_0^3",
            [
                ("Omega_xxz", octu0[0, 0, 2], octupole_au[0, 0, 2]),
                ("Omega_zzz", octu0[2, 2, 2], octupole_au[2, 2, 2]),
            ],
        )

        write_section(
            handle,
            "Hexadecapole",
            "e*a_0^4",
            [
                ("Phi_xxxx",hexa0[0, 0, 0, 0],hexadecapole_au[0, 0, 0, 0]),
                ("Phi_yyyy",hexa0[1, 1, 1, 1],hexadecapole_au[1, 1, 1, 1]),
                ("Phi_zzzz",hexa0[2, 2, 2, 2],hexadecapole_au[2, 2, 2, 2]),
            ],
        )

        write_section(
            handle,
            "Dipole polarizability",
            "a_0^3",
            [
                ("alpha_xx", alp0[0, 0], alpha_au[0, 0]),
                ("alpha_yy", alp0[1, 1], alpha_au[1, 1]),
                ("alpha_zz", alp0[2, 2], alpha_au[2, 2]),
            ],
        )

        write_section(
            handle,
            "First hyperpolarizability",
            "e^-1*a_0^5",
            [
                ("beta_zxx", beta0[2, 0, 0], beta_au[2, 0, 0]),
                ("beta_zyy", beta0[2, 1, 1], beta_au[2, 1, 1]),
                ("beta_zzz", beta0[2, 2, 2], beta_au[2, 2, 2]),
            ],
        )

        write_section(
            handle,
            "Second hyperpolarizability",
            "e^-2*a_0^7",
            [
                ("gamma_xxxx",gama0[0, 0, 0, 0],gamma_au[0, 0, 0, 0]),
                ("gamma_yyyy",gama0[1, 1, 1, 1],gamma_au[1, 1, 1, 1]),
                ("gamma_zzzz",gama0[2, 2, 2, 2],gamma_au[2, 2, 2, 2]),
                ("gamma_xxyy",gama0[0, 0, 1, 1],gamma_au[0, 0, 1, 1]),
                ("gamma_yyzz",gama0[1, 1, 2, 2],gamma_au[1, 1, 2, 2]),
                ("gamma_zzxx",gama0[2, 2, 0, 0],gamma_au[2, 2, 0, 0]),
            ],
        )

        write_section(
            handle,
            "Dipole-quadrupole polarizability A",
            "a_0^4",
            [
                ("A_x,zx", Aijk0[0, 2, 0], tensor_a_au[0, 2, 0]),
                ("A_y,zy", Aijk0[1, 2, 1], tensor_a_au[1, 2, 1]),
                ("A_z,xx", Aijk0[2, 0, 0], tensor_a_au[2, 0, 0]),
                ("A_z,zz", Aijk0[2, 2, 2], tensor_a_au[2, 2, 2]),
            ],
        )

        write_section(
            handle,
            "Dipole-dipole-quadrupole polarizability B",
            "e^-1*a_0^6",
            [
                ("B_xy,xy",Bijkl0[0, 1, 0, 1],tensor_b_au[0, 1, 0, 1]),
                ("B_xz,xz",Bijkl0[0, 2, 0, 2],tensor_b_au[0, 2, 0, 2]),
                ("B_yz,yz",Bijkl0[1, 2, 1, 2],tensor_b_au[1, 2, 1, 2]),
                ("B_xx,xx",Bijkl0[0, 0, 0, 0],tensor_b_au[0, 0, 0, 0]),
                ("B_xx,zz",Bijkl0[0, 0, 2, 2],tensor_b_au[0, 0, 2, 2]),
                ("B_yy,xx",Bijkl0[1, 1, 0, 0],tensor_b_au[1, 1, 0, 0]),
                ("B_yy,zz",Bijkl0[1, 1, 2, 2],tensor_b_au[1, 1, 2, 2]),
                ("B_zz,xx",Bijkl0[2, 2, 0, 0],tensor_b_au[2, 2, 0, 0]),
                ("B_zz,zz",Bijkl0[2, 2, 2, 2],tensor_b_au[2, 2, 2, 2]),
            ],
        )

        write_section(
            handle,
            "Quadrupole-quadrupole polarizability C",
            "a_0^5",
            [
                ("C_xx,xx",Cijkl0[0, 0, 0, 0],tensor_c_au[0, 0, 0, 0]),
                ("C_yy,yy",Cijkl0[1, 1, 1, 1],tensor_c_au[1, 1, 1, 1]),
                ("C_zz,zz",Cijkl0[2, 2, 2, 2],tensor_c_au[2, 2, 2, 2]),
                ("C_xy,xy",Cijkl0[0, 1, 0, 1],tensor_c_au[0, 1, 0, 1]),
                ("C_xz,xz",Cijkl0[0, 2, 0, 2],tensor_c_au[0, 2, 0, 2]),
                ("C_yz,yz",Cijkl0[1, 2, 1, 2],tensor_c_au[1, 2, 1, 2]),
            ],
        )

        write_section(
            handle,
            "Dipole-octupole polarizability D",
            "a_0^5",
            [
                ("D_x,xxx",Dijkl0[0, 0, 0, 0],tensor_d_au[0, 0, 0, 0]),
                ("D_x,xyy",Dijkl0[0, 0, 1, 1],tensor_d_au[0, 0, 1, 1]),
                ("D_y,yyy",Dijkl0[1, 1, 1, 1],tensor_d_au[1, 1, 1, 1]),
                ("D_y,yzz",Dijkl0[1, 1, 2, 2],tensor_d_au[1, 1, 2, 2]),
                ("D_z,zxx",Dijkl0[2, 2, 0, 0],tensor_d_au[2, 2, 0, 0]),
                ("D_z,zzz",Dijkl0[2, 2, 2, 2],tensor_d_au[2, 2, 2, 2]),
            ],
        )

params = read_force_field_parameters()

def read_xyz(
    filename: str | Path,
) -> tuple[torch.Tensor, list[str], list[int]]:
    """
    Read multiple water configurations from a multi-frame XYZ file.

    Returns
    -------
    qa_batch
        Shape: (Batch, 3*Nwater, 3)

    comments
        Original XYZ comment lines.
    """
    filename = Path(filename)

    with filename.open("r", encoding="utf-8") as handle:
        lines = handle.readlines()

    structures: list[torch.Tensor] = []
    comments: list[str] = []
    n_water_list: list[int] = []

    i = 0
    config_id = 1

    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        natoms = int(lines[i].strip())
        block_end = i + natoms + 2
        comment = lines[i + 1].strip()
        atom_coords: list[list[float]] = []
        n_water = 0
        for k in range(natoms):
            line_number = i + 2 + k
            parts = lines[line_number].split()
            element = parts[0].upper()
            xyz = [
                float(parts[1]),
                float(parts[2]),
                float(parts[3]),
            ]
            if element == "O":
                n_water += 1
            atom_coords.append(xyz)
        n_water_list.append(n_water)

        structures.append(torch.tensor(atom_coords,dtype=DTYPE,device=DEVICE))
        comments.append(comment)

        config_id += 1
        i = block_end

    qa_batch = torch.stack(structures, dim=0)
    return qa_batch, comments, n_water_list

qa_batch=read_xyz('monomer_config.xyz')
monomer_config=qa_batch[0]
monomer_result = calculate(
        monomer_config,
        params,
        lgrad=False,
        lfc=False,
        ldip=True,
        lpol=True,
        lpol2=True,
        lquad=True,
        lopt=True,
        lmode=True,
        lg=True,
    )
export_monomer_properties(monomer_result)

print(f"Calculation complete. The output file is 'monomer_equilibrium_properties_au.dat'!")