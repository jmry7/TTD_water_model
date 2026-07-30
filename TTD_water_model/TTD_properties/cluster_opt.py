from __future__ import annotations

from pathlib import Path
import re
import sys

import torch


CURRENT_DIR = Path(__file__).resolve().parent
DRUDE_DIR = (CURRENT_DIR / "Drude").resolve()

if str(DRUDE_DIR) not in sys.path:
    sys.path.append(str(DRUDE_DIR))

import drude_wat


DTYPE = torch.float64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PARAMETER_FILE = "drude_par.h"
INPUT_XYZ = "cluster.xyz"
OUTPUT_XYZ = "cluster_optimized.xyz"

def read_force_field_parameters(
    filename: str = PARAMETER_FILE,
) -> dict[str, torch.Tensor]:
    """
    Read force-field parameters from drude_par.h.

    Parameter order:
    1. q_O, q_H, q_dO
    2. b_cc, b_cd, b_dd
    3. O-dO spring coefficients
    4. H-dH spring coefficients
    5. O-H stretching parameters
    6. H-H stretching parameters
    7. H-O-H bending parameters
    8. O-O Buckingham parameters
    9. O-H Buckingham parameters
    10. H-H Buckingham parameters

    The H-Drude charge is derived from molecular charge neutrality.
    """
    number_pattern = r"[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?"

    with open(filename, "r", encoding="utf-8") as handle:
        numbers = re.findall(number_pattern, handle.read())

    values = torch.tensor(
        [float(value) for value in numbers],
        dtype=DTYPE,
        device=DEVICE,
    )

    if values.numel() < 42:
        raise ValueError(
            f"Only {values.numel()} numerical parameters were read from "
            f"{filename}; at least 42 are required."
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
        "bk": torch.stack(
            (
                values[30:34],
                values[34:38],
                values[38:42],
            )
        ),
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

def write_optimized_xyz(
    filename: str | Path,
    coordinates: torch.Tensor,
    comments: list[str],
    n_water_list: list[int],
) -> None:
    """
    Write optimized OHHDDD structures to a multi-frame XYZ file.

    The particle order for each water molecule is assumed to be:

        O, H1, H2, D1, D2, D3
    """
    filename = Path(filename)

    coordinates = coordinates.detach().cpu()

    if coordinates.ndim == 2:
        coordinates = coordinates.unsqueeze(0)

    batch_size = coordinates.shape[0]

    labels_per_water = ["O", "H", "H", "D", "D", "D"]

    with filename.open("w", encoding="utf-8") as handle:
        for batch_index in range(batch_size):
            xyz = coordinates[batch_index]
            n_water = n_water_list[batch_index]

            expected_particles = 6 * n_water

            labels = labels_per_water * n_water

            comment_parts = [
                f"configuration={batch_index + 1}",
                "optimized OHHDDD structure",
            ]

            original_comment = comments[batch_index]

            if original_comment:
                comment_parts.append(
                    f"original_comment={original_comment}"
                )

            handle.write(f"{expected_particles}\n")
            handle.write("; ".join(comment_parts) + "\n")

            for label, position in zip(labels, xyz):
                x, y, z = position.tolist()

                handle.write(
                    f"{label:<2s}"
                    f"{x:18.10f}"
                    f"{y:18.10f}"
                    f"{z:18.10f}\n"
                )

    print(
        f"Wrote {batch_size} optimized structure(s) to "
        f"{filename.resolve()}."
    )

params = read_force_field_parameters(PARAMETER_FILE)

qa_batch, comments, n_water_list = read_xyz(INPUT_XYZ)

print(f"Device: {DEVICE}")
print(f"Number of configurations: {qa_batch.shape[0]}")
print(f"Number of real atoms per configuration: {qa_batch.shape[1]}")
print(f"Input batch shape: {tuple(qa_batch.shape)}")

results = calculate(
        qa_batch,
        params,
        lgrad=False,
        lfc=False,
        ldip=False,
        lpol=False,
        lpol2=False,
        lquad=False,
        lopt=True,
        lmode=False,
        lg=False,
        loo=False,
    )

optimized_coordinates = results["dposition"]

print(f"Optimized coordinate shape: "
      f"{tuple(optimized_coordinates.shape)}"
      )

write_optimized_xyz(
        OUTPUT_XYZ,
        optimized_coordinates,
        comments,
        n_water_list,
    )
