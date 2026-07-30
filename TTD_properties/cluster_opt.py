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
) -> tuple[list[torch.Tensor], list[str], list[int]]:
    """
    Read water-cluster configurations from a multi-frame XYZ file.

    Configurations may contain different numbers of water molecules.
    The function returns a list of coordinate tensors rather than a
    single stacked tensor.
    """

    filename = Path(filename)

    with filename.open("r", encoding="utf-8") as handle:
        lines = handle.readlines()

    structures: list[torch.Tensor] = []
    comments: list[str] = []
    n_water_list: list[int] = []

    i = 0

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

            if len(parts) < 4:
                raise ValueError(
                    f"Invalid coordinate line {line_number + 1}: "
                    f"{lines[line_number].rstrip()}"
                )

            element = parts[0].upper()

            if element == "O":
                n_water += 1

            atom_coords.append(
                [
                    float(parts[1]),
                    float(parts[2]),
                    float(parts[3]),
                ]
            )

        if natoms != 3 * n_water:
            raise ValueError(
                f"Configuration {len(structures) + 1} contains "
                f"{natoms} real atoms but {n_water} oxygen atoms. "
                f"A water cluster should contain 3 atoms per molecule."
            )

        structures.append(
            torch.tensor(
                atom_coords,
                dtype=DTYPE,
                device=DEVICE,
            )
        )

        comments.append(comment)
        n_water_list.append(n_water)
        i = block_end

    return structures, comments, n_water_list

def optimize_structures(structures: list[torch.Tensor],
    params: dict[str, torch.Tensor],
) -> list[torch.Tensor]:
    """
    Optimize configurations with different sizes.
    """

    groups = {}

    for index, structure in enumerate(structures):
        n_atoms = structure.shape[0]
        groups.setdefault(n_atoms, []).append(index)

    optimized: list[torch.Tensor | None] = [None] * len(structures)

    for n_atoms, indices in groups.items():
        qa_batch = torch.stack(
            [structures[index] for index in indices],
            dim=0,
        )

        print(
            f"Optimizing {len(indices)} configuration(s) "
            f"containing {n_atoms // 3} water molecule(s)."
        )

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

        coordinates = results["dposition"]

        if coordinates.ndim == 2:
            coordinates = coordinates.unsqueeze(0)

        for batch_index, original_index in enumerate(indices):
            optimized[original_index] = coordinates[batch_index]

    if any(item is None for item in optimized):
        raise RuntimeError(
            "One or more configurations were not optimized."
        )

    return [item for item in optimized if item is not None]

def write_optimized_xyz(
    filename: str | Path,
    coordinates: list[torch.Tensor],
    comments: list[str],
    n_water_list: list[int],
) -> None:
    """
    Write optimized OHHDDD structures to a multi-frame XYZ file.

    Different configurations may contain different numbers of water
    molecules. The particle order for each molecule is assumed to be
    O, H1, H2, D1, D2, and D3.
    """

    filename = Path(filename)
    labels_per_water = ["O", "H", "H", "D", "D", "D"]


    with filename.open("w", encoding="utf-8") as handle:
        for config_index, xyz_tensor in enumerate(coordinates):
            xyz = xyz_tensor.detach().cpu()
            n_water = n_water_list[config_index]
            expected_particles = 6 * n_water

            labels = labels_per_water * n_water

            comment_parts = [
                f"configuration={config_index + 1}",
                f"n_water={n_water}",
                "optimized OHHDDD structure",
            ]

            original_comment = comments[config_index]

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
        f"Wrote {len(coordinates)} optimized structure(s) to "
        f"{filename.resolve()}."
    )

params = read_force_field_parameters(PARAMETER_FILE)

structures, comments, n_water_list = read_xyz(INPUT_XYZ)

print(f"Device: {DEVICE}")
print(f"Number of configurations: {len(structures)}")
print(f"Cluster sizes: {n_water_list}")

optimized_coordinates = optimize_structures(
    structures,
    params,
)

write_optimized_xyz(
    OUTPUT_XYZ,
    optimized_coordinates,
    comments,
    n_water_list,
)