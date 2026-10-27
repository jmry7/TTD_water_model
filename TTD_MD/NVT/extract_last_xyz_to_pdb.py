import argparse
import math
from pathlib import Path


SITE_NAMES = ("O", "H1", "H2", "DO", "DH1", "DH2")


def last_frame(path, names):
    count = len(names)
    with path.open("rb") as stream:
        stream.seek(0, 2)
        end = stream.tell()
        size = max(65536, (count + 2) * 80)

        while True:
            start = max(0, end - size)
            stream.seek(start)
            lines = stream.read().decode("utf-8", errors="replace").splitlines()
            if start:
                lines = lines[1:]

            for i in range(len(lines) - count - 2, -1, -1):
                if lines[i].strip() != str(count):
                    continue
                atoms = [line.split() for line in lines[i + 2 : i + 2 + count]]
                if any(len(atom) < 4 or atom[0].upper() != names[j]
                       for j, atom in enumerate(atoms)):
                    continue
                try:
                    coordinates = [tuple(map(float, atom[1:4])) for atom in atoms]
                except ValueError:
                    continue
                if all(math.isfinite(value) for xyz in coordinates for value in xyz):
                    return coordinates

            if start == 0:
                raise ValueError("No complete XYZ frame found")
            size *= 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", nargs="?", type=Path, default=Path("config.xyz"),
                        help="XYZ trajectory (default: config.xyz)")
    parser.add_argument("template", nargs="?", type=Path, default=Path("water_1.pdb"),
                        help="PDB template (default: water_6p.pdb)")
    parser.add_argument("-o", "--output", type=Path, default=Path("water_216.pdb"),
                        help="output PDB (default: water_216_1.pdb)")
    parser.add_argument("--force", action="store_true", help="overwrite the output")
    args = parser.parse_args()

    if args.output.resolve() in (args.trajectory.resolve(), args.template.resolve()):
        raise ValueError("Output path matches an input")

    lines = args.template.read_text(encoding="utf-8").splitlines()
    records = [(i, line[12:16].strip().upper()) for i, line in enumerate(lines)
               if line.startswith(("ATOM  ", "HETATM"))]
    if (not records or len(records) % 6
            or any(len(lines[i]) < 54 for i, _ in records)
            or any(tuple(name for _, name in records[j:j + 6]) != SITE_NAMES
                   for j in range(0, len(records), 6))):
        raise ValueError("Invalid PDB template")

    coordinates = last_frame(args.trajectory, [name for _, name in records])
    for i in range(0, len(coordinates), 6):
        coordinates[i + 3:i + 6] = coordinates[i:i + 3]

    for (line_index, _), xyz in zip(records, coordinates):
        fields = "".join(f"{value:8.3f}" for value in xyz)
        if len(fields) != 24:
            raise ValueError("Coordinate outside PDB range")
        line = lines[line_index]
        lines[line_index] = line[:30] + fields + line[54:]

    if args.output.exists() and not args.force:
        raise ValueError("Output exists; use --force to overwrite")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w" if args.force else "x", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(lines) + "\n")
    print(f"Saved {args.output} ({len(records)} atoms).")


if __name__ == "__main__":
    try:
        main()
    except OSError as exc:
        raise SystemExit(f"Error: Cannot access {exc.filename}") from None
    except ValueError as exc:
        raise SystemExit(f"Error: {exc}") from None
