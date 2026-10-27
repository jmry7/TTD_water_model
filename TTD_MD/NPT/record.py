"""XYZ trajectory output for the manually stepped NPT simulation."""

import numpy as np
import openmm.unit as unit


class TrajectoryWriter:
    def __init__(self, topology, coord_file, vel_file):
        self.coord = open(coord_file, "w")
        self.vel = open(vel_file, "w")
        self.atoms = [(atom.name, atom.index) for atom in topology.atoms()]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.coord.close()
        self.vel.close()

    def write(self, context, step):
        state = context.getState(getPositions=True, getVelocities=True)
        positions = state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
        velocities = state.getVelocities(asNumpy=True).value_in_unit(
            unit.angstrom / unit.picosecond
        )
        box = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.angstrom)
        lengths = np.linalg.norm(box, axis=1)
        time_ps = state.getTime().value_in_unit(unit.picosecond)
        box_comment = " ".join(f"{length:.6f}" for length in lengths)

        self.coord.write(
            f"{len(self.atoms)}\n"
            f"Step: {step} Time: {time_ps:.6f} ps Box(Angstrom): {box_comment}\n"
        )
        self.vel.write(
            f"{len(self.atoms)}\n"
            f"Velocity Step: {step} Time: {time_ps:.6f} ps "
            f"Box(Angstrom): {box_comment}\n"
        )
        for name, index in self.atoms:
            x, y, z = positions[index]
            vx, vy, vz = velocities[index]
            self.coord.write(f"{name:<4s} {x:12.6f} {y:12.6f} {z:12.6f}\n")
            self.vel.write(f"{name:<4s} {vx:12.6e} {vy:12.6e} {vz:12.6e}\n")
        self.coord.flush()
        self.vel.flush()
