import openmm.unit as unit
import time
import os


class TrajectoryReporter(object):

    def __init__(self, coord_file, vel_file, reportInterval, topology, append=False):
        mode = 'a' if append else 'w'
        self._coord_out = open(coord_file, mode)
        self._vel_out = open(vel_file, mode)
        self._reportInterval = reportInterval

        self._core_indices = []
        self._core_names = []

        for atom in topology.atoms():
                self._core_indices.append(atom.index)
                self._core_names.append(atom.name)

        self._num_core_atoms = len(self._core_indices)

    def __del__(self):
        try:
            self._coord_out.close()
            self._vel_out.close()
        except:
            pass

    def describeNextReport(self, simulation):
        steps = self._reportInterval - simulation.currentStep % self._reportInterval
        return (steps, True, True, False, False)

    def report(self, simulation, state):
        pos = state.getPositions().value_in_unit(unit.angstroms)
        vel = state.getVelocities().value_in_unit(unit.angstroms / unit.picosecond)
        time_ps = state.getTime().value_in_unit(unit.picosecond)

        self._coord_out.write(f"{self._num_core_atoms}\nStep: {simulation.currentStep} Time: {time_ps:.3f} ps\n")
        self._vel_out.write(f"{self._num_core_atoms}\nVelocity Step: {simulation.currentStep} Time: {time_ps:.3f} ps\n")

        coord_lines = []
        vel_lines = []

        for name, idx in zip(self._core_names, self._core_indices):
            coord_lines.append(f"{name:<4s} {pos[idx][0]:12.6f} {pos[idx][1]:12.6f} {pos[idx][2]:12.6f}\n")
            vel_lines.append(f"{name:<4s} {vel[idx][0]:12.6e} {vel[idx][1]:12.6e} {vel[idx][2]:12.6e}\n")

        self._coord_out.writelines(coord_lines)
        self._vel_out.writelines(vel_lines)
        self._coord_out.flush()
        self._vel_out.flush()


class ConservedThermoReporter(object):
    def __init__(self, file_name, reportInterval, target_temp, max_iters=1000, append=False):
        mode = 'a' if append else 'w'
        write_header = (not append) or (not os.path.exists(file_name)) or os.path.getsize(file_name) == 0
        self._out = open(file_name, mode)
        self._reportInterval = reportInterval
        self._target_temp = target_temp
        self._max_iters = max_iters
        if write_header:
            self._out.write("Step, Time(ps), PE(kJ/mol), KE(kJ/mol), T_Core(K), T_Drude(K), SCF_Iters\n")

    def __del__(self):
        try:
            self._out.close()
        except:
            pass

    def describeNextReport(self, simulation):
        steps = self._reportInterval - simulation.currentStep % self._reportInterval
        return (steps, False, False, False, True)

    def report(self, simulation, state):
        pe = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        ke = state.getKineticEnergy().value_in_unit(unit.kilojoules_per_mole)

        ke_core = simulation.integrator.getGlobalVariableByName("KE_core")
        sigma = simulation.integrator.getGlobalVariableByName("sigma")

        scf_iters = int(simulation.integrator.getGlobalVariableByName("iter"))

        if sigma > 0:
            t_core = self._target_temp * (2*ke_core / sigma)
        else:
            t_core = 0.0
        t_drude = 0.0

        step = simulation.currentStep
        time = state.getTime().value_in_unit(unit.picosecond)

        if scf_iters >= self._max_iters:
            print(
                f" Warning: The SCF iteration limit was reached at step {step} ({scf_iters}/{self._max_iters}).")

        self._out.write(f"{step}, {time:.3f}, {pe:.4f}, {ke:.4f}, {t_core:.4f}, {t_drude:.4f}, {scf_iters}\n")
        self._out.flush()


class ClockReporter(object):
    def __init__(self, reportInterval):
        self._reportInterval = reportInterval
        self._startTime = time.time()

    def describeNextReport(self, simulation):
        steps = self._reportInterval - simulation.currentStep % self._reportInterval
        return (steps, False, False, False, False)

    def report(self, simulation, state):
        elapsed = time.time() - self._startTime
        print(f"Step: {simulation.currentStep} | Total elapsed time: {elapsed:.2f} seconds ({elapsed / 60:.2f} minutes)")
