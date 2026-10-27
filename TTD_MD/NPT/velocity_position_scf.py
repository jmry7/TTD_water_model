import openmm as mm
from openmm import CustomIntegrator
import openmm.unit as unit


def create_momentum_half_integrator(
        dof_core,
        timestep=0.4 * unit.femtoseconds,
):
    """Create the pure force half-kick used by Appendix B.

    Barostat and thermostat velocity scaling belongs to ``NHCPISOINT``.  This
    operator therefore performs only ``v <- v + h*f/(2*m)`` on physical core
    sites.  It is called on both sides of the coordinate/cell drift.
    """
    dt_ps = timestep.value_in_unit(unit.picoseconds)
    if dof_core <= 0:
        raise ValueError("dof_core must be positive")

    integrator = CustomIntegrator(0.0)
    integrator.addPerDofVariable("is_core", 0)
    integrator.addGlobalVariable("h", dt_ps)
    integrator.addComputePerDof(
        "v", "v + 0.5*h*f/m*is_core"
    )
    return integrator
