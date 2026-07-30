import torch
import coul

def iter(ld, lcd, ldd, q, chg, b, pd, extfld, extfg, engpe, f, fcm):
    """
    Perform SCF relaxation of the Drude-particle coordinates using a
    third-order perturbative solution of

    2*k2*x + 3*k3*x^2 + 4*k4*x^3 = F_D,

    with

    x1 = F_D / (2*k2),
    x2 = -(3*k3 / (2*k2)) * x1^2,
    x3 = -(3*k3 / k2) * x1*x2 - (2*k4 / k2) * x1^3, and
    x_predicted = x1 + x2 + x3.

    The predicted displacement is updated using linear mixing, and convergence
    is determined from the RMS displacement correction.

    In OpenMM-based MD simulations, the SCF relaxation is implemented
    using the  gradient descent algorithm instead.
    """

    tiny = 1e-8
    qmix = 0.3

    batch_size = q.shape[0]
    N = q.shape[1]
    nd = ld.shape[0]
    device = q.device
    dtype = q.dtype

    qdd = torch.zeros((batch_size, nd, 3), device=device, dtype=dtype)

    nit = 0
    max_nit = 2000

    idx_drude = ld[:, 1]

    k2 = pd[:, 0].view(1, nd, 1)
    k3 = pd[:, 1].view(1, nd, 1)
    k4 = pd[:, 2].view(1, nd, 1)

    is_converged = False

    while nit < max_nit:
        nit += 1
        qdd_old = qdd

        f_step = torch.zeros((batch_size, N, 3), device=device, dtype=dtype)
        eng_step = torch.zeros((batch_size,), device=device, dtype=dtype)

        _, f_curr = coul.coul_drude(ld, lcd, ldd, q, chg, b, pd, extfld, extfg, eng_step, f_step, fcm)

        f_drude = f_curr[:, idx_drude, :]

        x1 = f_drude / (k2 * 2.0)

        x2 = -3.0 * k3 / k2 / 2.0 * (x1 ** 2)

        x3 = -3.0 * k3 / k2 * x1 * x2 - 2.0 * k4 / k2 * (x1 ** 3)

        qdd_new = x1 + x2 + x3

        current_displacement = qmix * qdd_old + (1.0 - qmix) * qdd_new

        qdd = current_displacement

        delta_q = torch.zeros_like(q)

        delta_q.index_add_(1, idx_drude, current_displacement)

        q = q + delta_q

        ssq = torch.sum(qdd_new ** 2, dim=(1, 2))
        dssq = torch.sqrt(ssq / (nd * 3.0))

        max_error = torch.max(dssq)

        if torch.isnan(max_error) or torch.isinf(max_error):
            is_converged = False
            break

        if max_error <= tiny:
            is_converged = True
            break

    if not is_converged:
        return torch.full_like(q, float('nan'))

    return q

