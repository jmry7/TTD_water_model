import torch
import bend
import stret
import buck
import coul

def engfor(lgrad, lfc,
           lst, lb, ld, lcc, lcd, ldd, lbk,
           q, chg, b, pst, pb, pd, pbk, extfld, extfg):
    """
    Calculate the total potential energy, atomic forces, and second-order
    force-constant matrix for a batch of molecular configurations.

    Returns engpe with shape (Batch,), f with shape (Batch, N, 3), and fcm
    with shape (Batch, 3N, 3N).
    """

    device = q.device
    dtype = q.dtype

    batch_size = q.shape[0]
    np_atoms = q.shape[1]

    engpe = torch.zeros(batch_size, device=device, dtype=dtype)
    f = torch.zeros((batch_size, np_atoms, 3), device=device, dtype=dtype)
    fcm = torch.zeros((batch_size, 3 * np_atoms, 3 * np_atoms), device=device, dtype=dtype)

    # stretching
    engpe, f, fcm = stret.stret(lgrad, lfc, lst, pst, q, engpe, f, fcm)

    # bending
    engpe, f, fcm = bend.bend(lgrad, lfc, q, lb, pb, engpe, f, fcm)

    # Buckingham
    engpe, f, fcm = buck.buck(lgrad, lfc, q, lbk, pbk, engpe, f, fcm)

    # core-core Coulomb interactions
    # b[0]: smearing factor for core-core
    engpe, f, fcm = coul.coul(lgrad, lfc, lcc, q, chg, b[0], engpe, f, fcm)

    # core-Drude Coulomb interactions
    # b[1]: smearing factor for core-Drude
    engpe, f, fcm = coul.coul(lgrad, lfc, lcd, q, chg, b[1], engpe, f, fcm)

    # Drude-Drude Coulomb interactions
    # b[2]: smearing factor for Drude-Drude
    engpe, f, fcm = coul.coul(lgrad, lfc, ldd, q, chg, b[2], engpe, f, fcm)

    # cd spring
    engpe, f, fcm = coul.cdspring(lgrad, lfc, ld, q, pd, engpe, f, fcm)

    # external field
    engpe, f = coul.extfield(q, chg, extfld, extfg, engpe, f)

    return engpe, f, fcm