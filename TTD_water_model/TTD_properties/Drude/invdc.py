import torch
import iter
import coul2
import cdspring2

def invdc(ld, lcd, ldd, q,
          chg, b, pd, extfld, extfg,
          engpe, f, fcm):
    """
    Relax the Drude-particle coordinates and calculate the inverse of the
    Drude-Drude block of the force-constant matrix.

    The Drude-Drude Hessian block is constructed from the core-Drude,
    Drude-Drude, and core-Drude spring contributions. The function returns
    F_dd^(-1) with shape (Batch, 3*N_Drude, 3*N_Drude).
    """

    q = iter.iter(ld, lcd, ldd, q, chg, b, pd, extfld, extfg, engpe, f, fcm)

    batch_size = q.shape[0]
    np_atoms = q.shape[1]
    nd = ld.shape[0]
    device = q.device
    dtype = q.dtype

    fcm_work = torch.zeros((batch_size, 3 * np_atoms, 3 * np_atoms), device=device, dtype=dtype)
    fc3 = torch.zeros((batch_size, 3 * np_atoms, 3 * np_atoms, 3 * np_atoms), device=device, dtype=dtype)
    fc4 = torch.zeros((batch_size, 3 * np_atoms, 3 * np_atoms, 3 * np_atoms, 3 * np_atoms), device=device, dtype=dtype)

    # core-Drude
    fcm_work, fc3, fc4 = coul2.coul2(True, False, False, lcd, q, fcm_work, fc3, fc4, chg, b[1])

    # Drude-Drude
    fcm_work, fc3, fc4 = coul2.coul2(True, False, False, ldd, q, fcm_work, fc3, fc4, chg, b[2])

    # cd spring
    fcm_work, fc3, fc4 = cdspring2.cdspring2(True, False, False, q, ld, pd, fcm_work, fc3, fc4)

    nmol = np_atoms // 6
    mol_indices = torch.arange(nmol, device=device)

    base_indices = mol_indices * 6
    drude_offsets = torch.tensor([3, 4, 5], device=device)

    particle_indices = (base_indices.unsqueeze(1) + drude_offsets).view(-1)

    dof_offsets = torch.arange(3, device=device)
    indices_d = (particle_indices.unsqueeze(1) * 3 + dof_offsets).view(-1)

    fdk = fcm_work[:, indices_d, :][:, :, indices_d]

    fdk_1 = torch.linalg.inv(fdk)

    return fdk_1