import torch
import iter
import engfor


def hessian(lst, lb, ld, lcc, lcd, ldd, lbk,
            q, chg, b, pst, pb, pd, pbk, extfld, extfg,
            wa, engpe, f, fcm):
    """
    Calculate the effective mass-weighted Hessian after optimizing and
    eliminating the Drude degrees of freedom.

    The effective core Hessian is obtained from the full force-constant
    matrix using

    K = F_cc - F_cd @ inv(F_dd) @ F_dc,

    in which F_cc, F_cd, F_dc, and F_dd are the core-core, core-Drude,
    Drude-core, and Drude-Drude blocks, respectively.

    Returns the mass-weighted effective Hessian with shape
    (Batch, 3*N_core, 3*N_core).
    """

    device = q.device
    dtype = q.dtype
    batch_size = q.shape[0]
    np_total = q.shape[1]

    na = np_total // 2
    nd = np_total // 2

    q = iter.iter(ld, lcd, ldd, q, chg, b, pd, extfld, extfg, engpe, f, fcm)

    engpe, f, fcm = engfor.engfor(True, True, lst, lb, ld, lcc, lcd, ldd, lbk, q, chg, b, pst, pb, pd, pbk, extfld,
                                  extfg)

    num_mols = np_total // 6
    mol_starts = torch.arange(num_mols, device=device) * 6

    atom_offsets = torch.tensor([0, 1, 2], device=device)
    drude_offsets = torch.tensor([3, 4, 5], device=device)

    idx_atom_particles = (mol_starts.unsqueeze(1) + atom_offsets).view(-1)
    idx_drude_particles = (mol_starts.unsqueeze(1) + drude_offsets).view(-1)

    dof_offsets = torch.arange(3, device=device)

    indices_a = (idx_atom_particles.unsqueeze(1) * 3 + dof_offsets).view(-1)
    indices_d = (idx_drude_particles.unsqueeze(1) * 3 + dof_offsets).view(-1)

    H_dd = fcm[:, indices_d, :][:, :, indices_d]

    H_aa = fcm[:, indices_a, :][:, :, indices_a]

    H_ad = fcm[:, indices_a, :][:, :, indices_d]

    H_da = fcm[:, indices_d, :][:, :, indices_a]

    fdk_inv = torch.linalg.inv(H_dd)

    temp = torch.matmul(H_ad, fdk_inv)
    correction = torch.matmul(temp, H_da)

    hess = H_aa - correction

    w_expanded = wa.repeat_interleave(3).to(device)

    sqrt_w = torch.sqrt(w_expanded)

    mass_matrix = torch.outer(sqrt_w, sqrt_w)

    hess = hess / mass_matrix.unsqueeze(0)

    return hess