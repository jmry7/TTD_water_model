import torch


def dipole(q, chg):
    """
    Calculate the dipole moment for each molecular configuration,

    mu = sum_i q_i * r_i.

    Returns the dipole magnitude with shape (Batch,) and the dipole vector
    with shape (Batch, 3).
        """

    dip = torch.sum(chg.view(1, -1, 1) * q, dim=1)

    dipole_val = torch.norm(dip, dim=-1)

    return dipole_val, dip


def quadrupole(q, chg):
    """
    Calculate the traceless quadrupole moment tensor for each molecular configuration,

    Returns the quadrupole tensor with shape (Batch, 3, 3).
    """

    quad1 = torch.einsum('n,bnj,bnk->bjk', chg, q, q)

    qq = torch.diagonal(quad1, dim1=-2, dim2=-1).sum(-1)

    eye = torch.eye(3, device=q.device, dtype=q.dtype).unsqueeze(0)  # (1, 3, 3)

    quad = 3.0 * quad1 - eye * qq.view(-1, 1, 1)
    quad = quad / 2.0

    return quad


def octupole(q, chg):
    """
    Calculate the traceless octupole moment tensor for each molecular configuration.

    Returns the octupole tensor with shape (Batch, 3, 3, 3).
        """

    r2 = torch.sum(q * q, dim=-1)

    term1 = torch.einsum('n,bnj,bnk,bnl->bjkl', chg, q, q, q)

    weighted_q = torch.einsum('n,bn,bnj->bj', chg, r2, q)

    eye = torch.eye(3, device=q.device, dtype=q.dtype)  # (3, 3)

    t2_1 = torch.einsum('bj,kl->bjkl', weighted_q, eye)
    t2_2 = torch.einsum('bk,jl->bjkl', weighted_q, eye)
    t2_3 = torch.einsum('bl,jk->bjkl', weighted_q, eye)

    term2 = t2_1 + t2_2 + t2_3

    ocu = 15.0 * term1 - 3.0 * term2
    ocu = ocu / 6.0

    return ocu


def hexadecapole(q, chg):
    """
    Calculate the traceless hexadecapole moment tensor for each molecular configuration.

    Returns the hexadecapole tensor with shape (Batch, 3, 3, 3, 3).
    """

    r2 = torch.sum(q * q, dim=-1)
    r4 = r2 * r2
    eye = torch.eye(3, device=q.device, dtype=q.dtype)

    term1 = torch.einsum('n,bnj,bnk,bnl,bnm->bjklm', chg, q, q, q, q)

    Q_mat = torch.einsum('n,bn,bnl,bnm->blm', chg, r2, q, q)

    t2_1 = torch.einsum('jk,blm->bjklm', eye, Q_mat)
    t2_2 = torch.einsum('jl,bkm->bjklm', eye, Q_mat)
    t2_3 = torch.einsum('jm,bkl->bjklm', eye, Q_mat)
    t2_4 = torch.einsum('kl,bjm->bjklm', eye, Q_mat)
    t2_5 = torch.einsum('km,bjl->bjklm', eye, Q_mat)
    t2_6 = torch.einsum('lm,bjk->bjklm', eye, Q_mat)

    term2 = t2_1 + t2_2 + t2_3 + t2_4 + t2_5 + t2_6

    sum_chg_r4 = torch.sum(chg * r4, dim=-1)

    e_p1 = torch.einsum('jk,lm->jklm', eye, eye)
    e_p2 = torch.einsum('jl,km->jklm', eye, eye)
    e_p3 = torch.einsum('jm,kl->jklm', eye, eye)
    eye_perms = e_p1 + e_p2 + e_p3

    term3 = sum_chg_r4.view(-1, 1, 1, 1, 1) * eye_perms.unsqueeze(0)

    hexa = 105.0 * term1 - 15.0 * term2 + 3.0 * term3
    hexa = hexa / 24.0

    return hexa