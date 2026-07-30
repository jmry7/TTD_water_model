import torch
import invdc
import coul2
import cdspring2

R4PIE0 = 13.89354835
# ---------------------------------------------------------------------------
# Notation used in the analytical response calculations
#     d, d', d'', e, e', ...    : Drude-particle indices
#     alpha, beta, gamma, ...   : Cartesian indices x, y, and z
#
# In the torch.einsum expressions:
#     z                         : batch index
#     i, j, a, b, c, ...        : Drude-particle indices
#     k, l, m, n, ...           : Cartesian indices x, y, and z
# ---------------------------------------------------------------------------

def get_drude_data(ld, chg, device):
    """
    Extract the global Drude-particle indices and their charges.
    ld[:, 1] contains the Drude-particle index d, and chg_d[d] corresponds
    to the Drude charge q_d used in the analytical response equations.
    """
    drude_global_indices = ld[:, 1]
    chg_d = chg[drude_global_indices]
    return drude_global_indices, chg_d


def pol(ld, lcd, ldd, q, engpe, f, fcm, chg, b, pd, extfld, extfg):
    """
    Calculate the dipole polarizability alpha,

    alpha_{alpha beta} = q_d q_d' f^(-1)_{d alpha, d' beta}.

    Code-index correspondence:

        alp[z, k, l]                        -> alpha_{alpha beta}
        chg_d[i], chg_d[j]                  -> q_d, q_d'
        fdk_inv_reshaped[z, i, k, j, l]     -> f^(-1)_{d alpha,d' beta}

    Here z is the batch index, i and j are Drude-particle indices, and
    k and l correspond to the Cartesian indices alpha and beta.

    Returns alp with shape (Batch, 3, 3).
    """

    device = q.device
    dtype = q.dtype
    batch_size = q.shape[0]
    nd = ld.shape[0]
    np_atoms = q.shape[1]

    f_local = torch.zeros((batch_size, np_atoms, 3), device=device, dtype=dtype)
    fcm_local = torch.zeros((batch_size, 3 * np_atoms, 3 * np_atoms), device=device, dtype=dtype)

    fdk_inv = invdc.invdc(ld, lcd, ldd, q, chg, b, pd, extfld, extfg, engpe, f_local, fcm_local)

    _, chg_d = get_drude_data(ld, chg, device)
    fdk_inv_reshaped = fdk_inv.view(batch_size, nd, 3, nd, 3)

    alp = torch.einsum('i,j,zikjl->zkl', chg_d, chg_d, fdk_inv_reshaped)
    alp = alp * R4PIE0
    return alp


def pol2(ld, lcd, ldd, q, chg, b, pd, extfld, extfg, engpe, f, fcm):
    """
    Calculate the first hyperpolarizability beta,

    beta_{alpha beta gamma} = -q_d q_d' q_d'' B_{d alpha, d' beta, d'' gamma},

    in which

    B_{d alpha, d' beta, d'' gamma} = f_{e eta, e' xi, e'' kappa}
                                      f^(-1)_{e eta, d alpha}
                                      f^(-1)_{e' xi, d' beta}
                                      f^(-1)_{e'' kappa, d'' gamma}.

    Code-index correspondence:

        f3_d[z, a, n, b, o, c, p] -> f_{e eta, e' xi, e'' kappa}

        fdk_r[z, a, n, d, k] -> f^(-1)_{e eta, d alpha}

        beta3[z, d, k, e, l, f, m] -> B_{d alpha, d' beta, d'' gamma}

        beta[z, k, l, m] -> beta_{alpha beta gamma}

    In the final contraction, d, e, and f are Drude-particle indices,
    whereas k, l, and m correspond to alpha, beta, and gamma.

    Returns beta with shape (Batch, 3, 3, 3) and beta3 with shape (Batch, 3*N_Drude, 3*N_Drude, 3*N_Drude).
        """

    device = q.device
    dtype = q.dtype
    batch_size = q.shape[0]
    np_atoms = q.shape[1]
    nd = ld.shape[0]

    f_local = torch.zeros((batch_size, np_atoms, 3), device=device, dtype=dtype)
    fcm_local = torch.zeros((batch_size, 3 * np_atoms, 3 * np_atoms), device=device, dtype=dtype)

    fdk_inv = invdc.invdc(ld, lcd, ldd, q, chg, b, pd, extfld, extfg, engpe, f_local, fcm_local)
    id_idx, chg_d = get_drude_data(ld, chg, device)

    fc3 = torch.zeros((batch_size, 3 * np_atoms, 3 * np_atoms, 3 * np_atoms), device=device, dtype=dtype)
    fc4 = torch.zeros((batch_size, 3 * np_atoms, 3 * np_atoms, 3 * np_atoms, 3 * np_atoms), device=device, dtype=dtype)
    fcm_tmp = torch.zeros_like(fcm_local)

    _, fc3, _ = coul2.coul2(False, True, False, lcd, q, fcm_tmp, fc3, fc4, chg, b[1])
    _, fc3, _ = coul2.coul2(False, True, False, ldd, q, fcm_tmp, fc3, fc4, chg, b[2])
    _, fc3, _ = cdspring2.cdspring2(False, True, False, q, ld, pd, fcm_tmp, fc3, fc4)

    dof_offsets = torch.arange(3, device=device)
    indices_d = (id_idx.unsqueeze(1) * 3 + dof_offsets).view(-1)

    f3_d_flat = fc3[:, indices_d, :, :][:, :, indices_d, :][:, :, :, indices_d]
    f3_d = f3_d_flat.view(batch_size, nd, 3, nd, 3, nd, 3)

    fdk_r = fdk_inv.view(batch_size, nd, 3, nd, 3)

    beta3 = torch.einsum('zanbocp,zandk,zboel,zcpfm->zdkelfm', f3_d, fdk_r, fdk_r, fdk_r)

    beta = -torch.einsum('zdkelfm,d,e,f->zklm', beta3, chg_d, chg_d, chg_d)

    return beta * (R4PIE0 ** 2), beta3.reshape(batch_size, 3 * nd, 3 * nd, 3 * nd)


def pol3(ld, lcd, ldd, q, engpe, f, fcm, chg, b, pd, extfld, extfg):
    """
    Calculate the second hyperpolarizability gamma.

    The intermediate tensors correspond to

        f3_d    -> third-order force constants f_{d alpha, d' beta, d'' gamma}

        f4_d    -> fourth-order force constants f_{d alpha, d' beta, d'' gamma, d''' delta}

        fdk_r   -> inverse Drude Hessian f^(-1)_{d alpha, d' beta}

        beta3_r -> B_{d alpha, d' beta, d'' gamma}

    Code-index correspondence:

        gama[z, k, l, m, n] -> gamma_{alpha beta gamma delta}

    in which k, l, m, and n correspond to alpha, beta, gamma, and delta.

    Returns gama with shape (Batch, 3, 3, 3, 3).
    """

    batch_size = q.shape[0]
    np_atoms = q.shape[1]
    nd = ld.shape[0]
    device = q.device
    dtype = q.dtype

    f_local = torch.zeros((batch_size, np_atoms, 3), device=device, dtype=dtype)
    fcm_local = torch.zeros((batch_size, 3 * np_atoms, 3 * np_atoms), device=device, dtype=dtype)

    fdk_inv = invdc.invdc(ld, lcd, ldd, q, chg, b, pd, extfld, extfg, engpe, f_local, fcm_local)
    _, beta3_flat = pol2(ld, lcd, ldd, q, chg, b, pd, extfld, extfg, engpe, f_local, fcm_local)

    f3 = torch.zeros((batch_size, 3 * np_atoms, 3 * np_atoms, 3 * np_atoms), device=device, dtype=dtype)
    f4 = torch.zeros((batch_size, 3 * np_atoms, 3 * np_atoms, 3 * np_atoms, 3 * np_atoms), device=device, dtype=dtype)
    fcm_tmp = torch.zeros_like(fcm_local)

    _, f3, f4 = coul2.coul2(True, True, True, lcd, q, fcm_tmp, f3, f4, chg, b[1])
    _, f3, f4 = coul2.coul2(True, True, True, ldd, q, fcm_tmp, f3, f4, chg, b[2])
    _, f3, f4 = cdspring2.cdspring2(True, True, True, q, ld, pd, fcm_tmp, f3, f4)

    id_idx, chg_d = get_drude_data(ld, chg, device)
    dof_offsets = torch.arange(3, device=device)
    indices_d = (id_idx.unsqueeze(1) * 3 + dof_offsets).view(-1)

    f3_d = f3[:, indices_d, :, :][:, :, indices_d, :][:, :, :, indices_d].view(batch_size, nd, 3, nd, 3, nd, 3)
    f4_d = f4[:, indices_d, :, :, :][:, :, indices_d, :, :][:, :, :, indices_d, :][:, :, :, :, indices_d].view(
        batch_size, nd, 3, nd, 3, nd, 3, nd, 3)

    fdk_r = fdk_inv.view(batch_size, nd, 3, nd, 3)
    beta3_r = beta3_flat.view(batch_size, nd, 3, nd, 3, nd, 3)

    term_a1 = torch.einsum('zeofpgq,zeoakdn,zfpbl,zgqcm->zakblcmdn', f3_d, beta3_r, fdk_r, fdk_r)
    term_a2 = torch.einsum('zeofpgq,zfpbldn,zeoak,zgqcm->zakblcmdn', f3_d, beta3_r, fdk_r, fdk_r)
    term_a3 = torch.einsum('zeofpgq,zgqcmdn,zeoak,zfpbl->zakblcmdn', f3_d, beta3_r, fdk_r, fdk_r)
    part_a = term_a1 + term_a2 + term_a3

    part_b = torch.einsum('zeofpgqhr,zeoak,zfpbl,zgqcm,zhrdn->zakblcmdn', f4_d, fdk_r, fdk_r, fdk_r, fdk_r)

    gama = torch.einsum('zakblcmdn,a,b,c,d->zklmn', (part_a - part_b), chg_d, chg_d, chg_d, chg_d)

    gama = gama * (R4PIE0 ** 3)
    return gama


def polA(ld, lcd, ldd, q, chg, b, pd, extfld, extfg, engpe, f, fcm):
    """
    Calculate the dipole-quadrupole polarizability.

    The primitive tensor is

        a_{gamma, alpha beta} = q_d q_d' [ f^(-1)_{d alpha, d' gamma} r_{d beta}
                                         + f^(-1)_{d beta, d' gamma} r_{d alpha} ].

    Code-index correspondence:

        A1[z, m, k, l]          -> a_{gamma, alpha beta}
        q_d[z, i, k]            -> r_{d alpha}
        fdk_inv[z, i, l, j, m]  -> f^(-1)_{d beta, d' gamma}

    m corresponds to gamma, while k and l correspond to alpha and beta.

    A1 is the primitive tensor a_{gamma,alpha beta}, and A is its traceless form.

    Returns A and A1, both with shape (Batch, 3, 3, 3).
    """

    device, dtype = q.device, q.dtype
    batch_size = q.shape[0]
    nd = ld.shape[0]
    np_atoms = q.shape[1]
    id_idx, chg_d = get_drude_data(ld, chg, device)

    f_local = torch.zeros((batch_size, np_atoms, 3), device=device, dtype=dtype)
    fcm_local = torch.zeros((batch_size, 3 * np_atoms, 3 * np_atoms), device=device, dtype=dtype)

    fdk_inv = invdc.invdc(ld, lcd, ldd, q, chg, b, pd, extfld, extfg, engpe, f_local, fcm_local).view(batch_size, nd, 3,
                                                                                                      nd, 3)
    q_d = q[:, id_idx, :]

    term1 = torch.einsum('i,j,zik,ziljm->zmkl', chg_d, chg_d, q_d, fdk_inv)
    term2 = torch.einsum('i,j,zil,zikjm->zmkl', chg_d, chg_d, q_d, fdk_inv)

    A1 = (term1 + term2) * R4PIE0

    qqq = torch.einsum('zmkk->zm', A1)
    eye = torch.eye(3, device=device, dtype=dtype).view(1, 1, 3, 3)

    A = 1.5 * A1 - 0.5 * qqq.view(batch_size, 3, 1, 1) * eye
    return A, A1


def polB(ld, lcd, ldd, q, chg, b, pd, extfld, extfg, engpe, f, fcm):
    """
    Calculate the dipole-dipole-quadrupole polarizability.

    The primitive tensor is

         b_{alpha beta, gamma delta} = q_d q_d' q_d'' [ f^(-1)_{d alpha, d' gamma} f^(-1)_{d' delta, d'' beta}
                                                      + f^(-1)_{d alpha, d' delta} f^(-1)_{d' gamma, d'' beta}
                                                      - B_{d alpha, d' gamma, d'' beta} r_{d' delta}
                                                      - B_{d alpha, d' delta, d'' beta} r_{d' gamma} ].

    Code-index correspondence:

        B1[z, m, n, k, l] -> b_{alpha beta, gamma delta}

    in which m, n, k, and l correspond to alpha, beta, gamma, and delta, respectively.
    fdk_r represents f^(-1)_{d alpha,d' beta},
    beta3_r represents B_{d alpha,d' beta,d'' gamma}, and
    q_d represents r_{d alpha}.

    B1 is the primitive tensor b_{alpha beta,gamma delta}. B is its traceless form.

    Returns B and B1, both with shape (Batch, 3, 3, 3, 3).
    """

    device, dtype = q.device, q.dtype
    batch_size = q.shape[0]
    nd = ld.shape[0]
    np_atoms = q.shape[1]
    id_idx, chg_d = get_drude_data(ld, chg, device)

    f_local = torch.zeros((batch_size, np_atoms, 3), device=device, dtype=dtype)
    fcm_local = torch.zeros((batch_size, 3 * np_atoms, 3 * np_atoms), device=device, dtype=dtype)

    q_d = q[:, id_idx, :]
    fdk_r = invdc.invdc(ld, lcd, ldd, q, chg, b, pd, extfld, extfg, engpe, f_local, fcm_local).view(batch_size, nd, 3,
                                                                                                    nd, 3)
    _, beta3_flat = pol2(ld, lcd, ldd, q, chg, b, pd, extfld, extfg, engpe, f_local, fcm_local)
    beta3_r = beta3_flat.view(batch_size, nd, 3, nd, 3, nd, 3)

    term_1 = torch.einsum('a,b,c,zakbm,zalcn->zmnkl', chg_d, chg_d, chg_d, fdk_r, fdk_r)
    term_2 = torch.einsum('a,b,c,zakcn,zalbm->zmnkl', chg_d, chg_d, chg_d, fdk_r, fdk_r)
    term_3 = torch.einsum('a,b,c,zakbmcn,zal->zmnkl', chg_d, chg_d, chg_d, beta3_r, q_d)
    term_4 = torch.einsum('a,b,c,zalbmcn,zak->zmnkl', chg_d, chg_d, chg_d, beta3_r, q_d)

    B1 = (term_1 + term_2 - term_3 - term_4) * (R4PIE0 ** 2)

    qqq = torch.einsum('zmnkk->zmn', B1)
    eye = torch.eye(3, device=device, dtype=dtype).view(1, 1, 1, 3, 3)
    B = 1.5 * B1 - 0.5 * qqq.view(batch_size, 3, 3, 1, 1) * eye
    return B, B1

def polC(ld, lcd, ldd, q, chg, b, pd, extfld, extfg, engpe, f, fcm):
    """
    Calculate the  quadrupole-quadrupole polarizability.

    The primitive tensor is

        c_{alpha beta, gamma delta} = (1/2) q_d q_d' [ (f^(-1)_{d alpha, d' delta} r_{d beta}
                                                     + f^(-1)_{d beta, d' delta} r_{d alpha}) r_{d' gamma}
                                                     + (f^(-1)_{d alpha, d' gamma} r_{d beta}
                                                     + f^(-1)_{d beta, d' gamma} r_{d alpha}) r_{d' delta} ].

    Code-index correspondence:

        C1[z, k, l, m, n] -> c_{alpha beta, gamma delta}

    in which k, l, m, and n correspond to alpha, beta, gamma, and delta, respectively.
    fdk_r represents f^(-1)_{d alpha,d' beta}, and q_d represents r_{d alpha}.

    C1 is the primitive tensor c_{alpha beta,gamma delta}. C is its traceless form.

    Returns C and C1, both with shape (Batch, 3, 3, 3, 3).
    """

    device, dtype = q.device, q.dtype
    batch_size = q.shape[0]
    nd = ld.shape[0]
    np_atoms = q.shape[1]
    id_idx, chg_d = get_drude_data(ld, chg, device)

    f_local = torch.zeros((batch_size, np_atoms, 3), device=device, dtype=dtype)
    fcm_local = torch.zeros((batch_size, 3 * np_atoms, 3 * np_atoms), device=device, dtype=dtype)

    q_d = q[:, id_idx, :]
    fdk_r = invdc.invdc(ld, lcd, ldd, q, chg, b, pd, extfld, extfg, engpe, f_local, fcm_local).view(batch_size, nd, 3,
                                                                                                    nd, 3)
    t1 = torch.einsum('a,b,zal,zakbn,zbm->zklmn', chg_d, chg_d, q_d, fdk_r, q_d)
    t2 = torch.einsum('a,b,zal,zakbm,zbn->zklmn', chg_d, chg_d, q_d, fdk_r, q_d)
    t3 = torch.einsum('a,b,zak,zalbn,zbm->zklmn', chg_d, chg_d, q_d, fdk_r, q_d)
    t4 = torch.einsum('a,b,zak,zalbm,zbn->zklmn', chg_d, chg_d, q_d, fdk_r, q_d)

    C1 = (t1 + t2 + t3 + t4) * (0.5 * R4PIE0)

    qqqq = torch.einsum('zkkll->z', C1)
    qqkl = torch.einsum('zklmm->zkl', C1)
    qqmn = torch.einsum('zkkmn->zmn', C1)

    eye = torch.eye(3, device=device, dtype=dtype)
    C = 1.5 * C1
    C = C - 0.5 * qqkl.view(batch_size, 3, 3, 1, 1) * eye.view(1, 1, 1, 3, 3)
    C = C - 0.5 * qqmn.view(batch_size, 1, 1, 3, 3) * eye.view(1, 3, 3, 1, 1)
    C = C + (qqqq.view(batch_size, 1, 1, 1, 1) / 6.0) * eye.view(1, 3, 3, 1, 1) * eye.view(1, 1, 1, 3, 3)

    return C, C1


def polD(ld, lcd, ldd, q, chg, b, pd, extfld, extfg, engpe, f, fcm):
    """
    Calculate the dipole–octupole polarizability.

    The primitive tensor is

        d_{delta, alpha beta gamma} = q_d q_d' [ f^(-1)_{d alpha, d' delta} r_{d beta} r_{d gamma}
                                               + f^(-1)_{d beta, d' delta} r_{d alpha} r_{d gamma}
                                               + f^(-1)_{d gamma, d' delta} r_{d alpha} r_{d beta} ].

    Code-index correspondence:

        D1[z, k, l, m, n] -> d_{delta, alpha beta gamma}

    in which k corresponds to delta, and l, m, and n correspond to alpha, beta, and gamma.

    D1 is the primitive tensor d_{delta,alpha beta gamma}. D is its traceless form.

    Returns D and D1, both with shape (Batch, 3, 3, 3, 3).
    """

    device, dtype = q.device, q.dtype
    batch_size = q.shape[0]
    nd = ld.shape[0]
    np_atoms = q.shape[1]
    id_idx, chg_d = get_drude_data(ld, chg, device)

    f_local = torch.zeros((batch_size, np_atoms, 3), device=device, dtype=dtype)
    fcm_local = torch.zeros((batch_size, 3 * np_atoms, 3 * np_atoms), device=device, dtype=dtype)

    q_d = q[:, id_idx, :]
    fdk_r = invdc.invdc(ld, lcd, ldd, q, chg, b, pd, extfld, extfg, engpe, f_local, fcm_local).view(batch_size, nd, 3,
                                                                                                    nd, 3)

    t1 = torch.einsum('a,b,zam,zan,zalbk->zklmn', chg_d, chg_d, q_d, q_d, fdk_r)
    t2 = torch.einsum('a,b,zal,zan,zambk->zklmn', chg_d, chg_d, q_d, q_d, fdk_r)
    t3 = torch.einsum('a,b,zal,zam,zanbk->zklmn', chg_d, chg_d, q_d, q_d, fdk_r)

    D1 = (t1 + t2 + t3) * R4PIE0

    qql = torch.einsum('zklmm->zkl', D1)
    qqm = torch.einsum('zklml->zkm', D1)
    qqn = torch.einsum('zklln->zkn', D1)

    eye = torch.eye(3, device=device, dtype=dtype)
    D = 2.5 * D1
    D = D - 0.5 * qql.view(batch_size, 3, 3, 1, 1) * eye.view(1, 1, 1, 3, 3)
    D = D - 0.5 * qqm.view(batch_size, 3, 1, 3, 1) * eye.view(1, 1, 3, 1, 3)
    D = D - 0.5 * qqn.view(batch_size, 3, 1, 1, 3) * eye.view(1, 1, 3, 3, 1)

    return D, D1