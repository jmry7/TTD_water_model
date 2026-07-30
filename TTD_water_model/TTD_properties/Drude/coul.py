import torch

def sf3(b, r1):
    if isinstance(b, torch.Tensor) and b.dim() == 0:
        b_val = b.item()
    elif isinstance(b, float):
        b_val = b
    else:
        b_val = b

    if b_val > 1e-8:
        return 1.0 - torch.exp(-b_val* r1** 3)
    return torch.ones_like(r1)


def sg3(b, r1, sf3_val):
    b_val = b.item() if isinstance(b, torch.Tensor) else b
    res = sf3_val * 3.0 / (r1 ** 2)
    if b_val > 1e-8:
        res = res - 3.0 * b_val* r1 * torch.exp(-b_val * r1 ** 3)
    return res


def vf3(b, r1):
    b_val = b.item() if isinstance(b, torch.Tensor) else b

    if b_val <= 1e-8:
        return torch.ones_like(r1)

    br3 = b_val * r1 ** 3
    a = 2.0 / 3.0
    a_tensor = torch.tensor(a, device=r1.device, dtype=r1.dtype)
    gamma_a = torch.exp(torch.lgamma(a_tensor))

    inc_gamma = torch.special.gammaincc(a_tensor, br3) * gamma_a

    return sf3(b, r1) + b_val**(1/3) * r1 * inc_gamma


def coul(lgrad, lfc, lc, q, chg, b, engpe, f, fcm, r4pie0=13.89354835):
    """
    Calculate the energy (engpe), forces (f), and force constant matrix (fcm)
    associated with the Thole-damped Coulomb interaction

    V_ij(r) = r4pie0*q_i*q_j*v3(r)/r,

    in which

    v3(r) = s3(r) + b^(1/3)*r*Gamma(2/3, b*r^3),
    s3(r) = 1 - exp(-b*r^3).

    Forces are evaluated when lgrad=True, and the force constant matrix is
    evaluated when lfc=True. q contains the particle coordinates, chg
    contains the particle charges, and lc contains the Coulomb interacting
    particle-pair indices.
    """

    device = q.device
    dtype = q.dtype
    batch_size = q.shape[0]

    nc = lc.shape[0]
    if nc == 0:
        return engpe, f, fcm

    idx_i = lc[:, 0]
    idx_j = lc[:, 1]

    r_vec = q[:, idx_i, :] - q[:, idx_j, :]
    r2 = torch.sum(r_vec ** 2, dim=-1)

    r2 = torch.clamp(r2, min=1e-12)
    r1 = torch.sqrt(r2)

    chgp = r4pie0 * chg[idx_i] * chg[idx_j]
    chgp = chgp.unsqueeze(0) / r1

    sf3_val = sf3(b, r1)
    vf3_val = vf3(b, r1)
    b_val = b.item() if isinstance(b, torch.Tensor) else b

    if b_val > 1e-8:
        eng_term = torch.sum(chgp * vf3_val, dim=1)
    else:
        eng_term = torch.sum(chgp, dim=1)

    new_engpe = engpe + eng_term

    if not lgrad and not lfc:
        return new_engpe, f, fcm

    fdum = sf3_val * chgp / r2

    new_f = f
    if lgrad:
        df = fdum.unsqueeze(-1) * r_vec

        delta_f = torch.zeros_like(f)

        delta_f.index_add_(1, idx_i, df)
        delta_f.index_add_(1, idx_j, -df)

        new_f = f + delta_f

    new_fcm = fcm
    if lfc:
        gdum = sg3(b, r1, sf3_val) * chgp / r2

        r_outer = torch.einsum('bni,bnj->bnij', r_vec, r_vec)
        term1 = gdum.view(batch_size, nc, 1, 1) * r_outer

        eye = torch.eye(3, device=device, dtype=dtype).view(1, 1, 3, 3)
        term2 = -fdum.view(batch_size, nc, 1, 1) * eye

        dfc = term1 + term2

        delta_fcm = torch.zeros_like(fcm)

        def scatter_block(target, r_idx, c_idx, val, sign=1.0):
            r_spatial = (r_idx * 3).view(-1, 1, 1) + torch.arange(3, device=device).view(1, 3, 1)
            c_spatial = (c_idx * 3).view(-1, 1, 1) + torch.arange(3, device=device).view(1, 1, 3)

            r_flat = r_spatial.expand(-1, 3, 3).reshape(-1)
            c_flat = c_spatial.expand(-1, 3, 3).reshape(-1)

            num_updates = r_flat.shape[0]
            batch_idx = torch.arange(batch_size, device=device).view(-1, 1).expand(-1, num_updates).reshape(-1)

            val_flat = (val * sign).reshape(-1)
            target.index_put_((batch_idx, r_flat.repeat(batch_size), c_flat.repeat(batch_size)),
                              val_flat, accumulate=True)

        scatter_block(delta_fcm, idx_i, idx_i, dfc, 1.0)
        scatter_block(delta_fcm, idx_j, idx_j, dfc, 1.0)
        scatter_block(delta_fcm, idx_i, idx_j, dfc, -1.0)
        scatter_block(delta_fcm, idx_j, idx_i, dfc, -1.0)

        new_fcm = fcm + delta_fcm

    return new_engpe, new_f, new_fcm


def extfield(q, chg, extfld, extfg, engpe, f, r4pie0=13.89354835):
    """
    Calculate the energy and forces associated with an external electric
    field and an external electric-field gradient.

    This function is not actually used in the current program
    """

    batch_size = q.shape[0]

    if extfld.dim() == 1:
        extfld = extfld.unsqueeze(0).expand(batch_size, -1)
    if extfg.dim() == 2:
        extfg = extfg.unsqueeze(0).expand(batch_size, -1, -1)

    chg_exp = chg.view(1, -1, 1)
    dip = torch.sum(chg_exp * q, dim=1)

    eng_dip = -r4pie0 * torch.sum(dip * extfld, dim=1)

    new_engpe = engpe + eng_dip

    f_dip = r4pie0 * chg.view(1, -1, 1) * extfld.unsqueeze(1)
    new_f = f + f_dip

    efgdum = torch.sum(torch.abs(extfg), dim=(1, 2))
    mask = efgdum > 1e-8

    if mask.any():
        q_outer = torch.einsum('bni,bnj->bnij', q, q)
        quad_tensor = torch.sum(chg.view(1, -1, 1, 1) * q_outer, dim=1)
        eng_quad = -0.5 * r4pie0 * torch.sum(quad_tensor * extfg, dim=(1, 2))

        new_engpe = torch.where(mask, new_engpe + eng_quad, new_engpe)

        V_sym = 0.5 * (extfg + extfg.transpose(1, 2))
        f_quad = torch.matmul(V_sym, q.transpose(1, 2)).transpose(1, 2)
        f_quad = r4pie0 * chg.view(1, -1, 1) * f_quad

        new_f = torch.where(mask.view(-1, 1, 1), new_f + f_quad, new_f)

    return new_engpe, new_f


def cdspring(lgrad, lfc, ld, q, pd, engpe, f, fcm):
    """
    Calculate the energy (engpe), forces (f), and force constant matrix (fcm)
    associated with the anharmonic core-Drude spring potential

    V(r) = k2*r^2 + k3*r^3 + k4*r^4 + k5*r^5 + k6*r^6.

    Forces are evaluated when lgrad=True, and the force constant matrix is
    evaluated when lfc=True. q contains the particle coordinates, ld
    contains the core-Drude pair indices, and pd contains the corresponding
    parameter list [k2, k3, k4, k5, k6].
    """

    device = q.device
    dtype = q.dtype
    batch_size = q.shape[0]

    nd = ld.shape[0]
    if nd == 0:
        return engpe, f, fcm

    idx_i, idx_j = ld[:, 0], ld[:, 1]
    r_vec = q[:, idx_i, :] - q[:, idx_j, :]
    r2 = torch.sum(r_vec ** 2, dim=-1)
    r1 = torch.sqrt(r2)

    pd_b = pd.unsqueeze(0)

    eng_term = pd_b[..., 0] * r2 + pd_b[..., 1] * r1 * r2 + pd_b[..., 2] * (r2 ** 2) + \
               pd_b[..., 3] * (r1 ** 5) + pd_b[..., 4] * (r1 ** 6)

    new_engpe = engpe + torch.sum(eng_term, dim=1)

    if not lgrad and not lfc:
        return new_engpe, f, fcm

    fdum = 2.0 * pd_b[..., 0] + 3.0 * pd_b[..., 1] * r1 + 4.0 * pd_b[..., 2] * r2 + \
           5.0 * pd_b[..., 3] * r1 * r2 + 6.0 * pd_b[..., 4] * (r2 ** 2)

    new_f = f
    if lgrad:
        df = -fdum.unsqueeze(-1) * r_vec

        delta_f = torch.zeros_like(f)
        delta_f.index_add_(1, idx_i, df)
        delta_f.index_add_(1, idx_j, -df)

        new_f = f + delta_f

    new_fcm = fcm
    if lfc:
        r1_safe = torch.where(r1 > 1e-8, r1, torch.ones_like(r1))
        val = 3.0 * pd_b[..., 1] / r1_safe + 8.0 * pd_b[..., 2] + \
              15.0 * pd_b[..., 3] * r1_safe + 24.0 * pd_b[..., 4] * r2
        gdum = torch.where(r1 > 1e-8, val, torch.zeros_like(r1))

        r_outer = torch.einsum('bni,bnj->bnij', r_vec, r_vec)
        term1 = gdum.view(batch_size, nd, 1, 1) * r_outer
        eye = torch.eye(3, device=device, dtype=dtype).view(1, 1, 3, 3)
        term2 = fdum.view(batch_size, nd, 1, 1) * eye
        dfc = term1 + term2

        delta_fcm = torch.zeros_like(fcm)

        def scatter_block(target, r_idx, c_idx, val, sign=1.0):
            r_spatial = (r_idx * 3).view(-1, 1, 1) + torch.arange(3, device=device).view(1, 3, 1)
            c_spatial = (c_idx * 3).view(-1, 1, 1) + torch.arange(3, device=device).view(1, 1, 3)
            r_flat = r_spatial.expand(-1, 3, 3).reshape(-1)
            c_flat = c_spatial.expand(-1, 3, 3).reshape(-1)

            num_updates = r_flat.shape[0]
            batch_idx = torch.arange(batch_size, device=device).view(-1, 1).expand(-1, num_updates).reshape(-1)
            val_flat = (val * sign).reshape(-1)
            target.index_put_((batch_idx, r_flat.repeat(batch_size), c_flat.repeat(batch_size)),
                              val_flat, accumulate=True)

        scatter_block(delta_fcm, idx_i, idx_i, dfc, 1.0)
        scatter_block(delta_fcm, idx_j, idx_j, dfc, 1.0)
        scatter_block(delta_fcm, idx_i, idx_j, dfc, -1.0)
        scatter_block(delta_fcm, idx_j, idx_i, dfc, -1.0)

        new_fcm = fcm + delta_fcm

    return new_engpe, new_f, new_fcm


def coul_drude(ld, lcd, ldd, q, chg, b, pd, extfld, extfg, engpe, f, fcm):
    """
    Calculate the Drude-related energy and forces by combining

        1. core-Drude Coulomb interactions,
        2. Drude-Drude Coulomb interactions,
        3. anharmonic core-Drude spring interactions, and
        4. interactions with the external electric field and field gradient.

    lcd and ldd contains the core-Drude and Drude-Drude interaction pairs,
    respectively. ld contains the core-Drude spring pairs. b contains the
    corresponding Thole damping factors, and pd contains the anharmonic
    spring parameters [k2, k3, k4, k5, k6].
    """

    curr_engpe = torch.zeros_like(engpe)
    curr_f = torch.zeros_like(f)
    curr_fcm = fcm

    # core-Drude
    curr_engpe, curr_f, _ = coul(True, False, lcd, q, chg, b[1], curr_engpe, curr_f, curr_fcm)

    # Drude-Drude
    curr_engpe, curr_f, _ = coul(True, False, ldd, q, chg, b[2], curr_engpe, curr_f, curr_fcm)

    # c-d Spring
    curr_engpe, curr_f, _ = cdspring(True, False, ld, q, pd, curr_engpe, curr_f, curr_fcm)

    # External Field
    curr_engpe, curr_f = extfield(q, chg, extfld, extfg, curr_engpe, curr_f)

    return curr_engpe, curr_f


def eng_drude(ld, lcd, ldd, q, chg, b, pd, extfld, extfg, engpe, f, fcm):
    """
    Calculate and return the total Drude-related potential energy, including
    the core-Drude and Drude-Drude Coulomb interactions, the anharmonic
    core-Drude spring energy, and the external-field interaction energy.
    """

    eng, _ = coul_drude(ld, lcd, ldd, q, chg, b, pd, extfld, extfg, engpe, f, fcm)
    return eng