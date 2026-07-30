import torch

def buck(lgrad, lfc, q, lbk, pbk, engpe, f, fcm):
    """
    Calculate the energy (engpe), forces (f), and force constant matrix (fcm)
    associated with the damped Buckingham potential

    V(r) = A * exp(-B*r) - C * f6(D*r) / r^6,
    f6(z) = 1 - exp(-z) * sum_{n=0}^{6} z^n / n!.

    Forces are evaluated when
    lgrad=True, and the force constant matrix is evaluated when lfc=True.
    q contains the particle coordinates, lbk contains the particle-pair indices
    for the Buckingham interactions, and pbk contains the parameter list of
    the Buckingham potential [A, B, C, D].
    """

    batch_size = q.shape[0]
    device = q.device
    dtype = q.dtype

    active_mask = pbk[:, 0] >= 1e-8

    if not active_mask.any():
        return engpe, f, fcm

    lbk_active = lbk[active_mask]
    pbk_active = pbk[active_mask]

    idx_i = lbk_active[:, 0]
    idx_j = lbk_active[:, 1]

    pk_a = pbk_active[:, 0].unsqueeze(0)
    pk_b = pbk_active[:, 1].unsqueeze(0)
    pk_c = pbk_active[:, 2].unsqueeze(0)
    pk_d = pbk_active[:, 3].unsqueeze(0)

    r_vec = q[:, idx_i, :] - q[:, idx_j, :]
    r1 = torch.norm(r_vec, dim=-1)
    r1 = torch.clamp(r1, min=1e-12)

    z = pk_d * r1
    exp_z = torch.exp(-z)
    exp_br = torch.exp(-pk_b * r1)

    poly_term = 1.0 + z + (z ** 2) / 2.0 + (z ** 3) / 6.0 + (z ** 4) / 24.0 + (z ** 5) / 120.0 + (z ** 6) / 720.0
    vf1 = 1.0 - exp_z * poly_term

    r2 = r1 * r1
    r6 = r2 * r2 * r2
    term_disp = (pk_c / r6) * vf1
    v_bks = pk_a * exp_br - term_disp

    new_engpe = engpe + torch.sum(v_bks, dim=1)

    if not lgrad and not lfc:
        return new_engpe, f, fcm

    sf1 = vf1 - exp_z * ((z ** 7) / 4320.0)
    r8 = r6 * r2

    fdum = (pk_a * pk_b / r1) * exp_br - (6.0 * pk_c / r8) * sf1

    new_f = f
    if lgrad or lfc:
        df = fdum.unsqueeze(-1) * r_vec

        delta_f = torch.zeros_like(f)

        delta_f.index_add_(1, idx_i, df)
        delta_f.index_add_(1, idx_j, -df)

        new_f = f + delta_f

    new_fcm = fcm
    if lfc:
        sf2 = 42.0 * vf1 - exp_z * ((z ** 7) / 120.0 + (z ** 8) / 720.0)

        gdum = pk_a * (pk_b ** 2) * exp_br - (pk_c / r8) * sf2

        r_outer = torch.einsum('bni,bnj->bnij', r_vec, r_vec)

        r2_view = r2.view(batch_size, -1, 1, 1)
        fdum_view = fdum.view(batch_size, -1, 1, 1)
        gdum_view = gdum.view(batch_size, -1, 1, 1)
        eye = torch.eye(3, device=device, dtype=dtype).view(1, 1, 3, 3)

        term1 = (fdum_view + gdum_view) * r_outer / r2_view
        term2 = -fdum_view * eye

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