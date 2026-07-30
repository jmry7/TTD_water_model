import torch


def stret(lgrad, lfc, lst, pst, q, engpe, f, fcm):
    """
    Calculate the energy, forces, and second-order force-constant matrix
    associated with the anharmonic bond-stretching potential

        V(r) = k2*(r-r0)^2
             + k3*(r-r0)^3
             + k4*(r-r0)^4
             + k5*(r-r0)^5
             + k6*(r-r0)^6.

    Forces and the force-constant matrix are calculated when lgrad and lfc are True, respectively.

    Returns the updated potential energy engpe, atomic forces f, and force-constant matrix fcm.
    """
    device = q.device
    dtype = q.dtype
    batch_size = q.shape[0]

    nst = lst.shape[0]
    if nst == 0:
        return engpe, f, fcm

    idx_i = lst[:, 0]
    idx_j = lst[:, 1]

    r_vec = q[:, idx_i, :] - q[:, idx_j, :]
    r1 = torch.norm(r_vec, dim=-1)

    pst_b = pst.unsqueeze(0)

    rr = r1 - pst_b[..., 0]

    e_terms = pst_b[..., 1] * (rr ** 2) + \
              pst_b[..., 2] * (rr ** 3) + \
              pst_b[..., 3] * (rr ** 4) + \
              pst_b[..., 4] * (rr ** 5) + \
              pst_b[..., 5] * (rr ** 6)

    new_engpe = engpe + torch.sum(e_terms, dim=1)

    if not lgrad and not lfc:
        return new_engpe, f, fcm

    dE_dr = 2.0 * pst_b[..., 1] * rr + \
            3.0 * pst_b[..., 2] * (rr ** 2) + \
            4.0 * pst_b[..., 3] * (rr ** 3) + \
            5.0 * pst_b[..., 4] * (rr ** 4) + \
            6.0 * pst_b[..., 5] * (rr ** 5)

    r1_safe = torch.clamp(r1, min=1e-12)

    fdum = -dE_dr / r1_safe

    new_f = f
    if lgrad:
        df = fdum.unsqueeze(-1) * r_vec

        delta_f = torch.zeros_like(f)
        delta_f.index_add_(1, idx_i, df)
        delta_f.index_add_(1, idx_j, -df)

        new_f = f + delta_f

    new_fcm = fcm
    if lfc:
        d2E_dr2 = 2.0 * pst_b[..., 1] + \
                  6.0 * pst_b[..., 2] * rr + \
                  12.0 * pst_b[..., 3] * (rr ** 2) + \
                  20.0 * pst_b[..., 4] * (rr ** 3) + \
                  30.0 * pst_b[..., 5] * (rr ** 4)

        fcdum = d2E_dr2

        r_unit = r_vec / r1_safe.unsqueeze(-1)
        rab = torch.einsum('bni,bnj->bnij', r_unit, r_unit)

        term_rab = (fcdum + fdum).view(batch_size, nst, 1, 1)
        term_eye = -fdum.view(batch_size, nst, 1, 1)

        eye = torch.eye(3, device=device, dtype=dtype).view(1, 1, 3, 3)
        dfc = term_rab * rab + term_eye * eye

        delta_fcm = torch.zeros_like(fcm)

        def scatter_block(r_idx, c_idx, val, sign=1.0):
            r_sp = (r_idx * 3).view(-1, 1, 1) + torch.arange(3, device=device).view(1, 3, 1)
            c_sp = (c_idx * 3).view(-1, 1, 1) + torch.arange(3, device=device).view(1, 1, 3)

            r_flat = r_sp.expand(-1, 3, 3).reshape(-1)
            c_flat = c_sp.expand(-1, 3, 3).reshape(-1)

            num_updates = r_flat.shape[0]
            b_idx = torch.arange(batch_size, device=device).view(-1, 1).expand(-1, num_updates).reshape(-1)

            val_flat = (val * sign).reshape(-1)
            delta_fcm.index_put_((b_idx, r_flat.repeat(batch_size), c_flat.repeat(batch_size)),
                                 val_flat, accumulate=True)

        scatter_block(idx_i, idx_i, dfc, 1.0)
        scatter_block(idx_j, idx_j, dfc, 1.0)
        scatter_block(idx_i, idx_j, dfc, -1.0)
        scatter_block(idx_j, idx_i, dfc, -1.0)

        new_fcm = fcm + delta_fcm

    return new_engpe, new_f, new_fcm