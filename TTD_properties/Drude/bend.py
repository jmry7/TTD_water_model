import torch

def bend(lgrad, lfc, q, lb, pb, engpe, f, fcm):
    """
    Calculate the energy (engpe), forces (f), and force constant matrix (fcm)
    associated with the bond-stretching potential.

    V(r) = k2 * (r - r0)^2
         + k3 * (r - r0)^3
         + k4 * (r - r0)^4
         + k5 * (r - r0)^5
         + k6 * (r - r0)^6.

    Forces are evaluated when lgrad=True, and the force constant matrix is
    evaluated when lfc=True. q contains the particle coordinates, lst contains
    the indices of the stretching interactions, and pst contains the parameter
    list of the stretching potential [r0, k2, k3, k4, k5, k6].
    """

    batch_size = q.shape[0]
    device = q.device
    dtype = q.dtype
    nb = lb.shape[0]

    if nb == 0:
        return engpe, f, fcm

    idx_i = lb[:, 0]
    idx_j = lb[:, 1]
    idx_k = lb[:, 2]

    rji = q[:, idx_j, :] - q[:, idx_i, :]
    rjk = q[:, idx_j, :] - q[:, idx_k, :]

    rji2 = torch.sum(rji ** 2, dim=-1)
    rji1 = torch.sqrt(rji2)
    rjk2 = torch.sum(rjk ** 2, dim=-1)
    rjk1 = torch.sqrt(rjk2)
    rrr = rji1 * rjk1

    csthe = torch.sum(rji * rjk, dim=-1) / rrr
    csthe = torch.clamp(csthe, -1.0 + 1e-8, 1.0 - 1e-8)
    theta = torch.acos(csthe)

    dang = theta - pb[:, 0].unsqueeze(0)
    v_bends = torch.zeros((batch_size, nb), device=device, dtype=dtype)

    for ll in range(2, 7):
        param = pb[:, ll - 1].unsqueeze(0)
        v_bends = v_bends + param * torch.pow(dang, float(ll))

    new_engpe = engpe + torch.sum(v_bends, dim=1)

    if not lgrad and not lfc:
        return new_engpe, f, fcm

    dg_raw = torch.zeros((batch_size, nb), device=device, dtype=dtype)
    dg2 = torch.zeros((batch_size, nb), device=device, dtype=dtype)

    for ll in range(2, 7):
        param = pb[:, ll - 1].unsqueeze(0)
        term = float(ll) * param * torch.pow(dang, float(ll - 1))
        dg_raw = dg_raw + term
        if lfc:
            dg2 = dg2 + float(ll * (ll - 1)) * param * torch.pow(dang, float(ll - 2))

    snthe = torch.sin(theta)
    snthe = torch.where(torch.abs(snthe) < 1e-8, torch.sign(snthe) * 1e-8, snthe)

    dg_final = dg_raw / snthe

    rrr_unsq = rrr.unsqueeze(-1)
    csthe_unsq = csthe.unsqueeze(-1)
    rji2_unsq = rji2.unsqueeze(-1)
    rjk2_unsq = rjk2.unsqueeze(-1)

    ajk = rjk / rrr_unsq - csthe_unsq * rji / rji2_unsq
    aji = rji / rrr_unsq - csthe_unsq * rjk / rjk2_unsq

    new_f = f
    if lgrad or lfc:
        dg_final_unsq = dg_final.unsqueeze(-1)
        fji = dg_final_unsq * ajk
        fjk = dg_final_unsq * aji

        delta_f = torch.zeros_like(f)

        delta_f.index_add_(1, idx_i, -fji)
        delta_f.index_add_(1, idx_k, -fjk)
        delta_f.index_add_(1, idx_j, fji + fjk)

        new_f = f + delta_f

    new_fcm = fcm
    if lfc:
        dgg = (dg2 - csthe * dg_final) / snthe ** 2

        m_ajk_ajk = torch.einsum('bni,bnj->bnij', ajk, ajk)
        m_aji_aji = torch.einsum('bni,bnj->bnij', aji, aji)
        m_ajk_aji = torch.einsum('bni,bnj->bnij', ajk, aji)
        m_aji_ajk = torch.einsum('bni,bnj->bnij', aji, ajk)

        rjk_rji = torch.einsum('bni,bnj->bnij', rjk, rji)
        rji_rjk = torch.einsum('bni,bnj->bnij', rji, rjk)
        rji_rji = torch.einsum('bni,bnj->bnij', rji, rji)
        rjk_rjk = torch.einsum('bni,bnj->bnij', rjk, rjk)

        rrr_view = rrr.view(batch_size, nb, 1, 1)
        rji2_view = rji2.view(batch_size, nb, 1, 1)
        rjk2_view = rjk2.view(batch_size, nb, 1, 1)
        csthe_view = csthe.view(batch_size, nb, 1, 1)

        abikik = (rjk_rji + rji_rjk) / rrr_view
        abiikk = rji_rji / rji2_view + rjk_rjk / rjk2_view

        coef_dgg = dgg.view(batch_size, nb, 1, 1)
        coef_dg = dg_final.view(batch_size, nb, 1, 1)

        term1 = coef_dgg * m_ajk_ajk
        term2_geom = (abikik - 3.0 * csthe_view * rji_rji / rji2_view) / rji2_view
        ca1 = term1 + coef_dg * term2_geom

        term1 = coef_dgg * m_aji_aji
        term2_geom = (abikik - 3.0 * csthe_view * rjk_rjk / rjk2_view) / rjk2_view
        ca2 = term1 + coef_dg * term2_geom

        term1 = coef_dgg * m_ajk_aji
        term2_geom = (abiikk - csthe_view * rji_rjk / rrr_view) / rrr_view
        ca3 = term1 + coef_dg * term2_geom

        term1 = coef_dgg * m_aji_ajk
        term2_geom = (abiikk - csthe_view * rjk_rji / rrr_view) / rrr_view
        ca4 = term1 + coef_dg * term2_geom

        diag = torch.eye(3, device=device, dtype=dtype).view(1, 1, 3, 3)
        ca1 = ca1 + (coef_dg * csthe_view / rji2_view) * diag
        ca2 = ca2 + (coef_dg * csthe_view / rjk2_view) * diag
        ca3 = ca3 - (coef_dg / rrr_view) * diag
        ca4 = ca4 - (coef_dg / rrr_view) * diag

        delta_fcm = torch.zeros_like(fcm)

        def add_block(atom_row, atom_col, block_val, sign=1.0):
            r_spatial = (atom_row * 3).view(-1, 1, 1) + torch.arange(3, device=device).view(1, 3, 1)
            c_spatial = (atom_col * 3).view(-1, 1, 1) + torch.arange(3, device=device).view(1, 1, 3)

            r_flat = r_spatial.expand(-1, 3, 3).reshape(-1)
            c_flat = c_spatial.expand(-1, 3, 3).reshape(-1)

            num_updates = r_flat.shape[0]
            batch_idx = torch.arange(batch_size, device=device).view(-1, 1).expand(-1, num_updates).reshape(-1)

            vals_flat = (sign * block_val).reshape(-1)

            delta_fcm.index_put_((batch_idx, r_flat.repeat(batch_size), c_flat.repeat(batch_size)),
                                 vals_flat, accumulate=True)

        add_block(idx_i, idx_i, ca1)
        add_block(idx_j, idx_j, ca1 + ca2 + ca3 + ca4)
        add_block(idx_k, idx_k, ca2)
        add_block(idx_i, idx_k, ca3)
        add_block(idx_k, idx_i, ca4)
        add_block(idx_j, idx_i, ca1 + ca4, sign=-1.0)
        add_block(idx_i, idx_j, ca1 + ca3, sign=-1.0)
        add_block(idx_j, idx_k, ca2 + ca3, sign=-1.0)
        add_block(idx_k, idx_j, ca2 + ca4, sign=-1.0)

        new_fcm = fcm + delta_fcm

    return new_engpe, new_f, new_fcm