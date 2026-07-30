import torch
import itertools

def cdspring2(l2, l3, l4, q, ld, pd, fc2, fc3, fc4):
    """
    Calculate the second-, third-, and fourth-order  force-constant tensors associated
    with the anharmonic Drude spring potential

    V(r) = k2*r^2 + k3*r^3 + k4*r^4 + k5*r^5 + k6*r^6.

    The corresponding tensors are evaluated when l2, l3, and l4 are True,
    respectively. q contains the particle coordinates, ld contains the
    core-Drude pair indices, and pd contains the  parameter list [k2, k3, k4, k5, k6].
    """

    device = q.device
    dtype = q.dtype
    batch_size = q.shape[0]
    nd = ld.shape[0]

    if nd == 0:
        return fc2, fc3, fc4

    idx_i = ld[:, 0]
    idx_j = ld[:, 1]

    r_vec = q[:, idx_i, :] - q[:, idx_j, :]
    r2 = torch.sum(r_vec ** 2, dim=-1)
    r1 = torch.sqrt(r2)

    k2 = pd[:, 0].unsqueeze(0)
    k3 = pd[:, 1].unsqueeze(0)
    k4 = pd[:, 2].unsqueeze(0)
    k5 = pd[:, 3].unsqueeze(0)
    k6 = pd[:, 4].unsqueeze(0)

    fdum = 2.0 * k2 + 3.0 * k3 * r1 + 4.0 * k4 * r2 + \
           5.0 * k5 * (r1 * r2) + 6.0 * k6 * (r2 ** 2)

    r1_safe = torch.where(r1 > 1e-8, r1, torch.ones_like(r1))

    gdum_val = 3.0 * k3 / r1_safe + 8.0 * k4 + 15.0 * k5 * r1_safe + 24.0 * k6 * r2
    gdum = torch.where(r1 > 1e-8, gdum_val, torch.zeros_like(r1))

    hdum_val = -3.0 * k3 / (r1_safe ** 3) + 15.0 * k5 / r1_safe + 48.0 * k6
    hdum = torch.where(r1 > 1e-8, hdum_val, torch.zeros_like(r1))

    rdum_val = 9.0 * k3 / (r1_safe ** 5) - 15.0 * k5 / (r1_safe ** 3)
    rdum = torch.where(r1 > 1e-8, rdum_val, torch.zeros_like(r1))

    eye = torch.eye(3, device=device, dtype=dtype)
    r_outer = torch.einsum('bpi,bpj->bpij', r_vec, r_vec)  # (Batch, nd, 3, 3)

    new_fc2 = fc2
    if l2:
        df2 = gdum.view(batch_size, nd, 1, 1) * r_outer + \
              fdum.view(batch_size, nd, 1, 1) * eye.view(1, 1, 3, 3)

        delta_fc2 = torch.zeros_like(fc2)

        def scatter_2d(r_idx, c_idx, val, sign):
            r_sp = (r_idx * 3).view(-1, 1, 1) + torch.arange(3, device=device).view(1, 3, 1)
            c_sp = (c_idx * 3).view(-1, 1, 1) + torch.arange(3, device=device).view(1, 1, 3)
            r_flat = r_sp.expand(-1, 3, 3).reshape(-1)
            c_flat = c_sp.expand(-1, 3, 3).reshape(-1)

            num = r_flat.shape[0]
            b_idx = torch.arange(batch_size, device=device).view(-1, 1).expand(-1, num).reshape(-1)

            target_val = (val * sign).reshape(-1)
            delta_fc2.index_put_((b_idx, r_flat.repeat(batch_size), c_flat.repeat(batch_size)),
                                 target_val, accumulate=True)

        scatter_2d(idx_i, idx_i, df2, 1.0)
        scatter_2d(idx_j, idx_j, df2, 1.0)
        scatter_2d(idx_i, idx_j, df2, -1.0)
        scatter_2d(idx_j, idx_i, df2, -1.0)

        new_fc2 = fc2 + delta_fc2

    new_fc3 = fc3
    if l3:
        rklm = torch.einsum('bpij,bpk->bpijk', r_outer, r_vec)
        df3 = hdum.view(batch_size, nd, 1, 1, 1) * rklm

        t1 = torch.einsum('kl,bpm->bpklm', eye, r_vec)
        t2 = torch.einsum('km,bpl->bpklm', eye, r_vec)
        t3 = torch.einsum('lm,bpk->bpklm', eye, r_vec)
        df3 = df3 + gdum.view(batch_size, nd, 1, 1, 1) * (t1 + t2 + t3)

        delta_fc3 = torch.zeros_like(fc3)

        def scatter_3d(i1, i2, i3, val, sign):
            sp1 = (i1 * 3).view(-1, 1, 1, 1) + torch.arange(3, device=device).view(1, 3, 1, 1)
            sp2 = (i2 * 3).view(-1, 1, 1, 1) + torch.arange(3, device=device).view(1, 1, 3, 1)
            sp3 = (i3 * 3).view(-1, 1, 1, 1) + torch.arange(3, device=device).view(1, 1, 1, 3)

            flat1 = sp1.expand(-1, 3, 3, 3).reshape(-1)
            flat2 = sp2.expand(-1, 3, 3, 3).reshape(-1)
            flat3 = sp3.expand(-1, 3, 3, 3).reshape(-1)

            num = flat1.shape[0]
            b_idx = torch.arange(batch_size, device=device).view(-1, 1).expand(-1, num).reshape(-1)

            target_val = (val * sign).reshape(-1)
            delta_fc3.index_put_((b_idx, flat1.repeat(batch_size), flat2.repeat(batch_size), flat3.repeat(batch_size)),
                                 target_val, accumulate=True)

        pairs = [(idx_i, 'i'), (idx_j, 'j')]
        for (p1, n1), (p2, n2), (p3, n3) in itertools.product(pairs, repeat=3):
            j_cnt = (n1 == 'j') + (n2 == 'j') + (n3 == 'j')
            sign = -1.0 if (j_cnt % 2 == 1) else 1.0
            scatter_3d(p1, p2, p3, df3, sign)

        new_fc3 = fc3 + delta_fc3

    new_fc4 = fc4
    if l3 or l4:
        rklmn = torch.einsum('bpijk,bpl->bpijkl', rklm, r_vec)
        df4 = rdum.view(batch_size, nd, 1, 1, 1, 1) * rklmn

        h_sum = torch.zeros_like(df4)
        h_sum = h_sum + torch.einsum('kl,bpmn->bpklmn', eye, r_outer)
        h_sum = h_sum + torch.einsum('km,bpln->bpklmn', eye, r_outer)
        h_sum = h_sum + torch.einsum('kn,bplm->bpklmn', eye, r_outer)
        h_sum = h_sum + torch.einsum('lm,bpkn->bpklmn', eye, r_outer)
        h_sum = h_sum + torch.einsum('ln,bpkm->bpklmn', eye, r_outer)
        h_sum = h_sum + torch.einsum('mn,bpkl->bpklmn', eye, r_outer)

        df4 = df4 + hdum.view(batch_size, nd, 1, 1, 1, 1) * h_sum

        ee1 = torch.einsum('kl,mn->klmn', eye, eye).unsqueeze(0).unsqueeze(0)
        ee2 = torch.einsum('km,ln->klmn', eye, eye).unsqueeze(0).unsqueeze(0)
        ee3 = torch.einsum('kn,lm->klmn', eye, eye).unsqueeze(0).unsqueeze(0)
        g_sum = ee1 + ee2 + ee3

        df4 = df4 + gdum.view(batch_size, nd, 1, 1, 1, 1) * g_sum

        delta_fc4 = torch.zeros_like(fc4)

        def scatter_4d(i1, i2, i3, i4, val, sign):
            sp1 = (i1 * 3).view(-1, 1, 1, 1, 1) + torch.arange(3, device=device).view(1, 3, 1, 1, 1)
            sp2 = (i2 * 3).view(-1, 1, 1, 1, 1) + torch.arange(3, device=device).view(1, 1, 3, 1, 1)
            sp3 = (i3 * 3).view(-1, 1, 1, 1, 1) + torch.arange(3, device=device).view(1, 1, 1, 3, 1)
            sp4 = (i4 * 3).view(-1, 1, 1, 1, 1) + torch.arange(3, device=device).view(1, 1, 1, 1, 3)

            f1 = sp1.expand(-1, 3, 3, 3, 3).reshape(-1)
            f2 = sp2.expand(-1, 3, 3, 3, 3).reshape(-1)
            f3 = sp3.expand(-1, 3, 3, 3, 3).reshape(-1)
            f4 = sp4.expand(-1, 3, 3, 3, 3).reshape(-1)

            num = f1.shape[0]
            b_idx = torch.arange(batch_size, device=device).view(-1, 1).expand(-1, num).reshape(-1)

            target_val = (val * sign).reshape(-1)
            delta_fc4.index_put_((b_idx, f1.repeat(batch_size), f2.repeat(batch_size),
                                  f3.repeat(batch_size), f4.repeat(batch_size)),
                                 target_val, accumulate=True)

        pairs = [(idx_i, 'i'), (idx_j, 'j')]
        for (p1, n1), (p2, n2), (p3, n3), (p4, n4) in itertools.product(pairs, repeat=4):
            j_cnt = (n1 == 'j') + (n2 == 'j') + (n3 == 'j') + (n4 == 'j')
            sign = 1.0 if (j_cnt % 2 == 0) else -1.0
            scatter_4d(p1, p2, p3, p4, df4, sign)

        new_fc4 = fc4 + delta_fc4

    return new_fc2, new_fc3, new_fc4
