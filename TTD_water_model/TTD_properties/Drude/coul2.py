import torch
import itertools


def sf3(b, r1):
    b_val = b.item() if isinstance(b, torch.Tensor) else b
    if b_val > 1e-8:
        return 1.0 - torch.exp(-b_val * r1 ** 3)
    return torch.ones_like(r1)

def sg3(b, r1, sf3_val):
    b_val = b.item() if isinstance(b, torch.Tensor) else b
    res = sf3_val * 3.0 / (r1 ** 2)
    if b_val > 1e-8:
        term = 3.0 * b_val * r1 * torch.exp(-b_val * r1** 3)
        res = res - term
    return res

def sh3(b, r1, sf3_val):
    b_val = b.item() if isinstance(b, torch.Tensor) else b
    sg3_val = sg3(b, r1, sf3_val)
    res = sg3_val * 5.0 / (r1 ** 2)
    if b_val > 1e-8:
        term = 9.0 * (b_val ** 2) * (r1 ** 2) * torch.exp(-b_val * r1 ** 3)
        res = res - term
    return res

def sr3(b, r1, sf3_val):
    b_val = b.item() if isinstance(b, torch.Tensor) else b
    sg3_val = sg3(b, r1, sf3_val)
    sh3_val = sh3(b, r1, sf3_val)
    res = sh3_val * 6.0 / (r1 ** 2) + sg3_val * 5.0 / (r1 ** 4)
    if b_val > 1e-8:
        term = 27.0 * (b_val ** 3) * (r1 ** 3) * torch.exp(-b_val * r1 ** 3)
        res = res - term
    return res

def coul2(l2, l3, l4, lc, q, fc2, fc3, fc4, chg, b, r4pie0=13.89354835):
    """
    Calculate the second-, third-, and fourth-order Cartesian force-constant
    tensors associated with the third-order Thole-damped Coulomb potential

    V_ij(r) = r4pie0 * q_i * q_j * v3(r) / r,

    in which

    v3(r) = s3(r) + b^(1/3)*r*Gamma(2/3, b*r^3),
    s3(r) = 1 - exp(-b*r^3).

    Gamma(a, x) denotes the incomplete gamma function. The calculated tensors are

    fc2 = d^2 V / d q^2,
    fc3 = d^3 V / d q^3,
    fc4 = d^4 V / d q^4,

    and are evaluated when l2, l3, and l4 are True, respectively.

    q contains the particle coordinates, chg contains the particle charges,
    and lc contains the Coulomb interacting particle-pair indices. b contains the
    Thole damping factors.
        """

    device = q.device
    dtype = q.dtype
    batch_size = q.shape[0]
    nc = lc.shape[0]

    if nc == 0:
        return fc2, fc3, fc4

    idx_i = lc[:, 0]
    idx_j = lc[:, 1]

    r_vec = q[:, idx_i, :] - q[:, idx_j, :]
    r2 = torch.sum(r_vec ** 2, dim=-1)
    r2 = torch.clamp(r2, min=1e-12)
    r1 = torch.sqrt(r2)

    chgp_base = r4pie0 * chg[idx_i] * chg[idx_j]
    chgp = chgp_base.unsqueeze(0) / r1

    sf3_val = sf3(b, r1)
    fdum = sf3_val * chgp / r2
    gdum = sg3(b, r1, sf3_val) * chgp / r2

    hdum = None
    if l3 or l4:
        hdum = sh3(b, r1, sf3_val) * chgp / r2

    rdum = None
    if l4:
        rdum = sr3(b, r1, sf3_val) * chgp / r2

    eye = torch.eye(3, device=device, dtype=dtype)
    r_outer = torch.einsum('bpi,bpj->bpij', r_vec, r_vec)

    new_fc2 = fc2
    if l2:
        df2 = gdum.view(batch_size, nc, 1, 1) * r_outer - \
              fdum.view(batch_size, nc, 1, 1) * eye.view(1, 1, 3, 3)

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
    if l3 or l4:
        rklm = torch.einsum('bpij,bpk->bpijk', r_outer, r_vec)
        df3 = -hdum.view(batch_size, nc, 1, 1, 1) * rklm

        t1 = torch.einsum('kl,bpm->bpklm', eye, r_vec)
        t2 = torch.einsum('km,bpl->bpklm', eye, r_vec)
        t3 = torch.einsum('lm,bpk->bpklm', eye, r_vec)
        df3 = df3 + gdum.view(batch_size, nc, 1, 1, 1) * (t1 + t2 + t3)

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
    if l4:
        rklmn = torch.einsum('bpijk,bpl->bpijkl', rklm, r_vec)
        df4 = rdum.view(batch_size, nc, 1, 1, 1, 1) * rklmn

        h_sum = torch.zeros_like(df4)
        h_sum = h_sum + torch.einsum('kl,bpmn->bpklmn', eye, r_outer)
        h_sum = h_sum + torch.einsum('km,bpln->bpklmn', eye, r_outer)
        h_sum = h_sum + torch.einsum('kn,bplm->bpklmn', eye, r_outer)
        h_sum = h_sum + torch.einsum('lm,bpkn->bpklmn', eye, r_outer)
        h_sum = h_sum + torch.einsum('ln,bpkm->bpklmn', eye, r_outer)
        h_sum = h_sum + torch.einsum('mn,bpkl->bpklmn', eye, r_outer)

        df4 = df4 - hdum.view(batch_size, nc, 1, 1, 1, 1) * h_sum

        ee1 = torch.einsum('kl,mn->klmn', eye, eye).unsqueeze(0).unsqueeze(0)
        ee2 = torch.einsum('km,ln->klmn', eye, eye).unsqueeze(0).unsqueeze(0)
        ee3 = torch.einsum('kn,lm->klmn', eye, eye).unsqueeze(0).unsqueeze(0)
        g_sum = ee1 + ee2 + ee3

        df4 = df4 + gdum.view(batch_size, nc, 1, 1, 1, 1) * g_sum

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
