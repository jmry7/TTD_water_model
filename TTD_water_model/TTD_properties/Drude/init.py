import torch

def init(qa, cd, ds, poh, phh, phoh, bk):
    """
    Initialize the coordinates, topology, charges, masses, and force-field
    parameters for a batch of Drude water configurations.

    Input:

    qa : torch.Tensor, shape (Batch, na, 3)
         Coordinates of the real atoms.

    cd : torch.Tensor, shape (4,)
         Particle charges ordered as [q_O, q_H, q_DO, q_DH].

    ds : torch.Tensor, shape (2, 5)
         Anharmonic Drude spring parameters. The first row is used for the
         O-DO spring and the second row for the H-DH springs.

    poh : torch.Tensor, shape (6,)
          O-H stretching parameters ordered as [r0, k2, k3, k4, k5, k6].

    phh : torch.Tensor
          H-H stretching parameters. The first two elements are the equilibrium
          distance and harmonic force constant.

    phoh : torch.Tensor, shape (6,)
           H-O-H bending parameters ordered as [theta0, k2, k3, k4, k5, k6], where theta0 is given in degrees.

    bk : torch.Tensor, shape (3, 4)
         Buckingham parameters [A, B, C, D] for O-O, O-H, and H-H interactions, respectively.

    Outputs:

    lst : torch.Tensor, shape (N_stretch, 2)
          Particle-index pairs for bond-stretching interactions.

    lb : torch.Tensor, shape (N_bend, 3)
        Particle-index triplets for angle-bending interactions.

    ld : torch.Tensor, shape (N_Drude, 2)
         Core-Drude particle-index pairs for the spring interactions.

    lcc : torch.Tensor, shape (N_cc, 2)
          Particle-index pairs for charge-charge Coulomb interactions.

    lcd : torch.Tensor, shape (N_cd, 2)
          Particle-index pairs for charge-Drude Coulomb interactions.

    ldd : torch.Tensor, shape (N_dd, 2)
          Particle-index pairs for Drude-Drude Coulomb interactions.

    lbk : torch.Tensor, shape (N_Buckingham, 2)
          Particle-index pairs for Buckingham interactions.

    q : torch.Tensor, shape (Batch, 2*na, 3)
        Full particle coordinates ordered as [O, H1, H2, DO, DH1, DH2]
        for each water molecule. Each Drude particle is initially placed
        at the position of its corresponding core particle.

    w : torch.Tensor, shape (2*na,)
        Particle masses. The Drude-particle masses are initialized to zero.

    chg : torch.Tensor, shape (2*na,)
          Charges of all core and Drude particles.

    pst : torch.Tensor, shape (N_stretch, 6)
          Parameters for the bond-stretching interactions.

    pb : torch.Tensor, shape (N_bend, 6)
         Parameters for the angle-bending interactions.

    pd : torch.Tensor, shape (N_Drude, 5)
        Parameters [k2, k3, k4, k5, k6] for the core-Drude springs.

    pbk : torch.Tensor, shape (N_Buckingham, 4)
          Buckingham parameters [A, B, C, D] for all intermolecular pairs.

    extfld : torch.Tensor, shape (Batch, 3)
             External electric field, initialized to zero.

    extfg : torch.Tensor, shape (Batch, 3, 3)
            External electric-field gradient, initialized to zero.
    """

    device = qa.device
    dtype = qa.dtype

    engunit = torch.tensor(0.04184, dtype=dtype, device=device)
    wo = torch.tensor(15.99491, dtype=dtype, device=device)
    wh = torch.tensor(1.00783, dtype=dtype, device=device)
    pi = 3.141592653589793

    na=qa.shape[1]
    nw = na // 3

    lst_idx = []
    lb_idx = []
    ld_idx = []
    lcc_idx = []
    lcd_idx = []
    ldd_idx = []
    lbk_idx = []

    chg_list = []
    w_list = []
    pst_list = []
    pb_list = []
    pd_list = []
    pbk_list = []

    q_list = []

    # =========================================================================
    # monomer
    # =========================================================================
    for iw in range(nw):
        io = 6 * iw
        jo = 3 * iw

        # --- A. coordinates, masses, and charges ---
        # order: O, H1, H2, DO, DH1, DH2

        # 1. O Atom
        q_list.append(qa[:, jo + 0, :])
        w_list.append(wo)
        chg_list.append(cd[0])

        # 2. H1 Atom
        q_list.append(qa[:, jo + 1, :])
        w_list.append(wh)
        chg_list.append(cd[1])

        # 3. H2 Atom
        q_list.append(qa[:, jo + 2, :])
        w_list.append(wh)
        chg_list.append(cd[1])

        # 4. DO (Drude O)
        q_list.append(qa[:, jo + 0, :])
        w_list.append(torch.tensor(0.0, dtype=dtype, device=device))
        chg_list.append(cd[2])

        # 5. DH1 (Drude H1)
        q_list.append(qa[:, jo + 1, :])
        w_list.append(torch.tensor(0.0, dtype=dtype, device=device))
        chg_list.append(cd[3])

        # 6. DH2 (Drude H2)
        q_list.append(qa[:, jo + 2, :])
        w_list.append(torch.tensor(0.0, dtype=dtype, device=device))
        chg_list.append(cd[3])

        # --- B. stretching ---
        # O-H1
        lst_idx.append([io + 0, io + 1])
        p_row_oh = torch.cat([poh[0].view(1), poh[1:6] * engunit])
        pst_list.append(p_row_oh)

        # O-H2
        lst_idx.append([io + 0, io + 2])
        pst_list.append(p_row_oh)

        # H1-H2
        lst_idx.append([io + 1, io + 2])
        p_row_hh = torch.cat([phh[0].view(1), phh[1].view(1) * engunit, torch.zeros(4, device=device, dtype=dtype)])
        pst_list.append(p_row_hh)

        # --- C. bending ---
        # H1-O-H2
        lb_idx.append([io + 1, io + 0, io + 2])
        p_row_ang = torch.cat([(phoh[0] * pi / 180.0).view(1), phoh[1:6] * engunit])
        pb_list.append(p_row_ang)

        # --- D. core-Drude spring ---
        # O-DO
        ld_idx.append([io + 0, io + 3])
        pd_list.append(ds[0, 0:5] * engunit)

        # H1-DH1
        ld_idx.append([io + 1, io + 4])
        pd_list.append(ds[1, 0:5] * engunit)

        # H2-DH2
        ld_idx.append([io + 2, io + 5])
        pd_list.append(ds[1, 0:5] * engunit)

        # --- E. intramolecular ---
        # core-Drude
        # O(0) with DH1(4), DH2(5)
        lcd_idx.append([io + 0, io + 4]);
        lcd_idx.append([io + 0, io + 5])
        # H1(1) with DO(3), DH2(5)
        lcd_idx.append([io + 1, io + 3]);
        lcd_idx.append([io + 1, io + 5])
        # H2(2) with DO(3), DH1(4)
        lcd_idx.append([io + 2, io + 3]);
        lcd_idx.append([io + 2, io + 4])

        # Drude-Drude
        ldd_idx.append([io + 3, io + 4])  # DO-DH1
        ldd_idx.append([io + 3, io + 5])  # DO-DH2
        ldd_idx.append([io + 4, io + 5])  # DH1-DH2

    # =========================================================================
    # 3. Intermolecular
    # =========================================================================
    for iw in range(nw - 1):
        io = 6 * iw
        for jw in range(iw + 1, nw):
            jo = 6 * jw

            # --- core-core and Buckingham ---
            for ia in range(3):
                for ja in range(3):
                    idx_pair = [io + ia, jo + ja]
                    lcc_idx.append(idx_pair)
                    lbk_idx.append(idx_pair)

                    if ia == 0 and ja == 0:  # O-O
                        bk_type = 0
                    elif ia == 0 or ja == 0:  # O-H
                        bk_type = 1
                    else:  # H-H
                        bk_type = 2

                    bk_row = torch.stack([
                        bk[bk_type, 0] * engunit,  # A
                        bk[bk_type, 1],  # B
                        bk[bk_type, 2] * engunit,  # C
                        bk[bk_type, 3]  # D
                    ])
                    pbk_list.append(bk_row)

            # --- core-Drude (Inter) ---
            # Mol i Atoms (0,1,2) - Mol j Drudes (3,4,5)
            for ia in range(3):
                for ja in range(3, 6):
                    lcd_idx.append([io + ia, jo + ja])

            # Mol j Atoms (0,1,2) - Mol i Drudes (3,4,5)
            for ia in range(3, 6):
                for ja in range(3):
                    lcd_idx.append([jo + ja, io + ia])

            # --- Drude-Drude (Inter) ---
            # Mol i Drudes (3,4,5) - Mol j Drudes (3,4,5)
            for ia in range(3, 6):
                for ja in range(3, 6):
                    ldd_idx.append([io + ia, jo + ja])

    # =========================================================================
    # 4. Assembly
    # =========================================================================

    q = torch.stack(q_list, dim=1)

    w = torch.stack(w_list)

    chg = torch.stack(chg_list)

    if len(lst_idx) > 0:
        lst = torch.tensor(lst_idx, dtype=torch.long, device=device)
    else:
        lst = torch.zeros((0, 2), dtype=torch.long, device=device)

    if len(lb_idx) > 0:
        lb = torch.tensor(lb_idx, dtype=torch.long, device=device)
    else:
        lb = torch.zeros((0, 3), dtype=torch.long, device=device)

    if len(ld_idx) > 0:
        ld = torch.tensor(ld_idx, dtype=torch.long, device=device)
    else:
        ld = torch.zeros((0, 2), dtype=torch.long, device=device)

    if len(lcc_idx) > 0:
        lcc = torch.tensor(lcc_idx, dtype=torch.long, device=device)
    else:
        lcc = torch.zeros((0, 2), dtype=torch.long, device=device)

    if len(lcd_idx) > 0:
        lcd = torch.tensor(lcd_idx, dtype=torch.long, device=device)
    else:
        lcd = torch.zeros((0, 2), dtype=torch.long, device=device)

    if len(ldd_idx) > 0:
        ldd = torch.tensor(ldd_idx, dtype=torch.long, device=device)
    else:
        ldd = torch.zeros((0, 2), dtype=torch.long, device=device)

    if len(lbk_idx) > 0:
        lbk = torch.tensor(lbk_idx, dtype=torch.long, device=device)
    else:
        lbk = torch.zeros((0, 2), dtype=torch.long, device=device)

    if len(pst_list) > 0:
        pst = torch.stack(pst_list)
    else:
        pst = torch.zeros((0, 6), dtype=dtype, device=device)

    if len(pb_list) > 0:
        pb = torch.stack(pb_list)
    else:
        pb = torch.zeros((0, 6), dtype=dtype, device=device)

    if len(pd_list) > 0:
        pd = torch.stack(pd_list)
    else:
        pd = torch.zeros((0, 5), dtype=dtype, device=device)

    if len(pbk_list) > 0:
        pbk = torch.stack(pbk_list)
    else:
        pbk = torch.zeros((0, 4), dtype=dtype, device=device)

    # external field == 0
    batch_size = qa.shape[0]
    extfld = torch.zeros((batch_size, 3), dtype=dtype, device=device)
    extfg = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

    return lst, lb, ld, lcc, lcd, ldd, lbk, \
        q, w, chg, pst, pb, pd, pbk, \
        extfld, extfg