import torch
import cmtrans
import engfor
import iter
import newton
import nmode
import init
import multipole
import pol
import distoo


def drude_wat(lgrad, lfc, ldip, lpol, lpol2, lquad, lopt, lmode, lg, loo,
                qa, cd, b, ds, poh, phh, phoh, bk):
    """
    Perform the requested calculations for a batch of Drude-water configurations.

    Parameters
    ----------
    lgrad : bool
            If True, calculate the forces on the atomic cores.

    lfc : bool
           If True, calculate the force-constant matrix during the energy and force evaluation.

    ldip : bool
           If True, calculate the dipole moment.

    lpol : bool
           If True, calculate the dipole polarizability alpha.

    lpol2 : bool
            If True, calculate beta, gamma, and the A, B, C, and D tensors.

    lquad : bool
            If True, calculate the quadrupole, octupole, and hexadecapole moments.

    lopt : bool
           If True, optimize the molecular structure before the other calculations.

    lmode : bool
            If True, perform normal-mode analysis.

    lg : bool
         If True, calculate the O-H bond length and H-O-H bond angle of the first water molecule.

    loo : bool
          If True, calculate all unique O-O distances.

    qa : torch.Tensor, shape (na, 3) or (Batch, na, 3)
         Coordinates of the cores. The atoms of each water molecule are ordered as O, H1, and H2.

    cd : torch.Tensor, shape (4,)
         Particle charges ordered as [q_O, q_H, q_DO, q_DH].

    b : torch.Tensor, shape (3,)
            Thole smearing factors for core-core, core-Drude, and Drude-Drude interactions, respectively.

    ds : torch.Tensor, shape (2, 5)
         Anharmonic Drude spring parameters [k2, k3, k4, k5, k6]. The first row is used for O-DO and the second row for H-DH.

    poh : torch.Tensor, shape (6,)
          O-H stretching parameters [r0, k2, k3, k4, k5, k6].

    phh : torch.Tensor
          H-H stretching parameters.[r0, k2]. In the present calculations,
          k2 is set to zero, so the H-H stretching term does not contribute to
          the energy, forces, or force-constant matrix.

    phoh : torch.Tensor, shape (6,)
           H-O-H bending parameters [theta0, k2, k3, k4, k5, k6], where theta0 is given in degrees.

    bk : torch.Tensor, shape (3, 4)
         Buckingham parameters [A, B, C, D] for O-O, O-H, and H-H interactions, respectively.

    Returns
    -------
    results : dict
              Dictionary containing the requested results.
              The following entries are always returned:

            'dposition'
                SCF-optimized coordinates with shape (Batch, 2*na, 3).

            'engpe'
                Potential energies with shape (Batch,).

            'f'
                Forces on all core and Drude particles with shape
                (Batch, 2*na, 3).

            Additional entries are included according to the logical flags:

            'opt'
                Optimized coordinates, returned when lopt=True.

            'rohf', 'ahohf'
                O-H bond length and H-O-H angle of the first water molecule,
                returned when lg=True.

            'freq'
                Tuple (frequencies, eigenvectors), returned when lmode=True.
                The corresponding shapes are (Batch, 3*na) and
                (Batch, 3*na, 3*na).

            'fa'
                Forces on the real atoms with shape (Batch, na, 3), returned
                when lgrad=True.

            'dip'
                Tuple containing the dipole magnitude and dipole vector, returned
                when ldip=True.

            'quad', 'octu', 'hexa'
                Quadrupole, octupole, and hexadecapole moment tensors, returned
                when lquad=True.

            'alp'
                Dipole polarizability tensor with shape (Batch, 3, 3), returned
                when lpol=True.

            'beta', 'gama', 'A', 'A1', 'B', 'B1', 'C', 'C1', 'D', 'D1'
                Hyperpolarizability and multipolar response tensors, returned
                when lpol2=True.

            'distoo'
                All unique O-O distances with shape (Batch, N_pairs), returned
                when loo=True.
        """


    if qa.dim() == 2:
        qa = qa.unsqueeze(0)

    na = qa.shape[1]
    device = qa.device
    dtype = qa.dtype

    lst, lb, ld, lcc, lcd, ldd, lbk, \
        q, w, chg, pst, pb, pd, pbk, \
        extfld, extfg = init.init(qa, cd, ds, poh, phh, phoh, bk)

    batch_size = q.shape[0]
    np = q.shape[1]

    engpe = torch.zeros(batch_size, dtype=dtype, device=device)
    f = torch.zeros((batch_size, np, 3), dtype=dtype, device=device)
    fcm = torch.zeros((batch_size, 3 * np, 3 * np), dtype=dtype, device=device)

    results = {}

    if lopt:
        q, _ = newton.newton(lst, lb, ld, lcc, lcd, ldd, lbk,
                             q, chg, b, pst, pb, pd, pbk)
        q = cmtrans.cmtrans(q, w)
        results['opt'] = q.clone()

    if lg:
        # q shape: (Batch, np, 3)
        # 0: O, 1: H1, 2: H2
        r_oh1 = q[:, 0, :] - q[:, 1, :]
        rohf = torch.norm(r_oh1, dim=1)
        results['rohf'] = rohf

        vec_oh1 = q[:, 1, :] - q[:, 0, :]
        vec_oh2 = q[:, 2, :] - q[:, 0, :]

        dot_prod = torch.sum(vec_oh1 * vec_oh2, dim=1)
        norm_oh1 = torch.norm(vec_oh1, dim=1)
        norm_oh2 = torch.norm(vec_oh2, dim=1)

        cos_theta = dot_prod / (norm_oh1 * norm_oh2 + 1e-12)
        cos_theta = torch.clamp(cos_theta, -1.0, 1.0)

        ahohf = torch.acos(cos_theta) * (180.0 / 3.141592653589793)
        results['ahohf'] = ahohf

    if lmode:
        freq, eigvec = nmode.nmode(lst, lb, ld, lcc, lcd, ldd, lbk,
                                   q, w, chg, b, pst, pb, pd, pbk,
                                   engpe, f, fcm, lg)
        results['freq'] = (freq, eigvec)

    q = iter.iter(ld, lcd, ldd, q, chg, b, pd, extfld, extfg, engpe, f, fcm)
    results['dposition'] = q.clone()

    engpe, f, _ = engfor.engfor(lgrad, lfc,
                                lst, lb, ld, lcc, lcd, ldd, lbk,
                                q, chg, b, pst, pb, pd, pbk, extfld, extfg)
    results['engpe'] = engpe.clone()
    results['f'] = f.clone()

    if lgrad:
        arange_na = torch.arange(na, device=device)
        idx_atoms = arange_na + torch.div(arange_na, 3, rounding_mode='floor') * 3

        # (Batch, na, 3)
        fa = f[:, idx_atoms, :]
        results['fa'] = fa.clone()

    if ldip:
        dip_val, dip_vec = multipole.dipole(q, chg)
        results['dip'] = (dip_val, dip_vec)

    if lquad:
        q_centered = cmtrans.cmtrans(q, w)
        quad = multipole.quadrupole(q_centered, chg)
        octu = multipole.octupole(q_centered, chg)
        hexa = multipole.hexadecapole(q_centered, chg)
        results['quad'] = quad
        results['octu'] = octu
        results['hexa'] = hexa

    if lpol:
        alp = pol.pol(ld, lcd, ldd, q, engpe, f, fcm, chg, b, pd, extfld, extfg)
        results['alp'] = alp

    if lpol2:
        beta, beta3 = pol.pol2(ld, lcd, ldd, q, chg, b, pd, extfld, extfg, engpe, f, fcm)
        results['beta'] = beta

        gama = pol.pol3(ld, lcd, ldd, q, engpe, f, fcm, chg, b, pd, extfld, extfg)
        results['gama'] = gama

        A, A1 = pol.polA(ld, lcd, ldd, q, chg, b, pd, extfld, extfg, engpe, f, fcm)
        results['A'] = A;
        results['A1'] = A1

        B, B1 = pol.polB(ld, lcd, ldd, q, chg, b, pd, extfld, extfg, engpe, f, fcm)
        results['B'] = B;
        results['B1'] = B1

        C, C1 = pol.polC(ld, lcd, ldd, q, chg, b, pd, extfld, extfg, engpe, f, fcm)
        results['C'] = C;
        results['C1'] = C1

        D, D1 = pol.polD(ld, lcd, ldd, q, chg, b, pd, extfld, extfg, engpe, f, fcm)
        results['D'] = D;
        results['D1'] = D1

    if loo:
        distenceoo = distoo.distoo(q)
        results['distoo'] = distenceoo

    return results