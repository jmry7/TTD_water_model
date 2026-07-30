import torch
import hessian
import projtr

def nmode(lst, lb, ld, lcc, lcd, ldd, lbk,
          q, w, chg, b, pst, pb, pd, pbk,
          engpe, f, fcm, lg):
    """
    Perform normal-mode analysis for a batch of molecular configurations.

    The function calculates the effective mass-weighted Hessian, projects
    out translational and rotational motions, and converts its eigenvalues
    into vibrational frequencies in cm^-1.

    Returns freq with shape (Batch, 3*N_atom) and eigvec with shape
    (Batch, 3*N_atom, 3*N_atom). Negative frequencies represent imaginary
    modes. When lg=False, the normal modes of the first configuration are
    written to the file 'normal.mode'.
    """

    device = q.device
    dtype = q.dtype
    batch_size = q.shape[0]
    np_particles = q.shape[1]

    # eig2cm = 1 / (2*pi*c*10^-14)^2
    eig2cm = 281838.0

    natms = int(np_particles // 2)

    nmol = np_particles // 6
    atom_indices = []
    for i in range(nmol):
        base = i * 6
        atom_indices.extend([base, base + 1, base + 2])

    idx_atoms = torch.tensor(atom_indices, device=device, dtype=torch.long)

    # wa: (natms,)
    wa = w[idx_atoms]

    # qa: (Batch, natms, 3)
    qa = q[:, idx_atoms, :]

    extfld = torch.zeros((batch_size, 3), device=device, dtype=dtype)
    extfg = torch.zeros((batch_size, 3, 3), device=device, dtype=dtype)

    hess = hessian.hessian(
        lst, lb, ld, lcc, lcd, ldd, lbk,
        q, chg, b, pst, pb, pd, pbk, extfld, extfg,
        wa, engpe, f, fcm
    )

    if natms > 1:
        hess = projtr.projtr(wa, qa, hess)

    eig, eigvec = torch.linalg.eigh(hess)

    freq_mag = torch.sqrt(torch.abs(eig * eig2cm))
    freq = torch.where(eig < 0, -freq_mag, freq_mag)

    if not lg:
        qa_0 = qa[0]
        freq_0 = freq[0]
        eigvec_0 = eigvec[0]

        ieig_0 = ['i' if x < 0 else ' ' for x in freq_0]

        wrtmode(natms, wa, qa_0, torch.abs(freq_0), eigvec_0, ieig_0)

    return freq, eigvec


def wrtmode(natms, wa, qa, freq, eigvec, ieig):

    with open('normal.mode', 'w') as f:
        f.write('number of atoms: \n')
        f.write(f'{natms:6d}\n\n')
        f.write(' index            x               y               z          weight\n')

        for i in range(natms):
            f.write(f'{i + 1:6d} {qa[i, 0]:12.6f} {qa[i, 1]:12.6f} {qa[i, 2]:12.6f} {wa[i]:9.4f}\n')

        f.write('\n\nfrequencies (cm^-1) and eigenvectors:\n')

        total_modes = 3 * natms
        for k_start in range(0, total_modes, 6):
            k_end = min(k_start + 6, total_modes)

            indices = " ".join([f"{i + 1:13d}" for i in range(k_start, k_end)])
            f.write(f"\n\n{indices}\n\n")

            freq_line = "freq "
            for i in range(k_start, k_end):
                freq_line += f"{freq[i]:10.2f}{ieig[i]}  "
            f.write(freq_line + "\n\n")

            for i in range(total_modes):
                vec_row = " ".join([f"{eigvec[i, j]:12.6f}" for j in range(k_start, k_end)])
                f.write(f"{i + 1:4d} {vec_row}\n")