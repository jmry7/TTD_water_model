import torch

def projtr(w, q, hess):
    """
    Project the translational and rotational degrees of freedom out of a
    batch of mass-weighted Hessian matrices.

    The projection matrix is constructed as

    P = I - B @ pinv(B.T @ B) @ B.T,

    where the columns of B represent the three translational and three
    rotational motions. The projected Hessian is calculated as

    H_proj = P @ H @ P.

    """

    device = q.device
    dtype = q.dtype
    batch_size = q.shape[0]
    natms = q.shape[1]
    n_dof = 3 * natms

    if torch.isnan(q).any() or torch.isinf(q).any() or torch.isnan(hess).any() or torch.isinf(hess).any():
        return hess

    sqrt_w = torch.sqrt(w).unsqueeze(0).expand(batch_size, -1)  # (Batch, N)

    q_w = q * sqrt_w.unsqueeze(-1)

    x = q_w[:, :, 0]
    y = q_w[:, :, 1]
    z = q_w[:, :, 2]

    B_tensor = torch.zeros((batch_size, natms, 3, 6), device=device, dtype=dtype)

    B_tensor[:, :, 0, 0] = sqrt_w
    B_tensor[:, :, 1, 1] = sqrt_w
    B_tensor[:, :, 2, 2] = sqrt_w

    B_tensor[:, :, 1, 3] = -z
    B_tensor[:, :, 2, 3] = y
    B_tensor[:, :, 0, 4] = z
    B_tensor[:, :, 2, 4] = -x
    B_tensor[:, :, 0, 5] = -y
    B_tensor[:, :, 1, 5] = x

    B_mat = B_tensor.reshape(batch_size, n_dof, 6)

    overlap = torch.matmul(B_mat.transpose(1, 2), B_mat)

    if torch.isnan(overlap).any() or torch.isinf(overlap).any():
        return hess

    try:
        overlap_inv = torch.linalg.pinv(overlap)
    except RuntimeError:
        return hess

    term = torch.matmul(B_mat, overlap_inv)
    proj_ext = torch.matmul(term, B_mat.transpose(1, 2))

    eye = torch.eye(n_dof, device=device, dtype=dtype).unsqueeze(0)
    P = eye - proj_ext

    temp = torch.matmul(hess, P)
    hess_proj = torch.matmul(P, temp)

    return hess_proj
