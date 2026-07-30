import torch

def distoo(q):
    """
    Calculate all O-O distances for each configuration in a batch.

    q contains the particle coordinates with shape (Batch, N, 3). The
    returned tensor roo has shape (Batch, N_pairs), where each row contains
    all O-O distances for one configuration.
    """

    device = q.device

    o_coords = q[:, 0::6, :]

    batch_size = o_coords.shape[0]
    nw = o_coords.shape[1]

    if nw <= 1:
        return torch.zeros((batch_size, 0), device=device, dtype=q.dtype)

    dist_mat = torch.cdist(o_coords, o_coords)

    r_idx, c_idx = torch.triu_indices(nw, nw, offset=1, device=device)

    roo = dist_mat[:, r_idx, c_idx]

    return roo