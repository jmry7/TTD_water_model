import torch

def cmtrans(q, w):
    """
    Transform molecular coordinates into a molecule-fixed coordinate system
    and translate the center of mass to the origin.

    The local orthonormal axes are defined using the first three particles,
    which are assumed to be ordered as O(0), H1(1), and H2(2):

    x_axis = (r_H2 - r_H1) / |r_H2 - r_H1|

    y_axis = [(r_H2 - r_O) x x_axis] / |(r_H2 - r_O) x x_axis|

    z_axis = x_axis x y_axis

    All particle coordinates are rotated into this local coordinate system and
    then the molecular center of mass is located at the origin.
    """

    if q.dim() == 2:
        q = q.unsqueeze(0)
        is_batch = False
    else:
        is_batch = True

    vec_x = q[:, 2, :] - q[:, 1, :]

    r1 = torch.norm(vec_x, dim=-1, keepdim=True)
    axis0 = vec_x / r1

    vec_z_prime = q[:, 2, :] - q[:, 0, :]

    vec_cross = torch.linalg.cross(vec_z_prime, axis0, dim=-1)
    r2 = torch.norm(vec_cross, dim=-1, keepdim=True)
    axis1 = vec_cross / r2

    axis2 = torch.linalg.cross(axis0, axis1, dim=-1)

    axes = torch.stack([axis0, axis1, axis2], dim=1)

    q_transformed = torch.matmul(q, axes.transpose(1, 2))

    w_expanded = w.view(1, -1, 1)
    wt = torch.sum(w)

    qcm = torch.sum(w_expanded * q_transformed, dim=1) / wt

    q_final = q_transformed - qcm.unsqueeze(1)

    if not is_batch:
        q_final = q_final.squeeze(0)

    return q_final