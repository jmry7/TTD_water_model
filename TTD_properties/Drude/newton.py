import torch
import engfor

def newton(lst, lb, ld, lcc, lcd, ldd, lbk, q, chg, b, pst, pb, pd, pbk,
           *, max_nit=200, tiny=1.0e-8, step_init=0.5, step_max=1.0, step_reset_threshold=1.0 / 256.0,
           step_reset_value=1.0 / 64.0, ratio_threshold=0.6, excellent_ratio=0.9, max_rescale=5):
    """
        Optimize the particle coordinates on the potential-energy surface using a
        Hessian-based modified Newton method.

        The function supports both single and batched configurations and returns
        the optimized coordinates and corresponding potential energies.

        References
        ----------
        [1] J. Simons, P. Jorgensen, H. Taylor, and J. Ozment,
            "Walking on Potential Energy Surfaces,"
            J. Phys. Chem. 87, 2745-2753 (1983).

        [2] A. Banerjee, N. Adams, J. Simons, and R. Shepard,
            "Search for Stationary Points on Surfaces,"
            J. Phys. Chem. 89, 52-57 (1985).

        [3] J. Nichols, H. Taylor, P. Schmidt, and J. Simons,
            "Walking on Potential Energy Surfaces,"
            J. Chem. Phys. 92, 340-346 (1990).
    """

    if q.ndim not in (2, 3) or q.shape[-1] != 3:
        raise ValueError("q must have shape (n_particles, 3) or (batch_size, n_particles, 3).")

    input_was_unbatched = q.ndim == 2
    q_work = q.unsqueeze(0).clone() if input_was_unbatched else q.clone()

    device = q_work.device
    dtype = q_work.dtype
    batch_size, n_particles, _ = q_work.shape
    ndim = 3 * n_particles
    tiny_t = torch.as_tensor(tiny, device=device, dtype=dtype)

    extfld = torch.zeros((batch_size, 3), device=device, dtype=dtype)  # zero external electric field
    extfg = torch.zeros((batch_size, 3, 3), device=device, dtype=dtype)  # zero external electric field gradient
    stepi = torch.full((batch_size,), step_init, device=device, dtype=dtype)  # maximum walk step

    valid_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
    converged_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
    energy = torch.full((batch_size,), float("nan"), device=device, dtype=dtype)

    for nit in range(1, max_nit + 1):
        active_mask = valid_mask & (~converged_mask)

        if not active_mask.any():
            break

        # save old positions
        q_old = q_work.clone()

        # particle forces and force constants
        engpe1, force, fcm = engfor.engfor(True, True, lst, lb, ld, lcc, lcd, ldd, lbk, q_old, chg, b, pst, pb, pd, pbk, extfld, extfg)

        finite_energy = torch.isfinite(engpe1)
        finite_force = torch.isfinite(force).reshape(batch_size, -1).all(dim=1)
        finite_fcm = torch.isfinite(fcm).reshape(batch_size, -1).all(dim=1)
        finite_now = finite_energy & finite_force & finite_fcm

        valid_mask = valid_mask & finite_now
        active_mask = active_mask & finite_now
        energy = torch.where(finite_now, engpe1, energy)

        if not active_mask.any():
            continue

        # get the gradient vector (grad = -force)
        grad = -force.reshape(batch_size, ndim)

        fcm_safe = fcm.clone()

        if (~finite_now).any():
            fcm_safe[~finite_now] = torch.eye(ndim, device=device, dtype=dtype)

        try:
            # diagonalize fcm to get eigenvalues (eig) and eigvec
            eig, eigvec = torch.linalg.eigh(fcm_safe)
        except RuntimeError:
            valid_mask[active_mask] = False
            break

        # transform gradient into normal coordinate
        gproj = torch.matmul(eigvec.transpose(1, 2), grad.unsqueeze(-1)).squeeze(-1)

        nonzero_mode = eig.abs() > tiny_t
        s0 = torch.where(nonzero_mode, -gproj / eig.abs().clamp_min(tiny_t), torch.zeros_like(gproj))
        s0 = torch.where(active_mask.unsqueeze(1), s0, torch.zeros_like(s0))

        # scale s so as to satisfy the trust radius stepi
        h1 = torch.sqrt(torch.sum(s0 * s0, dim=1) / n_particles)

        newly_converged = active_mask & (h1 <= tiny_t)
        converged_mask = converged_mask | newly_converged
        search_mask = active_mask & (~newly_converged)

        if not search_mask.any():
            continue

        qscale0 = torch.minimum(torch.ones_like(h1), stepi / h1.clamp_min(tiny_t))
        unresolved = search_mask.clone()
        q_selected = q_old.clone()
        energy_selected = engpe1.clone()
        cumulative_scale = qscale0.clone()
        ratio_for_update = torch.full((batch_size,), -1.0, device=device, dtype=dtype)
        irescale = torch.zeros(batch_size, dtype=torch.long, device=device)

        # scale step iteratively to get good agreement between the predicted
        # and the actual energy difference
        # see J. Simons et al. J. Chem. Phys. 92 (1990) 340–346, p. 344
        for _ in range(max_rescale + 1):
            if not unresolved.any():
                break

            s_trial = s0 * cumulative_scale.unsqueeze(1)

            # predicted energy change by Taylor expansion to 2nd order
            depredict = torch.sum(gproj * s_trial + 0.5 * eig * s_trial.square(), dim=1)

            # transform back to Cartesian coordinate
            dq_flat = torch.matmul(eigvec, s_trial.unsqueeze(-1)).squeeze(-1)
            dq = dq_flat.reshape(batch_size, n_particles, 3)

            # get new positions
            q_candidate = q_old + dq

            # actual energy
            engpe, _, _ = engfor.engfor(True, False, lst, lb, ld, lcc, lcd, ldd, lbk, q_candidate, chg, b, pst, pb, pd, pbk, extfld, extfg)

            finite_candidate = torch.isfinite(q_candidate).reshape(batch_size, -1).all(dim=1)
            finite_trial = torch.isfinite(engpe) & finite_candidate & torch.isfinite(depredict)
            dereal = engpe - engpe1  # actual energy difference

            safe_denominator = torch.where(depredict.abs() > tiny_t, depredict, torch.ones_like(depredict))
            deratio = dereal / safe_denominator  # ratio actual/predicted
            deratio = torch.where(deratio.abs() > 1.0, 1.0 / deratio, deratio)
            deratio = torch.where(dereal.abs() < tiny_t, torch.ones_like(deratio), deratio)
            deratio = torch.where(finite_trial, deratio, torch.full_like(deratio, -1.0))

            attempted = unresolved.clone()
            irescale = irescale + attempted.to(irescale.dtype)
            q_selected = torch.where(attempted.view(-1, 1, 1), q_candidate, q_selected)
            energy_selected = torch.where(attempted, engpe, energy_selected)
            ratio_for_update = torch.where(attempted, deratio, ratio_for_update)

            good = attempted & (deratio >= ratio_threshold)
            unresolved = unresolved & (~good)
            cumulative_scale = torch.where(unresolved, cumulative_scale * 0.5, cumulative_scale)

        q_work = torch.where(search_mask.view(-1, 1, 1), q_selected, q_work)
        energy = torch.where(search_mask, energy_selected, energy)

        # modified Fletcher's step-size scaling
        # see Jack Simons et al. J. Phys. Chem. 87 (1983) 2745, p. 2746
        stepi = torch.where(search_mask & (irescale > 1), stepi / 2.0, stepi)
        enlarge = search_mask & (irescale == 1) & (h1 > stepi) & (ratio_for_update > excellent_ratio)
        stepi = torch.where(enlarge, stepi * 2.0, stepi)
        stepi = torch.minimum(stepi, torch.full_like(stepi, step_max))
        stepi = torch.where(stepi < step_reset_threshold, torch.full_like(stepi, step_reset_value), stepi)

    final_finite = torch.isfinite(q_work).reshape(batch_size, -1).all(dim=1)

    if final_finite.any():
        final_energy, _, _ = engfor.engfor(True, False, lst, lb, ld, lcc, lcd, ldd, lbk, q_work, chg, b, pst, pb, pd, pbk, extfld, extfg)
        energy = torch.where(final_finite, final_energy, energy)

    invalid_mask = ~valid_mask
    unconverged_mask = valid_mask & (~converged_mask)

    if invalid_mask.any():
        indices = (torch.nonzero(invalid_mask, as_tuple=False).flatten() + 1).detach().cpu().tolist()
        print(f"ERROR: Invalid energy, force, or Hessian was encountered for configuration(s) {indices}. The last structures are returned.")

    if unconverged_mask.any():
        indices = (torch.nonzero(unconverged_mask, as_tuple=False).flatten() + 1).detach().cpu().tolist()
        print(f"ERROR: Geometry optimization did not converge for configuration(s) {indices} within {max_nit} iterations. The last structures are returned.")

    q_out = q_work.squeeze(0) if input_was_unbatched else q_work
    energy_out = energy.squeeze(0) if input_was_unbatched else energy

    return q_out, energy_out