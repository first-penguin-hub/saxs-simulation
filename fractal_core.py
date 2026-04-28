from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


def sphere_form_factor_amplitude(q: np.ndarray, radius_nm: float) -> np.ndarray:
    x = q * radius_nm
    F = np.ones_like(x, dtype=float)
    mask = x > 1e-12
    xm = x[mask]
    F[mask] = 3.0 * (np.sin(xm) - xm * np.cos(xm)) / (xm ** 3)
    return F


def sample_unit_vectors(n: int, rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(size=(n, 3))
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return v / norms


def sample_fractal_radii_from_reff(
    n: int,
    R_eff_nm: float,
    Df: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample radii so that cumulative count scales as N(<r) ~ r^Df within [0, R_eff_nm].
    """
    u = rng.random(n)
    return R_eff_nm * np.power(u, 1.0 / Df)


def estimate_max_particles(
    R_nm: float,
    d_nm: float,
    packing_fraction: float = 0.35,
) -> float:
    V_cluster = 4.0 / 3.0 * np.pi * R_nm**3
    V_particle = 4.0 / 3.0 * np.pi * (d_nm / 2.0) ** 3
    return packing_fraction * V_cluster / V_particle




def estimate_min_radius_for_particles(
    n_particles: int,
    d_nm: float,
    packing_fraction: float = 0.25,
) -> float:
    """
    Rough lower-bound radius needed to accommodate n hard spheres of diameter d_nm.
    """
    if n_particles < 1:
        raise ValueError("n_particles must be >= 1.")
    if packing_fraction <= 0 or packing_fraction >= 1:
        raise ValueError("packing_fraction must be in (0, 1).")
    return (n_particles / packing_fraction) ** (1.0 / 3.0) * (d_nm / 2.0)


def theoretical_rg_for_radial_fractal(R_eff_nm: float, Df: float) -> float:
    """
    Continuous isotropic estimate for a radial mass-fractal distribution with N(<r)~r^Df.
    For large N and centered distribution, Rg ≈ sqrt(<r^2>) = R_eff * sqrt(Df/(Df+2)).
    """
    return R_eff_nm * np.sqrt(Df / (Df + 2.0))


def infer_reff_from_rg_target(Rg_target_nm: float, Df: float) -> float:
    return Rg_target_nm * np.sqrt((Df + 2.0) / Df)


def generate_fractal_cluster_with_reff(
    R_nm: float,
    Df: float,
    d_nm: float,
    n_particles: int,
    R_eff_nm: float,
    seed: int = 2026,
    max_trials: int = 500000,
) -> np.ndarray:
    """
    Generate a finite mass-fractal cluster in a sphere of radius R_nm, but with radial sampling
    supported only up to R_eff_nm (<= R_nm - d/2). Lower R_eff leads to lower Rg.
    """
    if Df <= 0:
        raise ValueError("Df must be > 0.")
    if Df > 3:
        raise ValueError("Df should be <= 3 for a 3D mass fractal.")
    if d_nm >= 2 * R_nm:
        raise ValueError("Particle diameter is too large for the chosen cluster radius.")
    if n_particles < 1:
        raise ValueError("n_particles must be >= 1.")

    rng = np.random.default_rng(seed)
    R_center_max = R_nm - d_nm / 2.0
    if R_center_max <= 0:
        raise ValueError("R - d/2 must be positive.")
    if R_eff_nm <= 0:
        raise ValueError("R_eff_nm must be positive.")
    if R_eff_nm > R_center_max:
        raise ValueError("R_eff_nm must be <= R - d/2.")

    n_fit_est = estimate_max_particles(R_nm, d_nm)
    if n_particles > n_fit_est:
        print(
            f"[warning] n={n_particles} may be geometrically too dense for "
            f"R={R_nm:.1f} nm, d={d_nm:.1f} nm. Rough soft limit ~ {n_fit_est:.0f} particles."
        )

    pos = np.empty((0, 3), dtype=float)
    accepted = 0
    trials = 0

    while accepted < n_particles and trials < max_trials:
        trials += 1
        r = sample_fractal_radii_from_reff(1, R_eff_nm, Df, rng)[0]
        u = sample_unit_vectors(1, rng)[0]
        cand = r * u

        if accepted == 0:
            pos = np.vstack([pos, cand])
            accepted += 1
            continue

        dr = pos - cand
        dist2 = np.sum(dr * dr, axis=1)
        if np.all(dist2 >= d_nm**2):
            pos = np.vstack([pos, cand])
            accepted += 1

    if accepted < n_particles:
        raise RuntimeError(
            f"Could only place {accepted}/{n_particles} particles without overlap "
            f"after {trials} trials. Increase R, decrease d or n, or increase max_trials."
        )

    return pos


def compute_rg_from_positions(pos: np.ndarray) -> float:
    com = pos.mean(axis=0)
    rg2 = np.mean(np.sum((pos - com) ** 2, axis=1))
    return np.sqrt(rg2)


def pair_distances(pos: np.ndarray) -> np.ndarray:
    diff = pos[:, None, :] - pos[None, :, :]
    D = np.sqrt(np.sum(diff**2, axis=2))
    iu = np.triu_indices(len(pos), k=1)
    return D[iu]


def scattering_from_positions(pos: np.ndarray, d_nm: float, q: np.ndarray) -> np.ndarray:
    n = len(pos)
    rij = pair_distances(pos)

    qr = np.outer(q, rij)
    sinc_term = np.ones_like(qr)
    mask = qr > 1e-12
    sinc_term[mask] = np.sin(qr[mask]) / qr[mask]

    S_q = 1.0 + (2.0 / n) * np.sum(sinc_term, axis=1)
    radius_nm = d_nm / 2.0
    F_q = sphere_form_factor_amplitude(q, radius_nm)
    P_q = F_q**2

    I_q = P_q * S_q
    if I_q[0] == 0:
        raise RuntimeError("I(q=0) became zero unexpectedly.")
    I_q /= I_q[0]
    return I_q


@dataclass
class GuinierFitResult:
    qmax_used: float
    slope: float
    intercept: float
    rg_fit_nm: float
    I0_fit: float
    n_points: int
    mask: np.ndarray


def fit_guinier_iterative(
    q: np.ndarray,
    I: np.ndarray,
    qmax_init: float = 0.03,
    qRg_limit: float = 1.3,
    min_points: int = 8,
    max_iter: int = 20,
) -> GuinierFitResult:
    qmax = qmax_init

    q_sorted = np.sort(q[q > 0])
    if len(q_sorted) == 0:
        raise RuntimeError("q must contain positive values for Guinier fitting.")
    q_floor = q_sorted[min(min_points - 1, len(q_sorted) - 1)]

    for _ in range(max_iter):
        qmax = max(qmax, q_floor)
        mask = (q > 0) & (q <= qmax) & (I > 0)
        if np.count_nonzero(mask) < min_points:
            qmax = q_floor
            mask = (q > 0) & (q <= qmax) & (I > 0)
            if np.count_nonzero(mask) < min_points:
                raise RuntimeError("Too few points in Guinier region. Increase n_q or reduce q_min.")

        x = q[mask] ** 2
        y = np.log(I[mask])
        slope, intercept = np.polyfit(x, y, 1)
        if slope >= 0:
            raise RuntimeError("Guinier slope is non-negative; low-q region is not suitable.")

        rg_fit = np.sqrt(-3.0 * slope)
        new_qmax = max(qRg_limit / rg_fit, q_floor)

        if abs(new_qmax - qmax) / max(qmax, 1e-12) < 1e-3:
            qmax = new_qmax
            break
        qmax = new_qmax

    qmax = max(qmax, q_floor)
    mask = (q > 0) & (q <= qmax) & (I > 0)
    x = q[mask] ** 2
    y = np.log(I[mask])
    slope, intercept = np.polyfit(x, y, 1)
    rg_fit = np.sqrt(-3.0 * slope)
    I0_fit = np.exp(intercept)

    return GuinierFitResult(
        qmax_used=qmax,
        slope=slope,
        intercept=intercept,
        rg_fit_nm=rg_fit,
        I0_fit=I0_fit,
        n_points=np.count_nonzero(mask),
        mask=mask,
    )


def fit_mass_fractal_dimension(r: np.ndarray) -> Tuple[float, float]:
    rr = np.sort(r)
    Ncum = np.arange(1, len(rr) + 1)
    lo = max(3, int(0.05 * len(rr)))
    hi = max(lo + 3, int(0.90 * len(rr)))
    x = np.log(rr[lo:hi])
    y = np.log(Ncum[lo:hi])
    coef = np.polyfit(x, y, 1)
    return coef[0], coef[1]


def radial_concentration_profile(
    pos: np.ndarray,
    R_nm: float,
    n_bins: int = 30,
) -> Dict[str, np.ndarray]:
    if n_bins < 2:
        raise ValueError("n_bins must be >= 2.")
    r = np.sqrt(np.sum(pos**2, axis=1))
    edges = np.linspace(0.0, R_nm, n_bins + 1)
    counts, _ = np.histogram(r, bins=edges)
    shell_vol = (4.0 / 3.0) * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    density = counts / shell_vol
    centers = 0.5 * (edges[:-1] + edges[1:])
    cumulative_fraction = np.cumsum(counts) / max(len(r), 1)
    return {
        "r_center_nm": centers,
        "number_density_nm^-3": density,
        "counts": counts,
        "r_edges_nm": edges,
        "cumulative_fraction": cumulative_fraction,
    }


def generate_fractal_cluster_with_target_rg(
    R_nm: float,
    Df: float,
    d_nm: float,
    n_particles: int,
    Rg_target_nm: float,
    seed: int = 2026,
    max_trials: int = 500000,
    rg_tolerance_nm: float = 1.0,
    search_max_iter: int = 20,
    n_replicates: int = 3,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Generate a mass-fractal cluster with specified Df and target Rg by tuning the effective
    radial support R_eff <= R - d/2.

    Notes:
    - For fixed Df, not every Rg is feasible.
    - Approximate maximum is Rg_max ≈ (R - d/2) * sqrt(Df / (Df + 2)).
    - The search uses an analytical initial guess and then a bisection over R_eff,
      evaluating the median Rg over a few random seeds for robustness.
    """
    if Df <= 0 or Df > 3:
        raise ValueError("Df must satisfy 0 < Df <= 3.")
    R_center_max = R_nm - d_nm / 2.0
    if R_center_max <= 0:
        raise ValueError("R - d/2 must be positive.")
    rg_max_theoretical = theoretical_rg_for_radial_fractal(R_center_max, Df)
    if Rg_target_nm <= 0:
        raise ValueError("Rg_target_nm must be positive.")
    if Rg_target_nm > rg_max_theoretical:
        raise ValueError(
            f"Requested Rg_target={Rg_target_nm:.3f} nm is too large for mass-fractal Df={Df:.3f} "
            f"inside R={R_nm:.3f} nm with d={d_nm:.3f} nm. "
            f"Approximate maximum is {rg_max_theoretical:.3f} nm."
        )

    def eval_reff(reff: float, base_seed: int) -> Tuple[float, np.ndarray]:
        rgs = []
        successes = []
        for j in range(n_replicates):
            try:
                pos_j = generate_fractal_cluster_with_reff(
                    R_nm=R_nm,
                    Df=Df,
                    d_nm=d_nm,
                    n_particles=n_particles,
                    R_eff_nm=reff,
                    seed=base_seed + 997 * j,
                    max_trials=max_trials,
                )
            except RuntimeError:
                continue
            successes.append(pos_j)
            rgs.append(compute_rg_from_positions(pos_j))
        if not rgs:
            raise RuntimeError(
                f"Could not generate a non-overlapping configuration at R_eff={reff:.3f} nm. "
                "Increase R, reduce n or d, or increase max_trials."
            )
        order = np.argsort(rgs)
        median_idx = int(order[len(order) // 2])
        return float(np.median(rgs)), successes[median_idx]

    reff_guess = min(infer_reff_from_rg_target(Rg_target_nm, Df), R_center_max)
    reff_min_feasible = min(R_center_max, estimate_min_radius_for_particles(n_particles, d_nm, packing_fraction=0.22))
    low = max(reff_min_feasible, 0.6 * reff_guess)
    high = R_center_max

    rg_low, _ = eval_reff(low, seed + 10000)
    rg_high, best_pos = eval_reff(high, seed + 20000)
    if Rg_target_nm < rg_low:
        low = max(d_nm, 0.05 * R_center_max)
        rg_low, _ = eval_reff(low, seed + 11000)

    best_err = abs(rg_high - Rg_target_nm)
    best_reff = high
    best_rg = rg_high

    rg_guess, pos_guess = eval_reff(reff_guess, seed)
    err_guess = abs(rg_guess - Rg_target_nm)
    if err_guess < best_err:
        best_err = err_guess
        best_reff = reff_guess
        best_rg = rg_guess
        best_pos = pos_guess

    if best_err <= rg_tolerance_nm:
        info = {
            "R_eff_nm": float(best_reff),
            "Rg_target_nm": float(Rg_target_nm),
            "Rg_generated_nm": float(best_rg),
            "Rg_max_theoretical_nm": float(rg_max_theoretical),
            "search_iterations": 0,
            "Df": float(Df),
        }
        return best_pos, info

    cur_low, cur_high = low, high
    for it in range(search_max_iter):
        mid = 0.5 * (cur_low + cur_high)
        rg_mid, pos_mid = eval_reff(mid, seed + 30000 + 1000 * it)
        err_mid = abs(rg_mid - Rg_target_nm)
        if err_mid < best_err:
            best_err = err_mid
            best_reff = mid
            best_rg = rg_mid
            best_pos = pos_mid
        if err_mid <= rg_tolerance_nm:
            break
        if rg_mid < Rg_target_nm:
            cur_low = mid
        else:
            cur_high = mid

    info = {
        "R_eff_nm": float(best_reff),
        "Rg_target_nm": float(Rg_target_nm),
        "Rg_generated_nm": float(best_rg),
        "Rg_max_theoretical_nm": float(rg_max_theoretical),
        "search_iterations": int(search_max_iter),
        "Df": float(Df),
    }
    return best_pos, info


def build_summary_dict(
    R_nm: float,
    Df: float,
    d_nm: float,
    n_particles: int,
    fit: GuinierFitResult,
    rg_real_nm: float,
    Df_fit_radial: float,
) -> Dict[str, float]:
    return {
        "R_nm": float(R_nm),
        "Df_target": float(Df),
        "d_nm": float(d_nm),
        "n_particles": int(n_particles),
        "Df_fit_radial": float(Df_fit_radial),
        "Rg_real_nm": float(rg_real_nm),
        "Rg_guinier_nm": float(fit.rg_fit_nm),
        "Guinier_qmax_nm^-1": float(fit.qmax_used),
        "Guinier_n_points": int(fit.n_points),
    }


def generate_shell_like_cluster(
    R_nm: float,
    R_shell_nm: float,
    d_nm: float,
    n_particles: int,
    seed: int = 2026,
    max_trials: int = 500000,
    shell_width_factor: float = 2.0,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Generate a shell-like particle configuration inside a sphere.

    Input is intentionally simple: the user specifies only the shell center radius
    R_shell_nm. The shell thickness is automatically set from the particle size:

        shell_width_nm = shell_width_factor * d_nm
        sigma_shell_nm = shell_width_nm / 2

    Particle-center radii are sampled from a truncated normal distribution centered
    at R_shell_nm and constrained to 0 <= r <= R_nm - d_nm/2. Hard-sphere overlap is
    rejected.
    """
    if R_nm <= 0:
        raise ValueError("R_nm must be positive.")
    if d_nm <= 0:
        raise ValueError("d_nm must be positive.")
    if d_nm >= 2 * R_nm:
        raise ValueError("Particle diameter is too large for the chosen container radius.")
    if n_particles < 1:
        raise ValueError("n_particles must be >= 1.")
    if shell_width_factor <= 0:
        raise ValueError("shell_width_factor must be positive.")

    R_center_max = R_nm - d_nm / 2.0
    if R_shell_nm < 0 or R_shell_nm > R_center_max:
        raise ValueError(
            f"R_shell_nm must satisfy 0 <= R_shell_nm <= R - d/2 = {R_center_max:.3f} nm."
        )

    shell_width_nm = shell_width_factor * d_nm
    sigma_shell_nm = shell_width_nm / 2.0

    rng = np.random.default_rng(seed)
    pos = np.empty((0, 3), dtype=float)
    accepted = 0
    trials = 0

    while accepted < n_particles and trials < max_trials:
        trials += 1

        # Truncated Gaussian radius around the target shell position.
        r = rng.normal(loc=R_shell_nm, scale=sigma_shell_nm)
        if r < 0.0 or r > R_center_max:
            continue

        u = sample_unit_vectors(1, rng)[0]
        cand = r * u

        if accepted == 0:
            pos = np.vstack([pos, cand])
            accepted += 1
            continue

        dr = pos - cand
        dist2 = np.sum(dr * dr, axis=1)
        if np.all(dist2 >= d_nm**2):
            pos = np.vstack([pos, cand])
            accepted += 1

    if accepted < n_particles:
        raise RuntimeError(
            f"Could only place {accepted}/{n_particles} particles without overlap after {trials} trials. "
            "For a shell-like structure, try increasing R, moving R_shell away from very small radii, "
            "decreasing d or n, or increasing max_trials."
        )

    radii = np.sqrt(np.sum(pos**2, axis=1))
    info = {
        "R_nm": float(R_nm),
        "R_shell_nm": float(R_shell_nm),
        "shell_width_nm_auto": float(shell_width_nm),
        "sigma_shell_nm": float(sigma_shell_nm),
        "d_nm": float(d_nm),
        "n_particles": int(n_particles),
        "r_mean_nm": float(np.mean(radii)),
        "r_std_nm": float(np.std(radii)),
        "r_min_nm": float(np.min(radii)),
        "r_max_nm": float(np.max(radii)),
        "R_center_max_nm": float(R_center_max),
    }
    return pos, info


def generate_shell_like_cluster_with_target_rg(
    R_nm: float,
    Rg_target_nm: float,
    d_nm: float,
    n_particles: int,
    seed: int = 2026,
    max_trials: int = 500000,
    shell_width_factor: float = 2.0,
    rg_tolerance_nm: float = 2.0,
    search_max_iter: int = 12,
    n_replicates: int = 3,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Generate a shell-like particle configuration controlled by target Rg.

    User inputs are R_nm, Rg_target_nm, d_nm, and n_particles. The shell width is
    fixed internally as shell_width = shell_width_factor * d_nm, and R_shell is
    tuned automatically so that the generated coordinate Rg approaches Rg_target_nm.
    """
    if Rg_target_nm <= 0:
        raise ValueError("Rg_target_nm must be positive.")
    if R_nm <= 0:
        raise ValueError("R_nm must be positive.")
    if d_nm <= 0:
        raise ValueError("d_nm must be positive.")
    if d_nm >= 2 * R_nm:
        raise ValueError("Particle diameter is too large for the chosen container radius.")

    R_center_max = R_nm - d_nm / 2.0
    shell_width_nm = shell_width_factor * d_nm
    sigma_shell_nm = shell_width_nm / 2.0
    Rg_max_approx = np.sqrt(R_center_max**2 + sigma_shell_nm**2)
    if Rg_target_nm > Rg_max_approx:
        raise ValueError(
            f"Requested Rg_target={Rg_target_nm:.3f} nm is too large for this container. "
            f"Approximate maximum is {Rg_max_approx:.3f} nm. Increase R or decrease d."
        )

    def eval_rshell(rshell: float, base_seed: int) -> Tuple[float, np.ndarray, Dict[str, float]]:
        rgs = []
        poss = []
        infos = []
        for j in range(n_replicates):
            try:
                pos_j, info_j = generate_shell_like_cluster(
                    R_nm=R_nm,
                    R_shell_nm=rshell,
                    d_nm=d_nm,
                    n_particles=n_particles,
                    seed=base_seed + 997 * j,
                    max_trials=max_trials,
                    shell_width_factor=shell_width_factor,
                )
            except RuntimeError:
                continue
            poss.append(pos_j)
            infos.append(info_j)
            rgs.append(compute_rg_from_positions(pos_j))
        if not rgs:
            raise RuntimeError(
                f"Could not generate a non-overlapping shell-like configuration at R_shell={rshell:.3f} nm. "
                "Increase R, reduce n or d, or increase max_trials."
            )
        order = np.argsort(rgs)
        mid = int(order[len(order) // 2])
        return float(np.median(rgs)), poss[mid], infos[mid]

    rshell_guess = np.sqrt(max(Rg_target_nm**2 - sigma_shell_nm**2, 0.0))
    rshell_guess = float(np.clip(rshell_guess, 0.0, R_center_max))

    best_pos = None
    best_info = None
    best_rshell = None
    best_rg = None
    best_err = np.inf

    for idx, rs in enumerate([0.0, rshell_guess, R_center_max]):
        try:
            rg, pos, info = eval_rshell(rs, seed + 10000 * (idx + 1))
        except RuntimeError:
            continue
        err = abs(rg - Rg_target_nm)
        if err < best_err:
            best_err = err
            best_pos = pos
            best_info = info
            best_rshell = rs
            best_rg = rg

    if best_pos is None:
        raise RuntimeError("Could not generate any feasible initial shell-like configuration.")

    low, high = 0.0, R_center_max
    performed_iter = 0
    if best_err > rg_tolerance_nm:
        for it in range(search_max_iter):
            performed_iter = it + 1
            mid = 0.5 * (low + high)
            rg_mid, pos_mid, info_mid = eval_rshell(mid, seed + 30000 + 1000 * it)
            err_mid = abs(rg_mid - Rg_target_nm)
            if err_mid < best_err:
                best_err = err_mid
                best_pos = pos_mid
                best_info = info_mid
                best_rshell = mid
                best_rg = rg_mid
            if err_mid <= rg_tolerance_nm:
                break
            if rg_mid < Rg_target_nm:
                low = mid
            else:
                high = mid

    best_info.update({
        "Rg_target_nm": float(Rg_target_nm),
        "Rg_generated_nm": float(best_rg),
        "Rg_error_nm": float(best_rg - Rg_target_nm),
        "R_shell_optimized_nm": float(best_rshell),
        "Rg_max_approx_nm": float(Rg_max_approx),
        "search_iterations": int(performed_iter),
    })
    return best_pos, best_info
