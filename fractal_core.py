from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


# ============================================================
# Shared numerical functions
# ============================================================

def sphere_form_factor_amplitude(q: np.ndarray, radius_nm: float) -> np.ndarray:
    """Amplitude form factor of a homogeneous sphere, normalized so F(0)=1."""
    x = q * radius_nm
    F = np.ones_like(x, dtype=float)
    mask = x > 1e-12
    xm = x[mask]
    F[mask] = 3.0 * (np.sin(xm) - xm * np.cos(xm)) / (xm ** 3)
    return F


def sample_fractal_radii(
    n: int,
    R_nm: float,
    Df: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample radii so that cumulative count roughly scales as N(<r) ~ r^Df."""
    u = rng.random(n)
    r = R_nm * np.power(u, 1.0 / Df)
    return r


def sample_shell_radii(
    n: int,
    R_nm: float,
    alpha: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample radii with radial PDF p(r) ∝ r^alpha on [0, R_nm]."""
    if alpha <= -1:
        raise ValueError("alpha must be > -1.")
    u = rng.random(n)
    r = R_nm * np.power(u, 1.0 / (alpha + 1.0))
    return r


def sample_unit_vectors(n: int, rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(size=(n, 3))
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return v / norms


def estimate_max_particles(
    R_nm: float,
    d_nm: float,
    packing_fraction: float = 0.35,
) -> float:
    """Rough geometric sanity check for how many hard spheres might fit."""
    V_cluster = 4.0 / 3.0 * np.pi * R_nm**3
    V_particle = 4.0 / 3.0 * np.pi * (d_nm / 2.0) ** 3
    return packing_fraction * V_cluster / V_particle


def sphere_rg_uniform(R_nm: float) -> float:
    """Radius of gyration of a uniform solid sphere of radius R_nm."""
    return np.sqrt(3.0 / 5.0) * R_nm


def max_center_radius(R_nm: float, d_nm: float) -> float:
    """Maximum allowed radius for particle centers inside the container sphere."""
    return R_nm - d_nm / 2.0


def max_accessible_rg(R_nm: float, d_nm: float) -> float:
    """Practical upper bound if centers are concentrated near the outer edge."""
    return max_center_radius(R_nm, d_nm)


def _validate_cluster_inputs(R_nm: float, d_nm: float, n_particles: int) -> float:
    if d_nm >= 2 * R_nm:
        raise ValueError("Particle diameter is too large for the chosen cluster radius.")
    if n_particles < 1:
        raise ValueError("n_particles must be >= 1.")
    R_center_max = max_center_radius(R_nm, d_nm)
    if R_center_max <= 0:
        raise ValueError("R - d/2 must be positive.")
    return R_center_max


def generate_fractal_cluster(
    R_nm: float,
    Df: float,
    d_nm: float,
    n_particles: int,
    seed: int = 2026,
    max_trials: int = 500000,
) -> np.ndarray:
    """Generate an approximate finite mass-fractal cluster in a sphere."""
    if Df <= 0:
        raise ValueError("Df must be > 0.")
    if Df > 3:
        raise ValueError("Df should be <= 3 for a 3D mass fractal.")

    R_center_max = _validate_cluster_inputs(R_nm, d_nm, n_particles)
    rng = np.random.default_rng(seed)

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

        r = sample_fractal_radii(1, R_center_max, Df, rng)[0]
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


def generate_shell_cluster(
    R_nm: float,
    d_nm: float,
    n_particles: int,
    alpha: float,
    seed: int = 2026,
    max_trials: int = 500000,
) -> np.ndarray:
    """Generate a center-depleted / shell-like cluster in a sphere."""
    if alpha <= -1:
        raise ValueError("alpha must be > -1.")

    R_center_max = _validate_cluster_inputs(R_nm, d_nm, n_particles)
    rng = np.random.default_rng(seed)

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

        r = sample_shell_radii(1, R_center_max, alpha, rng)[0]
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


def generate_shell_cluster_target_rg(
    R_nm: float,
    d_nm: float,
    n_particles: int,
    Rg_target_nm: float,
    seed: int = 2026,
    max_trials: int = 500000,
    alpha_min: float = 2.0,
    alpha_max: float = 120.0,
    tolerance_nm: float = 1.0,
    max_iter: int = 16,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Tune a shell-like radial distribution p(r) ∝ r^alpha so that the generated
    cluster approaches the requested radius of gyration.
    """
    R_center_max = _validate_cluster_inputs(R_nm, d_nm, n_particles)
    rg_uniform = sphere_rg_uniform(R_nm)
    rg_upper = max_accessible_rg(R_nm, d_nm)

    if Rg_target_nm <= rg_uniform:
        raise ValueError(
            f"Rg_target_nm must be larger than the uniform-sphere value ({rg_uniform:.3f} nm)."
        )
    if Rg_target_nm >= rg_upper:
        raise ValueError(
            f"Rg_target_nm must be smaller than the practical upper bound ({rg_upper:.3f} nm)."
        )

    best_pos = None
    best_info: Dict[str, float] | None = None
    lo = alpha_min
    hi = alpha_max

    for i in range(max_iter):
        alpha = 0.5 * (lo + hi)
        pos = generate_shell_cluster(
            R_nm=R_nm,
            d_nm=d_nm,
            n_particles=n_particles,
            alpha=alpha,
            seed=seed + i,
            max_trials=max_trials,
        )
        rg = compute_rg_from_positions(pos)
        err = rg - Rg_target_nm

        info = {
            "alpha": float(alpha),
            "Rg_target_nm": float(Rg_target_nm),
            "Rg_real_nm": float(rg),
            "Rg_uniform_sphere_nm": float(rg_uniform),
            "Rg_upper_bound_nm": float(rg_upper),
            "R_center_max_nm": float(R_center_max),
            "abs_error_nm": float(abs(err)),
            "iterations_used": int(i + 1),
        }

        if best_info is None or abs(err) < best_info["abs_error_nm"]:
            best_pos = pos
            best_info = info

        if abs(err) <= tolerance_nm:
            break

        if err < 0:
            lo = alpha
        else:
            hi = alpha

    if best_pos is None or best_info is None:
        raise RuntimeError("Failed to generate a shell-like cluster for the requested Rg.")

    return best_pos, best_info


def pair_distances(pos: np.ndarray) -> np.ndarray:
    """All unique pair distances r_ij for i < j."""
    diff = pos[:, None, :] - pos[None, :, :]
    D = np.sqrt(np.sum(diff * diff, axis=2))
    iu = np.triu_indices(len(pos), k=1)
    return D[iu]


def compute_rg_from_positions(pos: np.ndarray) -> float:
    """Radius of gyration from coordinates."""
    com = pos.mean(axis=0)
    rg2 = np.mean(np.sum((pos - com) ** 2, axis=1))
    return np.sqrt(rg2)


def scattering_from_positions(
    pos: np.ndarray,
    d_nm: float,
    q: np.ndarray,
) -> np.ndarray:
    """Compute normalized scattering intensity I(q) = P(q) S(q)."""
    n = len(pos)
    rij = pair_distances(pos)

    qr = np.outer(q, rij)
    sinc_term = np.ones_like(qr)
    mask = qr > 1e-12
    sinc_term[mask] = np.sin(qr[mask]) / qr[mask]

    S_q = 1.0 + (2.0 / n) * np.sum(sinc_term, axis=1)

    radius_nm = d_nm / 2.0
    F_q = sphere_form_factor_amplitude(q, radius_nm)
    P_q = F_q ** 2

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
    """Iterative Guinier fit: ln I(q) = ln I0 - (Rg^2/3) q^2."""
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
    """Diagnostic fit of cumulative N(<r) ~ r^Df from generated positions."""
    rr = np.asarray(r, dtype=float)
    rr = rr[np.isfinite(rr)]
    rr = rr[rr > 0]
    if rr.size < 8:
        raise ValueError("Need at least 8 positive radii to fit an effective fractal dimension.")

    rr = np.sort(rr)
    Ncum = np.arange(1, len(rr) + 1, dtype=float)

    lo = max(3, int(0.05 * len(rr)))
    hi = max(lo + 3, int(0.90 * len(rr)))
    x = np.log(rr[lo:hi])
    y = np.log(Ncum[lo:hi])

    coef = np.polyfit(x, y, 1)
    return float(coef[0]), float(coef[1])


def radial_concentration_profile(
    pos: np.ndarray,
    R_nm: float,
    n_bins: int = 30,
) -> Dict[str, np.ndarray]:
    """Compute radial concentration profile inside the container sphere."""
    if n_bins < 2:
        raise ValueError("n_bins must be >= 2.")

    r = np.linalg.norm(pos, axis=1)
    edges = np.linspace(0.0, R_nm, n_bins + 1)
    counts, _ = np.histogram(r, bins=edges)
    shell_volumes = (4.0 * np.pi / 3.0) * (edges[1:] ** 3 - edges[:-1] ** 3)
    number_density = counts / shell_volumes
    centers = 0.5 * (edges[:-1] + edges[1:])
    count_fraction = counts / counts.sum()
    cumulative_fraction = np.cumsum(counts) / counts.sum()

    return {
        "r_center_nm": centers,
        "r_inner_nm": edges[:-1],
        "r_outer_nm": edges[1:],
        "count": counts,
        "number_density_nm^-3": number_density,
        "count_fraction": count_fraction,
        "cumulative_fraction": cumulative_fraction,
    }


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
