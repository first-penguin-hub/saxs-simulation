from __future__ import annotations

import io

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from fractal_core import (
    build_summary_dict,
    compute_rg_from_positions,
    fit_guinier_iterative,
    fit_mass_fractal_dimension,
    generate_fractal_cluster_with_reff,
    generate_shell_like_cluster_with_target_rg,
    infer_reff_from_rg_target,
    radial_concentration_profile,
    scattering_from_positions,
    theoretical_rg_for_radial_fractal,
)

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    }
)

st.set_page_config(page_title="Fractal SAXS / Shell-like Rg App", layout="wide")
st.title("Fractal and shell-like SAXS simulator")


def init_state() -> None:
    defaults = {
        "mode": "Fractal SAXS",
        "R_nm_saxs": 300.0,
        "Df_saxs": 2.2,
        "d_nm_saxs": 20.0,
        "n_particles_saxs": 500,
        "seed_saxs": 2026,
        "q_min_saxs": 0.001,
        "q_max_saxs": 0.30,
        "n_q_saxs": 250,
        "max_trials_saxs": 500000,
        "R_nm_rad": 300.0,
        "Rg_target_rad": 240.0,
        "d_nm_rad": 20.0,
        "n_particles_rad": 500,
        "seed_rad": 2026,
        "q_min_rad": 0.001,
        "q_max_rad": 0.30,
        "n_q_rad": 250,
        "max_trials_rad": 500000,
        "n_bins_rad": 40,
        "rg_tolerance_rad": 2.0,
        "search_max_iter_rad": 12,
        "run_saxs": False,
        "run_rad": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()

with st.sidebar:
    st.header("Mode")
    mode = st.radio("Select tool", ["Fractal SAXS", "Radial Profile"], key="mode")

    if mode == "Fractal SAXS":
        st.header("Fractal SAXS parameters")
        st.session_state["R_nm_saxs"] = st.number_input("Cluster radius R [nm]", min_value=1.0, value=float(st.session_state["R_nm_saxs"]), step=10.0)
        st.session_state["Df_saxs"] = st.number_input("Fractal dimension Df", min_value=0.1, max_value=3.0, value=float(st.session_state["Df_saxs"]), step=0.1)
        st.session_state["d_nm_saxs"] = st.number_input("Particle diameter d [nm]", min_value=0.1, value=float(st.session_state["d_nm_saxs"]), step=1.0)
        st.session_state["n_particles_saxs"] = st.number_input("Number of particles n", min_value=1, value=int(st.session_state["n_particles_saxs"]), step=10)
        st.subheader("Optional calculation settings")
        st.session_state["seed_saxs"] = st.number_input("Random seed", min_value=0, value=int(st.session_state["seed_saxs"]), step=1)
        st.session_state["q_min_saxs"] = st.number_input("q_min [nm^-1]", min_value=0.00001, value=float(st.session_state["q_min_saxs"]), step=0.00010, format="%.5f")
        st.session_state["q_max_saxs"] = st.number_input("q_max [nm^-1]", min_value=0.00010, value=float(st.session_state["q_max_saxs"]), step=0.01000, format="%.5f")
        st.session_state["n_q_saxs"] = st.number_input("Number of q points", min_value=10, value=int(st.session_state["n_q_saxs"]), step=10)
        st.session_state["max_trials_saxs"] = st.number_input("Max placement trials", min_value=1000, value=int(st.session_state["max_trials_saxs"]), step=1000)
        st.session_state["run_saxs"] = st.button("Run Fractal SAXS", type="primary", use_container_width=True)

    else:
        st.header("Shell-like Radial Profile parameters")
        st.session_state["R_nm_rad"] = st.number_input("Container radius R [nm]", min_value=1.0, value=float(st.session_state["R_nm_rad"]), step=10.0)
        st.session_state["Rg_target_rad"] = st.number_input("Target radius of gyration Rg [nm]", min_value=0.1, value=float(st.session_state["Rg_target_rad"]), step=5.0)
        st.session_state["d_nm_rad"] = st.number_input("Particle diameter d [nm]", min_value=0.1, value=float(st.session_state["d_nm_rad"]), step=1.0)
        st.session_state["n_particles_rad"] = st.number_input("Number of particles n", min_value=1, value=int(st.session_state["n_particles_rad"]), step=10)
        st.caption("Shell width is set automatically as shell_width = 2 × d. R_shell is optimized internally to match target Rg.")
        st.subheader("Optional calculation settings")
        st.session_state["seed_rad"] = st.number_input("Random seed", min_value=0, value=int(st.session_state["seed_rad"]), step=1)
        st.session_state["q_min_rad"] = st.number_input("q_min [nm^-1]", min_value=0.00001, value=float(st.session_state["q_min_rad"]), step=0.00010, format="%.5f")
        st.session_state["q_max_rad"] = st.number_input("q_max [nm^-1]", min_value=0.00010, value=float(st.session_state["q_max_rad"]), step=0.01000, format="%.5f")
        st.session_state["n_q_rad"] = st.number_input("Number of q points", min_value=10, value=int(st.session_state["n_q_rad"]), step=10)
        st.session_state["n_bins_rad"] = st.number_input("Number of radial bins", min_value=5, value=int(st.session_state["n_bins_rad"]), step=5)
        st.session_state["max_trials_rad"] = st.number_input("Max placement trials", min_value=1000, value=int(st.session_state["max_trials_rad"]), step=1000)
        st.session_state["rg_tolerance_rad"] = st.number_input("Rg tolerance [nm]", min_value=0.1, value=float(st.session_state["rg_tolerance_rad"]), step=0.5)
        st.session_state["search_max_iter_rad"] = st.number_input("Max Rg search iterations", min_value=1, value=int(st.session_state["search_max_iter_rad"]), step=1)
        st.session_state["run_rad"] = st.button("Run Shell-like Radial Profile", type="primary", use_container_width=True)


if mode == "Fractal SAXS":
    st.markdown(
        r"""
### Fractal SAXS
This mode generates a particle configuration from a **mass-fractal radial rule**,

\[N(<r) \sim r^{D_f}\]

and computes the corresponding **3D configuration**, **normalized scattering intensity** $I(q)$,
and **Guinier fit**.

Use this mode when you want to start from **R, Df, d, n** and inspect the structure mainly in reciprocal space.
"""
    )
    st.markdown("### Guinier fit equation")
    st.latex(r"\ln I(q) = \ln I_0 - \frac{R_g^2}{3} q^2")

    if st.session_state.get("run_saxs", False):
        try:
            with st.spinner("Generating structure and computing scattering..."):
                R_center_max = float(st.session_state["R_nm_saxs"]) - float(st.session_state["d_nm_saxs"]) / 2.0
                R_eff = min(
                    infer_reff_from_rg_target(theoretical_rg_for_radial_fractal(R_center_max, float(st.session_state["Df_saxs"])), float(st.session_state["Df_saxs"])),
                    R_center_max,
                )
                pos = generate_fractal_cluster_with_reff(
                    R_nm=float(st.session_state["R_nm_saxs"]),
                    Df=float(st.session_state["Df_saxs"]),
                    d_nm=float(st.session_state["d_nm_saxs"]),
                    n_particles=int(st.session_state["n_particles_saxs"]),
                    R_eff_nm=R_eff,
                    seed=int(st.session_state["seed_saxs"]),
                    max_trials=int(st.session_state["max_trials_saxs"]),
                )
                q = np.linspace(float(st.session_state["q_min_saxs"]), float(st.session_state["q_max_saxs"]), int(st.session_state["n_q_saxs"]))
                I_q = scattering_from_positions(pos, d_nm=float(st.session_state["d_nm_saxs"]), q=q)
                fit = fit_guinier_iterative(q, I_q, qmax_init=min(0.06, float(st.session_state["q_max_saxs"]) * 0.25))
                rg_real_nm = compute_rg_from_positions(pos)
                r_from_center = np.sqrt(np.sum(pos**2, axis=1))
                Df_fit_radial, _ = fit_mass_fractal_dimension(r_from_center)
                summary = build_summary_dict(
                    R_nm=float(st.session_state["R_nm_saxs"]),
                    Df=float(st.session_state["Df_saxs"]),
                    d_nm=float(st.session_state["d_nm_saxs"]),
                    n_particles=int(st.session_state["n_particles_saxs"]),
                    fit=fit,
                    rg_real_nm=rg_real_nm,
                    Df_fit_radial=Df_fit_radial,
                )
            st.success("Calculation finished.")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Rg (positions) [nm]", f"{summary['Rg_real_nm']:.3f}")
            c2.metric("Rg (Guinier) [nm]", f"{summary['Rg_guinier_nm']:.3f}")
            c3.metric("Df fitted", f"{summary['Df_fit_radial']:.3f}")
            c4.metric("Guinier qmax [nm^-1]", f"{summary['Guinier_qmax_nm^-1']:.5f}")

            fig1 = plt.figure(figsize=(4.2, 3.6))
            ax = fig1.add_subplot(111, projection="3d")
            ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=10)
            ax.set_xlabel("x [nm]"); ax.set_ylabel("y [nm]"); ax.set_zlabel("z [nm]")
            ax.set_title("Generated particle configuration")
            xyz_min = pos.min(axis=0); xyz_max = pos.max(axis=0); center = 0.5 * (xyz_min + xyz_max); span = np.max(xyz_max - xyz_min)
            ax.set_xlim(center[0] - span / 2, center[0] + span / 2); ax.set_ylim(center[1] - span / 2, center[1] + span / 2); ax.set_zlim(center[2] - span / 2, center[2] + span / 2)
            plt.tight_layout(); st.pyplot(fig1); plt.close(fig1)

            fig2 = plt.figure(figsize=(4.8, 3.4))
            plt.loglog(q, I_q, label="I(q)"); plt.xlabel(r"q [nm$^{-1}$]"); plt.ylabel("normalized intensity"); plt.title("Scattering intensity"); plt.legend(); plt.tight_layout(); st.pyplot(fig2); plt.close(fig2)

            fig3 = plt.figure(figsize=(4.8, 3.4))
            plt.plot(q**2, np.log(I_q), label="ln I(q)")
            plt.plot(q[fit.mask] ** 2, fit.intercept + fit.slope * q[fit.mask] ** 2, lw=1.8, label=fr"Guinier fit, $R_g$={fit.rg_fit_nm:.2f} nm")
            plt.xlabel(r"$q^2$ [nm$^{-2}$]"); plt.ylabel(r"$\ln I(q)$"); plt.title("Guinier plot"); plt.legend(); plt.tight_layout(); st.pyplot(fig3); plt.close(fig3)

            positions_csv = io.StringIO(); np.savetxt(positions_csv, pos, delimiter=",", header="x_nm,y_nm,z_nm", comments="")
            scattering_csv = io.StringIO(); np.savetxt(scattering_csv, np.column_stack([q, I_q]), delimiter=",", header="q_nm^-1,I_q", comments="")
            summary_txt = io.StringIO()
            for k, v in summary.items(): summary_txt.write(f"{k}={v}\n")
            st.subheader("Download outputs")
            d1, d2, d3 = st.columns(3)
            d1.download_button("Positions CSV", positions_csv.getvalue(), file_name="generated_positions_nm.csv", mime="text/csv", use_container_width=True)
            d2.download_button("I(q) CSV", scattering_csv.getvalue(), file_name="scattering_Iq.csv", mime="text/csv", use_container_width=True)
            d3.download_button("Summary TXT", summary_txt.getvalue(), file_name="guinier_fit_summary.txt", mime="text/plain", use_container_width=True)
            st.subheader("Summary"); st.json(summary)
        except Exception as e:
            st.error(str(e))
    else:
        st.info("Set parameters in the sidebar and click 'Run Fractal SAXS'.")

else:
    R_center_max = float(st.session_state["R_nm_rad"]) - float(st.session_state["d_nm_rad"]) / 2.0
    auto_width = 2.0 * float(st.session_state["d_nm_rad"])
    sigma_shell = auto_width / 2.0
    rg_max_approx = np.sqrt(R_center_max**2 + sigma_shell**2)
    st.markdown(
        r"""
### Shell-like Radial Profile controlled by Rg
This mode generates a **non-fractal shell-like particle configuration** inside a sphere of radius $R$.

The user specifies the target radius of gyration $R_g$. Internally, particle centers are sampled from a truncated Gaussian radial distribution around an optimized shell radius $R_{shell}$.

The shell thickness is not an input parameter; it is automatically set from particle diameter:
"""
    )
    st.latex(r"shell\_width = 2d")
    st.markdown(
        r"""
The algorithm tunes $R_{shell}$ so that the generated coordinate-based $R_g$ approaches the target $R_g$.
This mode outputs the 3D configuration, radial number-density profile, cumulative fraction profile, scattering intensity $I(q)$, and Guinier fit.
"""
    )
    st.markdown("### Feasibility check")
    st.write(
        f"Particle centers must remain within R - d/2 = **{R_center_max:.2f} nm**. "
        f"The automatic shell width is **{auto_width:.2f} nm**. "
        f"Approximate maximum achievable Rg is **{rg_max_approx:.2f} nm**."
    )

    if st.session_state.get("run_rad", False):
        try:
            with st.spinner("Generating shell-like structure with target Rg and computing profiles..."):
                pos, info = generate_shell_like_cluster_with_target_rg(
                    R_nm=float(st.session_state["R_nm_rad"]),
                    Rg_target_nm=float(st.session_state["Rg_target_rad"]),
                    d_nm=float(st.session_state["d_nm_rad"]),
                    n_particles=int(st.session_state["n_particles_rad"]),
                    seed=int(st.session_state["seed_rad"]),
                    max_trials=int(st.session_state["max_trials_rad"]),
                    shell_width_factor=2.0,
                    rg_tolerance_nm=float(st.session_state["rg_tolerance_rad"]),
                    search_max_iter=int(st.session_state["search_max_iter_rad"]),
                )
                q = np.linspace(float(st.session_state["q_min_rad"]), float(st.session_state["q_max_rad"]), int(st.session_state["n_q_rad"]))
                I_q = scattering_from_positions(pos, d_nm=float(st.session_state["d_nm_rad"]), q=q)
                fit = fit_guinier_iterative(q, I_q, qmax_init=min(0.06, float(st.session_state["q_max_rad"]) * 0.25))
                rg_real_nm = compute_rg_from_positions(pos)
                profile = radial_concentration_profile(pos, R_nm=float(st.session_state["R_nm_rad"]), n_bins=int(st.session_state["n_bins_rad"]))
                summary = {
                    "R_nm": float(st.session_state["R_nm_rad"]),
                    "Rg_target_nm": float(st.session_state["Rg_target_rad"]),
                    "Rg_real_nm": float(rg_real_nm),
                    "Rg_error_nm": float(rg_real_nm - float(st.session_state["Rg_target_rad"])),
                    "Rg_guinier_nm": float(fit.rg_fit_nm),
                    "R_shell_optimized_nm": float(info["R_shell_optimized_nm"]),
                    "shell_width_nm_auto": float(info["shell_width_nm_auto"]),
                    "sigma_shell_nm": float(info["sigma_shell_nm"]),
                    "d_nm": float(st.session_state["d_nm_rad"]),
                    "n_particles": int(st.session_state["n_particles_rad"]),
                    "r_mean_nm": float(info["r_mean_nm"]),
                    "r_std_nm": float(info["r_std_nm"]),
                    "search_iterations": int(info["search_iterations"]),
                    "Guinier_qmax_nm^-1": float(fit.qmax_used),
                    "Guinier_n_points": int(fit.n_points),
                }

            st.success("Calculation finished.")
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Rg target [nm]", f"{summary['Rg_target_nm']:.2f}")
            c2.metric("Rg (positions) [nm]", f"{summary['Rg_real_nm']:.2f}")
            c3.metric("Rg (Guinier) [nm]", f"{summary['Rg_guinier_nm']:.2f}")
            c4.metric("R_shell optimized [nm]", f"{summary['R_shell_optimized_nm']:.2f}")
            c5.metric("Shell width auto [nm]", f"{summary['shell_width_nm_auto']:.2f}")

            fig1 = plt.figure(figsize=(4.2, 3.6))
            ax = fig1.add_subplot(111, projection="3d")
            ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=10)
            ax.set_xlabel("x [nm]"); ax.set_ylabel("y [nm]"); ax.set_zlabel("z [nm]")
            ax.set_title("Generated Rg-controlled shell-like configuration")
            xyz_min = pos.min(axis=0); xyz_max = pos.max(axis=0); center = 0.5 * (xyz_min + xyz_max); span = np.max(xyz_max - xyz_min)
            ax.set_xlim(center[0] - span / 2, center[0] + span / 2); ax.set_ylim(center[1] - span / 2, center[1] + span / 2); ax.set_zlim(center[2] - span / 2, center[2] + span / 2)
            plt.tight_layout(); st.pyplot(fig1); plt.close(fig1)

            fig2 = plt.figure(figsize=(4.8, 3.4))
            plt.loglog(q, I_q, label="I(q)"); plt.xlabel(r"q [nm$^{-1}$]"); plt.ylabel("normalized intensity"); plt.title("Scattering intensity"); plt.legend(); plt.tight_layout(); st.pyplot(fig2); plt.close(fig2)

            colA, colB = st.columns(2)
            with colA:
                fig3 = plt.figure(figsize=(4.8, 3.4))
                plt.plot(profile["r_center_nm"], profile["number_density_nm^-3"], marker="o", ms=3)
                plt.axvline(summary["R_shell_optimized_nm"], linestyle="--", lw=1.2, label="optimized R_shell")
                plt.xlabel("r [nm]"); plt.ylabel(r"number density [nm$^{-3}$]"); plt.title("Radial number-density profile"); plt.legend(); plt.tight_layout(); st.pyplot(fig3); plt.close(fig3)
            with colB:
                fig4 = plt.figure(figsize=(4.8, 3.4))
                plt.plot(profile["r_center_nm"], profile["cumulative_fraction"], marker="o", ms=3)
                plt.axvline(summary["R_shell_optimized_nm"], linestyle="--", lw=1.2, label="optimized R_shell")
                plt.xlabel("r [nm]"); plt.ylabel("cumulative fraction"); plt.title("Cumulative fraction profile"); plt.legend(); plt.tight_layout(); st.pyplot(fig4); plt.close(fig4)

            fig5 = plt.figure(figsize=(4.8, 3.4))
            plt.plot(q**2, np.log(I_q), label="ln I(q)")
            plt.plot(q[fit.mask] ** 2, fit.intercept + fit.slope * q[fit.mask] ** 2, lw=1.8, label=fr"Guinier fit, $R_g$={fit.rg_fit_nm:.2f} nm")
            plt.xlabel(r"$q^2$ [nm$^{-2}$]"); plt.ylabel(r"$\ln I(q)$"); plt.title("Guinier plot"); plt.legend(); plt.tight_layout(); st.pyplot(fig5); plt.close(fig5)

            positions_csv = io.StringIO(); np.savetxt(positions_csv, pos, delimiter=",", header="x_nm,y_nm,z_nm", comments="")
            iq_csv = io.StringIO(); np.savetxt(iq_csv, np.column_stack([q, I_q]), delimiter=",", header="q_nm^-1,I_q", comments="")
            radial_csv = io.StringIO()
            np.savetxt(
                radial_csv,
                np.column_stack([profile["r_center_nm"], profile["number_density_nm^-3"], profile["cumulative_fraction"]]),
                delimiter=",",
                header="r_center_nm,number_density_nm^-3,cumulative_fraction",
                comments="",
            )
            summary_txt = io.StringIO()
            for k, v in summary.items(): summary_txt.write(f"{k}={v}\n")

            st.subheader("Download outputs")
            d1, d2, d3, d4 = st.columns(4)
            d1.download_button("Positions CSV", positions_csv.getvalue(), file_name="generated_shell_rg_positions_nm.csv", mime="text/csv", use_container_width=True)
            d2.download_button("I(q) CSV", iq_csv.getvalue(), file_name="shell_rg_scattering_Iq.csv", mime="text/csv", use_container_width=True)
            d3.download_button("Radial profile CSV", radial_csv.getvalue(), file_name="shell_rg_radial_profile.csv", mime="text/csv", use_container_width=True)
            d4.download_button("Summary TXT", summary_txt.getvalue(), file_name="shell_rg_summary.txt", mime="text/plain", use_container_width=True)
            st.subheader("Summary"); st.json(summary)
        except Exception as e:
            st.error(str(e))
    else:
        st.info("Set parameters in the sidebar and click 'Run Shell-like Radial Profile'.")
