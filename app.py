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
    generate_fractal_cluster,
    radial_concentration_profile,
    scattering_from_positions,
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

st.set_page_config(page_title="Fractal SAXS / Radial Profile App", layout="wide")


def init_state() -> None:
    defaults = {
        "mode": "Fractal SAXS",
        "R_nm_saxs": 300.0,
        "Df_saxs": 2.2,
        "d_nm_saxs": 20.0,
        "n_saxs": 500,
        "seed_saxs": 2026,
        "q_min_saxs": 0.001,
        "q_max_saxs": 0.300,
        "n_q_saxs": 250,
        "max_trials_saxs": 500000,
        "R_nm_rad": 300.0,
        "Df_rad": 2.2,
        "d_nm_rad": 20.0,
        "n_rad": 500,
        "seed_rad": 2026,
        "max_trials_rad": 500000,
        "n_bins_rad": 40,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()

st.title("Fractal analysis app")
st.caption(
    "Mass-fractal particle configuration, SAXS intensity I(q), Guinier fit, and radial concentration profile."
)

with st.sidebar:
    st.header("Tool")
    mode = st.radio(
        "Select mode",
        ["Fractal SAXS", "Radial Profile"],
        key="mode",
    )

    if mode == "Fractal SAXS":
        st.markdown("This mode generates a mass-fractal cluster and evaluates its scattering intensity **I(q)** and Guinier fit.")
        with st.form("saxs_form"):
            st.subheader("Structure parameters")
            R_nm = st.number_input("Cluster radius R [nm]", min_value=1.0, value=float(st.session_state.R_nm_saxs), step=10.0)
            Df = st.number_input("Fractal dimension Df", min_value=0.1, max_value=3.0, value=float(st.session_state.Df_saxs), step=0.1)
            d_nm = st.number_input("Particle diameter d [nm]", min_value=0.1, value=float(st.session_state.d_nm_saxs), step=1.0)
            n_particles = st.number_input("Number of particles n", min_value=1, value=int(st.session_state.n_saxs), step=10)

            st.subheader("Calculation parameters")
            seed = st.number_input("Random seed", min_value=0, value=int(st.session_state.seed_saxs), step=1)
            q_min = st.number_input("q_min [nm^-1]", min_value=0.00001, value=float(st.session_state.q_min_saxs), step=0.00010, format="%.5f")
            q_max = st.number_input("q_max [nm^-1]", min_value=0.00010, value=float(st.session_state.q_max_saxs), step=0.01000, format="%.5f")
            n_q = st.number_input("Number of q points", min_value=10, value=int(st.session_state.n_q_saxs), step=10)
            max_trials = st.number_input("Max placement trials", min_value=1000, value=int(st.session_state.max_trials_saxs), step=1000)

            run = st.form_submit_button("Run Fractal SAXS", use_container_width=True, type="primary")

        if run:
            st.session_state.R_nm_saxs = float(R_nm)
            st.session_state.Df_saxs = float(Df)
            st.session_state.d_nm_saxs = float(d_nm)
            st.session_state.n_saxs = int(n_particles)
            st.session_state.seed_saxs = int(seed)
            st.session_state.q_min_saxs = float(q_min)
            st.session_state.q_max_saxs = float(q_max)
            st.session_state.n_q_saxs = int(n_q)
            st.session_state.max_trials_saxs = int(max_trials)
            st.session_state["run_saxs"] = True

    else:
        st.markdown("This mode generates a mass-fractal cluster and analyzes its **real-space radial structure**.")
        with st.form("radial_form"):
            st.subheader("Structure parameters")
            R_nm = st.number_input("Cluster radius R [nm]", min_value=1.0, value=float(st.session_state.R_nm_rad), step=10.0)
            Df = st.number_input("Fractal dimension Df", min_value=0.1, max_value=3.0, value=float(st.session_state.Df_rad), step=0.1)
            d_nm = st.number_input("Particle diameter d [nm]", min_value=0.1, value=float(st.session_state.d_nm_rad), step=1.0)
            n_particles = st.number_input("Number of particles n", min_value=1, value=int(st.session_state.n_rad), step=10)

            st.subheader("Profile parameters")
            seed = st.number_input("Random seed", min_value=0, value=int(st.session_state.seed_rad), step=1)
            max_trials = st.number_input("Max placement trials", min_value=1000, value=int(st.session_state.max_trials_rad), step=1000)
            n_bins = st.number_input("Number of radial bins", min_value=5, value=int(st.session_state.n_bins_rad), step=1)

            run = st.form_submit_button("Run Radial Profile", use_container_width=True, type="primary")

        if run:
            st.session_state.R_nm_rad = float(R_nm)
            st.session_state.Df_rad = float(Df)
            st.session_state.d_nm_rad = float(d_nm)
            st.session_state.n_rad = int(n_particles)
            st.session_state.seed_rad = int(seed)
            st.session_state.max_trials_rad = int(max_trials)
            st.session_state.n_bins_rad = int(n_bins)
            st.session_state["run_radial"] = True


if mode == "Fractal SAXS":
    st.subheader("Fractal SAXS")
    st.markdown(
        "This program generates a **mass-fractal particle cluster** in a finite sphere and computes the normalized scattering intensity **I(q)** from the generated coordinates. "
        "It also performs an **iterative Guinier fit** to estimate the radius of gyration."
    )
    st.latex(r"\ln I(q) = \ln I_0 - \frac{R_g^2}{3} q^2")
    st.markdown(r"The fitting range is updated iteratively so that the upper limit approximately satisfies $q_{\max}R_g \lesssim 1.3$.")

    if st.session_state.get("run_saxs", False):
        try:
            with st.spinner("Generating structure and computing scattering..."):
                pos = generate_fractal_cluster(
                    R_nm=float(st.session_state.R_nm_saxs),
                    Df=float(st.session_state.Df_saxs),
                    d_nm=float(st.session_state.d_nm_saxs),
                    n_particles=int(st.session_state.n_saxs),
                    seed=int(st.session_state.seed_saxs),
                    max_trials=int(st.session_state.max_trials_saxs),
                )

                q = np.linspace(float(st.session_state.q_min_saxs), float(st.session_state.q_max_saxs), int(st.session_state.n_q_saxs))
                I_q = scattering_from_positions(pos, d_nm=float(st.session_state.d_nm_saxs), q=q)
                fit = fit_guinier_iterative(q, I_q, qmax_init=min(0.06, float(st.session_state.q_max_saxs) * 0.25))
                rg_real_nm = compute_rg_from_positions(pos)
                r_from_center = np.sqrt(np.sum(pos**2, axis=1))
                Df_fit_radial, _ = fit_mass_fractal_dimension(r_from_center)
                summary = build_summary_dict(
                    R_nm=float(st.session_state.R_nm_saxs),
                    Df=float(st.session_state.Df_saxs),
                    d_nm=float(st.session_state.d_nm_saxs),
                    n_particles=int(st.session_state.n_saxs),
                    fit=fit,
                    rg_real_nm=rg_real_nm,
                    Df_fit_radial=Df_fit_radial,
                )

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Rg (positions) [nm]", f"{summary['Rg_real_nm']:.3f}")
            c2.metric("Rg (Guinier) [nm]", f"{summary['Rg_guinier_nm']:.3f}")
            c3.metric("Df fitted", f"{summary['Df_fit_radial']:.3f}")
            c4.metric("Guinier qmax [nm^-1]", f"{summary['Guinier_qmax_nm^-1']:.5f}")

            left, right = st.columns(2)

            with left:
                fig1 = plt.figure(figsize=(4.2, 3.6))
                ax = fig1.add_subplot(111, projection="3d")
                ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=10)
                ax.set_xlabel("x [nm]")
                ax.set_ylabel("y [nm]")
                ax.set_zlabel("z [nm]")
                ax.set_title("Generated particle configuration")
                xyz_min = pos.min(axis=0)
                xyz_max = pos.max(axis=0)
                center = 0.5 * (xyz_min + xyz_max)
                span = np.max(xyz_max - xyz_min)
                ax.set_xlim(center[0] - span / 2, center[0] + span / 2)
                ax.set_ylim(center[1] - span / 2, center[1] + span / 2)
                ax.set_zlim(center[2] - span / 2, center[2] + span / 2)
                plt.tight_layout()
                st.pyplot(fig1, clear_figure=True)
                plt.close(fig1)

                fig2 = plt.figure(figsize=(4.8, 3.4))
                plt.loglog(q, I_q, label="I(q)")
                plt.xlabel(r"q [nm$^{-1}$]")
                plt.ylabel("normalized intensity")
                plt.title("Scattering intensity")
                plt.legend()
                plt.tight_layout()
                st.pyplot(fig2, clear_figure=True)
                plt.close(fig2)

            with right:
                fig3 = plt.figure(figsize=(4.8, 3.4))
                plt.plot(q**2, np.log(I_q), label="ln I(q)")
                plt.plot(
                    q[fit.mask] ** 2,
                    fit.intercept + fit.slope * q[fit.mask] ** 2,
                    lw=1.8,
                    label=fr"Guinier fit, $R_g$={fit.rg_fit_nm:.2f} nm",
                )
                plt.xlabel(r"$q^2$ [nm$^{-2}$]")
                plt.ylabel(r"$\ln I(q)$")
                plt.title("Guinier plot")
                plt.legend()
                plt.tight_layout()
                st.pyplot(fig3, clear_figure=True)
                plt.close(fig3)

                st.subheader("Summary")
                st.json(summary)

            positions_csv = io.StringIO()
            np.savetxt(positions_csv, pos, delimiter=",", header="x_nm,y_nm,z_nm", comments="")
            scattering_csv = io.StringIO()
            np.savetxt(scattering_csv, np.column_stack([q, I_q]), delimiter=",", header="q_nm^-1,I_q", comments="")
            summary_txt = io.StringIO()
            for k, v in summary.items():
                summary_txt.write(f"{k}={v}\n")

            st.subheader("Download outputs")
            d1, d2, d3 = st.columns(3)
            d1.download_button("Positions CSV", positions_csv.getvalue(), file_name="generated_positions_nm.csv", mime="text/csv", use_container_width=True)
            d2.download_button("I(q) CSV", scattering_csv.getvalue(), file_name="scattering_Iq.csv", mime="text/csv", use_container_width=True)
            d3.download_button("Summary TXT", summary_txt.getvalue(), file_name="guinier_fit_summary.txt", mime="text/plain", use_container_width=True)
        except Exception as e:
            st.error(str(e))
    else:
        st.info("Set parameters in the sidebar and click 'Run Fractal SAXS'.")

else:
    st.subheader("Radial Profile")
    st.markdown(
        "This program generates a **mass-fractal particle cluster** and analyzes its **real-space radial structure**. "
        "It plots the radial number density and cumulative particle fraction as functions of distance from the cluster center."
    )
    st.latex(r"\rho(r_k) = \frac{N_k}{\frac{4\pi}{3}(r_{k+1}^3-r_k^3)}")
    st.markdown("The cumulative fraction shows the fraction of particles contained within radius $r$.")

    if st.session_state.get("run_radial", False):
        try:
            with st.spinner("Generating structure and computing radial profile..."):
                pos = generate_fractal_cluster(
                    R_nm=float(st.session_state.R_nm_rad),
                    Df=float(st.session_state.Df_rad),
                    d_nm=float(st.session_state.d_nm_rad),
                    n_particles=int(st.session_state.n_rad),
                    seed=int(st.session_state.seed_rad),
                    max_trials=int(st.session_state.max_trials_rad),
                )

                profile = radial_concentration_profile(
                    pos,
                    R_nm=float(st.session_state.R_nm_rad),
                    n_bins=int(st.session_state.n_bins_rad),
                )

                r = profile["r_center_nm"]
                rho = profile["number_density_nm^-3"]
                cum = profile["cumulative_fraction"]
                rg_real_nm = compute_rg_from_positions(pos)
                r_from_center = np.sqrt(np.sum(pos**2, axis=1))
                Df_fit_radial, _ = fit_mass_fractal_dimension(r_from_center)

            c1, c2, c3 = st.columns(3)
            c1.metric("Rg (positions) [nm]", f"{rg_real_nm:.3f}")
            c2.metric("Df fitted", f"{Df_fit_radial:.3f}")
            c3.metric("Radial bins", f"{int(st.session_state.n_bins_rad)}")

            left, right = st.columns(2)
            with left:
                fig1 = plt.figure(figsize=(4.2, 3.6))
                ax = fig1.add_subplot(111, projection="3d")
                ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=10)
                ax.set_xlabel("x [nm]")
                ax.set_ylabel("y [nm]")
                ax.set_zlabel("z [nm]")
                ax.set_title("Generated particle configuration")
                xyz_min = pos.min(axis=0)
                xyz_max = pos.max(axis=0)
                center = 0.5 * (xyz_min + xyz_max)
                span = np.max(xyz_max - xyz_min)
                ax.set_xlim(center[0] - span / 2, center[0] + span / 2)
                ax.set_ylim(center[1] - span / 2, center[1] + span / 2)
                ax.set_zlim(center[2] - span / 2, center[2] + span / 2)
                plt.tight_layout()
                st.pyplot(fig1, clear_figure=True)
                plt.close(fig1)

            with right:
                fig2 = plt.figure(figsize=(4.8, 3.4))
                plt.plot(r, rho)
                plt.xlabel("r [nm]")
                plt.ylabel(r"number density [nm$^{-3}$]")
                plt.title("Radial number density")
                plt.tight_layout()
                st.pyplot(fig2, clear_figure=True)
                plt.close(fig2)

                fig3 = plt.figure(figsize=(4.8, 3.4))
                plt.plot(r, cum)
                plt.xlabel("r [nm]")
                plt.ylabel("cumulative fraction")
                plt.title("Cumulative particle fraction")
                plt.ylim(0.0, 1.02)
                plt.tight_layout()
                st.pyplot(fig3, clear_figure=True)
                plt.close(fig3)

            radial_csv = io.StringIO()
            np.savetxt(
                radial_csv,
                np.column_stack([r, rho, cum]),
                delimiter=",",
                header="r_center_nm,number_density_nm^-3,cumulative_fraction",
                comments="",
            )
            positions_csv = io.StringIO()
            np.savetxt(positions_csv, pos, delimiter=",", header="x_nm,y_nm,z_nm", comments="")

            st.subheader("Download outputs")
            d1, d2 = st.columns(2)
            d1.download_button("Radial profile CSV", radial_csv.getvalue(), file_name="radial_profile.csv", mime="text/csv", use_container_width=True)
            d2.download_button("Positions CSV", positions_csv.getvalue(), file_name="generated_positions_nm.csv", mime="text/csv", use_container_width=True)
        except Exception as e:
            st.error(str(e))
    else:
        st.info("Set parameters in the sidebar and click 'Run Radial Profile'.")
