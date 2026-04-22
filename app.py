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

st.set_page_config(page_title="Fractal SAXS / Mass-fractal Profile App", layout="wide")
st.title("Fractal cluster SAXS simulator")


def init_state() -> None:
    defaults = {
        "mode": "Fractal SAXS",
        "R_nm_saxs": 300.0,
        "Df_saxs": 2.2,
        "d_nm_saxs": 20.0,
        "n_saxs": 500,
        "seed_saxs": 2026,
        "q_min_saxs": 0.001,
        "q_max_saxs": 0.30,
        "n_q_saxs": 250,
        "max_trials_saxs": 500000,
        "R_nm_profile": 300.0,
        "Df_profile": 2.2,
        "d_nm_profile": 20.0,
        "n_profile": 500,
        "seed_profile": 2026,
        "q_min_profile": 0.001,
        "q_max_profile": 0.30,
        "n_q_profile": 250,
        "max_trials_profile": 500000,
        "n_bins_profile": 30,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()

with st.sidebar:
    st.header("Tool selection")
    mode = st.radio(
        "Choose analysis mode",
        ["Fractal SAXS", "Radial Profile"],
        key="mode",
    )

    if mode == "Fractal SAXS":
        st.header("Fractal SAXS parameters")
        with st.form("saxs_form"):
            R_nm = st.number_input("Cluster radius R [nm]", min_value=1.0, value=float(st.session_state.R_nm_saxs), step=10.0)
            Df = st.number_input("Fractal dimension Df", min_value=0.1, max_value=3.0, value=float(st.session_state.Df_saxs), step=0.1)
            d_nm = st.number_input("Particle diameter d [nm]", min_value=0.1, value=float(st.session_state.d_nm_saxs), step=1.0)
            n_particles = st.number_input("Number of particles n", min_value=1, value=int(st.session_state.n_saxs), step=10)
            st.subheader("Optional calculation settings")
            seed = st.number_input("Random seed", min_value=0, value=int(st.session_state.seed_saxs), step=1)
            q_min = st.number_input("q_min [nm^-1]", min_value=0.00001, value=float(st.session_state.q_min_saxs), step=0.00010, format="%.5f")
            q_max = st.number_input("q_max [nm^-1]", min_value=0.00010, value=float(st.session_state.q_max_saxs), step=0.01000, format="%.5f")
            n_q = st.number_input("Number of q points", min_value=10, value=int(st.session_state.n_q_saxs), step=10)
            max_trials = st.number_input("Max placement trials", min_value=1000, value=int(st.session_state.max_trials_saxs), step=1000)
            run = st.form_submit_button("Run calculation", type="primary", use_container_width=True)
        st.session_state.R_nm_saxs = R_nm
        st.session_state.Df_saxs = Df
        st.session_state.d_nm_saxs = d_nm
        st.session_state.n_saxs = n_particles
        st.session_state.seed_saxs = seed
        st.session_state.q_min_saxs = q_min
        st.session_state.q_max_saxs = q_max
        st.session_state.n_q_saxs = n_q
        st.session_state.max_trials_saxs = max_trials
    else:
        st.header("Mass-fractal profile parameters")
        with st.form("profile_form"):
            R_nm = st.number_input("Cluster radius R [nm]", min_value=1.0, value=float(st.session_state.R_nm_profile), step=10.0)
            Df = st.number_input("Fractal dimension Df", min_value=0.1, max_value=3.0, value=float(st.session_state.Df_profile), step=0.1)
            d_nm = st.number_input("Particle diameter d [nm]", min_value=0.1, value=float(st.session_state.d_nm_profile), step=1.0)
            n_particles = st.number_input("Number of particles n", min_value=1, value=int(st.session_state.n_profile), step=10)
            n_bins = st.number_input("Number of radial bins", min_value=5, value=int(st.session_state.n_bins_profile), step=1)
            st.subheader("Optional calculation settings")
            seed = st.number_input("Random seed", min_value=0, value=int(st.session_state.seed_profile), step=1)
            q_min = st.number_input("q_min [nm^-1]", min_value=0.00001, value=float(st.session_state.q_min_profile), step=0.00010, format="%.5f")
            q_max = st.number_input("q_max [nm^-1]", min_value=0.00010, value=float(st.session_state.q_max_profile), step=0.01000, format="%.5f")
            n_q = st.number_input("Number of q points", min_value=10, value=int(st.session_state.n_q_profile), step=10)
            max_trials = st.number_input("Max placement trials", min_value=1000, value=int(st.session_state.max_trials_profile), step=1000)
            run = st.form_submit_button("Run calculation", type="primary", use_container_width=True)
        st.session_state.R_nm_profile = R_nm
        st.session_state.Df_profile = Df
        st.session_state.d_nm_profile = d_nm
        st.session_state.n_profile = n_particles
        st.session_state.n_bins_profile = n_bins
        st.session_state.seed_profile = seed
        st.session_state.q_min_profile = q_min
        st.session_state.q_max_profile = q_max
        st.session_state.n_q_profile = n_q
        st.session_state.max_trials_profile = max_trials

if mode == "Fractal SAXS":
    st.caption("Input R, Df, d, and n to generate a finite mass-fractal particle cluster, calculate I(q), and estimate Rg by Guinier fitting.")
    st.markdown(
        r"""
**What this tool does**

This mode generates a **finite mass-fractal-like particle cluster** inside a sphere of radius $R$.
The radial sampling is designed so that the cumulative particle count approximately follows

a power-law relation $N(<r) \sim r^{D_f}$.

It then calculates:
- 3D particle configuration
- normalized scattering intensity $I(q)$
- Guinier plot using
  $\ln I(q) = \ln I_0 - \frac{R_g^2}{3}q^2$
- radius of gyration from both real-space coordinates and Guinier fitting
"""
    )
    st.latex(r"\ln I(q) = \ln I_0 - \frac{R_g^2}{3} q^2")
    st.markdown(r"The fitting range is updated iteratively so that the upper limit approximately satisfies $q_{\max}R_g \lesssim 1.3$.")

    if run:
        try:
            with st.spinner("Generating structure and computing scattering..."):
                pos = generate_fractal_cluster(float(R_nm), float(Df), float(d_nm), int(n_particles), int(seed), int(max_trials))
                q = np.linspace(float(q_min), float(q_max), int(n_q))
                I_q = scattering_from_positions(pos, d_nm=float(d_nm), q=q)
                fit = fit_guinier_iterative(q, I_q, qmax_init=min(0.06, float(q_max) * 0.25))
                rg_real_nm = compute_rg_from_positions(pos)
                r_from_center = np.sqrt(np.sum(pos**2, axis=1))
                Df_fit_radial, _ = fit_mass_fractal_dimension(r_from_center)
                summary = build_summary_dict(float(R_nm), float(Df), float(d_nm), int(n_particles), fit, rg_real_nm, Df_fit_radial)
            st.success("Calculation finished.")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Rg (positions) [nm]", f"{summary['Rg_real_nm']:.3f}")
            c2.metric("Rg (Guinier) [nm]", f"{summary['Rg_guinier_nm']:.3f}")
            c3.metric("Df fitted", f"{summary['Df_fit_radial']:.3f}")
            c4.metric("Guinier qmax [nm^-1]", f"{summary['Guinier_qmax_nm^-1']:.5f}")

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
            st.pyplot(fig1)
            plt.close(fig1)

            fig2 = plt.figure(figsize=(4.8, 3.4))
            plt.loglog(q, I_q, label="I(q)")
            plt.xlabel(r"q [nm$^{-1}$]")
            plt.ylabel("normalized intensity")
            plt.title("Scattering intensity")
            plt.legend()
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)

            fig3 = plt.figure(figsize=(4.8, 3.4))
            plt.plot(q**2, np.log(I_q), label="ln I(q)")
            plt.plot(q[fit.mask] ** 2, fit.intercept + fit.slope * q[fit.mask] ** 2, lw=1.8, label=fr"Guinier fit, $R_g$={fit.rg_fit_nm:.2f} nm")
            plt.xlabel(r"$q^2$ [nm$^{-2}$]")
            plt.ylabel(r"$\ln I(q)$")
            plt.title("Guinier plot")
            plt.legend()
            plt.tight_layout()
            st.pyplot(fig3)
            plt.close(fig3)

            positions_csv = io.StringIO()
            np.savetxt(positions_csv, pos, delimiter=",", header="x_nm,y_nm,z_nm", comments="")
            scattering_csv = io.StringIO()
            np.savetxt(scattering_csv, np.column_stack([q, I_q]), delimiter=",", header="q_nm^-1,I_q", comments="")
            summary_txt = io.StringIO()
            summary_txt.write("Input parameters\n")
            summary_txt.write(f"R_nm={summary['R_nm']}\n")
            summary_txt.write(f"Df_target={summary['Df_target']}\n")
            summary_txt.write(f"d_nm={summary['d_nm']}\n")
            summary_txt.write(f"n_particles={summary['n_particles']}\n\n")
            summary_txt.write("Guinier equation\n")
            summary_txt.write("ln I(q) = ln I0 - (Rg^2 / 3) q^2\n\n")
            summary_txt.write("Generated cluster diagnostics\n")
            summary_txt.write(f"Df_fit_radial={summary['Df_fit_radial']}\n")
            summary_txt.write(f"Rg_real_nm={summary['Rg_real_nm']}\n")
            summary_txt.write(f"Rg_guinier_nm={summary['Rg_guinier_nm']}\n")
            summary_txt.write(f"Guinier_qmax_nm^-1={summary['Guinier_qmax_nm^-1']}\n")
            summary_txt.write(f"Guinier_n_points={summary['Guinier_n_points']}\n")

            st.subheader("Download outputs")
            d1, d2, d3 = st.columns(3)
            d1.download_button("Positions CSV", positions_csv.getvalue(), file_name="generated_positions_nm.csv", mime="text/csv", use_container_width=True)
            d2.download_button("I(q) CSV", scattering_csv.getvalue(), file_name="scattering_Iq.csv", mime="text/csv", use_container_width=True)
            d3.download_button("Summary TXT", summary_txt.getvalue(), file_name="guinier_fit_summary.txt", mime="text/plain", use_container_width=True)

            st.subheader("Summary")
            st.json(summary)
        except Exception as e:
            st.error(str(e))
    else:
        st.info("Set parameters in the sidebar and click 'Run calculation'.")
else:
    st.caption("Input R, Df, d, and n to generate a finite mass-fractal particle cluster, then visualize its real-space radial density profile together with I(q) and Guinier fit.")
    st.markdown(
        r"""
**What this tool does**

This mode generates a **finite mass-fractal-like particle cluster** inside a sphere of radius $R$ using the input fractal dimension $D_f$.
The radial sampling is designed so that the cumulative particle count approximately follows

a power-law relation $N(<r) \sim r^{D_f}$.

It then evaluates the structure in **real space** and **reciprocal space** by calculating:
- 3D particle configuration
- radial number-density profile
- cumulative particle fraction
- normalized scattering intensity $I(q)$
- Guinier plot and estimated $R_g$
- fitted radial fractal dimension from $N(<r)$
"""
    )
    st.latex(r"N(<r) \sim r^{D_f}")
    st.latex(r"\rho(r_k) = \frac{N_k}{\frac{4\pi}{3}(r_{k+1}^3-r_k^3)}")
    st.latex(r"\ln I(q) = \ln I_0 - \frac{R_g^2}{3} q^2")

    if run:
        try:
            with st.spinner("Generating mass-fractal structure and computing profiles..."):
                pos = generate_fractal_cluster(float(R_nm), float(Df), float(d_nm), int(n_particles), int(seed), int(max_trials))
                q = np.linspace(float(q_min), float(q_max), int(n_q))
                I_q = scattering_from_positions(pos, d_nm=float(d_nm), q=q)
                fit = fit_guinier_iterative(q, I_q, qmax_init=min(0.06, float(q_max) * 0.25))
                rg_real_nm = compute_rg_from_positions(pos)
                r_from_center = np.sqrt(np.sum(pos**2, axis=1))
                Df_fit_radial, _ = fit_mass_fractal_dimension(r_from_center)
                profile = radial_concentration_profile(pos, R_nm=float(R_nm), n_bins=int(n_bins))
                summary = build_summary_dict(float(R_nm), float(Df), float(d_nm), int(n_particles), fit, rg_real_nm, Df_fit_radial)
            st.success("Calculation finished.")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Rg (positions) [nm]", f"{summary['Rg_real_nm']:.3f}")
            c2.metric("Rg (Guinier) [nm]", f"{summary['Rg_guinier_nm']:.3f}")
            c3.metric("Df input", f"{summary['Df_target']:.3f}")
            c4.metric("Df fitted", f"{summary['Df_fit_radial']:.3f}")

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
            st.pyplot(fig1)
            plt.close(fig1)

            col1, col2 = st.columns(2)
            with col1:
                fig2 = plt.figure(figsize=(4.8, 3.4))
                plt.loglog(q, I_q, label="I(q)")
                plt.xlabel(r"q [nm$^{-1}$]")
                plt.ylabel("normalized intensity")
                plt.title("Scattering intensity")
                plt.legend()
                plt.tight_layout()
                st.pyplot(fig2)
                plt.close(fig2)
            with col2:
                fig3 = plt.figure(figsize=(4.8, 3.4))
                plt.plot(q**2, np.log(I_q), label="ln I(q)")
                plt.plot(q[fit.mask] ** 2, fit.intercept + fit.slope * q[fit.mask] ** 2, lw=1.8, label=fr"Guinier fit, $R_g$={fit.rg_fit_nm:.2f} nm")
                plt.xlabel(r"$q^2$ [nm$^{-2}$]")
                plt.ylabel(r"$\ln I(q)$")
                plt.title("Guinier plot")
                plt.legend()
                plt.tight_layout()
                st.pyplot(fig3)
                plt.close(fig3)

            col3, col4 = st.columns(2)
            with col3:
                fig4 = plt.figure(figsize=(4.8, 3.4))
                plt.plot(profile["r_center_nm"], profile["number_density_nm^-3"], marker="o", ms=3)
                plt.xlabel("r [nm]")
                plt.ylabel(r"number density [nm$^{-3}$]")
                plt.title("Radial concentration profile")
                plt.tight_layout()
                st.pyplot(fig4)
                plt.close(fig4)
            with col4:
                fig5 = plt.figure(figsize=(4.8, 3.4))
                plt.plot(profile["r_center_nm"], profile["cumulative_fraction"], marker="o", ms=3)
                plt.xlabel("r [nm]")
                plt.ylabel("cumulative fraction")
                plt.title("Cumulative radial fraction")
                plt.tight_layout()
                st.pyplot(fig5)
                plt.close(fig5)

            positions_csv = io.StringIO()
            np.savetxt(positions_csv, pos, delimiter=",", header="x_nm,y_nm,z_nm", comments="")
            scattering_csv = io.StringIO()
            np.savetxt(scattering_csv, np.column_stack([q, I_q]), delimiter=",", header="q_nm^-1,I_q", comments="")
            profile_csv = io.StringIO()
            np.savetxt(profile_csv, np.column_stack([profile["r_center_nm"], profile["count"], profile["number_density_nm^-3"], profile["cumulative_fraction"]]), delimiter=",", header="r_center_nm,count,number_density_nm^-3,cumulative_fraction", comments="")
            summary_txt = io.StringIO()
            summary_txt.write("Input parameters\n")
            summary_txt.write(f"R_nm={summary['R_nm']}\n")
            summary_txt.write(f"Df_target={summary['Df_target']}\n")
            summary_txt.write(f"d_nm={summary['d_nm']}\n")
            summary_txt.write(f"n_particles={summary['n_particles']}\n\n")
            summary_txt.write("Diagnostics\n")
            summary_txt.write(f"Df_fit_radial={summary['Df_fit_radial']}\n")
            summary_txt.write(f"Rg_real_nm={summary['Rg_real_nm']}\n")
            summary_txt.write(f"Rg_guinier_nm={summary['Rg_guinier_nm']}\n")
            summary_txt.write(f"Guinier_qmax_nm^-1={summary['Guinier_qmax_nm^-1']}\n")
            summary_txt.write(f"Guinier_n_points={summary['Guinier_n_points']}\n")

            st.subheader("Download outputs")
            d1, d2, d3, d4 = st.columns(4)
            d1.download_button("Positions CSV", positions_csv.getvalue(), file_name="generated_positions_nm.csv", mime="text/csv", use_container_width=True)
            d2.download_button("I(q) CSV", scattering_csv.getvalue(), file_name="scattering_Iq.csv", mime="text/csv", use_container_width=True)
            d3.download_button("Radial profile CSV", profile_csv.getvalue(), file_name="radial_profile.csv", mime="text/csv", use_container_width=True)
            d4.download_button("Summary TXT", summary_txt.getvalue(), file_name="profile_summary.txt", mime="text/plain", use_container_width=True)

            st.subheader("Summary")
            st.json(summary)
        except Exception as e:
            st.error(str(e))
    else:
        st.info("Set parameters in the sidebar and click 'Run calculation'.")
