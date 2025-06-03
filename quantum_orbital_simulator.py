import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.special import sph_harm
from math import pi

# Constants
h = 6.626e-34  # Planck's constant (J·s)
c = 3e8        # Speed of light (m/s)
eV = 1.602e-19 # Electron volt (J)
R_H = 13.6     # Rydberg constant for hydrogen (eV)
hbar = h / (2 * np.pi)
R_inf = 1.097e7  # Rydberg constant (1/m)

# Element names for Z
element_names = {1: "Hydrogen", 2: "Helium⁺", 3: "Lithium²⁺", 4: "Beryllium³⁺"}

st.set_page_config(layout="wide")
st.title(" ⚛️ Qantum Orbital Simulator: Orbitals, Transitions & Spectra")

# Sidebar inputs
Z = st.sidebar.slider("Atomic Number (Z)", 1, 4, 1)
st.markdown(f"### Selected Element: **{element_names[Z]}**")

n_max = 5
n_options = list(range(1, n_max + 1)) + ["∞"]
n1 = st.sidebar.selectbox("Initial Energy Level n₁ (Shell)", options=n_options[:-1], index=0)
n2 = st.sidebar.selectbox("Final Energy Level n₂ (Shell)", options=n_options, index=1)

# Convert n1 and n2 to integers
n1_int = int(n1)
n2_int = np.inf if n2 == "∞" else int(n2)

# Allowed l values
def allowed_l(n):
    return list(range(min(n, 4)))  # s=0 to f=3, max l = n-1

l1_allowed = allowed_l(n1_int)
l2_allowed = allowed_l(n2_int if np.isfinite(n2_int) else 5)

l1 = st.sidebar.selectbox(f"Initial Subshell l₁", options=l1_allowed, format_func=lambda x: ["s", "p", "d", "f"][x])
l2 = st.sidebar.selectbox(f"Final Subshell l₂", options=l2_allowed, format_func=lambda x: ["s", "p", "d", "f"][x])

# Determine transition type
if n1_int != n2_int:
    delta_E_J = R_H * eV * Z**2 * (1 / n2_int**2 - 1 / n1_int**2)
    delta_E_eV = delta_E_J / eV
    wavelength = 1 / (R_inf * Z**2 * abs(1 / n2_int**2 - 1 / n1_int**2))  # in meters
    wavelength_nm = wavelength * 1e9

    # Format scientific notation
    def sci_notation(val, unit):
        if val == 0:
            return f"0 {unit}"
        exp = int(np.floor(np.log10(abs(val))))
        coeff = val / 10**exp
        return f"{coeff:.3f}×10^{exp} {unit}"

    st.markdown("### 📡 Photon Emission / Absorption")
    if delta_E_J > 0:
        st.markdown(f"- **Absorption:** Electron absorbs a photon.")
    else:
        st.markdown(f"- **Emission:** Electron emits a photon.")
    st.markdown(f"- **Energy:** {abs(delta_E_eV):.3f} eV ({sci_notation(abs(delta_E_J), 'J) Excluding Subshell Transition')}")
    st.markdown(f"- **Wavelength:** {wavelength_nm:.2f} nm")
else:
    st.markdown("### ⚠️ No Transition")
    st.markdown("- **Same initial and final level**: no photon emitted or absorbed.")
    delta_E_J = 0
    wavelength_nm = None

# Generate orbital surfaces
theta, phi = np.mgrid[0:pi:100j, 0:2 * pi:100j]

def orbital_surface(l, m):
    Y_lm = sph_harm(m, l, phi, theta).real
    density = np.abs(Y_lm) ** 2
    R = density / np.max(density)
    x = R * np.sin(theta) * np.cos(phi)
    y = R * np.sin(theta) * np.sin(phi)
    z = R * np.cos(theta)
    return x, y, z, density

# Initial orbital
m1 = 0  # fixed m for simplicity
x1, y1, z1, density1 = orbital_surface(l1, m1)
fig1 = go.Figure(data=[go.Surface(x=x1, y=y1, z=z1, surfacecolor=density1, colorscale='Viridis', showscale=False)])
fig1.update_layout(scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False), aspectmode='cube'))
st.subheader(f"Initial Orbital: n = {n1_int}, Subshell = {['s', 'p', 'd', 'f'][l1]}")
st.plotly_chart(fig1, use_container_width=True, key="orbital1")

# Final orbital
if n1_int != n2_int:
    m2 = 0
    x2, y2, z2, density2 = orbital_surface(l2, m2)
    fig2 = go.Figure(data=[go.Surface(x=x2, y=y2, z=z2, surfacecolor=density2, colorscale='Plasma', showscale=False)])
    fig2.update_layout(scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False), aspectmode='cube'))
    st.subheader(f"Final Orbital: n = {n2}, Subshell = {['s', 'p', 'd', 'f'][l2]}")
    st.plotly_chart(fig2, use_container_width=True, key="orbital2")

    # Transition animation
    st.subheader("Transition Animation")
    frames = []
    steps = 20
    for i in range(steps + 1):
        alpha = i / steps
        x = (1 - alpha) * x1 + alpha * x2
        y = (1 - alpha) * y1 + alpha * y2
        z = (1 - alpha) * z1 + alpha * z2
        density = (1 - alpha) * density1 + alpha * density2
        frame = go.Frame(data=[go.Surface(x=x, y=y, z=z, surfacecolor=density, colorscale='Viridis', showscale=False)])
        frames.append(frame)

    fig_anim = go.Figure(data=[go.Surface(x=x1, y=y1, z=z1, surfacecolor=density1, colorscale='Viridis', showscale=False)],
                         frames=frames)
    fig_anim.update_layout(scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False), aspectmode='cube'),
                           updatemenus=[dict(type='buttons',
                                             showactive=False,
                                             buttons=[dict(label='Play',
                                                           method='animate',
                                                           args=[None, dict(frame=dict(duration=100, redraw=True), fromcurrent=True)])])])
    st.plotly_chart(fig_anim, use_container_width=True, key="transition_animation")

# Spectrum display
show_spectrum = st.sidebar.checkbox("Show Emission/Absorption Lines", value=True)
if show_spectrum and n1_int != n2_int:
    st.subheader("Spectral Lines")
    fig_spec = go.Figure()
    for n_hi in range(n1_int + 1, n1_int + 6):
        if n_hi > n1_int:
            inv_lambda = R_inf * Z**2 * abs(1 / n1_int**2 - 1 / n_hi**2)
            wavelength_nm = 1e9 / inv_lambda
            fig_spec.add_trace(go.Scatter(x=[wavelength_nm, wavelength_nm], y=[0, 1], mode="lines",
                                          line=dict(color='blue', width=3), name=f"{wavelength_nm:.1f} nm"))
    fig_spec.update_layout(title=f"Emission/Absorption Lines for Z = {Z}",
                           xaxis_title="Wavelength (nm)", yaxis_visible=False, showlegend=True)
    st.plotly_chart(fig_spec, use_container_width=True, key="spectrum")
