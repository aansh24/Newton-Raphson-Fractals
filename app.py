import streamlit as st
import numpy as np
import sympy as sp
from PIL import Image

# --- Core Math Logic (Identical to your Tkinter version) ---
def generate_newton_fractal(formula_str, max_iter, epsilon, width=600, height=600):
    z_sym = sp.Symbol("z")
    try:
        f_sym = sp.parse_expr(formula_str) 
        f_prime_sym = sp.diff(f_sym, z_sym)
        f = sp.lambdify(z_sym, f_sym, "numpy")
        f_prime = sp.lambdify(z_sym, f_prime_sym, "numpy")
        all_roots = sp.solve(f_sym, z_sym)
        
        ROOTS = np.array([complex(p.evalf()) if hasattr(p, 'evalf') else complex(p) for p in all_roots], dtype=complex)
    except Exception as e:
        raise ValueError(f"Math Parsing Error: {e}")

    x = np.linspace(-2.0, 2.0, width)
    y = np.linspace(-2.0, 2.0, height)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y

    root_grid = np.zeros(Z.shape, dtype=int)
    iter_grid = np.zeros(Z.shape, dtype=float)

    for i in range(max_iter):
        df = f_prime(Z)
        
        if not isinstance(df, np.ndarray):
            df = np.full(Z.shape, df, dtype=complex)
        else:
            df = df.astype(complex)
             
        df[df == 0] = 1e-12
        Z_next = Z - f(Z) / df

        for idx, root in enumerate(ROOTS):
            converged = np.abs(Z_next - root) < epsilon
            unassigned = root_grid == 0
            updating = converged & unassigned
            root_grid[updating] = idx + 1
            iter_grid[updating] = i
        Z = Z_next

    light_effect = root_grid + (iter_grid / max_iter) * 0.4

    if light_effect.max() > 0:
        normalized = light_effect / light_effect.max() * 255
    else:
        normalized = np.zeros(light_effect.shape)

    r_chan = np.clip((normalized * 3) % 256, 0, 255).astype(np.uint8)
    g_chan = np.clip((normalized * 7) % 256, 0, 255).astype(np.uint8)
    b_chan = np.clip((normalized * 13) % 256, 0, 255).astype(np.uint8)

    rgb_img = np.stack([r_chan, g_chan, b_chan], axis=-1)

    return Image.fromarray(rgb_img)


# --- Streamlit UI Configuration ---
st.set_page_config(page_title="Newton Fractal Generator", layout="centered", page_icon="🌀")

st.title("🌀 Newton Fractal Generator")
st.markdown("Visualize the mathematical beauty of the Newton-Raphson method finding complex roots. Adjust the parameters below to render the fractal.")

# --- Control Panel ---
# Using columns to layout the inputs neatly side-by-side
col1, col2, col3 = st.columns(3)

with col1:
    formula = st.selectbox(
        "Select Equation:",
        ("z**3 - 1", "z**4 - 1", "z**5 - z - 1", "z**3 - 2*z + 2"),
        index=0
    )

with col2:
    max_iter = int(st.selectbox(
        "Max Iterations:",
        ("5", "15", "30", "50", "80", "100"),
        index=3
    ))

with col3:
    raw_eps = st.number_input(
        "Epsilon Precision:",
        min_value=0.0000001,
        max_value=0.9,
        value=0.001,
        format="%f"
    )

# --- Execution ---
if st.button("🔄 Generate Fractal", type="primary"):
    # The spinner acts as a loading indicator while the server calculates the roots
    with st.spinner(f"Crunching complex numbers for {formula}... this takes a few seconds!"):
        try:
            img = generate_newton_fractal(
                formula, max_iter, raw_eps, width=600, height=520
            )
            st.image(img, use_container_width=True, caption=f"Newton Fractal: {formula}")
