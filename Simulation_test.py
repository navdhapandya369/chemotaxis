import pandas as pd
import pde
from scipy.stats.qmc import LatinHypercube
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
import matplotlib.pyplot as plt
import numpy as np
import time
#from GPR.py import run_simulation

# ============================================================
# 1. FIXED CONSTANTS
# ============================================================
K_eq  = 100.0
delta = 0.8
D_E   = 40.0
D_EL  = delta * D_E       # 32.0
D_S   = 10.0 * D_E        # 400.0
D_P   = 10.0 * D_E        # 400.0 
L     = 20.0

E_over_S = 1e-3

print("Fixed constants:")
print(f"  K_eq  = {K_eq}")
print(f"  delta = {delta}")
print(f"  D_E   = {D_E}  (μm)²/s")
print(f"  D_EL  = {D_EL} (μm)²/s")
print(f"  D_S   = {D_S}  (μm)²/s")
print(f"  D_P   = {D_P}  (μm)²/s")
print(f"  E0/S0 = {E_over_S}  (E0 = {E_over_S} × S0 in each run)")
print(f"  L     = {L}    μm")



def run_simulation(k1, km1, S0, kcat0, alpha):
    """
    Runs py-pde simulation with:
      k2  = kcat0/(1+alpha*S)        — varies spatially
      km2 = k1*kcat0/(km1*Keq*(1+alpha*S)) — varies spatially

    Returns:
      x_max    — location of E+EL peak (None if no interior peak)
      strength — max(E+EL) - min(E+EL)
      label    — 1 if interior peak exists, 0 if not
    """
    E0 = E_over_S * S0
    grid = pde.CartesianGrid([[0, L]], [200], periodic=False)

    eq = pde.PDE(
        {
            # k2  = kcat0/(1+alpha*S)
            # km2 = k1*kcat0/(km1*Keq*(1+alpha*S))
            # both depend on local S(x,t)

            "E":  """D_E*laplace(E) - k1*E*S + km1*EL - (k1*kcat0/(km1*Keq*(1+alpha*S)))*P*E + (kcat0/(1+alpha*S))*EL""",

            "EL": """D_EL*laplace(EL) + k1*E*S + (k1*kcat0/(km1*Keq*(1+alpha*S)))*P*E - km1*EL - (kcat0/(1+alpha*S))*EL""",

            "S":  """D_S*laplace(S) - k1*E*S + km1*EL""",

            "P":  """D_P*laplace(P) + (kcat0/(1+alpha*S))*EL - (k1*kcat0/(km1*Keq*(1+alpha*S)))*P*E""",
        },
        consts={
            "D_E":   D_E,
            "D_EL":  D_EL,
            "D_S":   D_S,
            "D_P":   D_P,
            "k1":    k1,
            "km1":   km1,
            "kcat0": kcat0,
            "alpha": alpha,
            "Keq":   K_eq,
        },
        bc_ops={
            "E:*":  "neumann",
            "EL:*": "neumann",
            "S:*":  ({"value": 0.9*S0}, {"value": 0.1*S0}),
            "P:*":  ({"value": 0.1*S0}, {"value": 0.9*S0}),
        },
    )

    # initial conditions
    x = grid.axes_coords[0]                 # the 200 positions, 0.05 ... 19.95 μm

    S_init = 0.9*S0 - 0.8*S0 * x / L        # 0.9*S0 at left → 0.1*S0 at right
    P_init = 0.1*S0 + 0.8*S0 * x / L        # 0.1*S0 at left → 0.9*S0 at right

    E_f   = pde.ScalarField(grid, E0,  label="E")
    EL_f  = pde.ScalarField(grid, 0.0, label="EL")
    S_f   = pde.ScalarField(grid, S_init, label="S")
    P_f   = pde.ScalarField(grid, P_init, label="P")
    state = pde.FieldCollection([E_f, EL_f, S_f, P_f])

    # solve
    try:
        result = eq.solve(state, t_range=100, dt=None)
    except Exception as e:
        print(f"  Simulation failed: {e}")
        return None, 0.0, 0, np.zeros(200)

    # extract E+EL profile
    x_coords = grid.axes_coords[0]
    E_total  = result[0].data + result[1].data

    # check for unphysical results
    if np.any(np.isnan(E_total)) or np.any(E_total < 0):
        print(f"  Unphysical result — skipping")
        return None, 0.0, 0, np.zeros(200)  # ← 4 values now

    # find interior extrema
    
    margin_pts = int(0.1 * len(x_coords))
    interior_E = E_total[margin_pts:-margin_pts]
    interior_x = x_coords[margin_pts:-margin_pts]
    gradient     = np.gradient(interior_E, interior_x)
    sign_changes = np.where(np.diff(np.sign(gradient)))[0]

    if len(sign_changes) > 0:
        x_max_idx = np.argmax(E_total)
        x_max_val = float(x_coords[x_max_idx])
        margin_um = 0.1 * L

        if margin_um < x_max_val < (L - margin_um):
            strength = float(E_total.max() - E_total.min())
            if strength / E_total.mean() > 0.001:
                return x_max_val, strength, 1, E_total
            else:
                return None, 0.0, 0, E_total
        else:
            return None, 0.0, 0, E_total
    else:
        return None, 0.0, 0, E_total
xmax, strength, label, Etotal = run_simulation(141.065,209.901,6.441,1222.408,4.592)
print(f"x_max    = {xmax}")
print(f"strength = {strength}")
print(f"label    = {label}")

# ── plot E_total vs x ────────────────────────────────────────
grid_plot = pde.CartesianGrid([[0, L]], [200], periodic=False)
x_coords  = grid_plot.axes_coords[0]

plt.figure(figsize=(10, 5))
plt.plot(x_coords, Etotal, 'k-', linewidth=2, label="E + EL")

if xmax is not None:
    plt.axvline(x=xmax, color='red', linestyle='--',
                label=f"x_max = {xmax:.3f} μm")
    plt.scatter([xmax], [Etotal.max()],
                color='red', s=100, zorder=5)

plt.axvline(x=0.1*L, color='gray', linestyle=':',
            alpha=0.5, label="10% margin")
plt.axvline(x=0.9*L, color='gray', linestyle=':',
            alpha=0.5)

plt.xlabel("x (μm)")
plt.ylabel("E + EL (total enzyme, mM)")
plt.title(f"Total Enzyme Profile\n"
          f"k1={141.065}, km1={209.901}, "
          f"S0={6.441}, kcat0={1222.408}, α={4.592}")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
