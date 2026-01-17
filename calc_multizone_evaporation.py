
import numpy as np

# ============================================================
# Physical Parameters
# ============================================================
L_zone = 8.0        # Zone length (m)
W_web = 1.2         # Web width (m)
u_bulk = 0.69       # Bulk velocity (m/s)
nu = 1.7e-5         # N2 kinematic viscosity (m²/s)
D_ab = 7.5e-6       # Diffusion coefficient (m²/s)
Sc = nu / D_ab      # Schmidt number

# Constants for Antoine Equation
A_antoine = 4.17726
B_antoine = 1489.756
C_antoine = -69.607

def calc_saturation_concentration(T_celsius):
    """Calculate saturation concentration (g/m³)"""
    T_K = T_celsius + 273.15
    log_P_bar = A_antoine - B_antoine / (T_K + C_antoine)
    P_sat_bar = 10 ** log_P_bar
    P_sat_Pa = P_sat_bar * 1e5
    M_anisole = 108.14  # g/mol
    R = 8.314
    c_sat = P_sat_Pa * M_anisole / (R * T_K)  # g/m³
    return c_sat

def compute_surface_flux_analytical(x_points, C_sat, D=D_ab):
    """
    Compute surface flux using laminar/mixed boundary layer Sherwood correlation
    (matching evaporation_flat_plate_correlation.py)
    
    Laminar (Re < 5e5): Sh = 0.664 * Re^0.5 * Sc^(1/3)
    Mixed (Re > 5e5): Sh = (0.037 * Re^0.8 - 871) * Sc^(1/3)
    """
    # Characteristic length (matching reference code)
    L_char = L_zone  # 8.0 m
    Re_L = u_bulk * L_char / nu
    Re_crit = 5e5
    
    if Re_L < Re_crit:
        # Laminar average Sherwood
        Sh = 0.664 * (Re_L ** 0.5) * (Sc ** (1/3))
    else:
        # Mixed boundary layer
        Sh = (0.037 * (Re_L ** 0.8) - 871) * (Sc ** (1/3))
    
    # Mass transfer coefficient h_m = Sh * D / L [m/s]
    h_m = Sh * D / L_char
    
    # Evaporation flux J = h_m * (C_sat - C_bulk) [g/(m²·s)]
    C_bulk = 0.0
    J = h_m * (C_sat - C_bulk)  # g/(m²·s)
    
    # Return constant flux array (average value)
    return np.ones_like(x_points) * J

# ============================================================
# Multi-Zone Calculation
# ============================================================
print(f"{'='*60}")
print("Multi-Zone Removal Rate Analysis")
print(f"{'='*60}")

# Constants
initial_solvent_mass = 360.0  # g
zones = [40, 60, 80]
residence_time = 24.0 # s
double_sided = 2
x_flux = np.linspace(0.001, L_zone, 200)

total_removed_mass = 0.0

print(f"Initial Solvent Mass: {initial_solvent_mass:.2f} g")
print(f"Web Speed: 20 m/min")
print(f"Residence Time per Zone: {residence_time:.2f} s")
print("-" * 60)

for i, T_zone_val in enumerate(zones):
    # 1. Calculate C_sat for this temperature
    C_sat_zone = calc_saturation_concentration(T_zone_val)
    
    # 2. Calculate flux distribution (Analytical - Laminar/Mixed)
    J_zone = compute_surface_flux_analytical(x_flux, C_sat_zone)
    
    # 3. Integrate to get total mass rate (g/s)
    # Integral J dx * W_web * double_sided
    evap_rate_zone = np.sum((J_zone[:-1] + J_zone[1:]) / 2 * np.diff(x_flux)) * W_web * double_sided
    
    # 4. Total mass removed in this zone (g)
    mass_removed_zone = evap_rate_zone * residence_time
    total_removed_mass += mass_removed_zone
    
    print(f"Zone {i+1} @ {T_zone_val}°C:")
    print(f"  - Saturation Conc: {C_sat_zone:.2f} g/m³")
    print(f"  - Avg Flux: {np.mean(J_zone):.4f} g/(m²·s)")
    print(f"  - Evap Rate: {evap_rate_zone:.4f} g/s ({evap_rate_zone*3.6:.2f} kg/h)")
    print(f"  - Mass Removed: {mass_removed_zone:.2f} g")

print("-" * 60)
removal_rate_pct = (total_removed_mass / initial_solvent_mass) * 100
final_mass = max(0, initial_solvent_mass - total_removed_mass)

print(f"Total Mass Removed: {total_removed_mass:.2f} g")
print(f"Final Residual Mass: {final_mass:.2f} g")
print(f"Total Removal Rate: {removal_rate_pct:.2f}%")
print(f"{'='*60}")
