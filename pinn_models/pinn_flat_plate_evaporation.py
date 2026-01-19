
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
import os

# ============================================================
# 1. Device Configuration
# ============================================================
print("=" * 60)
print("PINN Flat Plate Evaporation Solver (Multi-Zone)")
print("=" * 60)

torch.manual_seed(42)
np.random.seed(42)

def get_device():
    try:
        import torch_directml
        dml_device = torch_directml.device()
        print(f"DirectML device detected: {torch_directml.device_name(0)}")
        return dml_device, "DirectML"
    except ImportError:
        pass
    
    if torch.cuda.is_available():
        return torch.device('cuda'), "CUDA"
    
    return torch.device('cpu'), "CPU"

device, device_type = get_device()
print(f"Device: {device} ({device_type})")

# ============================================================
# 2. Physics & Parameters (Dynamic)
# ============================================================

# Geometry
L_zone = 8.0        # Zone length (m)
W_web = 1.2         # Web width (m)
y_max = 0.4         # Domain height (m)
u_bulk = 0.69       # Bulk velocity (m/s)

# Antoine Constants (Anisole)
A_ant = 4.17726
B_ant = 1489.756
C_ant = -69.607
M_sol = 108.14
R_gas = 8.314

def get_zone_properties(T_celsius):
    """Calculate physical properties for a given temperature"""
    T_K = T_celsius + 273.15
    
    # 1. Saturation Concentration (Antoine)
    log_P_bar = A_ant - B_ant / (T_K + C_ant)
    P_sat_Pa = (10 ** log_P_bar) * 1e5
    C_sat = P_sat_Pa * M_sol / (R_gas * T_K)  # g/m³
    
    # 2. Density of N2 (Ideal Gas)
    # P = 101325 Pa, M_N2 = 0.028 kg/mol
    rho_n2 = (101325 * 0.028) / (8.314 * T_K)
    
    # 3. Dynamic Viscosity of N2 (Interpolated/Approximated from reference)
    # Reference values: 40C->1.85e-5, 60C->1.94e-5, 80C->2.02e-5 (Pa.s)
    # We can use a linear fit or just match the reference code logic
    if T_celsius <= 40:
        mu_n2 = 1.8484e-05
    elif T_celsius <= 60:
        mu_n2 = 1.9376e-05
    else:
        mu_n2 = 2.0246e-05
        
    nu = mu_n2 / rho_n2  # Kinematic viscosity (m²/s)
    
    # 4. Diffusion Coefficient (Fuller Method)
    # References: evaporation_flat_plate_correlation.py
    # D_AB (298K) ≈ 0.76e-5 m^2/s
    # Formula: D_ab = 0.76e-5 * ((T_K / 298) ** 1.75)
    D_ab = 0.76e-5 * ((T_K / 298.0) ** 1.75)
    
    return {
        'C_sat': C_sat,
        'nu': nu,
        'D_ab': D_ab,
        'Sc': nu / D_ab
    }

# ============================================================
# 3. Physics Functions
# ============================================================

def boundary_layer_thickness(x, nu):
    Re_x = u_bulk * x / nu
    Re_x = np.maximum(Re_x, 1e-6)
    delta = 0.37 * x * (Re_x ** (-0.2))
    return delta

def boundary_layer_thickness_torch(x, nu):
    Re_x = u_bulk * x / nu
    Re_x = torch.clamp(Re_x, min=1e-6)
    delta = 0.37 * x * (Re_x ** (-0.2))
    return delta

def analytical_solution_power_law(x, y, C_sat, nu, C_bulk=0):
    delta = boundary_layer_thickness(x, nu)
    eta = y / np.maximum(delta, 1e-6)
    C_infinity = C_bulk + C_sat * 0.3 * (x / L_zone)
    C = np.where(
        eta <= 1,
        C_infinity + (C_sat - C_infinity) * (1 - eta**(1/7)),
        C_infinity
    )
    return C

# ============================================================
# 4. PINN Model
# ============================================================

class PINN_FlatPlate(nn.Module):
    def __init__(self, layers, C_scale):
        super(PINN_FlatPlate, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
        
        for layer in self.layers:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
            
        self.activation = nn.Tanh()
        self.x_min, self.x_max = 0.1, L_zone
        self.y_min, self.y_max = 0.0, y_max
        self.C_scale = C_scale  # Scaling factor (C_sat)
        
    def forward(self, x, y):
        # Normalize inputs
        x_norm = (x - self.x_min) / (self.x_max - self.x_min)
        y_norm = (y - self.y_min) / (self.y_max - self.y_min)
        
        inputs = torch.cat([x_norm, y_norm], dim=1)
        
        for i in range(len(self.layers) - 1):
            inputs = self.activation(self.layers[i](inputs))
            
        output = self.layers[-1](inputs)
        # Scale output to physical range [0, C_sat]
        output = torch.sigmoid(output) * self.C_scale
        return output

# ============================================================
# 5. Loss Calculation
# ============================================================

def compute_pde_residual(model, x, y, D_ab, C_sat):
    x = x.requires_grad_(True)
    y = y.requires_grad_(True)
    
    C = model(x, y)
    
    ones = torch.ones_like(C)
    C_x = torch.autograd.grad(C, x, grad_outputs=ones, create_graph=True)[0]
    C_y = torch.autograd.grad(C, y, grad_outputs=ones, create_graph=True)[0]
    C_yy = torch.autograd.grad(C_y, y, grad_outputs=ones, create_graph=True)[0]
    
    # PDE: u * dC/dx = D * d2C/dy2
    # Residual normalized by C_sat
    residual = (u_bulk * C_x - D_ab * C_yy) / C_sat
    return residual

def compute_losses(model, batch_data, props):
    C_sat = props['C_sat']
    D_ab = props['D_ab']
    nu = props['nu']
    
    # Unpack batch data
    (x_pde, y_pde, x_bc_surf, y_bc_surf, 
     x_bc_bulk, y_bc_bulk, x_inlet, y_inlet) = batch_data
     
    # 1. PDE Loss
    residual = compute_pde_residual(model, x_pde, y_pde, D_ab, C_sat)
    loss_pde = torch.mean(residual ** 2)
    
    # 2. Surface BC Loss (C = C_sat)
    C_surface = model(x_bc_surf, y_bc_surf)
    loss_bc_surface = torch.mean((C_surface - C_sat) ** 2) / (C_sat ** 2)
    
    # 3. Bulk BC Loss (C = C_infinity)
    delta_bulk = boundary_layer_thickness_torch(x_bc_bulk, nu)
    C_infinity = C_sat * 0.3 * (x_bc_bulk / L_zone)
    C_bulk_pred = model(x_bc_bulk, delta_bulk) # Evaluate at BL edge
    loss_bc_bulk = torch.mean((C_bulk_pred - C_infinity) ** 2) / (C_sat ** 2)
    
    # 4. Inlet BC Loss (1/7 power law profile)
    # At x=0.1, boundary layer is thin but not zero.
    C_inlet = model(x_inlet, y_inlet)
    delta_inlet = boundary_layer_thickness_torch(x_inlet, nu)
    
    # Target profile
    mask_in_bl = (y_inlet < delta_inlet).float()
    eta_inlet = y_inlet / torch.clamp(delta_inlet, min=1e-6)
    C_inlet_target = C_sat * (1 - eta_inlet ** (1/7)) * mask_in_bl
    
    loss_bc_inlet = torch.mean((C_inlet - C_inlet_target) ** 2) / (C_sat ** 2)
    
    return loss_pde, loss_bc_surface, loss_bc_bulk, loss_bc_inlet

# ============================================================
# 6. Training Function (Per Zone)
# ============================================================

def train_zone_pinn(T_celsius, epochs=5000):
    print(f"\n{'-'*60}")
    print(f"Training PINN for Zone @ {T_celsius}°C")
    print(f"{'-'*60}")
    
    # Get properties
    props = get_zone_properties(T_celsius)
    C_sat = props['C_sat']
    print(f"Params: C_sat={C_sat:.2f} g/m³, D={props['D_ab']:.2e}, nu={props['nu']:.2e}")
    
    # Initialize Model
    model = PINN_FlatPlate([2, 64, 64, 64, 64, 1], C_scale=C_sat).to(device)
    # Disable foreach to prevent DirectML CPU fallback for lerp operation
    optimizer = optim.Adam(model.parameters(), lr=0.001, foreach=False)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
    # Data sizes (Reduced for stability)
    N_pde = 10000
    N_bc = 1000
    
    # Weights
    lambda_surf = 20.0
    lambda_bulk = 10.0
    lambda_inlet = 5.0
    
    start_time = time.time()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Sampling (Resample every epoch for stochasticity)
        # PDE Points (Randomly in domain)
        # Bias sampling towards boundary layer (0 < y < delta)
        x_pde_np = np.random.uniform(0.1, L_zone, N_pde)
        delta_pde_np = boundary_layer_thickness(x_pde_np, props['nu'])
        # Sample mostly within BL (80%) and some outside (20%)
        y_pde_bl = np.random.uniform(0, 1.2, N_pde) * delta_pde_np
        y_pde_np = np.minimum(y_pde_bl, y_max)
        
        x_pde = torch.tensor(x_pde_np, dtype=torch.float32, device=device).view(-1, 1)
        y_pde = torch.tensor(y_pde_np, dtype=torch.float32, device=device).view(-1, 1)
        
        # BC Points
        x_bc_surf = torch.tensor(np.random.uniform(0.1, L_zone, N_bc), dtype=torch.float32, device=device).view(-1, 1)
        y_bc_surf = torch.zeros_like(x_bc_surf)
        
        x_bc_bulk = torch.tensor(np.random.uniform(0.1, L_zone, N_bc), dtype=torch.float32, device=device).view(-1, 1)
        y_bc_bulk = torch.tensor(np.ones(N_bc) * y_max, dtype=torch.float32, device=device).view(-1, 1) # This is domain top, not just BL edge? 
        # Actually in compute_losses we evaluate at BL edge using delta function. 
        # But let's pass dummy y_bc_bulk here as compute_losses recalculates delta_bulk.
        
        x_inlet = torch.ones((N_bc, 1), device=device) * 0.1
        y_inlet_np = np.random.uniform(0, boundary_layer_thickness(0.1, props['nu'])*1.2, N_bc)
        y_inlet = torch.tensor(y_inlet_np, dtype=torch.float32, device=device).view(-1, 1)
        
        batch = (x_pde, y_pde, x_bc_surf, y_bc_surf, x_bc_bulk, y_bc_bulk, x_inlet, y_inlet)
        
        # Compute Loss
        l_pde, l_surf, l_bulk, l_inlet = compute_losses(model, batch, props)
        loss = l_pde + lambda_surf * l_surf + lambda_bulk * l_bulk + lambda_inlet * l_inlet
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if (epoch + 1) % 1000 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] Loss: {loss.item():.2e} (PDE: {l_pde.item():.2e}, Surf: {l_surf.item():.2e})")
            
    print(f"Training Time: {time.time() - start_time:.1f}s")
    
    # Calculate Evaporation Flux
    model.eval()
    x_flux_np = np.linspace(0.01, L_zone, 200)
    x_flux = torch.tensor(x_flux_np, dtype=torch.float32, device=device).view(-1, 1)
    y_flux = torch.zeros_like(x_flux).requires_grad_(True)
    
    C_surf = model(x_flux, y_flux)
    C_y = torch.autograd.grad(C_surf, y_flux, grad_outputs=torch.ones_like(C_surf), create_graph=False)[0]
    
    # J = -D * dC/dy
    J_pred = -props['D_ab'] * C_y.cpu().detach().numpy().flatten() # g/(m2.s)
    
    return model, J_pred, x_flux_np

# ============================================================
# 7. Main Execution
# ============================================================

if __name__ == "__main__":
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.weight'] = 'bold'
    
    zones = [40, 60, 80]
    results = {}
    
    total_evap_mass = 0.0
    residence_time = 24.0 # s
    W_web = 1.2 # m
    double_sided = 2
    
    # Create Figure for Flux Comparison
    plt.figure(figsize=(10, 6))
    
    for T in zones:
        model, J_pred, x_axis = train_zone_pinn(T, epochs=8000)
        
        # Calculate Mass Removed
        # Integrate J over Area
        # Rate (g/s) = Integral(J dx) * W * 2
        # Manual trapezoidal integration (compatible with all numpy versions)
        evap_rate_gs = np.sum((J_pred[:-1] + J_pred[1:]) / 2 * np.diff(x_axis)) * W_web * double_sided
        mass_removed = evap_rate_gs * residence_time
        
        total_evap_mass += mass_removed
        
        results[T] = {
            'J_mean': np.mean(J_pred),
            'EvapRate_kg_h': evap_rate_gs * 3.6,
            'MassRemoved_g': mass_removed
        }
        
        plt.plot(x_axis, J_pred, linewidth=2, label=f'T={T}°C (Rate={evap_rate_gs*3.6:.2f} kg/h)')

    plt.title('PINN Predicted Surface Evaporation Flux', fontweight='bold')
    plt.xlabel('Position x (m)', fontweight='bold')
    plt.ylabel('Flux J (g/m²·s)', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('d:/VScode file/PINN/pinn_multizone_flux.png')
    
    print(f"\n{'='*60}")
    print("Multi-Zone PINN Calculation Results")
    print(f"{'='*60}")
    print(f"Initial Solvent Mass: 360.0 g")
    
    for T in zones:
        r = results[T]
        print(f"Zone {T}°C:")
        print(f"  Flux (Mean): {r['J_mean']:.4f} g/m²s")
        print(f"  Evap Rate:   {r['EvapRate_kg_h']:.3f} kg/h")
        print(f"  Mass Removed:{r['MassRemoved_g']:.2f} g")
        
    print("-" * 60)
    print(f"Total Mass Removed: {total_evap_mass:.2f} g")
    print(f"Removal Rate: {total_evap_mass / 360.0 * 100:.2f}%")
    print(f"{'='*60}")
