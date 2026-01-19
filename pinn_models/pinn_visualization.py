
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib

# ============================================================
# 1. Load trained model and parameters
# ============================================================
print("=" * 60)
print("PINN Visualization Generator")
print("=" * 60)

# Set font
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'

# Device
torch.manual_seed(42)
np.random.seed(42)

def get_device():
    try:
        import torch_directml
        return torch_directml.device(), "DirectML"
    except ImportError:
        pass
    if torch.cuda.is_available():
        return torch.device('cuda'), "CUDA"
    return torch.device('cpu'), "CPU"

device, device_type = get_device()
print(f"Device: {device} ({device_type})")

# ============================================================
# 2. Physics Parameters (60°C as example)
# ============================================================
L_zone = 8.0
y_max = 0.4
u_bulk = 0.69
T_celsius = 60.0
T_K = T_celsius + 273.15

# Antoine
A_ant, B_ant, C_ant = 4.17726, 1489.756, -69.607
M_sol, R_gas = 108.14, 8.314

log_P_bar = A_ant - B_ant / (T_K + C_ant)
P_sat_Pa = (10 ** log_P_bar) * 1e5
C_sat = P_sat_Pa * M_sol / (R_gas * T_K)

# N2 properties
rho_n2 = (101325 * 0.028) / (8.314 * T_K)
mu_n2 = 1.9376e-05
nu = mu_n2 / rho_n2
D_ab = 0.76e-5 * ((T_K / 298.0) ** 1.75)

print(f"T = {T_celsius}°C, C_sat = {C_sat:.2f} g/m³, D = {D_ab:.2e} m²/s")

# ============================================================
# 3. Model Definition (Must match training)
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
        self.C_scale = C_scale
        
    def forward(self, x, y):
        x_norm = (x - self.x_min) / (self.x_max - self.x_min)
        y_norm = (y - self.y_min) / (self.y_max - self.y_min)
        inputs = torch.cat([x_norm, y_norm], dim=1)
        for i in range(len(self.layers) - 1):
            inputs = self.activation(self.layers[i](inputs))
        output = self.layers[-1](inputs)
        output = torch.sigmoid(output) * self.C_scale
        return output

def boundary_layer_thickness(x, nu_val):
    Re_x = u_bulk * x / nu_val
    Re_x = np.maximum(Re_x, 1e-6)
    delta = 0.37 * x * (Re_x ** (-0.2))
    return delta

# ============================================================
# 4. Train a quick model for visualization
# ============================================================
print("\nTraining PINN for visualization (8000 epochs)...")

model = PINN_FlatPlate([2, 64, 64, 64, 64, 1], C_scale=C_sat).to(device)
# Use SGD with momentum to avoid DirectML CPU fallback (Adam uses lerp internally)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=8000, eta_min=1e-4)

N_pde = 10000
N_bc = 1000
loss_history = []

# Loss weights
lambda_surf = 20.0
lambda_bulk = 15.0
lambda_inlet = 5.0

def boundary_layer_thickness_torch(x, nu_val):
    Re_x = u_bulk * x / nu_val
    Re_x = torch.clamp(Re_x, min=1e-6)
    delta = 0.37 * x * (Re_x ** (-0.2))
    return delta

for epoch in range(8000):
    optimizer.zero_grad()
    
    # === Sampling PDE points ===
    x_pde_np = np.random.uniform(0.1, L_zone, N_pde)
    delta_pde = boundary_layer_thickness(x_pde_np, nu)
    y_pde_np = np.random.uniform(0, 1.0, N_pde) * delta_pde  # Within BL only
    
    x_pde = torch.tensor(x_pde_np, dtype=torch.float32, device=device).view(-1, 1)
    y_pde = torch.tensor(y_pde_np, dtype=torch.float32, device=device).view(-1, 1)
    
    # === Surface BC: C = C_sat at y=0 ===
    x_bc_surf_np = np.random.uniform(0.1, L_zone, N_bc)
    x_bc_surf = torch.tensor(x_bc_surf_np, dtype=torch.float32, device=device).view(-1, 1)
    y_bc_surf = torch.zeros_like(x_bc_surf)
    
    # === Bulk BC: C = C_infinity at y=delta(x) ===
    x_bc_bulk_np = np.random.uniform(0.1, L_zone, N_bc)
    delta_bulk_np = boundary_layer_thickness(x_bc_bulk_np, nu)
    x_bc_bulk = torch.tensor(x_bc_bulk_np, dtype=torch.float32, device=device).view(-1, 1)
    y_bc_bulk = torch.tensor(delta_bulk_np, dtype=torch.float32, device=device).view(-1, 1)
    # C_infinity increases along x (accumulation effect) - matches main PINN script
    C_infinity = C_sat * 0.3 * (x_bc_bulk / L_zone)  # Unified coefficient: 0.3
    
    # === Inlet BC: 1/7 power law profile at x=0.1 ===
    x_inlet = torch.ones((N_bc, 1), device=device) * 0.1
    delta_inlet = boundary_layer_thickness(0.1, nu)
    y_inlet_np = np.random.uniform(0, delta_inlet, N_bc)
    y_inlet = torch.tensor(y_inlet_np, dtype=torch.float32, device=device).view(-1, 1)
    eta_inlet = y_inlet / delta_inlet
    C_inlet_target = C_sat * (1 - eta_inlet ** (1/7))  # 1/7 power law
    
    # === PDE Loss ===
    x_pde.requires_grad_(True)
    y_pde.requires_grad_(True)
    C = model(x_pde, y_pde)
    ones = torch.ones_like(C)
    C_x = torch.autograd.grad(C, x_pde, grad_outputs=ones, create_graph=True)[0]
    C_y = torch.autograd.grad(C, y_pde, grad_outputs=ones, create_graph=True)[0]
    C_yy = torch.autograd.grad(C_y, y_pde, grad_outputs=ones, create_graph=True)[0]
    residual = (u_bulk * C_x - D_ab * C_yy) / C_sat
    loss_pde = torch.mean(residual ** 2)
    
    # === Surface BC Loss ===
    C_surface = model(x_bc_surf, y_bc_surf)
    loss_bc_surf = torch.mean((C_surface - C_sat) ** 2) / (C_sat ** 2)
    
    # === Bulk BC Loss ===
    C_bulk_pred = model(x_bc_bulk, y_bc_bulk)
    loss_bc_bulk = torch.mean((C_bulk_pred - C_infinity) ** 2) / (C_sat ** 2)
    
    # === Inlet BC Loss ===
    C_inlet_pred = model(x_inlet, y_inlet)
    loss_bc_inlet = torch.mean((C_inlet_pred - C_inlet_target) ** 2) / (C_sat ** 2)
    
    # === Total Loss ===
    loss = loss_pde + lambda_surf * loss_bc_surf + lambda_bulk * loss_bc_bulk + lambda_inlet * loss_bc_inlet
    loss.backward()
    optimizer.step()
    scheduler.step()
    
    loss_history.append(loss.item())
    
    if (epoch + 1) % 2000 == 0:
        print(f"Epoch [{epoch+1}/8000] Loss: {loss.item():.2e} (PDE:{loss_pde.item():.2e}, Surf:{loss_bc_surf.item():.2e}, Bulk:{loss_bc_bulk.item():.2e})")

print("Training complete!")
model.eval()

# ============================================================
# 5. Generate Visualizations
# ============================================================
print("\nGenerating visualizations...")

fig = plt.figure(figsize=(16, 12))
gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

# --- 5.1 Concentration Field ---
ax1 = fig.add_subplot(gs[0, :2])
nx, ny = 200, 100
x_grid = np.linspace(0.1, L_zone, nx)
# Limit y range to slightly above max boundary layer thickness
delta_max = boundary_layer_thickness(L_zone, nu)  # ~0.25m at x=8m
y_limit = delta_max * 1.3  # Show just above BL
y_grid = np.linspace(0, y_limit, ny)

X, Y = np.meshgrid(x_grid, y_grid)
X_flat = torch.tensor(X.flatten(), dtype=torch.float32, device=device).view(-1, 1)
Y_flat = torch.tensor(Y.flatten(), dtype=torch.float32, device=device).view(-1, 1)

with torch.no_grad():
    C_pred = model(X_flat, Y_flat).cpu().numpy().reshape(ny, nx)

# Mask outside boundary layer - set to bulk concentration (near zero)
delta_grid = boundary_layer_thickness(x_grid, nu)
C_bulk_approx = 0  # Outside BL concentration
for i in range(nx):
    for j in range(ny):
        if y_grid[j] > delta_grid[i]:
            C_pred[j, i] = C_bulk_approx  # Set to bulk, not NaN

cf = ax1.contourf(X, Y * 1000, C_pred, levels=50, cmap='jet')  # Convert y to mm
ax1.plot(x_grid, delta_grid * 1000, 'w--', linewidth=2, label='Boundary Layer δ(x)')
ax1.set_xlabel('Position x (m)', fontweight='bold')
ax1.set_ylabel('Height y (mm)', fontweight='bold')  # Changed to mm
ax1.set_title('(a) PINN Predicted Concentration Field C(x,y)', fontweight='bold')
ax1.legend(loc='upper right')
plt.colorbar(cf, ax=ax1, label='C (g/m³)')

# --- 5.2 Loss Curve ---
ax2 = fig.add_subplot(gs[0, 2])
ax2.semilogy(loss_history, 'b-', linewidth=1.5)
ax2.set_xlabel('Epoch', fontweight='bold')
ax2.set_ylabel('Loss', fontweight='bold')
ax2.set_title('(b) Training Loss Convergence', fontweight='bold')
ax2.grid(True, alpha=0.3)

# --- 5.3 Concentration Profiles at Different x ---
ax3 = fig.add_subplot(gs[1, 0])
x_positions = [1.0, 4.0, 7.0]
colors = ['blue', 'green', 'red']
y_profile = np.linspace(0, 0.15, 100)

for i, x_pos in enumerate(x_positions):
    x_tensor = torch.ones((100, 1), device=device) * x_pos
    y_tensor = torch.tensor(y_profile, dtype=torch.float32, device=device).view(-1, 1)
    with torch.no_grad():
        C_profile = model(x_tensor, y_tensor).cpu().numpy().flatten()
    ax3.plot(C_profile, y_profile * 1000, colors[i], linewidth=2, label=f'x = {x_pos} m')
    
ax3.set_xlabel('Concentration C (g/m³)', fontweight='bold')
ax3.set_ylabel('Height y (mm)', fontweight='bold')
ax3.set_title('(c) Concentration Profiles', fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

# Add annotation showing gradient decrease
ax3.annotate('Gradient\ndecreases', xy=(C_sat*0.7, 20), fontsize=9, 
             ha='center', color='darkred', fontweight='bold')

# --- 5.4 Surface Flux J(x) ---
ax4 = fig.add_subplot(gs[1, 1])
x_flux = np.linspace(0.1, L_zone, 100)
x_flux_tensor = torch.tensor(x_flux, dtype=torch.float32, device=device).view(-1, 1)
y_flux_tensor = torch.zeros_like(x_flux_tensor).requires_grad_(True)

C_surf = model(x_flux_tensor, y_flux_tensor)
C_y = torch.autograd.grad(C_surf, y_flux_tensor, grad_outputs=torch.ones_like(C_surf), create_graph=False)[0]
# Flux J = -D * dC/dy. Take abs to ensure positive (physical evaporation)
J_pred = np.abs(-D_ab * C_y.cpu().detach().numpy().flatten())

ax4.plot(x_flux, J_pred, 'b-', linewidth=2, label='PINN Flux J(x)')
ax4.fill_between(x_flux, 0, J_pred, alpha=0.3)
ax4.set_xlabel('Position x (m)', fontweight='bold')
ax4.set_ylabel('Flux J (g/m²·s)', fontweight='bold')
ax4.set_title('(d) Surface Evaporation Flux', fontweight='bold')
ax4.grid(True, alpha=0.3)

# Add annotation showing flux decay
max_flux = np.max(J_pred)
mid_flux = J_pred[50]
ax4.annotate('Flux decreases\\ndownstream', 
             xy=(5, mid_flux*1.2), fontsize=9, ha='center', color='darkblue', fontweight='bold')

# --- 5.5 Boundary Layer Thickness ---
ax5 = fig.add_subplot(gs[1, 2])
delta_x = boundary_layer_thickness(x_flux, nu)
ax5.plot(x_flux, delta_x * 1000, 'g-', linewidth=2)
ax5.fill_between(x_flux, 0, delta_x * 1000, alpha=0.3, color='green')
ax5.set_xlabel('Position x (m)', fontweight='bold')
ax5.set_ylabel('δ (mm)', fontweight='bold')
ax5.set_title('(e) Boundary Layer Growth', fontweight='bold')
ax5.grid(True, alpha=0.3)

# --- 5.6 PINN Architecture Diagram ---
ax6 = fig.add_subplot(gs[2, :])
ax6.axis('off')

# Draw neural network diagram
layer_sizes = [2, 64, 64, 64, 64, 1]
layer_names = ['Input\n(x, y)', 'Hidden\n64', 'Hidden\n64', 'Hidden\n64', 'Hidden\n64', 'Output\nC']
layer_x = np.linspace(0.1, 0.9, len(layer_sizes))

for i, (size, name) in enumerate(zip(layer_sizes, layer_names)):
    circle = plt.Circle((layer_x[i], 0.5), 0.06, color='steelblue', alpha=0.8)
    ax6.add_patch(circle)
    ax6.text(layer_x[i], 0.5, name, ha='center', va='center', fontsize=9, fontweight='bold', color='white')
    
    if i < len(layer_sizes) - 1:
        ax6.annotate('', xy=(layer_x[i+1]-0.08, 0.5), xytext=(layer_x[i]+0.08, 0.5),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

# Add loss function boxes
ax6.text(0.5, 0.15, 'Loss = PDE Residual + λ·BC Error', ha='center', fontsize=12, fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='orange'))
ax6.text(0.5, -0.05, r'$\mathcal{L}_{PDE} = \| u \frac{\partial C}{\partial x} - D \frac{\partial^2 C}{\partial y^2} \|^2$   +   $\mathcal{L}_{BC} = \| C|_{y=0} - C_{sat} \|^2$',
         ha='center', fontsize=10, fontweight='bold')

ax6.set_xlim(0, 1)
ax6.set_ylim(-0.2, 0.8)
ax6.set_title('(f) PINN Architecture & Loss Function', fontweight='bold', fontsize=12, y=0.95)

plt.suptitle(f'PINN Flat Plate Evaporation Visualization (T = {T_celsius}°C)', fontweight='bold', fontsize=14)

# Save
output_path = 'd:/VScode file/PINN/pinn_visualization_comprehensive.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\nVisualization saved to: {output_path}")
plt.close(fig)  # Close figure to free memory
