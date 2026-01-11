"""
PINN（物理信息神经网络）示例 - 1D热传导方程求解
==================================================

本代码演示如何使用PINN求解一维热传导方程：
    ∂u/∂t = α * ∂²u/∂x²

边界条件: u(0,t) = u(1,t) = 0
初始条件: u(x,0) = sin(πx)

解析解: u(x,t) = sin(πx) * exp(-α*π²*t)

作者: PINN入门教程
日期: 2026年1月
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import time

# ============================================================
# 1. 设备配置（支持DirectML/CUDA/CPU）
# ============================================================
print("=" * 60)
print("PINN 1D热传导方程求解示例")
print("=" * 60)

# 设置随机种子以确保可重复性
torch.manual_seed(42)
np.random.seed(42)

# 自动选择设备：优先DirectML(AMD) -> CUDA(NVIDIA) -> CPU
def get_device():
    """检测并返回最佳可用设备"""
    # 1. 尝试DirectML（AMD显卡）
    try:
        import torch_directml
        dml_device = torch_directml.device()
        print(f"检测到DirectML设备: {torch_directml.device_name(0)}")
        return dml_device, "DirectML"
    except ImportError:
        pass
    except Exception as e:
        print(f"DirectML初始化失败: {e}")
    
    # 2. 尝试CUDA（NVIDIA显卡）
    if torch.cuda.is_available():
        return torch.device('cuda'), "CUDA"
    
    # 3. 使用CPU
    return torch.device('cpu'), "CPU"

device, device_type = get_device()

print(f"\n[设备信息]")
print(f"PyTorch版本: {torch.__version__}")
print(f"使用设备: {device} ({device_type})")
if device_type == "CUDA":
    print(f"GPU型号: {torch.cuda.get_device_name(0)}")

# ============================================================
# 2. 物理参数设置
# ============================================================
alpha = 0.01        # 热扩散系数 (m²/s)
x_min, x_max = 0.0, 1.0  # 空间域
t_min, t_max = 0.0, 1.0  # 时间域

print(f"\n[物理参数]")
print(f"热扩散系数 α = {alpha}")
print(f"空间域: x ∈ [{x_min}, {x_max}]")
print(f"时间域: t ∈ [{t_min}, {t_max}]")

# ============================================================
# 3. 解析解（用于验证）
# ============================================================
def analytical_solution(x, t, alpha=0.01):
    """
    热传导方程的解析解
    u(x,t) = sin(πx) * exp(-α*π²*t)
    """
    return np.sin(np.pi * x) * np.exp(-alpha * np.pi**2 * t)

# ============================================================
# 4. 神经网络定义
# ============================================================
class PINN(nn.Module):
    """
    物理信息神经网络
    
    结构: 输入(x,t) -> 隐藏层 -> 输出u
    激活函数: Tanh (对于PINN效果较好，因为其导数平滑)
    """
    def __init__(self, layers):
        """
        参数:
            layers: 网络层结构，如 [2, 32, 32, 32, 1]
                   输入层2个神经元(x,t)，输出层1个神经元(u)
        """
        super(PINN, self).__init__()
        
        self.layers = nn.ModuleList()
        self.num_layers = len(layers)
        
        # 创建网络层
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
        
        # Xavier初始化（对于Tanh激活函数效果好）
        for layer in self.layers:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
        
        # 激活函数
        self.activation = nn.Tanh()
    
    def forward(self, x, t):
        """
        前向传播
        
        参数:
            x: 空间坐标 (N, 1)
            t: 时间坐标 (N, 1)
        返回:
            u: 温度场 (N, 1)
        """
        # 拼接输入
        inputs = torch.cat([x, t], dim=1)
        
        # 通过隐藏层
        for i in range(len(self.layers) - 1):
            inputs = self.activation(self.layers[i](inputs))
        
        # 输出层（不加激活函数）
        output = self.layers[-1](inputs)
        return output

# ============================================================
# 5. 损失函数定义
# ============================================================
def compute_pde_residual(model, x, t, alpha):
    """
    计算PDE残差: r = ∂u/∂t - α * ∂²u/∂x²
    
    使用PyTorch自动微分计算偏导数
    """
    # 确保需要梯度
    x = x.requires_grad_(True)
    t = t.requires_grad_(True)
    
    # 前向传播
    u = model(x, t)
    
    # 计算一阶偏导数 ∂u/∂t
    u_t = torch.autograd.grad(
        outputs=u, 
        inputs=t, 
        grad_outputs=torch.ones_like(u),
        create_graph=True,  # 保留计算图以计算高阶导数
        retain_graph=True
    )[0]
    
    # 计算一阶偏导数 ∂u/∂x
    u_x = torch.autograd.grad(
        outputs=u, 
        inputs=x, 
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True
    )[0]
    
    # 计算二阶偏导数 ∂²u/∂x²
    u_xx = torch.autograd.grad(
        outputs=u_x, 
        inputs=x, 
        grad_outputs=torch.ones_like(u_x),
        create_graph=True,
        retain_graph=True
    )[0]
    
    # PDE残差: ∂u/∂t - α * ∂²u/∂x² = 0
    residual = u_t - alpha * u_xx
    
    return residual

def compute_losses(model, x_pde, t_pde, x_bc, t_bc, x_ic, alpha):
    """
    计算总损失函数
    
    Total Loss = Loss_PDE + λ_bc * Loss_BC + λ_ic * Loss_IC
    """
    # ----------------------
    # 1. PDE残差损失（物理约束）
    # ----------------------
    residual = compute_pde_residual(model, x_pde, t_pde, alpha)
    loss_pde = torch.mean(residual**2)
    
    # ----------------------
    # 2. 边界条件损失
    # ----------------------
    # 左边界: u(0, t) = 0
    x_left = torch.zeros_like(t_bc)
    u_left = model(x_left, t_bc)
    
    # 右边界: u(1, t) = 0
    x_right = torch.ones_like(t_bc)
    u_right = model(x_right, t_bc)
    
    loss_bc = torch.mean(u_left**2) + torch.mean(u_right**2)
    
    # ----------------------
    # 3. 初始条件损失
    # ----------------------
    # u(x, 0) = sin(πx)
    t_zero = torch.zeros_like(x_ic)
    u_ic_pred = model(x_ic, t_zero)
    u_ic_exact = torch.sin(np.pi * x_ic)
    loss_ic = torch.mean((u_ic_pred - u_ic_exact)**2)
    
    return loss_pde, loss_bc, loss_ic

# ============================================================
# 6. 训练配置
# ============================================================
# 网络结构: 输入2 -> 隐藏层64x4 -> 输出1
network_layers = [2, 64, 64, 64, 64, 1]

# 创建模型
model = PINN(network_layers).to(device)

# 优化器
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 学习率调度器（每2000步衰减）
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)

# 采样点数量
N_pde = 10000   # 域内配点
N_bc = 500      # 边界配点
N_ic = 500      # 初始条件配点

# 训练参数
epochs = 10000
print_every = 1000

# 损失权重
lambda_bc = 10.0
lambda_ic = 10.0

print(f"\n[网络结构]")
print(f"层结构: {network_layers}")
print(f"总参数量: {sum(p.numel() for p in model.parameters())}")

print(f"\n[训练配置]")
print(f"学习率: {learning_rate}")
print(f"训练轮数: {epochs}")
print(f"PDE配点: {N_pde}, 边界配点: {N_bc}, 初条配点: {N_ic}")
print(f"损失权重 - λ_bc: {lambda_bc}, λ_ic: {lambda_ic}")

# ============================================================
# 7. 训练循环
# ============================================================
print(f"\n{'='*60}")
print("开始训练...")
print(f"{'='*60}")

# 记录训练历史
history = {
    'loss_total': [],
    'loss_pde': [],
    'loss_bc': [],
    'loss_ic': []
}

start_time = time.time()

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    
    # ----------------------
    # 采样训练点（每个epoch重新采样）
    # ----------------------
    # 域内随机采样
    x_pde = torch.rand(N_pde, 1, device=device) * (x_max - x_min) + x_min
    t_pde = torch.rand(N_pde, 1, device=device) * (t_max - t_min) + t_min
    
    # 边界采样
    t_bc = torch.rand(N_bc, 1, device=device) * (t_max - t_min) + t_min
    
    # 初始条件采样
    x_ic = torch.rand(N_ic, 1, device=device) * (x_max - x_min) + x_min
    
    # ----------------------
    # 计算损失
    # ----------------------
    loss_pde, loss_bc, loss_ic = compute_losses(
        model, x_pde, t_pde, x_bc=None, t_bc=t_bc, x_ic=x_ic, alpha=alpha
    )
    
    # 总损失
    loss_total = loss_pde + lambda_bc * loss_bc + lambda_ic * loss_ic
    
    # ----------------------
    # 反向传播和优化
    # ----------------------
    loss_total.backward()
    optimizer.step()
    scheduler.step()
    
    # ----------------------
    # 记录历史
    # ----------------------
    history['loss_total'].append(loss_total.item())
    history['loss_pde'].append(loss_pde.item())
    history['loss_bc'].append(loss_bc.item())
    history['loss_ic'].append(loss_ic.item())
    
    # ----------------------
    # 打印进度
    # ----------------------
    if (epoch + 1) % print_every == 0:
        elapsed = time.time() - start_time
        print(f"Epoch [{epoch+1:5d}/{epochs}] | "
              f"Loss: {loss_total.item():.2e} | "
              f"PDE: {loss_pde.item():.2e} | "
              f"BC: {loss_bc.item():.2e} | "
              f"IC: {loss_ic.item():.2e} | "
              f"Time: {elapsed:.1f}s")

total_time = time.time() - start_time
print(f"\n训练完成! 总用时: {total_time:.1f}秒")

# ============================================================
# 8. 结果可视化
# ============================================================
print(f"\n{'='*60}")
print("生成可视化结果...")
print(f"{'='*60}")

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# 创建评估网格
n_x, n_t = 100, 100
x_eval = np.linspace(x_min, x_max, n_x)
t_eval = np.linspace(t_min, t_max, n_t)
X, T = np.meshgrid(x_eval, t_eval)

# 转换为张量
x_tensor = torch.tensor(X.flatten(), dtype=torch.float32).view(-1, 1).to(device)
t_tensor = torch.tensor(T.flatten(), dtype=torch.float32).view(-1, 1).to(device)

# 预测
model.eval()
with torch.no_grad():
    u_pred = model(x_tensor, t_tensor).cpu().numpy().reshape(n_t, n_x)

# 解析解
u_exact = analytical_solution(X, T, alpha)

# 计算误差
error = np.abs(u_pred - u_exact)
l2_error = np.sqrt(np.mean((u_pred - u_exact)**2))
max_error = np.max(error)

print(f"\n[误差分析]")
print(f"L2相对误差: {l2_error:.6e}")
print(f"最大绝对误差: {max_error:.6e}")

# ----------------------
# 创建图表
# ----------------------
fig = plt.figure(figsize=(16, 12))

# 1. 损失曲线
ax1 = fig.add_subplot(2, 3, 1)
epochs_arr = np.arange(1, epochs + 1)
ax1.semilogy(epochs_arr, history['loss_total'], 'b-', label='Total Loss', linewidth=1.5)
ax1.semilogy(epochs_arr, history['loss_pde'], 'r--', label='PDE Loss', linewidth=1)
ax1.semilogy(epochs_arr, history['loss_bc'], 'g--', label='BC Loss', linewidth=1)
ax1.semilogy(epochs_arr, history['loss_ic'], 'm--', label='IC Loss', linewidth=1)
ax1.set_xlabel('Epoch', fontweight='bold')
ax1.set_ylabel('Loss', fontweight='bold')
ax1.set_title('Training Loss Curve', fontweight='bold', fontsize=12)
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. PINN预测结果
ax2 = fig.add_subplot(2, 3, 2)
c2 = ax2.contourf(X, T, u_pred, levels=50, cmap='jet')
plt.colorbar(c2, ax=ax2, label='u')
ax2.set_xlabel('x', fontweight='bold')
ax2.set_ylabel('t', fontweight='bold')
ax2.set_title('PINN Prediction', fontweight='bold', fontsize=12)

# 3. 解析解
ax3 = fig.add_subplot(2, 3, 3)
c3 = ax3.contourf(X, T, u_exact, levels=50, cmap='jet')
plt.colorbar(c3, ax=ax3, label='u')
ax3.set_xlabel('x', fontweight='bold')
ax3.set_ylabel('t', fontweight='bold')
ax3.set_title('Analytical Solution', fontweight='bold', fontsize=12)

# 4. 绝对误差
ax4 = fig.add_subplot(2, 3, 4)
c4 = ax4.contourf(X, T, error, levels=50, cmap='hot')
plt.colorbar(c4, ax=ax4, label='|Error|')
ax4.set_xlabel('x', fontweight='bold')
ax4.set_ylabel('t', fontweight='bold')
ax4.set_title(f'Absolute Error (Max: {max_error:.2e})', fontweight='bold', fontsize=12)

# 5. 特定时刻比较
ax5 = fig.add_subplot(2, 3, 5)
t_indices = [0, 25, 50, 75, 99]  # t = 0, 0.25, 0.5, 0.75, 1.0
colors = ['b', 'g', 'r', 'c', 'm']
for idx, t_idx in enumerate(t_indices):
    t_val = t_eval[t_idx]
    ax5.plot(x_eval, u_pred[t_idx, :], colors[idx] + '-', 
             label=f'PINN t={t_val:.2f}', linewidth=2)
    ax5.plot(x_eval, u_exact[t_idx, :], colors[idx] + 'o', 
             markersize=4, markevery=10)
ax5.set_xlabel('x', fontweight='bold')
ax5.set_ylabel('u', fontweight='bold')
ax5.set_title('Solution at Different Times\n(Lines: PINN, Dots: Exact)', 
              fontweight='bold', fontsize=12)
ax5.legend(loc='upper right', fontsize=8)
ax5.grid(True, alpha=0.3)

# 6. 3D表面图
ax6 = fig.add_subplot(2, 3, 6, projection='3d')
surf = ax6.plot_surface(X, T, u_pred, cmap='viridis', alpha=0.8)
ax6.set_xlabel('x', fontweight='bold')
ax6.set_ylabel('t', fontweight='bold')
ax6.set_zlabel('u', fontweight='bold')
ax6.set_title('PINN Solution (3D View)', fontweight='bold', fontsize=12)

plt.tight_layout()
plt.savefig('d:/VScode file/PINN/pinn_heat_equation_results.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\n结果已保存至: d:/VScode file/PINN/pinn_heat_equation_results.png")

# ============================================================
# 9. 模型保存
# ============================================================
model_save_path = 'd:/VScode file/PINN/pinn_heat_model.pth'
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'network_layers': network_layers,
    'alpha': alpha,
    'history': history
}, model_save_path)

print(f"模型已保存至: {model_save_path}")

# ============================================================
# 10. 总结
# ============================================================
print(f"\n{'='*60}")
print("训练总结")
print(f"{'='*60}")
print(f"问题: 1D热传导方程 ∂u/∂t = {alpha} * ∂²u/∂x²")
print(f"边界条件: u(0,t) = u(1,t) = 0")
print(f"初始条件: u(x,0) = sin(πx)")
print(f"网络结构: {network_layers}")
print(f"训练轮数: {epochs}")
print(f"最终损失: {history['loss_total'][-1]:.6e}")
print(f"L2误差: {l2_error:.6e}")
print(f"最大误差: {max_error:.6e}")
print(f"训练时间: {total_time:.1f}秒")
print(f"{'='*60}")
