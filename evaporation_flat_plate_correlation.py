import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap

# =========================================================================
# 全局绘图风格设置 (参考 Turbulent Flat Plate Correlation.py)
# =========================================================================
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.weight'] = 'bold'
plt.close('all')

# =========================================================================
# Geometry and input parameters 
# =========================================================================
L_oven_total = 10.0 # 烘箱总长 (m)
L_drying_zone = 8.0 # 有效干燥区长度 (m)
W_oven = 1.5  # 烘箱宽度 (m)
H_oven_total = 0.8 # 烘箱总高度 (m)
H_web_position = 0.4 # 涂布膜位置（距底部高度, m）

W_coated = 1.2 # 涂布宽度 (m)
Web_speed_m_min = 20 # 线速度 (m/min)
Web_speed = Web_speed_m_min / 60 # 转换为 m/s

Flow_rate_m3h = 3000 # 气体流量 (m^3/h)
Liquid_mass_initial_g = 360.0 # 初始溶剂质量 (g)

# =========================================================================
# Thermodynamics constants (热力学常数)
# =========================================================================
M_sol = 108.14  # 溶液相对分子质量 (g/mol) - 用于Antoine方程和理想气体定律
R_gas = 8.314   # 气体常数 (J/(mol·K))

# Antoine方程常数: log10(P_bar) = A - B/(T+C)
# 这些常数针对该特定溶液(OCH3类物质)的蒸汽压数据拟合而来
A_ant = 4.17726     # Antoine A常数
B_ant = 1489.756    # Antoine B常数
C_ant = -69.607     # Antoine C常数

# =========================================================================
# Derived flow geometry (派生的流动几何参数)
# =========================================================================
# 有效过流截面积: 气流通过的有效空间面积
Area_flow = W_oven * H_oven_total # (m²)

# 近表面气流速度: 实际作用于涂布膜表面的气流速度
V_air_effective = (Flow_rate_m3h / 3600) / Area_flow

# 水力直径: 虽然在平板模型中不再作为特征长度，但保留用于参考
Dh = 2 * W_oven * H_oven_total / (W_oven + H_oven_total)

# =========================================================================
# Discretization (数值离散参数)
# =========================================================================
POINTS = 100        # 沿干燥区离散的计算点数
DX = L_drying_zone / (POINTS - 1)  # 空间离散步长 (m)
DT = DX / Web_speed # 时间离散步长 (s) = 涂布膜通过一个空间步长所需的时间
Area_drying_total = L_drying_zone * W_coated  # 单面涂布面积 (m²)

# =========================================================================
# [修改] 双面蒸发设置
# =========================================================================
# N2气体从涂布膜上下两面都会带走蒸汽，因此有效蒸发面积应为2倍
N_surfaces = 2  # 涂布膜的蒸发面数(上下两面)
Area_drying_effective = Area_drying_total * N_surfaces  # 有效蒸发总面积 = 单面面积 × 2

# =========================================================================
# Zone settings (温区设置)
# =========================================================================
# 三个温区的配置：温度递增，从低温预热到高温快速干燥
ZONES = [
    {
        'name': 'Zone 1',           # 温区名称
        'temp_c': 40.0,             # 温区温度 (°C)
        'Nv_n2': 1.8484e-05         # 动态粘度 (Pa·s) - N2在该温度的粘度
    },
    {
        'name': 'Zone 2',
        'temp_c': 60.0,
        'Nv_n2': 1.9376e-05
    },
    {
        'name': 'Zone 3',
        'temp_c': 80.0,
        'Nv_n2': 2.0246e-05
    } 
]

# =========================================================================
# Calculate kinematic viscosity and thermal properties for each zone
# =========================================================================
for zone in ZONES:
    # ============ 温度转换 ============
    temp_k = zone['temp_c'] + 273.15
    
    # ============ 气体密度计算(理想气体定律) ============
    rho_n2 = (101325 * 0.028) / (R_gas * temp_k)
    
    # ============ 运动粘度计算 ============
    zone['nu_n2'] = zone['Nv_n2'] / rho_n2

# =========================================================================
# Core calculations
# =========================================================================

def calc_zone_properties(temp_c, nu_n2):
    """
    计算蒸发区的热质传递特性参数。
    
    [重大更新] 物理模型变更:
    由 "管流模型 (Pipe Flow)" 变更为 "平板边界层模型 (Flat Plate Boundary Layer)"。
    这更符合宽大烘箱中涂布膜实际的物理场景，能解决原模型高估蒸发量的问题。
    """
    # ============ 温度转换 ============
    temp_k = temp_c + 273.15

    # ============ Antoine方程：饱和蒸汽压计算 ============
    log_p_bar = A_ant - B_ant / (temp_k + C_ant)
    p_sat_pa = (10 ** log_p_bar) * 1e5  # 转换为Pa (1 bar = 1e5 Pa)

    # ============ 饱和浓度计算 ============
    c_sat = (p_sat_pa * (M_sol / 1000)) / (R_gas * temp_k)

    # ============ 气相扩散系数 ============
    # Fuller Method Calculation for Anisole in N2:
    # D_AB (298K) ≈ 0.76e-5 m^2/s
    d_ab = 0.76e-5 * ((temp_k / 298) ** 1.75)

    # ============ 特征长度与雷诺数 (Physics Update) ============
    # 旧模型: Re_Dh = V * Dh / nu (管流)
    # 新模型: Re_L = V * L / nu (平板流)
    # 不再使用水力直径 Dh，而是使用温区长度 L 作为特征尺度
    L_char = L_drying_zone # 8.0 m
    
    re_L = V_air_effective * L_char / nu_n2
    
    # Schmidt数
    sc = nu_n2 / d_ab
    
    # ============ Sherwood数 (Correlation Update) ============
    # 物理分析:
    # Zone 1-2 的 Re_L 约为 3e5，低于平板临界雷诺数 (5e5)。
    # 这意味着流动主要处于 层流 (Laminar) 或 过渡区。
    # 既然 COMSOL 结果显著低于全湍流公式，说明它捕捉到了层流/过渡区的影响。
    # 因此，我们应使用 "混合边界层 (Mixed Boundary Layer)" 关联式，
    # 或者如果不超过临界值，直接使用层流公式。
    
    Re_crit = 5e5
    
    if re_L < Re_crit:
        # 层流平均 Sherwood 数
        # Sh = 0.664 * Re^0.5 * Sc^(1/3)
        sh = 0.664 * (re_L ** 0.5) * (sc ** (1/3))
    else:
        # 混合边界层 (前段层流，后段湍流)
        # Sh = (0.037 * Re^0.8 - A) * Sc^(1/3)
        # A = 0.037*Re_crit^0.8 - 0.664*Re_crit^0.5 ≈ 871
        sh = (0.037 * (re_L ** 0.8) - 871) * (sc ** (1/3))

    # ============ 质量传递系数和蒸发通量 (Update) ============
    # hm = Sh * D_AB / L_char
    # 注意: 分母变为 L_char 而不是 Dh
    hm = sh * d_ab / L_char
    
    # 蒸发通量
    flux_g_m2_s = hm * c_sat * 1000

    return {
        'temp_k': temp_k,
        'p_sat_pa': p_sat_pa,
        'c_sat': c_sat,
        'd_ab': d_ab,
        'nu_n2': nu_n2,
        're': re_L,     # Updated to Re_L
        'sc': sc,
        'sh': sh,
        'hm': hm,
        'flux_g_m2_s': flux_g_m2_s,
    }

def simulate_zone(start_mass_g, temp_c, nu_n2, x_start_m, name):
    """
    模拟单个温区的干燥过程。
    """
    # 计算该温区的热质传递参数
    props = calc_zone_properties(temp_c, nu_n2)
    
    # 沿干燥区长度均匀离散POINTS个计算点
    x_axis = np.linspace(x_start_m, x_start_m + L_drying_zone, POINTS)

    # 初始化质量数组，存储沿路线的剩余质量
    mass_remaining = np.zeros_like(x_axis)
    mass_remaining[0] = start_mass_g  # 入口质量

    # ============ 总蒸发速率计算 (双面蒸发版本) ============
    total_evap_rate_g_s = props['flux_g_m2_s'] * Area_drying_effective

    # ============ 沿干燥区分步计算质量损失 ============
    for i in range(1, len(x_axis)):
        loss = total_evap_rate_g_s * DT
        mass_remaining[i] = mass_remaining[i - 1] - loss

    # 确保质量不会变为负数
    mass_remaining[mass_remaining < 0] = 0
    
    # 温区出口的剩余质量
    residue_g = mass_remaining[-1]
    
    # ============ 干燥速率计算 ============
    removal_rate_pct = ((start_mass_g - residue_g) / start_mass_g * 100) if start_mass_g > 0 else 0

    return {
        'name': name,
        'temp_c': temp_c,
        'props': props,
        'x': x_axis,                    # 沿干燥区的位置(m)
        'mass': mass_remaining,          # 沿路线的剩余质量(g)
        'start_mass_g': start_mass_g,   # 进入质量(g)
        'residue_g': residue_g,         # 出口剩余质量(g)
        'removal_rate_pct': removal_rate_pct,  # 干燥速率(%)
    }


# =========================================================================
# Zone-by-zone simulation
# =========================================================================
# Zone 1
zone1 = simulate_zone(Liquid_mass_initial_g, ZONES[0]['temp_c'], ZONES[0]['nu_n2'], 0.0, ZONES[0]['name'])

# Zone 2
zone2 = simulate_zone(zone1['residue_g'], ZONES[1]['temp_c'], ZONES[1]['nu_n2'], L_drying_zone, ZONES[1]['name'])

# Zone 3
zone3 = simulate_zone(zone2['residue_g'], ZONES[2]['temp_c'], ZONES[2]['nu_n2'], L_drying_zone * 2, ZONES[2]['name'])

zones = [zone1, zone2, zone3]

# ============ 单温区估算(仅用Zone 1参考) ============
Residence_time = L_drying_zone / Web_speed  # 停留时间(s)
Mass_Evaporated_Lumped = zone1['props']['flux_g_m2_s'] * Area_drying_effective * Residence_time

# ============ 总体干燥结果 ============
Residue_Final = zone3['residue_g']  # 最终出口质量(g)
Total_Removal_Rate = ((Liquid_mass_initial_g - Residue_Final) / Liquid_mass_initial_g) * 100

# ============ 干燥完全位置检测 ============
dry_indices = np.where(zone3['mass'] == 0)[0]  # 查找质量为0的所有点
dry_pos = zone3['x'][dry_indices[0]] if dry_indices.size > 0 else None

# =========================================================================
# Console summary
# =========================================================================
print('\n' + '=' * 70)
print('蒸发干燥模拟结果 (混合/层流平板模型 - COMSOL 对齐版)')
print('=' * 70)
print(f"\n【模型配置】")
print(f"  物理模型: Mixed/Laminar Flat Plate Correlation (Re < 5e5 -> Laminar)")
print(f"  特征长度: {L_drying_zone} m (温区长度)")
print(f"  蒸发面数: {N_surfaces} (上下两面)")
print(f"  有效蒸发面积(双面): {Area_drying_effective:.4f} m²")
print(f"\n【初始条件】")
print(f"  初始溶剂质量: {Liquid_mass_initial_g:.2f} g")
print(f"  目标残留率: 5% ({Liquid_mass_initial_g * 0.05:.2f} g)")
print(f"\n【温区逐级结果】")
print('-' * 70)
for z in zones:
    p = z['props']
    print(f"{z['name']} ({z['temp_c']:.0f}°C):")
    print(f"  入口质量: {z['start_mass_g']:.2f} g → 出口质量: {z['residue_g']:.2f} g")
    print(f"  去除效率: {z['removal_rate_pct']:.2f}% | 蒸发通量: {p['flux_g_m2_s']:.4f} g/(m²·s)")
    print(f"  质量传递系数: {p['hm']:.5f} m/s | 饱和浓度: {p['c_sat']:.4f} kg/m³")
    print(f"  (Re_L: {p['re']:.1e})")
print('-' * 70)
print(f"\n【最终干燥结果】")
print(f"  单温区估算蒸发量(Zone 1): {Mass_Evaporated_Lumped:.2f} g")
print(f"  最终剩余质量: {Residue_Final:.2f} g")
print(f"  总去除效率: {Total_Removal_Rate:.2f}%")
if Residue_Final <= Liquid_mass_initial_g * 0.05:
    print(f"  ✅ 干燥状态: 满足要求 (剩余量 ≤ 5%)")
else:
    print(f"  ❌ 干燥状态: 不满足要求 (剩余量 > 5%)")
if dry_pos is not None:
    print(f"  干燥完全位置: {dry_pos:.2f} m")
print('=' * 70 + '\n')

# =========================================================================
# Plotting - Styled
# =========================================================================

# 定义温区背景颜色 (浅黄 -> 浅橙 -> 浅红，表示温度升高)
zone_colors = ['#FFECB3', '#FFCC80', '#FF8A65']
zone_temps = [z['temp_c'] for z in zones]
zone_names = [z['name'] for z in zones]

# ============ 图1: 三温区干燥动力学曲线 ============
fig1, ax1 = plt.subplots(figsize=(14, 7))

# 绘制温区背景色块
for i in range(3):
    start_x = i * L_drying_zone
    rect = patches.Rectangle((start_x, 0), L_drying_zone, Liquid_mass_initial_g*1.1, 
                             facecolor=zone_colors[i], alpha=0.3, edgecolor='none')
    ax1.add_patch(rect)
    # 添加温区文字标签
    ax1.text(start_x + L_drying_zone/2, Liquid_mass_initial_g*1.02, f'Zone {i+1}\n{zone_temps[i]}°C', 
             ha='center', fontsize=12, fontweight='bold')

# 合并数据
X_full = np.concatenate((zone1['x'], zone2['x'], zone3['x']))
M_full = np.concatenate((zone1['mass'], zone2['mass'], zone3['mass']))

# 绘制主干燥曲线 (深红色粗线)
ax1.plot(X_full, M_full, color='#D32F2F', linewidth=3.5, label='Residual Solvent Mass', zorder=5)

# 在温区交界处添加关键数据点标注
zone_boundaries = [0, L_drying_zone, L_drying_zone*2, L_drying_zone*3]
mass_at_boundary = [Liquid_mass_initial_g] + [z['residue_g'] for z in zones]

for i, (x, m) in enumerate(zip(zone_boundaries, mass_at_boundary)):
    ax1.plot(x, m, 'ko', markersize=10, zorder=6, markerfacecolor='w', markeredgewidth=2) # 白色填充的黑色圆点
    # 添加文本说明
    if i == 0:
        continue # 起点不标
    ax1.annotate(f'{m:.1f} g\n({(1-m/Liquid_mass_initial_g)*100:.1f}%)', 
                 xy=(x, m), xytext=(x-0.5, m + 25),
                 fontsize=10, fontweight='bold', ha='right',
                 arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

# 绘制 5% 残留目标线
target_mass = Liquid_mass_initial_g * 0.05
ax1.axhline(target_mass, color='green', linestyle='-.', linewidth=2, label=f'5% Target ({target_mass:.0f} g)')

# 干燥完全位置标记
if dry_pos is not None:
    ax1.plot(dry_pos, 0, 'h', markersize=14, markerfacecolor='g', markeredgecolor='k', zorder=10)
    ax1.text(dry_pos, 15, f'Dry-out\n{dry_pos:.1f} m', color='g', fontweight='bold', ha='center')

# 设置标题和坐标轴
ax1.set_title(f'Drying Kinetics (Flat Plate Model) - 20 m/min\n'
              f'Initial: {Liquid_mass_initial_g}g', 
              fontsize=14, fontweight='bold')
ax1.set_xlabel('Oven Position (m)', fontsize=13, fontweight='bold')
ax1.set_ylabel('Residual Solvent Mass (g)', fontsize=13, fontweight='bold')
ax1.set_xlim(-0.5, L_drying_zone*3 + 1)
ax1.set_ylim(0, Liquid_mass_initial_g * 1.15)
ax1.legend(loc='upper right', fontsize=11)
ax1.grid(True, alpha=0.3)

# 结果汇总框
status = "PASS" if Residue_Final <= target_mass else "FAIL"
info_text = (f"═══ RESULTS ═══\n"
             f"Final Mass: {Residue_Final:.1f} g\n"
             f"Removal: {Total_Removal_Rate:.1f}%\n"
             f"Status: {status}")
props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='black', linewidth=2)
ax1.text(L_drying_zone*3-2, Liquid_mass_initial_g*0.6, info_text, fontsize=11, fontweight='bold',
         verticalalignment='top', ha='right', bbox=props, family='monospace')

plt.tight_layout()

# ============ 图2: 浓度场分布热图 (Vapor Concentration Field) ============
# 使用 contourf + jet 配色 (参考风格)
fig2, ax2 = plt.subplots(figsize=(16, 5))

# 网格设置
n_x = 300
n_y = 80
X = np.linspace(0, L_drying_zone * 3, n_x)  # 全长
Y = np.linspace(H_web_position, H_oven_total, n_y) # 高度方向 0.4 - 0.8
X_grid, Y_grid = np.meshgrid(X, Y)
concentration_field = np.zeros_like(X_grid)
boundary_layer_y = []

# 边界层厚度计算函数
def get_bl_thickness(x_pos, nu):
    if x_pos < 0.1: x_pos = 0.1
    re_x = V_air_effective * x_pos / nu
    delta = 0.37 * x_pos / (re_x ** 0.2)
    return min(delta, 0.3)

# 计算浓度场
# 计算最大饱和浓度用于归一化颜色
vmax = max(z['props']['c_sat'] for z in zones) * 1000

# 逐点计算
for j, x_curr in enumerate(X):
    # 确定当前温区
    zone_idx = min(int(x_curr // L_drying_zone), 2)
    nu = zones[zone_idx]['props']['nu_n2']
    c_sat = zones[zone_idx]['props']['c_sat'] * 1000
    
    # 当前温区内的相对x坐标
    x_local = x_curr % L_drying_zone
    delta = get_bl_thickness(x_local + 0.1, nu)
    boundary_layer_y.append(H_web_position + delta)
    
    for i, y_curr in enumerate(Y):
        dist = y_curr - H_web_position
        # 简单指数衰减模型模拟浓度分布
        concentration_field[i, j] = c_sat * np.exp(-3 * dist / delta) if dist < delta * 1.5 else 0

# 绘制等高线填充图
cf = ax2.contourf(X_grid, Y_grid, concentration_field, 50, cmap='jet', vmin=0, vmax=vmax)

# 添加颜色条
cbar = plt.colorbar(cf, ax=ax2, pad=0.02)
cbar.set_label('Concentration (g/m³)', fontsize=11, fontweight='bold')

# 绘制边界层曲线 (白色虚线)
ax2.plot(X, boundary_layer_y, 'w--', linewidth=2, label='Boundary Layer')

# 绘制涂布膜位置 (黑色粗线)
ax2.plot([0, L_drying_zone*3], [H_web_position, H_web_position], 'k-', linewidth=3)
ax2.text(L_drying_zone*1.5, H_web_position + 0.02, 'Web Surface', fontsize=12, fontweight='bold', ha='center')

# 温区分界线
ax2.axvline(L_drying_zone, color='white', linestyle='--', linewidth=1)
ax2.axvline(L_drying_zone*2, color='white', linestyle='--', linewidth=1)

# 坐标轴设置
ax2.set_xlabel('Total line length (m)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Height in Oven (m)', fontsize=12, fontweight='bold')
ax2.set_title('Vapor Concentration Field', fontsize=13, fontweight='bold')
ax2.set_ylim(H_web_position, H_oven_total)
ax2.set_xlim(0, L_drying_zone*3)

# 温区标签
for i, z in enumerate(zones):
    ax2.text(i*L_drying_zone + L_drying_zone/2, H_oven_total*0.9, 
             f"{z['name']} ({z['temp_c']}°C)", 
             color='white', fontsize=12, fontweight='bold', ha='center',
             bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))

plt.tight_layout()

# ============ 图3: 综合参数分析 (改为柱状图风格) ============
fig3, axes = plt.subplots(2, 2, figsize=(14, 10))

# 提取数据
fluxes = [z['props']['flux_g_m2_s'] for z in zones]
hms = [z['props']['hm'] for z in zones]
res = [z['props']['re'] for z in zones]
shs = [z['props']['sh'] for z in zones]
p_sats = [z['props']['p_sat_pa']/1000 for z in zones]

# [2,2,1] 蒸发通量 (Bar Chart)
ax = axes[0, 0]
bars = ax.bar(zone_names, fluxes, width=0.5, color=['#42A5F5', '#66BB6A', '#EF5350'], edgecolor='black', linewidth=1.5)
for bar, val in zip(bars, fluxes):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.4f}', 
            ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_ylabel('Evaporation Flux (g/m²·s)', fontsize=11, fontweight='bold')
ax.set_title('Evaporation Flux by Zone', fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# [2,2,2] 传质系数 (Bar Chart)
ax = axes[0, 1]
bars = ax.bar(zone_names, [h for h in hms], width=0.5, color=['#42A5F5', '#66BB6A', '#EF5350'], edgecolor='black', linewidth=1.5)
# Set Y limit to ensure labels fit (add 20% margin)
ax.set_ylim(0, max(hms) * 1.2)
for bar, val in zip(bars, hms):
    # Adjust offset relative to value size
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(hms)*0.02, f'{val:.5f}', 
            ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_ylabel('Mass Transfer Coeff (m/s)', fontsize=11, fontweight='bold')
ax.set_title('Mass Transfer Coefficient (Flat Plate)', fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# [2,2,3] 无量纲数 Re & Sh (Double Axis Bar)
ax = axes[1, 0]
x = np.arange(len(zone_names))
width = 0.35
bars1 = ax.bar(x - width/2, [r/1e5 for r in res], width, label='Re (×10⁵)', color='#7E57C2', edgecolor='black')
ax2_sub = ax.twinx()
bars2 = ax2_sub.bar(x + width/2, shs, width, label='Sh', color='#26A69A', edgecolor='black')

ax.set_xticks(x)
ax.set_xticklabels(zone_names)
# Update Y-label to reflect Re_L
ax.set_ylabel('Reynolds Number Re_L (x1e5)', fontsize=11, fontweight='bold', color='#7E57C2')
ax2_sub.set_ylabel('Sherwood Number', fontsize=11, fontweight='bold', color='#26A69A')
ax.set_title('Dimensionless Numbers (Plate Model)', fontsize=12, fontweight='bold')

for bar, val in zip(bars1, res):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{val/1e5:.2f}', 
            ha='center', va='bottom', fontsize=10, fontweight='bold')
for bar, val in zip(bars2, shs):
    ax2_sub.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{val:.0f}', 
            ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.legend(loc='upper left')
ax2_sub.legend(loc='upper right')

# [2,2,4] 饱和蒸汽压 (Bar Chart)
ax = axes[1, 1]
bars = ax.bar(zone_names, p_sats, width=0.5, color='#FFCA28', edgecolor='black', linewidth=1.5)
for bar, val in zip(bars, p_sats):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{val:.2f}', 
            ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_ylabel('Sat. Pressure (kPa)', fontsize=11, fontweight='bold')
ax.set_title('Saturation Vapor Pressure', fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

fig3.suptitle('Evaporation Parameters Summary (Flat Plate)', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()

# ============ 图4: 雷达图 (保留图表类型但适配字体) ============
fig4 = plt.figure(figsize=(10, 8))
ax = fig4.add_subplot(111, polar=True)

labels = np.array(['Flux', 'Hm', 'P_sat', 'C_sat'])
num_vars = len(labels)
raw_data = []
for z in zones:
    raw_data.append([
        z['props']['flux_g_m2_s'],
        z['props']['hm'],
        z['props']['p_sat_pa'] / 1000,
        z['props']['c_sat']
    ])
raw_data = np.array(raw_data)
max_values = raw_data.max(axis=0)
normalized_data = raw_data / max_values
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

colors = ['#1f77b4', '#ff7f0e', '#d62728']
widths = [2, 2, 3]

for i, z in enumerate(zones):
    values = normalized_data[i].tolist()
    values += values[:1]
    ax.plot(angles, values, color=colors[i], linewidth=widths[i], label=f"{z['name']}")
    ax.fill(angles, values, color=colors[i], alpha=0.1 + i*0.1)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, size=11, weight='bold')
plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["20%", "40%", "60%", "80%", "100%"], color="grey", size=10)
plt.title('Normalized Comparison', size=15, weight='bold', y=1.08)
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

plt.show()
