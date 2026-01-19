"""
三温区干燥曲线：最低限度湍流区 (5000 m³/h), 50/70/90°C
"""
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.weight'] = 'bold'
plt.close('all')

# =========================================================================
# 基础参数
# =========================================================================
L_drying_zone = 8.0
W_oven = 1.5
H_oven_total = 0.8
W_coated = 1.2
Web_speed = 20 / 60  # m/s (20 m/min)

Liquid_mass_initial_g = 360.0
Area_flow = W_oven * H_oven_total
N_surfaces = 2
Area_drying_effective = L_drying_zone * W_coated * N_surfaces

M_sol = 108.14
R_gas = 8.314
A_ant, B_ant, C_ant = 4.17726, 1489.756, -69.607

POINTS = 100
DX = L_drying_zone / (POINTS - 1)
DT = DX / Web_speed

# =========================================================================
# 流量设定：最低限度湍流区
# =========================================================================
Flow_rate = 5000  # m³/h
V_air = (Flow_rate / 3600) / Area_flow

# =========================================================================
# 三温区配置
# =========================================================================
ZONES = [
    {'name': 'Zone 1', 'temp_c': 50.0},
    {'name': 'Zone 2', 'temp_c': 70.0},
    {'name': 'Zone 3', 'temp_c': 90.0},
]

# =========================================================================
# 计算函数
# =========================================================================
def get_nu_n2(temp_c):
    temp_k = temp_c + 273.15
    mu = 1.7e-5 * (temp_k / 293) ** 0.7
    rho = (101325 * 0.028) / (R_gas * temp_k)
    return mu / rho

def calc_zone_properties(temp_c):
    temp_k = temp_c + 273.15
    
    # Antoine方程
    log_p_bar = A_ant - B_ant / (temp_k + C_ant)
    p_sat_pa = (10 ** log_p_bar) * 1e5
    c_sat = (p_sat_pa * (M_sol / 1000)) / (R_gas * temp_k)
    
    # 扩散系数
    d_ab = 0.76e-5 * ((temp_k / 298) ** 1.75)
    nu = get_nu_n2(temp_c)
    
    # 雷诺数
    re_L = V_air * L_drying_zone / nu
    sc = nu / d_ab
    
    # Sherwood数 (湍流)
    if re_L < 5e5:
        sh = 0.664 * (re_L ** 0.5) * (sc ** (1/3))
        regime = 'Laminar'
    else:
        sh = (0.037 * (re_L ** 0.8) - 871) * (sc ** (1/3))
        regime = 'Turbulent'
    
    hm = sh * d_ab / L_drying_zone
    flux = hm * c_sat * 1000
    
    return {
        'temp_c': temp_c, 'p_sat_kpa': p_sat_pa/1000, 'c_sat': c_sat*1000,
        'hm': hm, 'flux': flux, 're': re_L, 'sh': sh, 'regime': regime
    }

def simulate_zone(start_mass, temp_c, x_start):
    props = calc_zone_properties(temp_c)
    x_axis = np.linspace(x_start, x_start + L_drying_zone, POINTS)
    mass_remaining = np.zeros_like(x_axis)
    mass_remaining[0] = start_mass
    
    evap_rate = props['flux'] * Area_drying_effective
    
    for i in range(1, len(x_axis)):
        loss = evap_rate * DT
        mass_remaining[i] = max(0, mass_remaining[i-1] - loss)
    
    return {
        'x': x_axis,
        'mass': mass_remaining,
        'start': start_mass,
        'end': mass_remaining[-1],
        'props': props
    }

# =========================================================================
# 模拟三温区
# =========================================================================
zone1 = simulate_zone(Liquid_mass_initial_g, ZONES[0]['temp_c'], 0.0)
zone2 = simulate_zone(zone1['end'], ZONES[1]['temp_c'], L_drying_zone)
zone3 = simulate_zone(zone2['end'], ZONES[2]['temp_c'], L_drying_zone * 2)

zones = [zone1, zone2, zone3]

# =========================================================================
# 绘图
# =========================================================================
fig, ax = plt.subplots(figsize=(14, 7))

# 温区背景色
zone_colors = ['#E3F2FD', '#FFF3E0', '#FFEBEE']
zone_label_y = [Liquid_mass_initial_g * 1.05, Liquid_mass_initial_g * 1.05, Liquid_mass_initial_g * 0.85]
for i, z in enumerate(ZONES):
    start_x = i * L_drying_zone
    ax.axvspan(start_x, start_x + L_drying_zone, alpha=0.4, color=zone_colors[i])
    ax.text(start_x + L_drying_zone/2, zone_label_y[i], 
            f"Zone {i+1}\n{z['temp_c']:.0f}C", 
            ha='center', fontsize=12, fontweight='bold')

# 合并曲线数据
X_full = np.concatenate([z['x'] for z in zones])
M_full = np.concatenate([z['mass'] for z in zones])

# 绘制干燥曲线
ax.plot(X_full, M_full, color='#D32F2F', linewidth=3.5, label='Residual Solvent Mass')

# 温区边界标注
boundaries = [0, L_drying_zone, L_drying_zone*2, L_drying_zone*3]
masses = [Liquid_mass_initial_g, zone1['end'], zone2['end'], zone3['end']]

for i, (x, m) in enumerate(zip(boundaries, masses)):
    ax.plot(x, m, 'ko', markersize=10, markerfacecolor='white', markeredgewidth=2, zorder=6)
    if i > 0:
        pct = (1 - m / Liquid_mass_initial_g) * 100
        ax.annotate(f'{m:.1f}g\n({pct:.1f}%)', 
                   xy=(x, m), xytext=(x-0.3, m+20),
                   fontsize=10, fontweight='bold', ha='right',
                   arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

# 5%目标线
target = Liquid_mass_initial_g * 0.05
ax.axhline(target, color='green', linestyle='-.', linewidth=2, label=f'5% Target ({target:.0f}g)')

# 标题和标签
ax.set_title(f'Drying Kinetics: 50/70/90C @ {Flow_rate} m3/h (Turbulent)\n'
             f'Initial: {Liquid_mass_initial_g}g, Web Speed: 20 m/min', 
             fontsize=14, fontweight='bold')
ax.set_xlabel('Oven Position (m)', fontsize=13, fontweight='bold')
ax.set_ylabel('Residual Solvent Mass (g)', fontsize=13, fontweight='bold')
ax.set_xlim(-0.5, L_drying_zone*3 + 0.5)
ax.set_ylim(0, Liquid_mass_initial_g * 1.15)
ax.legend(loc='upper right', fontsize=11)
ax.grid(True, alpha=0.3)

# 结果汇总框
final_mass = zone3['end']
removal_pct = (1 - final_mass / Liquid_mass_initial_g) * 100
status = "PASS" if final_mass <= target else "FAIL"

info_text = (f"=== RESULTS ===\n"
             f"Flow: {Flow_rate} m3/h\n"
             f"Temps: 50/70/90 C\n"
             f"Final: {final_mass:.1f}g\n"
             f"Removal: {removal_pct:.1f}%\n"
             f"Status: {status}")

props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='black', linewidth=2)
ax.text(L_drying_zone*1.5, Liquid_mass_initial_g*0.50, info_text, 
        fontsize=11, fontweight='bold', verticalalignment='top', ha='center', 
        bbox=props, family='monospace')

plt.tight_layout()
plt.savefig('d:/VScode file/PINN/output_images/drying_curve_50_70_90.png', dpi=150, bbox_inches='tight')
plt.show()

# 控制台输出
print('=' * 70)
print('三温区干燥曲线 (50/70/90°C @ 5000 m³/h 湍流区)')
print('=' * 70)
for i, (z, zone) in enumerate(zip(ZONES, zones)):
    p = zone['props']
    print(f"\n{z['name']} ({z['temp_c']:.0f}°C):")
    print(f"  Re_L = {p['re']:.2e} ({p['regime']})")
    print(f"  蒸发通量: {p['flux']:.4f} g/(m²·s)")
    print(f"  入口: {zone['start']:.1f}g → 出口: {zone['end']:.1f}g")

print(f"\n最终剩余: {final_mass:.1f}g ({removal_pct:.1f}% 去除)")
print('=' * 70)
