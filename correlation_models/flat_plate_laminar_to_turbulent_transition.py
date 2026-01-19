"""
苯甲醚蒸发量分析：从层流区到湍流区的流量优化
Flat Plate Boundary Layer Model - 层流/湍流过渡分析
"""
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.weight'] = 'bold'
plt.close('all')

# =========================================================================
# Geometry and input parameters 
# =========================================================================
L_drying_zone = 8.0   # 有效干燥区长度 (m)
W_oven = 1.5          # 烘箱宽度 (m)
H_oven_total = 0.8    # 烘箱总高度 (m)
W_coated = 1.2        # 涂布宽度 (m)
Web_speed = 20 / 60   # 线速度 (m/s)

Liquid_mass_initial_g = 360.0  # 初始溶剂质量 (g)

# =========================================================================
# Thermodynamics constants
# =========================================================================
M_sol = 108.14
R_gas = 8.314
A_ant, B_ant, C_ant = 4.17726, 1489.756, -69.607

# =========================================================================
# Geometry derived
# =========================================================================
Area_flow = W_oven * H_oven_total
N_surfaces = 2
Area_drying_effective = L_drying_zone * W_coated * N_surfaces

# =========================================================================
# Zone settings (取Zone 1的40°C作为分析基准)
# =========================================================================
ZONES = [
    {'name': 'Zone 1', 'temp_c': 40.0, 'Nv_n2': 1.8484e-05},
    {'name': 'Zone 2', 'temp_c': 60.0, 'Nv_n2': 1.9376e-05},
    {'name': 'Zone 3', 'temp_c': 80.0, 'Nv_n2': 2.0246e-05},
]

# =========================================================================
# 计算运动粘度
# =========================================================================
for z in ZONES:
    temp_k = z['temp_c'] + 273.15
    rho_n2 = (101325 * 0.028) / (R_gas * temp_k)
    z['nu_n2'] = z['Nv_n2'] / rho_n2

# =========================================================================
# Core calculation with laminar/turbulent transition
# =========================================================================
def calc_zone_properties(temp_c, nu_n2, V_air):
    """计算给定温度和风速下的蒸发特性参数"""
    temp_k = temp_c + 273.15
    
    # Antoine方程
    log_p_bar = A_ant - B_ant / (temp_k + C_ant)
    p_sat_pa = (10 ** log_p_bar) * 1e5
    c_sat = (p_sat_pa * (M_sol / 1000)) / (R_gas * temp_k)
    
    # 扩散系数
    d_ab = 0.76e-5 * ((temp_k / 298) ** 1.75)
    
    # 雷诺数
    L_char = L_drying_zone
    re_L = V_air * L_char / nu_n2
    sc = nu_n2 / d_ab
    
    # Sherwood数 - 层流/混合边界层判定
    Re_crit = 5e5
    if re_L < Re_crit:
        # 层流: Sh = 0.664 * Re^0.5 * Sc^(1/3)
        sh = 0.664 * (re_L ** 0.5) * (sc ** (1/3))
        flow_regime = 'Laminar'
    else:
        # 混合边界层: Sh = (0.037 * Re^0.8 - 871) * Sc^(1/3)
        sh = (0.037 * (re_L ** 0.8) - 871) * (sc ** (1/3))
        flow_regime = 'Turbulent'
    
    hm = sh * d_ab / L_char
    flux_g_m2_s = hm * c_sat * 1000
    
    return {
        'temp_k': temp_k, 'p_sat_pa': p_sat_pa, 'c_sat': c_sat,
        'd_ab': d_ab, 're': re_L, 'sc': sc, 'sh': sh, 'hm': hm,
        'flux_g_m2_s': flux_g_m2_s, 'flow_regime': flow_regime,
    }

def simulate_evaporation(flow_rate_m3h):
    """模拟给定N2流量下的三温区蒸发过程"""
    V_air = (flow_rate_m3h / 3600) / Area_flow
    
    results = []
    current_mass = Liquid_mass_initial_g
    
    for z in ZONES:
        props = calc_zone_properties(z['temp_c'], z['nu_n2'], V_air)
        
        total_evap_rate = props['flux_g_m2_s'] * Area_drying_effective
        residence_time = L_drying_zone / Web_speed
        evaporated = total_evap_rate * residence_time
        
        end_mass = max(0, current_mass - evaporated)
        
        results.append({
            'name': z['name'], 'temp_c': z['temp_c'],
            'start_mass': current_mass, 'end_mass': end_mass,
            'evaporated': current_mass - end_mass,
            'removal_pct': (current_mass - end_mass) / current_mass * 100 if current_mass > 0 else 0,
            'flux': props['flux_g_m2_s'], 'hm': props['hm'],
            're': props['re'], 'sh': props['sh'],
            'flow_regime': props['flow_regime'], 'V_air': V_air,
        })
        current_mass = end_mass
    
    return {
        'flow_rate': flow_rate_m3h, 'V_air': V_air, 'zones': results,
        'final_mass': current_mass,
        'total_evaporated': Liquid_mass_initial_g - current_mass,
        'total_removal_pct': (Liquid_mass_initial_g - current_mass) / Liquid_mass_initial_g * 100,
    }

# =========================================================================
# 计算临界流量（使Zone 1进入湍流区）
# =========================================================================
# Zone 1: Re_crit = 5e5, Re = V * L / nu
# V_crit = Re_crit * nu / L
nu_zone1 = ZONES[0]['nu_n2']
V_crit = 5e5 * nu_zone1 / L_drying_zone
Flow_crit = V_crit * Area_flow * 3600  # m³/h

print('=' * 80)
print('苯甲醚蒸发分析：层流 → 湍流过渡')
print('=' * 80)
print(f"\n【临界流量计算 (Zone 1 @ 40°C)】")
print(f"  运动粘度 ν = {nu_zone1:.3e} m²/s")
print(f"  临界雷诺数 Re_crit = 5.0×10⁵")
print(f"  临界气流速度 V_crit = {V_crit:.3f} m/s")
print(f"  ⭐ 临界流量 = {Flow_crit:.0f} m³/h")

# =========================================================================
# 多流量对比计算
# =========================================================================
flow_rates = [3000, 4000, 4500, 5000, 5500, 6000, 7000, 8000]
all_results = {fr: simulate_evaporation(fr) for fr in flow_rates}

print(f"\n{'─' * 80}")
print(f"【多流量蒸发效果对比】")
print(f"{'─' * 80}")
print(f"\n{'流量(m³/h)':<12} {'气流速度':<12} {'Zone1流态':<12} {'Re_L(Z1)':<14} {'总蒸发量':<12} {'去除率':<10}")
print(f"{'-' * 78}")

for fr in flow_rates:
    res = all_results[fr]
    z1 = res['zones'][0]
    regime_mark = '🔴层流' if z1['flow_regime'] == 'Laminar' else '🟢湍流'
    print(f"{fr:<12} {res['V_air']:.4f} m/s{'':<3} {regime_mark:<10} {z1['re']:.2e}{'':<4} "
          f"{res['total_evaporated']:.1f} g{'':<5} {res['total_removal_pct']:.1f}%")

# =========================================================================
# 详细对比：3000 vs 6000 m³/h
# =========================================================================
print(f"\n{'═' * 80}")
print(f"【详细对比：3000 m³/h (层流) vs 6000 m³/h (湍流)】")
print(f"{'═' * 80}")

for fr in [3000, 6000]:
    res = all_results[fr]
    print(f"\n【N₂流量: {fr} m³/h】")
    print(f"  气流速度: {res['V_air']:.4f} m/s")
    for z in res['zones']:
        print(f"  {z['name']} ({z['temp_c']:.0f}°C): "
              f"蒸发 {z['evaporated']:.1f}g, {z['flow_regime']}, Re={z['re']:.2e}, Sh={z['sh']:.0f}")
    print(f"  → 总蒸发: {res['total_evaporated']:.1f}g ({res['total_removal_pct']:.1f}%)")

# 效率提升
res_3000 = all_results[3000]
res_6000 = all_results[6000]
improvement = (res_6000['total_evaporated'] / res_3000['total_evaporated'] - 1) * 100

print(f"\n【效率提升分析】")
print(f"  流量翻倍 (3000→6000): 蒸发量从 {res_3000['total_evaporated']:.1f}g → {res_6000['total_evaporated']:.1f}g")
print(f"  ⬆ 蒸发效率提升: +{improvement:.1f}%")
print(f"\n  物理解释:")
print(f"    层流区: Sh ∝ Re^0.5 → 翻倍流量仅增加 ~41%")
print(f"    湍流区: Sh ∝ Re^0.8 → 非线性增强效应")
print(f"    进入湍流区后，边界层被有效破坏，传质阻力大幅降低")

# =========================================================================
# 可视化
# =========================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 图1: 蒸发量 vs 流量
ax1 = axes[0]
evap_values = [all_results[fr]['total_evaporated'] for fr in flow_rates]
removal_values = [all_results[fr]['total_removal_pct'] for fr in flow_rates]

colors = ['#FF6B6B' if all_results[fr]['zones'][0]['flow_regime'] == 'Laminar' else '#4ECDC4' 
          for fr in flow_rates]

bars = ax1.bar(range(len(flow_rates)), evap_values, color=colors, edgecolor='black', linewidth=1.5)

# 添加数值标签
for i, (bar, val, pct) in enumerate(zip(bars, evap_values, removal_values)):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3, 
             f'{val:.0f}g\n({pct:.0f}%)', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax1.axvline(x=2.5, color='gray', linestyle='--', linewidth=2, label=f'Critical Flow (~{Flow_crit:.0f} m3/h)')
ax1.set_xticks(range(len(flow_rates)))
ax1.set_xticklabels([str(fr) for fr in flow_rates])
ax1.set_xlabel('N2 Flow Rate (m3/h)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Total Evaporated Mass (g)', fontsize=12, fontweight='bold')
ax1.set_title('Evaporation vs N2 Flow Rate\n(Red=Laminar, Green=Turbulent)', fontsize=13, fontweight='bold')
ax1.set_ylim(0, max(evap_values) * 1.25)
ax1.legend(loc='upper left')
ax1.grid(axis='y', alpha=0.3)

# 图2: Re数和Sh数变化
ax2 = axes[1]
re_values = [all_results[fr]['zones'][0]['re'] / 1e5 for fr in flow_rates]
sh_values = [all_results[fr]['zones'][0]['sh'] for fr in flow_rates]

ax2.plot(flow_rates, re_values, 'o-', color='#7E57C2', linewidth=2.5, markersize=8, label='Re_L / 1e5')
ax2.axhline(y=5, color='red', linestyle='--', linewidth=2, label='Critical Re = 5e5')

ax2_twin = ax2.twinx()
ax2_twin.plot(flow_rates, sh_values, 's-', color='#26A69A', linewidth=2.5, markersize=8, label='Sherwood No.')

ax2.set_xlabel('N2 Flow Rate (m3/h)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Reynolds Number (x1e5)', fontsize=12, fontweight='bold', color='#7E57C2')
ax2_twin.set_ylabel('Sherwood Number', fontsize=12, fontweight='bold', color='#26A69A')
ax2.set_title('Dimensionless Numbers vs Flow Rate', fontsize=13, fontweight='bold')

ax2.legend(loc='upper left')
ax2_twin.legend(loc='upper right')
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('d:/VScode file/PINN/evaporation_turbulent_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\n{'═' * 80}")
print(f"✅ 分析完成！图表已保存至 evaporation_turbulent_analysis.png")
print(f"{'═' * 80}")
