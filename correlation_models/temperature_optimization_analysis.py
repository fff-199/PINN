"""
温度优化分析：在最低限度湍流区下，三温区温度配置优化
分析如何调整温度以最大化蒸发效率
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
Web_speed = 20 / 60  # m/s

Liquid_mass_initial_g = 360.0
Area_flow = W_oven * H_oven_total
N_surfaces = 2
Area_drying_effective = L_drying_zone * W_coated * N_surfaces

M_sol = 108.14
R_gas = 8.314
A_ant, B_ant, C_ant = 4.17726, 1489.756, -69.607

# =========================================================================
# 临界流量计算 (刚进入湍流区)
# =========================================================================
# 取较低温度(40°C)的粘度作为参考
def get_nu_n2(temp_c):
    """计算N2运动粘度"""
    temp_k = temp_c + 273.15
    # 动态粘度近似 (Sutherland公式简化)
    mu = 1.7e-5 * (temp_k / 293) ** 0.7
    rho = (101325 * 0.028) / (R_gas * temp_k)
    return mu / rho

# 临界雷诺数对应的流量
nu_ref = get_nu_n2(40)  # 参考粘度
Re_crit = 5e5
V_crit = Re_crit * nu_ref / L_drying_zone
Flow_crit = V_crit * Area_flow * 3600

# 使用刚超过临界值的流量
Flow_turbulent = 5000  # m³/h (刚进入湍流区)
V_air = (Flow_turbulent / 3600) / Area_flow

print('=' * 80)
print('温度优化分析：最低限度湍流区下的三温区配置')
print('=' * 80)
print(f"\n【流量设定】")
print(f"  临界流量: {Flow_crit:.0f} m³/h")
print(f"  选用流量: {Flow_turbulent} m³/h (刚进入湍流区)")
print(f"  气流速度: {V_air:.4f} m/s")

# =========================================================================
# 核心计算函数
# =========================================================================
def calc_evaporation(temp_c, V_air):
    """计算给定温度和风速下的蒸发参数"""
    temp_k = temp_c + 273.15
    
    # Antoine方程
    log_p_bar = A_ant - B_ant / (temp_k + C_ant)
    p_sat_pa = (10 ** log_p_bar) * 1e5
    c_sat = (p_sat_pa * (M_sol / 1000)) / (R_gas * temp_k)
    
    # 扩散系数
    d_ab = 0.76e-5 * ((temp_k / 298) ** 1.75)
    
    # 运动粘度
    nu = get_nu_n2(temp_c)
    
    # 雷诺数
    re_L = V_air * L_drying_zone / nu
    sc = nu / d_ab
    
    # Sherwood数
    if re_L < 5e5:
        sh = 0.664 * (re_L ** 0.5) * (sc ** (1/3))
        regime = 'Laminar'
    else:
        sh = (0.037 * (re_L ** 0.8) - 871) * (sc ** (1/3))
        regime = 'Turbulent'
    
    hm = sh * d_ab / L_drying_zone
    flux = hm * c_sat * 1000  # g/(m²·s)
    
    # 单温区蒸发量
    residence_time = L_drying_zone / Web_speed
    evaporated = flux * Area_drying_effective * residence_time
    
    return {
        'temp_c': temp_c, 'temp_k': temp_k,
        'p_sat_kpa': p_sat_pa / 1000, 'c_sat': c_sat * 1000,  # g/m³
        'd_ab': d_ab, 'nu': nu, 're': re_L, 'sc': sc, 'sh': sh,
        'hm': hm, 'flux': flux, 'evaporated': evaporated, 'regime': regime
    }

def simulate_three_zones(temps, V_air):
    """模拟三温区蒸发"""
    results = []
    current_mass = Liquid_mass_initial_g
    
    for i, t in enumerate(temps):
        props = calc_evaporation(t, V_air)
        evap = min(props['evaporated'], current_mass)  # 不能超过剩余量
        end_mass = current_mass - evap
        
        results.append({
            'zone': i + 1, 'temp': t,
            'start': current_mass, 'end': end_mass,
            'evaporated': evap,
            'removal_pct': evap / Liquid_mass_initial_g * 100,
            'props': props
        })
        current_mass = end_mass
    
    total_evap = Liquid_mass_initial_g - current_mass
    return {
        'temps': temps,
        'zones': results,
        'final_mass': current_mass,
        'total_evaporated': total_evap,
        'total_removal_pct': total_evap / Liquid_mass_initial_g * 100
    }

# =========================================================================
# 温度对蒸发的影响分析
# =========================================================================
print(f"\n{'─' * 80}")
print(f"【温度对蒸发参数的影响】")
print(f"{'─' * 80}")

temps_range = np.arange(40, 121, 10)
print(f"\n{'温度(°C)':<10} {'饱和蒸汽压(kPa)':<18} {'饱和浓度(g/m³)':<18} {'蒸发通量(g/m²s)':<18} {'单区蒸发量(g)':<15}")
print('-' * 80)

temp_data = []
for t in temps_range:
    props = calc_evaporation(t, V_air)
    temp_data.append(props)
    print(f"{t:<10} {props['p_sat_kpa']:<18.3f} {props['c_sat']:<18.2f} {props['flux']:<18.4f} {props['evaporated']:<15.1f}")

# =========================================================================
# 不同温区配置对比
# =========================================================================
print(f"\n{'═' * 80}")
print(f"【不同温区配置对比】")
print(f"{'═' * 80}")

configurations = [
    ([40, 60, 80], "Original"),
    ([50, 70, 90], "T+10C"),
    ([60, 80, 100], "T+20C"),
    ([60, 90, 100], "Mid-boost"),
    ([40, 80, 100], "Wide-span"),
    ([70, 90, 110], "High-T"),
    ([80, 100, 120], "Max-T"),
    ([60, 80, 110], "Rear-boost"),
    ([50, 80, 110], "Balanced"),
]

results_all = []
for temps, name in configurations:
    result = simulate_three_zones(temps, V_air)
    result['name'] = name
    results_all.append(result)
    
print(f"\n{'配置名称':<20} {'Zone1':<8} {'Zone2':<8} {'Zone3':<8} {'总蒸发量(g)':<14} {'去除率(%)':<10}")
print('-' * 80)

for res in results_all:
    temps = res['temps']
    print(f"{res['name']:<20} {temps[0]:<8} {temps[1]:<8} {temps[2]:<8} "
          f"{res['total_evaporated']:<14.1f} {res['total_removal_pct']:<10.1f}")

# 找出最优配置
best = max(results_all, key=lambda x: x['total_removal_pct'])
print(f"\n⭐ 最优配置: {best['name']} ({best['temps']}°C)")
print(f"   总蒸发量: {best['total_evaporated']:.1f}g, 去除率: {best['total_removal_pct']:.1f}%")

# =========================================================================
# 详细对比：原配置 vs 最优配置
# =========================================================================
print(f"\n{'═' * 80}")
print(f"【详细对比】")
print(f"{'═' * 80}")

original = results_all[0]
print(f"\n原配置 {original['temps']}°C:")
for z in original['zones']:
    print(f"  Zone {z['zone']} ({z['temp']}°C): 蒸发 {z['evaporated']:.1f}g ({z['removal_pct']:.1f}%)")
print(f"  → 总去除率: {original['total_removal_pct']:.1f}%")

print(f"\n最优配置 {best['temps']}°C:")
for z in best['zones']:
    print(f"  Zone {z['zone']} ({z['temp']}°C): 蒸发 {z['evaporated']:.1f}g ({z['removal_pct']:.1f}%)")
print(f"  → 总去除率: {best['total_removal_pct']:.1f}%")

improvement = best['total_removal_pct'] - original['total_removal_pct']
print(f"\n⬆ 温度优化提升: +{improvement:.1f}%")

# =========================================================================
# 可视化
# =========================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 图1: 温度 vs 饱和蒸汽压和蒸发通量
ax1 = axes[0, 0]
temps_plot = [d['temp_c'] for d in temp_data]
p_sat_plot = [d['p_sat_kpa'] for d in temp_data]
flux_plot = [d['flux'] for d in temp_data]

ax1.plot(temps_plot, p_sat_plot, 'o-', color='#E74C3C', linewidth=2.5, markersize=8, label='P_sat (kPa)')
ax1_twin = ax1.twinx()
ax1_twin.plot(temps_plot, flux_plot, 's-', color='#3498DB', linewidth=2.5, markersize=8, label='Flux (g/m2s)')

ax1.set_xlabel('Temperature (C)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Saturation Pressure (kPa)', fontsize=12, fontweight='bold', color='#E74C3C')
ax1_twin.set_ylabel('Evaporation Flux (g/m2s)', fontsize=12, fontweight='bold', color='#3498DB')
ax1.set_title('Temperature Effect on Evaporation', fontsize=13, fontweight='bold')
ax1.legend(loc='upper left')
ax1_twin.legend(loc='upper right')
ax1.grid(alpha=0.3)

# 图2: 配置对比柱状图
ax2 = axes[0, 1]
config_names = [r['name'][:10] for r in results_all]
removal_rates = [r['total_removal_pct'] for r in results_all]
colors = ['#27AE60' if r == best else '#95A5A6' for r in results_all]

bars = ax2.bar(range(len(config_names)), removal_rates, color=colors, edgecolor='black', linewidth=1.5)
ax2.set_xticks(range(len(config_names)))
ax2.set_xticklabels(config_names, rotation=45, ha='right', fontsize=9)
ax2.set_ylabel('Total Removal Rate (%)', fontsize=12, fontweight='bold')
ax2.set_title('Configuration Comparison', fontsize=13, fontweight='bold')
ax2.set_ylim(0, 110)
for bar, val in zip(bars, removal_rates):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val:.0f}%', 
             ha='center', fontsize=9, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

# 图3: 温度敏感性 (指数增长)
ax3 = axes[1, 0]
evap_amounts = [d['evaporated'] for d in temp_data]
ax3.bar(temps_plot, evap_amounts, color='#9B59B6', edgecolor='black', linewidth=1, width=8)
ax3.set_xlabel('Zone Temperature (C)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Single Zone Evaporation (g)', fontsize=12, fontweight='bold')
ax3.set_title('Evaporation vs Temperature (Exponential Growth)', fontsize=13, fontweight='bold')
ax3.grid(axis='y', alpha=0.3)

# 添加指数增长说明
ax3.text(60, max(evap_amounts)*0.8, 'P_sat ~ exp(T)\nEvaporation increases\nexponentially with T', 
         fontsize=10, fontweight='bold', color='#9B59B6',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# 图4: 原配置 vs 最优配置
ax4 = axes[1, 1]
x = np.arange(3)
width = 0.35
orig_evap = [z['evaporated'] for z in original['zones']]
best_evap = [z['evaporated'] for z in best['zones']]

bars1 = ax4.bar(x - width/2, orig_evap, width, label=f"Original {original['temps']}", color='#E74C3C', edgecolor='black')
bars2 = ax4.bar(x + width/2, best_evap, width, label=f"Optimized {best['temps']}", color='#27AE60', edgecolor='black')

ax4.set_xticks(x)
ax4.set_xticklabels(['Zone 1', 'Zone 2', 'Zone 3'])
ax4.set_ylabel('Evaporated Mass (g)', fontsize=12, fontweight='bold')
ax4.set_title('Zone-by-Zone Comparison', fontsize=13, fontweight='bold')
ax4.legend()
ax4.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('d:/VScode file/PINN/output_images/temperature_optimization.png', dpi=150, bbox_inches='tight')
plt.show()

# =========================================================================
# 物理解释与建议
# =========================================================================
print(f"\n{'═' * 80}")
print(f"【物理解释与优化建议】")
print(f"{'═' * 80}")

print("""
1. 【温度对蒸发的影响机制】
   - 饱和蒸汽压 P_sat 随温度呈指数增长 (Antoine方程)
   - 每升高10°C，P_sat 约增加 1.5-2倍
   - 蒸发通量 J = hm × C_sat，C_sat ∝ P_sat
   
2. 【温度提升的边际效益】
   - 低温区(40-60°C): 蒸发量增幅较小
   - 高温区(80-120°C): 蒸发量增幅显著(指数效应)
   
3. 【优化策略建议】
   ✅ 优先提高后段温区(Zone 3)温度 → 指数效应更明显
   ✅ 中间温区适当提高 → 加速溶剂蒸发
   ✅ 前段温区可保持较低 → 避免表面结皮
   
4. 【实际限制考虑】
   ⚠️ 温度上限受材料耐热性限制
   ⚠️ 过高温度可能导致溶剂闪蒸、气泡等问题
   ⚠️ 需平衡能耗与效率
""")

print(f"{'═' * 80}")
print(f"✅ 分析完成！推荐配置: {best['temps']}°C (去除率 {best['total_removal_pct']:.1f}%)")
print(f"{'═' * 80}")
