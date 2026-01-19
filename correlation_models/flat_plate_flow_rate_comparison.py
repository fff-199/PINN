"""
苯甲醚蒸发量对比计算：N2流量 3000 vs 4000 m³/h
基于平板边界层模型 (Flat Plate Boundary Layer)
"""
import numpy as np

# =========================================================================
# Geometry and input parameters 
# =========================================================================
L_oven_total = 10.0   # 烘箱总长 (m)
L_drying_zone = 8.0   # 有效干燥区长度 (m)
W_oven = 1.5          # 烘箱宽度 (m)
H_oven_total = 0.8    # 烘箱总高度 (m)

W_coated = 1.2        # 涂布宽度 (m)
Web_speed_m_min = 20  # 线速度 (m/min)
Web_speed = Web_speed_m_min / 60  # 转换为 m/s

Liquid_mass_initial_g = 360.0  # 初始溶剂质量 (g)

# =========================================================================
# Thermodynamics constants
# =========================================================================
M_sol = 108.14   # 苯甲醚相对分子质量 (g/mol)
R_gas = 8.314    # 气体常数 (J/(mol·K))

# Antoine方程常数
A_ant = 4.17726
B_ant = 1489.756
C_ant = -69.607

# =========================================================================
# Derived geometry
# =========================================================================
Area_flow = W_oven * H_oven_total  # 有效过流截面积 (m²)
N_surfaces = 2  # 双面蒸发
Area_drying_total = L_drying_zone * W_coated
Area_drying_effective = Area_drying_total * N_surfaces

# =========================================================================
# Discretization
# =========================================================================
POINTS = 100
DX = L_drying_zone / (POINTS - 1)
DT = DX / Web_speed

# =========================================================================
# Zone settings
# =========================================================================
ZONES = [
    {'name': 'Zone 1', 'temp_c': 40.0, 'Nv_n2': 1.8484e-05},
    {'name': 'Zone 2', 'temp_c': 60.0, 'Nv_n2': 1.9376e-05},
    {'name': 'Zone 3', 'temp_c': 80.0, 'Nv_n2': 2.0246e-05},
]

# =========================================================================
# Core calculation function
# =========================================================================
def calc_zone_properties(temp_c, nu_n2, V_air):
    """计算给定温度和风速下的蒸发特性参数"""
    temp_k = temp_c + 273.15
    
    # Antoine方程: 饱和蒸汽压
    log_p_bar = A_ant - B_ant / (temp_k + C_ant)
    p_sat_pa = (10 ** log_p_bar) * 1e5
    
    # 饱和浓度
    c_sat = (p_sat_pa * (M_sol / 1000)) / (R_gas * temp_k)
    
    # 气相扩散系数 (Fuller Method)
    d_ab = 0.76e-5 * ((temp_k / 298) ** 1.75)
    
    # 雷诺数 (平板流)
    L_char = L_drying_zone
    re_L = V_air * L_char / nu_n2
    
    # Schmidt数
    sc = nu_n2 / d_ab
    
    # Sherwood数 (层流/混合边界层)
    Re_crit = 5e5
    if re_L < Re_crit:
        sh = 0.664 * (re_L ** 0.5) * (sc ** (1/3))
    else:
        sh = (0.037 * (re_L ** 0.8) - 871) * (sc ** (1/3))
    
    # 传质系数
    hm = sh * d_ab / L_char
    
    # 蒸发通量
    flux_g_m2_s = hm * c_sat * 1000
    
    return {
        'temp_k': temp_k,
        'p_sat_pa': p_sat_pa,
        'c_sat': c_sat,
        'd_ab': d_ab,
        're': re_L,
        'sc': sc,
        'sh': sh,
        'hm': hm,
        'flux_g_m2_s': flux_g_m2_s,
    }

def simulate_evaporation(flow_rate_m3h):
    """
    模拟给定N2流量下的三温区蒸发过程
    返回各温区结果和总蒸发量
    """
    # 计算近表面气流速度
    V_air = (flow_rate_m3h / 3600) / Area_flow
    
    # 计算各温区的运动粘度
    zones_data = []
    for z in ZONES:
        temp_k = z['temp_c'] + 273.15
        rho_n2 = (101325 * 0.028) / (R_gas * temp_k)
        nu_n2 = z['Nv_n2'] / rho_n2
        zones_data.append({
            'name': z['name'],
            'temp_c': z['temp_c'],
            'nu_n2': nu_n2,
        })
    
    # 逐温区模拟
    results = []
    current_mass = Liquid_mass_initial_g
    
    for zd in zones_data:
        props = calc_zone_properties(zd['temp_c'], zd['nu_n2'], V_air)
        
        # 该温区蒸发量
        total_evap_rate = props['flux_g_m2_s'] * Area_drying_effective
        residence_time = L_drying_zone / Web_speed
        evaporated = total_evap_rate * residence_time
        
        end_mass = max(0, current_mass - evaporated)
        
        results.append({
            'name': zd['name'],
            'temp_c': zd['temp_c'],
            'start_mass': current_mass,
            'end_mass': end_mass,
            'evaporated': current_mass - end_mass,
            'removal_pct': (current_mass - end_mass) / current_mass * 100 if current_mass > 0 else 0,
            'flux': props['flux_g_m2_s'],
            'hm': props['hm'],
            're': props['re'],
            'sh': props['sh'],
            'V_air': V_air,
        })
        
        current_mass = end_mass
    
    total_evaporated = Liquid_mass_initial_g - current_mass
    total_removal_pct = total_evaporated / Liquid_mass_initial_g * 100
    
    return {
        'flow_rate': flow_rate_m3h,
        'V_air': V_air,
        'zones': results,
        'final_mass': current_mass,
        'total_evaporated': total_evaporated,
        'total_removal_pct': total_removal_pct,
    }

# =========================================================================
# 运行对比计算
# =========================================================================
flow_rates = [3000, 4000]  # m³/h
all_results = {}

for fr in flow_rates:
    all_results[fr] = simulate_evaporation(fr)

# =========================================================================
# 输出结果
# =========================================================================
print('=' * 80)
print('苯甲醚蒸发量对比计算：N₂流量 3000 vs 4000 m³/h')
print('=' * 80)
print(f"\n【模型配置】")
print(f"  物理模型: Flat Plate Boundary Layer (Re < 5e5 → 层流)")
print(f"  特征长度: {L_drying_zone} m")
print(f"  有效蒸发面积(双面): {Area_drying_effective:.2f} m²")
print(f"  初始溶剂质量: {Liquid_mass_initial_g:.0f} g")

for fr in flow_rates:
    res = all_results[fr]
    print(f"\n{'─' * 80}")
    print(f"【N₂流量: {fr} m³/h】")
    print(f"  气流速度: {res['V_air']:.4f} m/s")
    print(f"\n  温区详情:")
    print(f"  {'Zone':<10} {'Temp':<8} {'入口质量':<12} {'出口质量':<12} {'蒸发量':<12} {'去除率':<10} {'Re_L':<12}")
    print(f"  {'-'*76}")
    for z in res['zones']:
        print(f"  {z['name']:<10} {z['temp_c']:.0f}°C{'':<4} {z['start_mass']:.2f} g{'':<4} "
              f"{z['end_mass']:.2f} g{'':<4} {z['evaporated']:.2f} g{'':<4} "
              f"{z['removal_pct']:.2f}%{'':<4} {z['re']:.2e}")
    
    print(f"\n  【汇总】")
    print(f"    最终剩余质量: {res['final_mass']:.2f} g")
    print(f"    总蒸发量: {res['total_evaporated']:.2f} g")
    print(f"    总去除率: {res['total_removal_pct']:.2f}%")

# =========================================================================
# 对比分析
# =========================================================================
res_3000 = all_results[3000]
res_4000 = all_results[4000]

delta_evap = res_4000['total_evaporated'] - res_3000['total_evaporated']
delta_pct = (res_4000['total_evaporated'] / res_3000['total_evaporated'] - 1) * 100

print(f"\n{'═' * 80}")
print(f"【对比分析】")
print(f"{'═' * 80}")
print(f"\n  流量变化: 3000 → 4000 m³/h (+33.3%)")
print(f"  气流速度变化: {res_3000['V_air']:.4f} → {res_4000['V_air']:.4f} m/s")
print(f"\n  蒸发量对比:")
print(f"    3000 m³/h: 总蒸发 {res_3000['total_evaporated']:.2f} g (去除率 {res_3000['total_removal_pct']:.2f}%)")
print(f"    4000 m³/h: 总蒸发 {res_4000['total_evaporated']:.2f} g (去除率 {res_4000['total_removal_pct']:.2f}%)")
print(f"\n  ⬆ 蒸发量增加: {delta_evap:.2f} g (+{delta_pct:.1f}%)")

# 雷诺数变化分析
print(f"\n  雷诺数变化 (Zone 1):")
re_3000 = res_3000['zones'][0]['re']
re_4000 = res_4000['zones'][0]['re']
print(f"    3000 m³/h: Re_L = {re_3000:.2e}")
print(f"    4000 m³/h: Re_L = {re_4000:.2e}")
print(f"    临界Re: 5.0e+05")
if re_4000 < 5e5:
    print(f"    → 4000 m³/h 仍处于层流区 (Re < 5e5)")
else:
    print(f"    → 4000 m³/h 已进入湍流区 (Re > 5e5)，传质效率显著提升!")

# 边界层物理解释
print(f"\n  【物理解释】")
print(f"    在层流区，Sh ∝ Re^0.5，因此:")
print(f"    蒸发量增幅 ≈ (V₂/V₁)^0.5 = (4000/3000)^0.5 = {(4000/3000)**0.5:.3f} ≈ +15.5%")
print(f"    实际计算增幅: +{delta_pct:.1f}% (符合理论预期)")

print(f"\n{'═' * 80}")
