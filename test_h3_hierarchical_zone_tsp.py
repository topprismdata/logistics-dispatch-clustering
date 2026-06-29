import warnings
warnings.filterwarnings("ignore")

import sys
import os
import math
import numpy as np
import pandas as pd
from collections import defaultdict
import h3

sys.path.insert(0, os.path.abspath("src"))
from taihe_dc.ppm_mine import SimplePPM

report_excel = '/Users/ghb/Downloads/全流程报表2026.1.1-5.31.xlsx'
coords_path = '/Users/ghb/Downloads/经纬度.csv'

def load_gps_coords():
    df_coords = pd.read_csv(coords_path)
    df_coords['clean_code'] = pd.to_numeric(df_coords['code'], errors='coerce').fillna(-2).astype(int)
    coords_dict = {}
    for idx, row in df_coords.iterrows():
        code = str(int(row['clean_code']))
        coords_dict[code] = (float(row['lng']), float(row['lat']))
    return coords_dict

def calculate_sequence_deviation(true_seq, pred_seq):
    if len(true_seq) <= 1 or len(pred_seq) <= 1: return 0.0
    pos_true = {node: i for i, node in enumerate(true_seq)}
    pos_pred = {node: i for i, node in enumerate(pred_seq)}
    common_nodes = set(true_seq) & set(pred_seq)
    if len(common_nodes) <= 1: return 0.0
    n = len(common_nodes)
    diff_sum = 0
    for u in common_nodes:
        for v in common_nodes:
            if u != v:
                true_dir = 1 if pos_true[u] < pos_true[v] else -1
                pred_dir = 1 if pos_pred[u] < pos_pred[v] else -1
                if true_dir != pred_dir: diff_sum += 1
    max_diffs = n * (n - 1)
    return diff_sum / max_diffs if max_diffs > 0 else 0.0

def main():
    print("==========================================================================")
    print("🎯 完整 Amazon 架构验证: H3 Zone 二级分层排序 (Hierarchical Zone TSP)")
    print("   Stage 1: H3 Zone 跨区跳转 PPM 序列预测 (Zone-level PPM Sequence)")
    print("   Stage 2: Zone 内部客户微观 TSP 路线规划 (Intra-Zone Micro TSP)")
    print("==========================================================================\n")

    df = pd.read_excel(report_excel)
    df['clean_cid'] = df['送达方编号'].astype(str).str.strip()
    df['clean_shipment'] = df['装运单号'].astype(str).str.strip()
    
    coords_dict = load_gps_coords()
    
    # 采用 H3 Resolution 7 (区域级 Zone，约 5 km² 覆盖范围) 作为真正的 Amazon Zone 抽象
    cust_to_zone = {}
    for cid in df['clean_cid'].unique():
        if cid in coords_dict:
            lng, lat = coords_dict[cid]
            try: cust_to_zone[cid] = h3.latlng_to_cell(lat, lng, 7)
            except: pass

    # 提取独立车次
    vehicle_trips = []
    for shipment_id, group in df.groupby('clean_shipment'):
        cust_list = list(group['clean_cid'].unique())
        if 4 <= len(cust_list) <= 30:
            vehicle_trips.append(cust_list)

    print(f"1. 成功对齐 Amazon 理论: 采用 H3 Res-7 聚类构建网格 Zone。提取得到 {len(vehicle_trips)} 个车次。")

    # 训练 Zone-level PPM 模型
    zone_sequences = []
    for seq in vehicle_trips:
        z_seq = []
        for c in seq:
            z = cust_to_zone.get(c)
            if z and (not z_seq or z_seq[-1] != z):
                z_seq.append(z)
        if len(z_seq) >= 2:
            zone_sequences.append(z_seq)

    ppm_zone_model = SimplePPM(order=3)
    for z_seq in zone_sequences:
        ppm_zone_model.add_sequence(z_seq)
        
    print(f"2. 成功训练 Zone-level PPM 序列规划器，包含 {len(zone_sequences)} 条跨 Zone 轨迹。")

    print("\n--------------------------------------------------------------------------")
    print("评估 H3 Zone 二级分层排序策略 (Amazon 2-Level Hierarchical Routing)...")
    print("--------------------------------------------------------------------------")

    sd_hierarchical_list = []
    sample_trips = vehicle_trips[:40]

    for trip_idx, assigned_customers in enumerate(sample_trips):
        # 将该车次内的客户归类到各自的 H3 Zone
        zone_to_custs = defaultdict(list)
        for c in assigned_customers:
            z = cust_to_zone.get(c, "unknown_zone")
            zone_to_custs[z].append(c)
            
        # 1. 宏观阶段：预测 H3 Zone 的访问顺序
        start_zone = cust_to_zone.get(assigned_customers[0], list(zone_to_custs.keys())[0])
        unvisited_zones = set(zone_to_custs.keys()) - {start_zone}
        curr_z = start_zone
        ordered_zones = [curr_z]
        context = [curr_z]
        
        while unvisited_zones:
            best_z, best_score = None, -99999.0
            for nxt_z in unvisited_zones:
                if curr_z != "unknown_zone" and nxt_z != "unknown_zone":
                    try: dist = h3.grid_distance(curr_z, nxt_z)
                    except: dist = 5
                else: dist = 5
                spatial_cost = math.exp(-3.8 * (dist / 5.0))
                pred_z, prob = ppm_zone_model.predict_next(context)
                ppm_prob = (prob ** 1.04) if (pred_z and pred_z == nxt_z) else 0.02
                score = spatial_cost + 2.5 * math.log(ppm_prob + 1e-6)
                if score > best_score: best_score, best_z = score, nxt_z
            if not best_z: best_z = list(unvisited_zones)[0]
            ordered_zones.append(best_z); unvisited_zones.remove(best_z); curr_z = best_z
            context.append(curr_z)

        # 2. 微观阶段：在每个 Zone 内部按距离微观切分排序 (Intra-Zone Ordering)
        final_predicted_seq = []
        for z in ordered_zones:
            c_list = zone_to_custs[z]
            if len(c_list) == 1:
                final_predicted_seq.append(c_list[0])
            else:
                # Intra-Zone Nearest Neighbor
                curr_c = c_list[0]
                unv_c = set(c_list[1:])
                z_seq = [curr_c]
                while unv_c:
                    c1_coords = coords_dict.get(curr_c)
                    best_next_c, min_d = None, 99999.0
                    for nxt_c in unv_c:
                        c2_coords = coords_dict.get(nxt_c)
                        if c1_coords and c2_coords:
                            d = (c1_coords[0]-c2_coords[0])**2 + (c1_coords[1]-c2_coords[1])**2
                        else: d = 1.0
                        if d < min_d: min_d, best_next_c = d, nxt_c
                    if not best_next_c: best_next_c = list(unv_c)[0]
                    z_seq.append(best_next_c); unv_c.remove(best_next_c); curr_c = best_next_c
                final_predicted_seq.extend(z_seq)

        sd = calculate_sequence_deviation(assigned_customers, final_predicted_seq)
        sd_hierarchical_list.append(sd)

    print("\n==========================================================================")
    print("📊 完整 H3 Zone 二级分层排序 (Amazon Hierarchical Zone Routing) 评估结果:")
    print(f"  • H3 Res-7 Zone 分层预测平均 SD: {np.mean(sd_hierarchical_list):.4f} 🎯")
    print("==========================================================================")

if __name__ == "__main__":
    main()
