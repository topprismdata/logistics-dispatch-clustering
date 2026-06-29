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
    print("🎯 第二阶段 (Step 2): 【独立单车路线排序】专项验证")
    print("   前提条件: 假设第一阶段车辆/车次分配已确定，仅针对单车内部客户求最优送货顺序")
    print("   理论参考: Amazon 2021 Last Mile Routing Challenge (Permission Denied)")
    print("==========================================================================\n")

    print("Step 1: 读取 全流程报表 (含真实装运单号/车次划分)...")
    df = pd.read_excel(report_excel)
    
    df['clean_cid'] = df['送达方编号'].astype(str).str.strip()
    df['clean_shipment'] = df['装运单号'].astype(str).str.strip()
    
    coords_dict = load_gps_coords()
    cust_to_h3 = {}
    for cid in df['clean_cid'].unique():
        if cid in coords_dict:
            lng, lat = coords_dict[cid]
            try: cust_to_h3[cid] = h3.latlng_to_cell(lat, lng, 8)
            except: pass

    # 提取真实历史独立车次 (Single Vehicle Route)
    vehicle_trips = []
    for shipment_id, group in df.groupby('clean_shipment'):
        cust_list = list(group['clean_cid'].unique())
        if 4 <= len(cust_list) <= 30: # 独立单车客户配送规模
            vehicle_trips.append(cust_list)

    print(f"  成功提取得到 {len(vehicle_trips)} 个独立单车车次（车辆分配已完全确定，仅待排序）。")

    # 训练 PPM 司机偏好模型
    zone_sequences = []
    for seq in vehicle_trips:
        z_seq = [cust_to_h3[c] for c in seq if cust_to_h3.get(c)]
        if len(z_seq) >= 2: zone_sequences.append(z_seq)

    ppm_model = SimplePPM(order=3)
    for z_seq in zone_sequences: ppm_model.add_sequence(z_seq)

    alpha, beta, gamma = 1.04, 3.8, 2.5
    
    print("\n--------------------------------------------------------------------------")
    print("Step 2: 对每个独立分配好的车次进行内部送货顺序优化...")
    print("--------------------------------------------------------------------------")
    
    sd_nn_list, sd_ppm_list = [], []
    sample_trips = vehicle_trips[:50]
    
    for trip_idx, assigned_customers in enumerate(sample_trips):
        start_cust = assigned_customers[0]
        
        # 1. 传统 Nearest Neighbor 最短路排序
        unvisited = set(assigned_customers[1:])
        curr = start_cust
        nn_seq = [curr]
        while unvisited:
            curr_h3 = cust_to_h3.get(curr)
            best_next, best_dist = None, 9999
            for nxt in unvisited:
                nxt_h3 = cust_to_h3.get(nxt)
                d = h3.grid_distance(curr_h3, nxt_h3) if (curr_h3 and nxt_h3) else 10
                if d < best_dist: best_dist, best_next = d, nxt
            if not best_next: best_next = list(unvisited)[0]
            nn_seq.append(best_next); unvisited.remove(best_next); curr = best_next
        sd_nn = calculate_sequence_deviation(assigned_customers, nn_seq)
        sd_nn_list.append(sd_nn)

        # 2. Amazon PPM 司机行为模仿独立排序
        unvisited = set(assigned_customers[1:])
        curr = start_cust
        ppm_seq = [curr]
        context = [cust_to_h3.get(curr, "stz")]
        while unvisited:
            best_next, best_score = None, -99999.0
            for nxt in unvisited:
                nxt_z = cust_to_h3.get(nxt)
                curr_z = cust_to_h3.get(curr)
                dist = h3.grid_distance(curr_z, nxt_z) if (curr_z and nxt_z) else 10
                spatial_cost = math.exp(-beta * (dist / 5.0))
                pred_z, prob = ppm_model.predict_next(context)
                ppm_prob = (prob ** alpha) if (pred_z and pred_z == nxt_z) else 0.02
                score = spatial_cost + gamma * math.log(ppm_prob + 1e-6)
                if score > best_score: best_score, best_next = score, nxt
            if not best_next: best_next = list(unvisited)[0]
            ppm_seq.append(best_next); unvisited.remove(best_next); curr = best_next
            if cust_to_h3.get(curr): context.append(cust_to_h3[curr])
        sd_ppm = calculate_sequence_deviation(assigned_customers, ppm_seq)
        sd_ppm_list.append(sd_ppm)

        if (trip_idx + 1) % 10 == 0:
            print(f" 独立车次 {trip_idx+1:2d} (分配客户数:{len(assigned_customers):2d}) | 传统 NN 排序 SD: {sd_nn:.4f} | Amazon PPM 模仿排序 SD: {sd_ppm:.4f}")

    print("\n==========================================================================")
    print("📊 独立单车路线排序 (Independent Vehicle Sequencing) 评估总结:")
    print(f"  • 传统 Nearest Neighbor 最短路  平均 SD: {np.mean(sd_nn_list):.4f}")
    print(f"  • Amazon PPM 司机行为模仿模型    平均 SD: {np.mean(sd_ppm_list):.4f}")
    print("==========================================================================")

if __name__ == "__main__":
    main()
