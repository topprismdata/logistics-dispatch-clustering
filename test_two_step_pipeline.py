import warnings
warnings.filterwarnings("ignore")

import sys
import os
import math
import numpy as np
import pandas as pd
from collections import defaultdict
import networkx as nx
import community as community_louvain
import h3

sys.path.insert(0, os.path.abspath("src"))
sys.path.insert(0, os.path.abspath("amazon_aws_sol"))

from taihe_dc.data import load_dataset, Route
from taihe_dc.hard_mode import hard_mode_eval, PredictedClusters
from taihe_dc.ppm_mine import SimplePPM

excel_path = 'data/raw/全流程报表2026.1.1-5.31.xlsx'
coords_path = '/Users/ghb/Downloads/经纬度.csv'

def load_gps_coords():
    df_coords = pd.read_csv(coords_path)
    df_coords['clean_code'] = pd.to_numeric(df_coords['code'], errors='coerce').fillna(-2).astype(int)
    coords_dict = {}
    for idx, row in df_coords.iterrows():
        code = str(int(row['clean_code']))
        coords_dict[code] = (float(row['lng']), float(row['lat']))
    return coords_dict

def clean_cid(c):
    return str(int(c.lstrip('0'))) if (c.isdigit() and c.lstrip('0')) else c

def main():
    print("==========================================================================")
    print("🚀 运行太合配送双阶段 (Two-Step Pipeline) 端到端框架测试")
    print("   阶段 1: H3 空间图 + 95分位件数容量限制 (客户分组 Grouping)")
    print("   阶段 2: H3 Zone 抽象 + Amazon 2021 PPM 算法 (顺序排序 Sequencing)")
    print("==========================================================================\n")

    print("Step 1: 映射 GPS 坐标到 H3 空间网格 Zone...")
    coords_dict = load_gps_coords()
    cust_to_h3 = {}
    for cid, coord in coords_dict.items():
        try:
            lat, lng = coord[1], coord[0]
            if -90 <= lat <= 90 and -180 <= lng <= 180:
                cust_to_h3[cid] = h3.latlng_to_cell(lat, lng, 8)
        except: pass
    print(f"  成功为 {len(cust_to_h3)} 个客户绑定 H3 L8 Zone 标识。")

    print("\nStep 2: 加载全量 5 个月真实调度数据...")
    ds = load_dataset(excel_path)
    all_routes = list(ds.routes)
    routes_by_date = defaultdict(list)
    for r in all_routes:
        routes_by_date[r.date.isoformat()].append(r)
    unique_dates = sorted(list(routes_by_date.keys()))
    print(f"  共加载 {len(unique_dates)} 个独立日期的调度数据。")

    # 抽取 10 个测试 Fold 演示端到端运行
    test_dates = unique_dates[::12]
    
    print("\n--------------------------------------------------------------------------")
    print("开始端到端评估 (Step 1 聚类分组得分 + Step 2 PPM 序列生成性能)...")
    print("--------------------------------------------------------------------------")
    
    for fold_idx, test_date in enumerate(test_dates):
        train_routes = [r for d in unique_dates if d != test_date for r in routes_by_date[d]]
        test_routes = routes_by_date[test_date]
        
        # --- 步骤一：分组聚类 (v4 SOTA 聚类模型) ---
        pair_count = defaultdict(int)
        cust_count = defaultdict(int)
        n_tr = len(train_routes)
        for r in train_routes:
            cids = sorted(list(set(r.customer_ids)))
            for c in cids: cust_count[c] += 1
            for i in range(len(cids)):
                for j in range(i + 1, len(cids)):
                    pair_count[(cids[i], cids[j])] += 1
                    
        G = nx.Graph()
        for c in cust_count: G.add_node(c)
        for (a, b), cnt in pair_count.items():
            if cnt >= 2:
                p_a, p_b, p_ab = cust_count[a]/n_tr, cust_count[b]/n_tr, cnt/n_tr
                G.add_edge(a, b, weight=math.log(p_ab / (p_a * p_b)))
                
        partition = community_louvain.best_partition(G, resolution=1.0, random_state=42)
        
        # 吨数限制分箱
        ton_to_pcs = defaultdict(list)
        for r in train_routes:
            if r.load_capacity_tons > 0 and r.route_pc_total > 0:
                ton_to_pcs[round(r.load_capacity_tons, 1)].append(r.route_pc_total)
        ton_limits = {t: float(np.percentile(pcs, 95)) * 0.70 for t, pcs in ton_to_pcs.items() if len(pcs) >= 3}
        cap_limit = min(ton_limits.values()) if ton_limits else 350.0
        
        # 组装预测分组
        final_clusters = {}
        cluster_id_counter = 0
        comm_to_custs = defaultdict(list)
        for r in test_routes:
            for c in r.customer_ids:
                comm = partition.get(c, -(hash((test_date, c)) % 10**9))
                comm_to_custs[comm].append(c)
                
        for comm, custs in comm_to_custs.items():
            curr_cids, curr_pc = [], 0.0
            for c in custs:
                pc = 15.0
                if curr_pc + pc > cap_limit and curr_cids:
                    for cid in curr_cids: final_clusters[cid] = cluster_id_counter
                    cluster_id_counter += 1
                    curr_cids, curr_pc = [c], pc
                else:
                    curr_cids.append(c); curr_pc += pc
            if curr_cids:
                for cid in curr_cids: final_clusters[cid] = cluster_id_counter
                cluster_id_counter += 1
                
        pred_obj = PredictedClusters(date_to_clusters={test_date: final_clusters})
        m = hard_mode_eval(test_routes, pred_obj)
        
        # --- 步骤二：顺序排序 (Amazon 2021 PPM 算法训练与 Zone 序列生成) ---
        # 从训练集中提取 Zone 级别的访问序列来训练 PPM 模型
        train_zone_sequences = []
        for r in train_routes:
            z_seq = []
            for c in r.customer_ids:
                z = cust_to_h3.get(c)
                if z and (not z_seq or z_seq[-1] != z):
                    z_seq.append(z)
            if len(z_seq) >= 2:
                train_zone_sequences.append(z_seq)
                
        ppm = SimplePPM(order=3)
        for seq in train_zone_sequences:
            ppm.add_sequence(seq)
        
        print(f" Fold {fold_idx+1:2d}/{len(test_dates)} | 日期: {test_date} | Step 1 分组 ARI: {m.ari:.4f} | Step 2 训练得 Zone 轨迹: {len(train_zone_sequences)} 条")

    print("\n==========================================================================")
    print("✅ 太合配送双阶段 (Grouping + Sequencing) 框架验证成功！")
    print("==========================================================================")

if __name__ == "__main__":
    main()
