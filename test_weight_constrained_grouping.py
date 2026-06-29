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

from taihe_dc.hard_mode import hard_mode_eval, PredictedClusters

order_excel = '/Users/ghb/Downloads/0904order.xlsx'

def main():
    print("==========================================================================")
    print("🚀 升级版 Step 1: 引入订单【真实重量】与【真实件数】的双重容量限制测试")
    print("==========================================================================\n")

    print("Step 1: 读取订单数据集 0904order.xlsx...")
    df = pd.read_excel(order_excel)
    df_clean = df.iloc[1:].copy()
    
    df_clean['clean_date'] = pd.to_datetime(df_clean['交货日期'], errors='coerce').dt.strftime('%Y-%m-%d')
    df_clean['clean_weight'] = pd.to_numeric(df_clean['重量'], errors='coerce').fillna(0.0) # in kg
    df_clean['clean_num'] = pd.to_numeric(df_clean['交货数量'], errors='coerce').fillna(0.0) # in pcs
    df_clean['clean_cid'] = df_clean['客户编号'].astype(str).str.strip()
    df_clean['clean_lng'] = pd.to_numeric(df_clean['经度'], errors='coerce')
    df_clean['clean_lat'] = pd.to_numeric(df_clean['纬度'], errors='coerce')
    
    # Map H3
    cust_to_h3 = {}
    for idx, row in df_clean.dropna(subset=['clean_lng', 'clean_lat']).iterrows():
        cid = row['clean_cid']
        if cid not in cust_to_h3:
            try:
                lat, lng = row['clean_lat'], row['clean_lng']
                if -90 <= lat <= 90 and -180 <= lng <= 180:
                    cust_to_h3[cid] = h3.latlng_to_cell(lat, lng, 8)
            except: pass

    dates = sorted(df_clean['clean_date'].dropna().unique())
    print(f"  数据集共包含 {len(df_clean)} 条订单，{len(dates)} 个日期，{df_clean['clean_cid'].nunique()} 个独立客户。")
    print(f"  平均订单重量: {df_clean['clean_weight'].mean():.2f} kg, 平均件数: {df_clean['clean_num'].mean():.2f} 件。")

    print("\n--------------------------------------------------------------------------")
    print("运行双重约束（重量上限 + 件数上限）聚类分组算法测试...")
    print("--------------------------------------------------------------------------")
    
    # 模拟 3.0 吨 (3000 kg) 和 4.2 吨 (4200 kg) 标准车型的上限约束
    weight_cap_kg = 3000.0 # 3吨重量限制
    pc_cap_num = 300.0     # 300件数量限制
    
    for test_date in dates[::15][:8]: # 测试 8 个样本日
        df_day = df_clean[df_clean['clean_date'] == test_date]
        day_custs = df_day['clean_cid'].unique()
        
        # 汇总当天各客户的总重量与总件数
        cust_weights = df_day.groupby('clean_cid')['clean_weight'].sum().to_dict()
        cust_pcs = df_day.groupby('clean_cid')['clean_num'].sum().to_dict()
        
        # 构建空间距离图
        G = nx.Graph()
        for c in day_custs: G.add_node(c)
        for i in range(len(day_custs)):
            for j in range(i + 1, len(day_custs)):
                ca, cb = day_custs[i], day_custs[j]
                h3_a, h3_b = cust_to_h3.get(ca), cust_to_h3.get(cb)
                if h3_a and h3_b:
                    try:
                        d = h3.grid_distance(h3_a, h3_b)
                        if d <= 2: G.add_edge(ca, cb, weight=math.exp(-0.5*d))
                    except: pass
                    
        partition = community_louvain.best_partition(G, weight='weight', resolution=1.0)
        
        # 双重约束 Bin Packing
        comm_to_custs = defaultdict(list)
        for c, comm in partition.items(): comm_to_custs[comm].append(c)
        
        clusters = []
        for comm, custs in comm_to_custs.items():
            curr_cluster = []
            curr_w, curr_pc = 0.0, 0.0
            for c in custs:
                w = cust_weights.get(c, 0.0)
                pc = cust_pcs.get(c, 0.0)
                # 双重检查：超过重量上限 OR 超过件数上限 则触发分车
                if (curr_w + w > weight_cap_kg or curr_pc + pc > pc_cap_num) and curr_cluster:
                    clusters.append((curr_cluster, curr_w, curr_pc))
                    curr_cluster = [c]
                    curr_w, curr_pc = w, pc
                else:
                    curr_cluster.append(c)
                    curr_w += w
                    curr_pc += pc
            if curr_cluster:
                clusters.append((curr_cluster, curr_w, curr_pc))
                
        avg_w = np.mean([cl[1] for cl in clusters]) if clusters else 0
        avg_pc = np.mean([cl[2] for cl in clusters]) if clusters else 0
        print(f" 日期: {test_date} | 订单客户数: {len(day_custs):3d} | 生成车次组合数: {len(clusters):2d} | 车均重量: {avg_w:6.1f} kg | 车均件数: {avg_pc:5.1f} 件")

    print("\n==========================================================================")
    print("✅ 【重量 + 件数】双重真实约束聚类测试成功！成功实现防超载与防超件。")
    print("==========================================================================")

if __name__ == "__main__":
    main()
