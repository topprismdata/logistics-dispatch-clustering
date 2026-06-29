import warnings
warnings.filterwarnings("ignore")

import sys
import os
import math
import numpy as np
import pandas as pd
from shapely.geometry import Point
from collections import defaultdict
import networkx as nx
import community as community_louvain
import h3

# Ensure src/ is in python path
sys.path.insert(0, os.path.abspath("src"))

from taihe_dc.data import load_dataset
from taihe_dc.hard_mode import hard_mode_eval, PredictedClusters

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

def get_h3_ring(h3_idx):
    try:
        return h3.grid_disk(h3_idx, 1)
    except AttributeError:
        return h3.k_ring(h3_idx, 1)

def main():
    print("Loading coordinates and mapping H3 cells...")
    coords_dict = load_gps_coords()
    cust_to_h3 = {}
    for cid, coord in coords_dict.items():
        try:
            lat, lng = coord[1], coord[0]
            if -90 <= lat <= 90 and -180 <= lng <= 180:
                try:
                    cust_to_h3[cid] = h3.latlng_to_cell(lat, lng, 8)
                except AttributeError:
                    cust_to_h3[cid] = h3.geo_to_h3(lat, lng, 8)
        except: pass

    print("Loading full 5-month dataset into memory...")
    ds = load_dataset(excel_path)
    all_routes = list(ds.routes)
    
    # Pre-cache cleaned ID mappings
    route_cids = set()
    for r in all_routes:
        route_cids.update(r.customer_ids)
    clean_to_orig = defaultdict(list)
    for c in route_cids:
        clean_to_orig[clean_cid(c)].append(c)
        
    # Group routes by date
    routes_by_date = defaultdict(list)
    for r in all_routes:
        routes_by_date[r.date.isoformat()].append(r)
        
    unique_dates = sorted(list(routes_by_date.keys()))
    print(f"Total unique dates for LODO-CV: {len(unique_dates)} days")
    
    # Safety multipliers to test
    # We will run LODO-CV for Safety Multiplier = 0.70 (represents a good balance)
    mult = 0.70
    print(f"\nRunning 5-Month LODO-CV with Safety Multiplier = {mult:.2f}...")
    
    daily_aris = []
    daily_f1s = []
    daily_precisions = []
    daily_recalls = []
    
    for fold_idx, test_date in enumerate(unique_dates):
        train_routes = []
        for d in unique_dates:
            if d != test_date:
                train_routes.extend(routes_by_date[d])
        test_routes = routes_by_date[test_date]
        
        if not train_routes or not test_routes:
            continue
            
        # 1. Learn Tonnage PC limits from Train routes (95th percentile per tonnage)
        ton_to_pcs = defaultdict(list)
        for r in train_routes:
            if r.load_capacity_tons > 0.0 and r.route_pc_total > 0.0:
                ton_to_pcs[round(r.load_capacity_tons, 1)].append(r.route_pc_total)
                
        ton_caps = {}
        for t, pcs in ton_to_pcs.items():
            ton_caps[t] = np.percentile(pcs, 95)
            
        def get_local_vehicle_cap(tonnage):
            t = round(tonnage, 1)
            # Find closest tonnage in trained caps
            if not ton_caps:
                return 600.0 # fallback
            closest_t = min(ton_caps.keys(), key=lambda x: abs(x - t))
            return ton_caps[closest_t] * mult

        # 2. Build PMI Co-occurrence Graph from Train
        pair_count = defaultdict(int)
        cust_count = defaultdict(int)
        n_train = len(train_routes)
        
        for r in train_routes:
            cids = sorted(set(r.customer_ids))
            for c in cids:
                cust_count[c] += 1
            for i in range(len(cids)):
                for j in range(i + 1, len(cids)):
                    pair_count[(cids[i], cids[j])] += 1
                    
        G = nx.Graph()
        for c in cust_count: G.add_node(c)
        for (a, b), cnt in pair_count.items():
            if cnt < 2: continue
            p_a = cust_count[a] / n_train
            p_b = cust_count[b] / n_train
            p_ab = cnt / n_train
            pmi = math.log(p_ab / (p_a * p_b))
            G.add_edge(a, b, weight=pmi)
            
        # 3. Add H3 Neighbors edges
        h3_groups = defaultdict(list)
        for cid, h3_idx in cust_to_h3.items():
            h3_groups[h3_idx].append(cid)
            
        active_cells = list(h3_groups.keys())
        cell_neighbors = {cell: get_h3_ring(cell) for cell in active_cells}
        h3_weight = 0.5
        neighbor_discount = 0.25
        
        for cell_i in active_cells:
            cids_in_hex = h3_groups[cell_i]
            orig_cids = []
            for cc in cids_in_hex: orig_cids.extend(clean_to_orig[cc])
            
            for i in range(len(orig_cids)):
                for j in range(i + 1, len(orig_cids)):
                    u, v = orig_cids[i], orig_cids[j]
                    if not G.has_node(u): G.add_node(u)
                    if not G.has_node(v): G.add_node(v)
                    curr_w = G.get_edge_data(u, v, default={}).get('weight', 0.0)
                    G.add_edge(u, v, weight=curr_w + h3_weight)
                    
            neighbors_i = cell_neighbors.get(cell_i, [])
            for cell_j in neighbors_i:
                if cell_j == cell_i or cell_j not in h3_groups: continue
                cids_j = h3_groups[cell_j]
                orig_cids_j = []
                for cc in cids_j: orig_cids_j.extend(clean_to_orig[cc])
                for u in orig_cids:
                    for v in orig_cids_j:
                        if u == v: continue
                        if not G.has_node(u): G.add_node(u)
                        if not G.has_node(v): G.add_node(v)
                        curr_w = G.get_edge_data(u, v, default={}).get('weight', 0.0)
                        G.add_edge(u, v, weight=curr_w + h3_weight * neighbor_discount)

        # 4. Detect Communities
        partition = community_louvain.best_partition(G, resolution=1.0, random_state=42)
        
        # 5. Dynamic Tonnage Capacitated Bin Packing
        date_to_clusters = {}
        next_cluster_id = 0
        
        demands = {}
        cust_to_unload = {}
        for r in test_routes:
            for d in r.delivery_rows:
                cust_to_unload.setdefault(d.customer_id, d.unload_time)
            for c in r.customer_ids:
                demands[c] = r.pc_per_customer.get(c, 0.0)
                
        fleet_tons = [r.load_capacity_tons for r in test_routes if r.load_capacity_tons > 0.0]
        if not fleet_tons:
            fleet_tons = [6.0] * len(test_routes)
            
        vehicle_capacities = sorted([get_local_vehicle_cap(t) for t in fleet_tons], reverse=True)
        
        by_comm = defaultdict(list)
        for c in demands:
            comm_id = partition.get(c)
            if comm_id is None:
                comm_id = -(hash((test_date, c)) % (10**9))
            pc = demands[c]
            unload = cust_to_unload.get(c)
            by_comm[comm_id].append((c, pc, unload))
            
        for comm_id, custs in by_comm.items():
            solo_threshold = 260.0
            solo = [(c, pc, t) for c, pc, t in custs if pc > solo_threshold]
            group = [(c, pc, t) for c, pc, t in custs if pc <= solo_threshold]
            
            for c, _, _ in solo:
                date_to_clusters[c] = next_cluster_id
                next_cluster_id += 1
                
            if not group: continue
                
            with_time = sorted([(c, pc, t) for c, pc, t in group if t is not None], key=lambda x: x[2])
            no_time = [(c, pc, t) for c, pc, t in group if t is None]
            
            time_bins = []
            if with_time:
                cur = [with_time[0]]
                for c, pc, t in with_time[1:]:
                    last_t = cur[-1][2]
                    gap_h = (t - last_t).total_seconds() / 3600
                    if gap_h > 2.0:
                        time_bins.append(cur)
                        cur = [(c, pc, t)]
                    else:
                        cur.append((c, pc, t))
                time_bins.append(cur)
            if no_time:
                time_bins.append(no_time)
                
            for bin_items in time_bins:
                items_for_pack = [(c, pc) for c, pc, _ in bin_items]
                total_bin_pc = sum(pc for _, pc in items_for_pack)
                
                max_cap = vehicle_capacities[0] if vehicle_capacities else 600.0
                if len(items_for_pack) == 1 or total_bin_pc <= max_cap:
                    for c, _ in items_for_pack:
                        date_to_clusters[c] = next_cluster_id
                    next_cluster_id += 1
                else:
                    sorted_items = sorted(items_for_pack, key=lambda x: -x[1])
                    bins = []
                    for c, pc in sorted_items:
                        placed = False
                        for b in bins:
                            if sum(x[1] for x in b) + pc <= max_cap:
                                b.append((c, pc))
                                placed = True
                                break
                        if not placed:
                            bins.append([(c, pc)])
                            
                    for b in bins:
                        for c, _ in b:
                            date_to_clusters[c] = next_cluster_id
                        next_cluster_id += 1
                        
        test_preds = PredictedClusters(date_to_clusters={test_date: date_to_clusters})
        test_metrics = hard_mode_eval(test_routes, test_preds)
        
        daily_aris.append(test_metrics.ari)
        daily_f1s.append(test_metrics.partition_f1)
        daily_precisions.append(test_metrics.partition_precision)
        daily_recalls.append(test_metrics.partition_recall)
        
        if (fold_idx + 1) % 15 == 0 or (fold_idx + 1) == len(unique_dates):
            print(f"  Processed {fold_idx + 1:3d}/{len(unique_dates)} folds | Running Avg ARI: {np.mean(daily_aris):.5f} | Running Avg F1: {np.mean(daily_f1s)*100:.2f}%")

    print("\n================ LODO CROSS-VALIDATION SUMMARY (v4 Model) ================")
    print(f"Total days evaluated:             {len(daily_aris)}")
    print(f"Average Daily ARI:                {np.mean(daily_aris):.5f}")
    print(f"Average Daily F1-Score:           {np.mean(daily_f1s)*100:.2f}%")
    print(f"Average Daily Precision:          {np.mean(daily_precisions)*100:.2f}%")
    print(f"Average Daily Recall:             {np.mean(daily_recalls)*100:.2f}%")
    print(f"Daily ARI Standard Deviation:     {np.std(daily_aris):.5f}")
    print("=========================================================================")

if __name__ == "__main__":
    main()
