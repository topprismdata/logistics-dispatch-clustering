"""Trained stop position model for Amazon 2021.

Learns to predict each stop's relative position (0-1) in the route.
Sort stops by predicted position → predicted sequence.

This is NOT a Transformer, but it IS real ML (trained, generalizable).
Next step: upgrade to Pointer Network (Transformer decoder).

Features: zone_embedding (learned) + lat/lng
Output: predicted position (0=start, 1=end)
Loss: MSE
"""

import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def load_amazon(data_dir="data/amazon2021"):
    d = Path(data_dir)
    with open(d / "train_route_data.json") as f:
        route_data = json.load(f)
    with open(d / "train_actual_sequences.json") as f:
        actual_seq = json.load(f)
    with open(d / "eval_real_route_data.json") as f:
        eval_routes = json.load(f)
    with open(d / "eval_real_actual.json") as f:
        eval_actual = json.load(f)
    return route_data, actual_seq, eval_routes, eval_actual


def build_zone_vocab(route_data):
    zones = set()
    for rd in route_data.values():
        for stop in rd.get("stops", {}).values():
            z = stop.get("zone_id")
            if isinstance(z, str) and z and z != "nan":
                zones.add(z)
    zone2idx = {z: i + 1 for i, z in enumerate(sorted(zones))}  # 0 = unknown
    return zone2idx, len(zones) + 1


class RouteSequenceDataset(Dataset):
    def __init__(self, route_data, actual_seq, zone2idx, max_stops=200):
        self.samples = []  # (zone_idx, lat_norm, lng_norm, position_norm)
        for rid, seq_data in actual_seq.items():
            actual = seq_data.get("actual", [])
            n = len(actual)
            if n < 2 or n > max_stops:
                continue
            rd = route_data.get(rid, {})
            stops = rd.get("stops", {})
            for i, sid in enumerate(actual):
                stop = stops.get(sid, {})
                z = stop.get("zone_id")
                zi = zone2idx.get(z, 0) if isinstance(z, str) else 0
                lat = float(stop.get("lat") or 0)
                lng = float(stop.get("lng") or 0)
                # Normalize lat/lng (Seattle area roughly)
                lat_n = (lat - 47.0) / 1.0
                lng_n = (lng + 122.5) / 1.0
                pos = i / max(1, n - 1)  # 0 to 1
                self.samples.append((zi, lat_n, lng_n, pos))
        print(f"  Dataset: {len(self.samples):,} stop-position samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        zi, lat, lng, pos = self.samples[idx]
        return (
            torch.tensor(zi, dtype=torch.long),
            torch.tensor([lat, lng], dtype=torch.float32),
            torch.tensor(pos, dtype=torch.float32),
        )


class StopPositionModel(nn.Module):
    def __init__(self, n_zones, zone_dim=64, hidden=128):
        super().__init__()
        self.zone_emb = nn.Embedding(n_zones, zone_dim, padding_idx=0)
        self.mlp = nn.Sequential(
            nn.Linear(zone_dim + 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),  # output 0-1
        )

    def forward(self, zone_idx, coords):
        z = self.zone_emb(zone_idx)
        x = torch.cat([z, coords], dim=-1)
        return self.mlp(x).squeeze(-1)


def train_position_model(route_data, actual_seq, epochs=10, batch_size=512, lr=1e-3):
    zone2idx, n_zones = build_zone_vocab(route_data)
    print(f"Zones: {n_zones - 1}")

    dataset = RouteSequenceDataset(route_data, actual_seq, zone2idx)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = StopPositionModel(n_zones)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        n = 0
        for zone_idx, coords, pos in loader:
            opt.zero_grad()
            pred = model(zone_idx, coords)
            loss = F.mse_loss(pred, pos)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(pos)
            n += len(pos)
        if (epoch + 1) % 2 == 0:
            print(f"  epoch {epoch+1}/{epochs}: loss={total_loss/n:.6f}")

    return model, zone2idx


def predict_sequence(model, stops_dict, zone2idx):
    model.eval()
    stop_list = list(stops_dict.keys())
    if len(stop_list) < 2:
        return stop_list

    with torch.no_grad():
        zone_idxs = []
        coords = []
        for sid in stop_list:
            stop = stops_dict[sid]
            z = stop.get("zone_id")
            zi = zone2idx.get(z, 0) if isinstance(z, str) else 0
            lat = (float(stop.get("lat") or 0) - 47.0) / 1.0
            lng = (float(stop.get("lng") or 0) + 122.5) / 1.0
            zone_idxs.append(zi)
            coords.append([lat, lng])

        zi = torch.tensor(zone_idxs, dtype=torch.long)
        c = torch.tensor(coords, dtype=torch.float32)
        scores = model(zi, c).numpy()

    # Sort by predicted position score
    order = sorted(range(len(stop_list)), key=lambda i: scores[i])
    return [stop_list[i] for i in order]


def sd_metric(actual, predicted):
    n = len(actual)
    if n < 2: return 0
    pa = {s: i for i, s in enumerate(actual)}
    pp = {s: i for i, s in enumerate(predicted)}
    return sum(abs(pa[s] - pp.get(s, 0)) for s in actual if s in pp) / (n * (n - 1) / 2)


def run():
    print("Loading data...")
    route_data, actual_seq, eval_routes, eval_actual = load_amazon()

    print("\nTraining stop position model (10 epochs)...")
    model, zone2idx = train_position_model(route_data, actual_seq, epochs=10)

    print("\nEvaluating on eval set (first 200 routes)...")
    sds = []
    for rid in list(eval_routes.keys())[:200]:
        actual = eval_actual.get(rid, {}).get("actual", [])
        stops = eval_routes[rid].get("stops", {})
        pred = predict_sequence(model, stops, zone2idx)
        sds.append(sd_metric(actual, pred))

    mean_sd = sum(sds) / len(sds)
    s = sorted(sds)
    print(f"\n{'='*60}")
    print(f"  Trained Stop Position Model — Results")
    print(f"{'='*60}")
    print(f"  Routes: {len(sds)}")
    print(f"  SD mean={mean_sd:.4f}, median={s[len(s)//2]:.4f}")
    print(f"  SD p25={s[len(s)//4]:.4f}, p75={s[3*len(s)//4]:.4f}")
    print(f"\n  Reference: random≈0.67, top teams 0.025-0.037")
    return mean_sd


if __name__ == "__main__":
    run()