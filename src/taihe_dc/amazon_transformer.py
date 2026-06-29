"""Pointer-style Transformer for Amazon 2021 route sequencing.

Key insight from diagnostics:
  - Simple position regression fails (SD=0.67) because stops scored independently
  - Need SELF-ATTENTION so each stop sees ALL other stops on the route
  - This captures context: "given these zones, where does stop X go?"

Architecture:
  1. Embed each stop: zone_embedding + [lat, lng] → d_model
  2. Transformer encoder (self-attention over all stops in route)
  3. Linear head → position score per stop
  4. Sort stops by score → predicted sequence

This is NOT full Pointer Network (no autoregressive decoder), but it IS
a Transformer with self-attention. The next upgrade would add autoregressive
decoding for exact Pointer Network behavior.
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
    return {z: i + 1 for i, z in enumerate(sorted(zones))}, len(zones) + 1


class RouteDataset(Dataset):
    """Each item = one route (variable length, padded in collate)."""
    def __init__(self, route_data, actual_seq, zone2idx, max_stops=150):
        self.routes = []
        for rid, seq_data in actual_seq.items():
            actual = seq_data.get("actual", [])
            n = len(actual)
            if n < 5 or n > max_stops:
                continue
            rd = route_data.get(rid, {})
            stops = rd.get("stops", {})
            zone_idxs = []
            coords = []
            positions = []
            for i, sid in enumerate(actual):
                stop = stops.get(sid, {})
                z = stop.get("zone_id")
                zi = zone2idx.get(z, 0) if isinstance(z, str) else 0
                lat = (float(stop.get("lat") or 0) - 47.0)
                lng = (float(stop.get("lng") or 0) + 122.5)
                zone_idxs.append(zi)
                coords.append([lat, lng])
                positions.append(i / max(1, n - 1))
            self.routes.append((zone_idxs, coords, positions))
        print(f"  Routes: {len(self.routes)}")

    def __len__(self):
        return len(self.routes)

    def __getitem__(self, idx):
        z, c, p = self.routes[idx]
        return (
            torch.tensor(z, dtype=torch.long),
            torch.tensor(c, dtype=torch.float32),
            torch.tensor(p, dtype=torch.float32),
        )


def collate_routes(batch):
    """Pad routes to same length within batch."""
    max_len = max(len(z) for z, _, _ in batch)
    B = len(batch)
    zone_idxs = torch.zeros(B, max_len, dtype=torch.long)
    coords = torch.zeros(B, max_len, 2, dtype=torch.float32)
    positions = torch.zeros(B, max_len, dtype=torch.float32)
    mask = torch.ones(B, max_len, dtype=torch.bool)  # True = padding
    for i, (z, c, p) in enumerate(batch):
        L = len(z)
        zone_idxs[i, :L] = z
        coords[i, :L] = c
        positions[i, :L] = p
        mask[i, :L] = False
    return zone_idxs, coords, positions, mask


class RouteTransformer(nn.Module):
    """Transformer encoder that predicts position score for each stop.

    Self-attention lets each stop see all other stops → context-dependent scoring.
    """
    def __init__(self, n_zones, zone_dim=64, d_model=128, n_heads=4, n_layers=2):
        super().__init__()
        self.zone_emb = nn.Embedding(n_zones, zone_dim, padding_idx=0)
        self.coord_proj = nn.Linear(2, d_model - zone_dim)
        self.d_model = d_model

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=256,
            dropout=0.1, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, zone_idxs, coords, mask):
        # Embed
        z = self.zone_emb(zone_idxs)  # (B, L, zone_dim)
        c = self.coord_proj(coords)   # (B, L, d_model - zone_dim)
        x = torch.cat([z, c], dim=-1)  # (B, L, d_model)

        # Self-attention (mask padding)
        x = self.encoder(x, src_key_padding_mask=mask)  # (B, L, d_model)

        # Position score
        scores = self.head(x).squeeze(-1)  # (B, L)
        scores = scores.masked_fill(mask, -1e9)  # padding → very low
        return torch.sigmoid(scores)  # (B, L) in [0, 1]


def train_transformer(route_data, actual_seq, epochs=5, batch_size=32, lr=1e-4):
    zone2idx, n_zones = build_zone_vocab(route_data)
    print(f"Zones: {n_zones - 1}")

    dataset = RouteDataset(route_data, actual_seq, zone2idx)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_routes)

    model = RouteTransformer(n_zones)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {total_params:,}")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        n_routes = 0
        for zone_idxs, coords, positions, mask in loader:
            opt.zero_grad()
            pred = model(zone_idxs, coords, mask)  # (B, L)
            # MSE loss on non-padded positions
            loss = F.mse_loss(pred[~mask], positions[~mask])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item() * (~mask).sum().item()
            n_routes += (~mask).sum().item()
        avg = total_loss / max(1, n_routes)
        print(f"  epoch {epoch+1}/{epochs}: loss={avg:.6f}")

    return model, zone2idx


def predict_and_eval(model, eval_routes, eval_actual, zone2idx, n_eval=200):
    model.eval()
    sds = []

    for rid in list(eval_routes.keys())[:n_eval]:
        actual = eval_actual.get(rid, {}).get("actual", [])
        stops = eval_routes[rid].get("stops", {})
        if len(actual) < 5:
            continue

        # Prepare input
        stop_list = list(stops.keys())
        zone_idxs = []
        coords = []
        for sid in stop_list:
            stop = stops[sid]
            z = stop.get("zone_id")
            zi = zone2idx.get(z, 0) if isinstance(z, str) else 0
            lat = float(stop.get("lat") or 0) - 47.0
            lng = float(stop.get("lng") or 0) + 122.5
            zone_idxs.append(zi)
            coords.append([lat, lng])

        with torch.no_grad():
            zi = torch.tensor([zone_idxs], dtype=torch.long)
            c = torch.tensor([coords], dtype=torch.float32)
            m = torch.zeros(1, len(stop_list), dtype=torch.bool)
            scores = model(zi, c, m).squeeze(0).numpy()

        order = sorted(range(len(stop_list)), key=lambda i: scores[i])
        predicted = [stop_list[i] for i in order]

        # SD
        n = len(actual)
        pa = {s: i for i, s in enumerate(actual)}
        pp = {s: i for i, s in enumerate(predicted)}
        diff = sum(abs(pa[s] - pp.get(s, 0)) for s in actual if s in pp)
        sd = diff / (n * (n - 1) / 2) if n > 1 else 0
        sds.append(sd)

    mean_sd = sum(sds) / len(sds) if sds else 0
    s = sorted(sds)
    median = s[len(s) // 2] if sds else 0
    return mean_sd, median, len(sds)


def run():
    print("Loading Amazon data...")
    route_data, actual_seq, eval_routes, eval_actual = load_amazon()

    print("\nTraining Route Transformer (5 epochs, batch 32)...")
    model, zone2idx = train_transformer(route_data, actual_seq, epochs=5, batch_size=32, lr=1e-4)

    print("\nEvaluating...")
    mean_sd, median_sd, n = predict_and_eval(model, eval_routes, eval_actual, zone2idx, n_eval=200)

    print(f"\n{'='*60}")
    print(f"  Route Transformer (Self-Attention) — Results")
    print(f"{'='*60}")
    print(f"  Routes: {n}")
    print(f"  SD mean={mean_sd:.4f}, median={median_sd:.4f}")
    print(f"\n  Reference: random≈0.67, top teams 0.025-0.037")
    print(f"  Previous (no attention): 0.6710")
    print(f"  This (Transformer):       {mean_sd:.4f}")


if __name__ == "__main__":
    run()