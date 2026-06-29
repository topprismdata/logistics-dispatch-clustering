"""B1 baseline: Pairwise Siamese Network.

This is the CORRECT formulation per NotebookLM:
  - Input: pair of customers (c1, c2)
  - Output: probability they are on the same route

Different from BC (which predicts plate ID):
  - BC: vehicle ID direct → fails on dynamic fleet
  - Pairwise: pair binary classification → learns generalizable similarity

Expected: significantly higher Pair Recall than BC baseline.
"""

from __future__ import annotations

import random
from itertools import combinations
from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from taihe_dc.data import Route, DispatchDataset
from taihe_dc.evaluator import Predictions, evaluate_predictions


def build_pair_training_data(
    routes: list[Route],
    neg_pos_ratio: float = 3.0,
    seed: int = 42,
) -> tuple[list[tuple[str, str, int]], set[str]]:
    """Build positive + negative pair samples from routes.

    Positive: customer pairs on the same route
    Negative: customer pairs on different routes (sampled)
    """
    rng = random.Random(seed)
    customers: set[str] = set()
    pos_pairs: set[tuple[str, str]] = set()
    for r in routes:
        cids = list(set(r.customer_ids))
        customers.update(cids)
        for i in range(len(cids)):
            for j in range(i + 1, len(cids)):
                a, b = sorted([cids[i], cids[j]])
                pos_pairs.add((a, b))

    # Build customer -> route_ids mapping for negative sampling
    cust_to_routes: dict[str, set[str]] = {}
    for r in routes:
        for c in r.customer_ids:
            cust_to_routes.setdefault(c, set()).add(r.route_id)

    all_cust = sorted(customers)
    n_pos = len(pos_pairs)
    n_neg = int(n_pos * neg_pos_ratio)
    neg_pairs: set[tuple[str, str]] = set()
    attempts = 0
    while len(neg_pairs) < n_neg and attempts < n_neg * 20:
        attempts += 1
        a, b = rng.sample(all_cust, 2)
        if a == b:
            continue
        x, y = sorted([a, b])
        if (x, y) in pos_pairs or (x, y) in neg_pairs:
            continue
        # Verify they really never co-occur
        if cust_to_routes.get(x, set()) & cust_to_routes.get(y, set()):
            continue
        neg_pairs.add((x, y))

    samples = [(p[0], p[1], 1) for p in pos_pairs] + [(p[0], p[1], 0) for p in neg_pairs]
    rng.shuffle(samples)
    return samples, customers


class PairDataset(Dataset):
    def __init__(self, samples: list[tuple[str, str, int]], cust2idx: dict[str, int]):
        self.samples = samples
        self.cust2idx = cust2idx

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        c1, c2, label = self.samples[i]
        return (
            torch.tensor([self.cust2idx.get(c1, 0)], dtype=torch.long),
            torch.tensor([self.cust2idx.get(c2, 0)], dtype=torch.long),
            torch.tensor(label, dtype=torch.float32),
        )


class SiamesePairNet(nn.Module):
    def __init__(self, n_customers: int, d_model: int = 64):
        super().__init__()
        self.cust_emb = nn.Embedding(n_customers + 1, d_model, padding_idx=0)
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 2 + 1, 64),  # +1 for PC distance / co-occurrence
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def encode(self, c_idx):
        return self.cust_emb(c_idx).squeeze(1)  # (B, d_model)

    def forward(self, c1_idx, c2_idx, extra_feat):
        e1 = self.encode(c1_idx)
        e2 = self.encode(c2_idx)
        x = torch.cat([e1, e2, extra_feat], dim=-1)
        return torch.sigmoid(self.mlp(x).squeeze(-1))


def train_pairwise(
    train_routes: list[Route],
    val_routes: list[Route],
    epochs: int = 10,
    lr: float = 1e-3,
    batch_size: int = 256,
    seed: int = 42,
    device: str = "cpu",
) -> tuple[SiamesePairNet, dict[str, int]]:
    """Train Siamese network for same-route prediction."""
    samples, customers = build_pair_training_data(train_routes, neg_pos_ratio=3.0, seed=seed)
    cust2idx = {c: i + 1 for i, c in enumerate(sorted(customers))}
    ds = PairDataset(samples, cust2idx)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    # Build validation pairs
    val_samples, val_customers = build_pair_training_data(val_routes, neg_pos_ratio=3.0, seed=seed + 1)
    # Make sure val customers are in train vocab (filter unknowns)
    val_samples_filtered = [(c1, c2, l) for c1, c2, l in val_samples
                            if c1 in cust2idx and c2 in cust2idx]
    val_ds = PairDataset(val_samples_filtered, cust2idx)
    val_dl = DataLoader(val_ds, batch_size=batch_size)

    model = SiamesePairNet(n_customers=len(cust2idx))
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    best_state = None
    for ep in range(epochs):
        model.train()
        for c1, c2, label in dl:
            opt.zero_grad()
            # extra feature = co-occurrence proxy: not modeled here (just 0)
            extra = torch.zeros(len(c1), 1)
            pred = model(c1, c2, extra)
            loss = F.binary_cross_entropy(pred, label)
            loss.backward()
            opt.step()

        model.eval()
        val_loss = 0.0
        n = 0
        with torch.no_grad():
            for c1, c2, label in val_dl:
                extra = torch.zeros(len(c1), 1)
                pred = model(c1, c2, extra)
                loss = F.binary_cross_entropy(pred, label)
                val_loss += loss.item() * len(c1)
                n += len(c1)
        val_loss /= max(1, n)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    return model, cust2idx


def predict_pairs(
    model: SiamesePairNet,
    routes: list[Route],
    cust2idx: dict[str, int],
    threshold: float = 0.5,
) -> Predictions:
    """Predict same-route pairs per route using trained model.

    For each route, enumerate all customer pairs and predict P(same-route).
    Threshold > threshold → predict same-route.
    """
    model.eval()
    preds = Predictions()

    with torch.no_grad():
        for r in routes:
            cids = list(dict.fromkeys(r.customer_ids))  # preserve order
            pairs = list(combinations(cids, 2))
            if not pairs:
                preds.per_route_pairs[r.route_id] = frozenset()
                continue
            # Filter to known customers
            known_pairs = [(a, b) for a, b in pairs if a in cust2idx and b in cust2idx]
            if not known_pairs:
                preds.per_route_pairs[r.route_id] = frozenset()
                continue
            c1_idx = torch.tensor([cust2idx[a] for a, _ in known_pairs], dtype=torch.long)
            c2_idx = torch.tensor([cust2idx[b] for _, b in known_pairs], dtype=torch.long)
            extra = torch.zeros(len(known_pairs), 1)
            probs = model(c1_idx, c2_idx, extra).cpu().numpy()
            predicted: set[tuple[str, str]] = set()
            for (a, b), p in zip(known_pairs, probs):
                if p > threshold:
                    predicted.add(tuple(sorted([a, b])))
            preds.per_route_pairs[r.route_id] = frozenset(predicted)
    return preds


def run_pairwise_baseline(
    train_routes: list[Route],
    val_routes: list[Route],
    test_routes: list[Route],
    epochs: int = 10,
    threshold: float = 0.5,
) -> "tuple[Metrics, Metrics]":
    """End-to-end Pairwise baseline."""
    model, cust2idx = train_pairwise(train_routes, val_routes, epochs=epochs)
    val_preds = predict_pairs(model, val_routes, cust2idx, threshold=threshold)
    test_preds = predict_pairs(model, test_routes, cust2idx, threshold=threshold)
    val_m = evaluate_predictions(val_routes, val_preds)
    test_m = evaluate_predictions(test_routes, test_preds)
    return val_m, test_m