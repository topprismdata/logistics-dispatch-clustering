"""B3 baseline: BC model (Behavior Cloning — vehicle ID prediction).

This is the 太和 DC approach (MultiheadAttention directly predicts vehicle ID).
On 郑东 DC, expected to FAIL (NotebookLM prediction: Pair Recall < 30%)
because 24.8% co-occurrence + dynamic vehicles makes vehicle-ID classification
collapse.

We implement it as the "反证 baseline" — its failure proves 太和's approach
doesn't transfer.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from taihe_dc.data import Route, DispatchDataset
from taihe_dc.evaluator import Predictions, evaluate_predictions


class RouteDataset(Dataset):
    """Build per-date training examples.

    For each route on each date:
        features: list of (customer_id) → embedding + customer features
        label: vehicle_id (plate) for that route
    """

    def __init__(self, routes: list[Route], customer_to_idx: dict[str, int], plate_to_idx: dict[str, int]):
        self.routes = routes
        self.cust2idx = customer_to_idx
        self.plate2idx = plate_to_idx
        # pre-compute per-customer PC stats from training data
        cust_pc_sum: dict[str, float] = {}
        cust_pc_count: dict[str, int] = {}
        for r in routes:
            for cid, pc in r.pc_per_customer.items():
                cust_pc_sum[cid] = cust_pc_sum.get(cid, 0.0) + pc
                cust_pc_count[cid] = cust_pc_count.get(cid, 0) + 1
        self.cust_pc_avg = {cid: cust_pc_sum[cid] / max(1, cust_pc_count[cid]) for cid in cust_pc_sum}

    def __len__(self):
        return len(self.routes)

    def __getitem__(self, idx):
        r = self.routes[idx]
        cust_idxs = [self.cust2idx.get(c, -1) for c in r.customer_ids]
        cust_idxs = [i for i in cust_idxs if i >= 0]
        pc_avg = sum(self.cust_pc_avg.get(c, 0.0) for c in r.customer_ids) / max(1, len(r.customer_ids))
        return {
            "cust_idxs": torch.tensor(cust_idxs, dtype=torch.long),
            "pc_avg": torch.tensor([pc_avg / 1000.0], dtype=torch.float32),
            "n_customers": torch.tensor([min(len(r.customer_ids), 30)], dtype=torch.long),
            "label": torch.tensor(self.plate2idx.get(r.plate, 0), dtype=torch.long),
        }


def collate(batch):
    cust_idxs = [b["cust_idxs"] for b in batch]
    return {
        "cust_idxs": cust_idxs,  # list of variable-length tensors
        "pc_avg": torch.stack([b["pc_avg"] for b in batch]),
        "n_customers": torch.stack([b["n_customers"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
    }


class BCVehicleIDModel(nn.Module):
    """太和 DC BC baseline — MultiheadAttention over customer embeddings.

    Note: this predicts plate ID directly. Expected to overfit on 郑东 DC
    because of dynamic vehicles + low co-occurrence (NotebookLM).
    """

    def __init__(self, n_customers: int, n_plates: int, d_model: int = 64, n_heads: int = 4):
        super().__init__()
        self.cust_emb = nn.Embedding(n_customers + 1, d_model, padding_idx=0)  # +1 for padding/unknown
        self.pc_proj = nn.Linear(1, d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.plate_head = nn.Linear(d_model, n_plates)

    def forward(self, cust_idxs, pc_avg, n_customers):
        # cust_idxs: list of length B, each is a long tensor of customer indices (variable length)
        # pc_avg: (B, 1) float tensor (per-route avg PC normalized)
        # Build padded sequence (B, L, d_model) where L = max length in batch
        max_len = max(len(c) for c in cust_idxs)
        B = len(cust_idxs)
        seq = torch.zeros(B, max_len, self.cust_emb.embedding_dim)
        mask = torch.ones(B, max_len, dtype=torch.bool)
        for i, c in enumerate(cust_idxs):
            L = len(c)
            if L > 0:
                seq[i, :L] = self.cust_emb(c)
                mask[i, :L] = False
        seq = seq + self.pc_proj(pc_avg).unsqueeze(1)  # broadcast PC feature across positions
        # Self-attention
        out, _ = self.attn(seq, seq, seq, key_padding_mask=mask)
        # Pool: mean over real positions
        mask_float = (~mask).float().unsqueeze(-1)
        pooled = (out * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1)
        logits = self.plate_head(pooled)
        return logits


def train_bc(
    train_routes: list[Route],
    val_routes: list[Route],
    n_customers: int,
    n_plates: int,
    epochs: int = 30,
    lr: float = 1e-3,
    device: str = "cpu",
) -> BCVehicleIDModel:
    """Train BC model on training routes, return best on val loss."""
    # Build vocabularies from train only
    all_cust = set()
    all_plate = set()
    for r in train_routes:
        for c in r.customer_ids:
            all_cust.add(c)
        all_plate.add(r.plate)
    cust2idx = {c: i + 1 for i, c in enumerate(sorted(all_cust))}  # 0 = padding
    plate2idx = {p: i for i, p in enumerate(sorted(all_plate))}
    n_plates = len(plate2idx)

    model = BCVehicleIDModel(n_customers=len(cust2idx), n_plates=n_plates)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    train_ds = RouteDataset(train_routes, cust2idx, plate2idx)
    val_ds = RouteDataset(val_routes, cust2idx, plate2idx)
    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate)
    val_dl = DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=collate)

    best_val_loss = float("inf")
    best_state = None
    for ep in range(epochs):
        model.train()
        train_loss = 0.0
        n = 0
        for batch in train_dl:
            opt.zero_grad()
            logits = model(batch["cust_idxs"], batch["pc_avg"], batch["n_customers"])
            loss = F.cross_entropy(logits, batch["label"])
            loss.backward()
            opt.step()
            train_loss += loss.item() * len(batch["label"])
            n += len(batch["label"])
        train_loss /= max(1, n)

        model.eval()
        val_loss = 0.0
        n = 0
        with torch.no_grad():
            for batch in val_dl:
                logits = model(batch["cust_idxs"], batch["pc_avg"], batch["n_customers"])
                loss = F.cross_entropy(logits, batch["label"])
                val_loss += loss.item() * len(batch["label"])
                n += len(batch["label"])
        val_loss /= max(1, n)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    return model


def bc_predict_routes(
    model: BCVehicleIDModel,
    routes: list[Route],
    cust2idx: dict[str, int],
    idx2plate: dict[int, str],
    device: str = "cpu",
) -> Predictions:
    """Run BC model on routes, return Predictions.

    IMPORTANT: BC predicts PLATE ID, but predictions are evaluated at the
    ROUTE pair level. To get pair predictions, we group routes by predicted
    plate — all routes assigned to same plate become "same predicted group".
    BUT this is the wrong semantic — multiple routes on same plate should NOT
    be merged (one truck can only do one route per day). So actually BC
    predicts single-route pair = no same-route pairs at all (each plate is one route).

    To make BC competitive as a baseline, we use the **customer embedding**
    to derive pair predictions: customers with similar embeddings are predicted
    to be on the same plate. We cluster by embedding similarity.

    This makes BC baseline a "soft" same-route predictor, not a hard ID classifier.
    """
    model.eval()
    # Note: cust2idx was built from train only. Routes in val/test may have
    # customers not in train. Filter those out (BC can't embed them anyway).
    ds = RouteDataset(routes, cust2idx, {p: i for i, p in enumerate(sorted(set(r.plate for r in routes)))})
    dl = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=collate)

    route_plate_preds: dict[str, str] = {}
    route_cust_emb: dict[str, dict[str, torch.Tensor]] = {}

    with torch.no_grad():
        for batch_idx, batch in enumerate(dl):
            batch_cust_idxs = batch["cust_idxs"]
            logits = model(batch_cust_idxs, batch["pc_avg"], batch["n_customers"])
            preds = logits.argmax(dim=-1)
            for i, ridx in enumerate(range(batch_idx * 64, min((batch_idx + 1) * 64, len(routes)))):
                if ridx >= len(routes):
                    break
                r = routes[ridx]
                cust_idxs_i = batch_cust_idxs[i]
                # Filter to known customers (in train vocabulary)
                known_cids = [c for c in r.customer_ids if c in cust2idx]
                route_cust_emb[r.route_id] = {
                    c: model.cust_emb(cust_idxs_i[k]).detach()
                    for k, c in enumerate(known_cids)
                }
                route_plate_preds[r.route_id] = idx2plate.get(preds[i].item(), "?")

    # Build pair predictions from customer embeddings (cosine similarity threshold)
    pair_preds: dict[str, frozenset] = {}
    SIM_THRESHOLD = 0.7
    for r in routes:
        embs = route_cust_emb.get(r.route_id, {})
        pairs = set()
        cids = list(embs.keys())
        for i in range(len(cids)):
            for j in range(i + 1, len(cids)):
                e1, e2 = embs[cids[i]], embs[cids[j]]
                cos = F.cosine_similarity(e1.unsqueeze(0), e2.unsqueeze(0)).item()
                if cos > SIM_THRESHOLD:
                    a, b = sorted([cids[i], cids[j]])
                    pairs.add((a, b))
        pair_preds[r.route_id] = frozenset(pairs)

    return Predictions(
        route_to_vehicle=route_plate_preds,
        per_route_pairs=pair_preds,
    )


def build_bc_artifacts(train_routes: list[Route]):
    """Build vocabulary from training routes. Used by both train and predict."""
    all_cust = set()
    all_plate = set()
    for r in train_routes:
        for c in r.customer_ids:
            all_cust.add(c)
        all_plate.add(r.plate)
    cust2idx = {c: i + 1 for i, c in enumerate(sorted(all_cust))}
    plate2idx = {p: i for i, p in enumerate(sorted(all_plate))}
    idx2plate = {i: p for p, i in plate2idx.items()}
    return cust2idx, plate2idx, idx2plate


def run_bc_baseline(
    train_routes: list[Route],
    val_routes: list[Route],
    test_routes: list[Route],
    epochs: int = 30,
) -> "tuple[Metrics, Metrics]":
    """End-to-end BC baseline: train + val → predict on test, eval."""
    cust2idx, plate2idx, idx2plate = build_bc_artifacts(train_routes)
    model = train_bc(train_routes, val_routes, n_customers=len(cust2idx),
                     n_plates=len(plate2idx), epochs=epochs)
    val_preds = bc_predict_routes(model, val_routes, cust2idx, idx2plate)
    test_preds = bc_predict_routes(model, test_routes, cust2idx, idx2plate)
    val_m = evaluate_predictions(val_routes, val_preds)
    test_m = evaluate_predictions(test_routes, test_preds)
    return val_m, test_m