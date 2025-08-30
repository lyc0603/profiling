#!/usr/bin/env python3
"""
Train Node2Vec-style SGNS (float32) using ONLY the 'first-node' vocabulary,
and force the job to use the **second A100** (GPU index 1 on the host).

It:
- reads walk lines (space-separated node IDs),
- restricts vocabulary to the set of first nodes you provide,
- builds a unigram^0.75 negative sampler,
- trains SGNS in float32 on **GPU 1**,
- saves embeddings and the vocab mapping.

USAGE
-----
python scripts/train.py \
  --walks /home/yichen/profiling/processed_data/random_walk/weighted_walks.txt \
  --first_nodes /home/yichen/profiling/processed_data/random_walk/weighted_walks_firstnodes.txt \
  --dim 128 --window 10 --neg 10 --epochs 3 --batch_size 4096 \
  --out_dir /home/yichen/profiling/processed_data/random_walk/node2vec_firstnodes \
  --gpu_index 1


Notes
-----
- This script **forces CUDA_VISIBLE_DEVICES=1** before importing torch,
  so it will not touch GPU 0 (which your other job is using).
- Float32 memory for embeddings: ~ (2 * V * dim * 4) bytes.
  For V=1,000,000 and dim=128 => ~1.0 GB (plus optimizer/activations).
"""

import os
import argparse
import random
import gzip
import json
from collections import Counter
from typing import Dict, List, Iterator, Tuple

import numpy as np
from tqdm import tqdm


# --------------------- minimal IO helpers ---------------------
def open_any(path: str):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path, "r")


def load_first_nodes(path: str) -> List[int]:
    with open(path, "r") as f:
        return [int(line.strip()) for line in f if line.strip()]


def build_vocab_from_firstnodes(first_nodes: List[int]) -> Dict[int, int]:
    # Map node_id -> vocab_index [0..V-1]
    return {nid: i for i, nid in enumerate(sorted(first_nodes))}


def count_freqs(walks_path: str, vocab: Dict[int, int]) -> np.ndarray:
    counts = Counter()
    with open_any(walks_path) as f:
        for line in tqdm(f, desc="Counting token frequencies"):
            toks = line.strip().split()
            if not toks:
                continue
            for t in toks:
                try:
                    nid = int(t)
                except:
                    continue
                if nid in vocab:
                    counts[nid] += 1
    V = len(vocab)
    freqs = np.zeros(V, dtype=np.int64)
    for nid, c in counts.items():
        freqs[vocab[nid]] = c
    freqs[freqs == 0] = 1
    return freqs


def build_alias_from_freqs(freqs: np.ndarray):
    # unigram^0.75 alias table
    prob = freqs.astype(np.float64)
    prob = np.power(prob, 0.75)
    prob /= prob.sum()
    n = len(prob)
    scaled = prob * n
    q = np.zeros(n, dtype=np.float64)
    J = np.zeros(n, dtype=np.int64)
    small, large = [], []
    for i, p in enumerate(scaled):
        q[i] = p
        (small if p < 1.0 else large).append(i)
    while small and large:
        s, l = small.pop(), large.pop()
        J[s] = l
        q[l] -= 1.0 - q[s]
        (small if q[l] < 1.0 else large).append(l)
    for l in large:
        q[l] = 1.0
    for s in small:
        q[s] = 1.0
    return J, q


def alias_draw(J, q, rng: np.random.Generator) -> int:
    k = rng.integers(0, len(J))
    return int(k if rng.random() < q[k] else J[k])


def pair_stream(
    walks_path: str, vocab: Dict[int, int], window: int, epochs: int
) -> Iterator[Tuple[int, int]]:
    ids = vocab
    for _ in range(epochs):
        with open_any(walks_path) as f:
            for line in f:
                toks = line.strip().split()
                if not toks:
                    continue
                seq = []
                for t in toks:
                    try:
                        nid = int(t)
                    except:
                        continue
                    idx = ids.get(nid)
                    if idx is not None:
                        seq.append(idx)
                L = len(seq)
                if L == 0:
                    continue
                for i in range(L):
                    c = seq[i]
                    w = random.randint(1, window)  # dynamic window
                    left = max(0, i - w)
                    right = min(L, i + w + 1)
                    for j in range(left, right):
                        if j == i:
                            continue
                        yield c, seq[j]


# --------------------- main (sets GPU=1 before torch import) ---------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--walks", required=True)
    ap.add_argument(
        "--first_nodes",
        required=True,
        help="text file of first nodes (one ID per line)",
    )
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--neg", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=0.025)
    ap.add_argument("--out_dir", required=True)
    # Optional override if you want a different GPU later
    ap.add_argument(
        "--gpu_index", type=int, default=1, help="Use this host GPU index (default=1)"
    )
    args = ap.parse_args()

    # Force this process to see ONLY the selected GPU before importing torch
    if args.gpu_index >= 0:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)

    # Now import torch so it binds to the masked device set
    import torch
    from torch import nn

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU name:", torch.cuda.get_device_name(0))

    # 1) Load fixed vocab from first nodes
    first_nodes = load_first_nodes(args.first_nodes)
    print(f"Loaded first-node list: {len(first_nodes):,} IDs")
    vocab = build_vocab_from_firstnodes(first_nodes)
    V = len(vocab)

    # 2) Count frequencies & build alias table
    freqs = count_freqs(args.walks, vocab)
    J, q = build_alias_from_freqs(freqs)
    rng = np.random.default_rng(1234)

    # 3) Model (float32) + optimizer
    class SGNS(nn.Module):
        def __init__(self, vocab_size: int, dim: int):
            super().__init__()
            self.in_emb = nn.Embedding(vocab_size, dim)  # float32
            self.out_emb = nn.Embedding(vocab_size, dim)
            nn.init.uniform_(self.in_emb.weight, -0.5 / dim, 0.5 / dim)
            nn.init.zeros_(self.out_emb.weight)
            self.logsigmoid = nn.LogSigmoid()

        def forward(
            self, centers: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor
        ) -> torch.Tensor:
            cen = self.in_emb(centers)  # [B, D]
            pos_w = self.out_emb(pos)  # [B, D]
            neg_w = self.out_emb(neg)  # [B, K, D]
            pos_score = torch.sum(cen * pos_w, dim=1)  # [B]
            neg_score = torch.bmm(neg_w, cen.unsqueeze(2)).squeeze(2)  # [B, K]
            loss = -(
                self.logsigmoid(pos_score).mean()
                + self.logsigmoid(-neg_score).mean(dim=1).mean()
            )
            return loss

    model = SGNS(V, args.dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 4) Training stream
    stream = pair_stream(args.walks, vocab, window=args.window, epochs=args.epochs)

    B = args.batch_size
    K = args.neg
    centers_buf: List[int] = []
    ctx_buf: List[int] = []

    pbar = tqdm(desc="Training", unit="batch")

    def flush_batch():
        if not centers_buf:
            return
        centers = torch.tensor(centers_buf, dtype=torch.long, device=device)
        pos = torch.tensor(ctx_buf, dtype=torch.long, device=device)
        # negatives via alias sampling
        neg_np = np.stack(
            [
                [alias_draw(J, q, rng) for _ in range(K)]
                for _ in range(len(centers_buf))
            ],
            axis=0,
        )
        neg = torch.tensor(neg_np, dtype=torch.long, device=device)

        opt.zero_grad(set_to_none=True)
        loss = model(centers, pos, neg)
        loss.backward()
        opt.step()

        pbar.set_postfix(loss=float(loss.detach().cpu()))
        pbar.update(1)
        centers_buf.clear()
        ctx_buf.clear()

    for c, p in stream:
        centers_buf.append(c)
        ctx_buf.append(p)
        if len(centers_buf) >= B:
            flush_batch()
    flush_batch()
    pbar.close()

    # 5) Save embeddings + vocab
    os.makedirs(args.out_dir, exist_ok=True)
    emb_path = os.path.join(args.out_dir, "embeddings.pt")
    vocab_path = os.path.join(args.out_dir, "vocab_firstnodes.json")
    torch.save(model.in_emb.weight.detach().cpu(), emb_path)
    with open(vocab_path, "w") as f:
        json.dump({str(k): int(v) for k, v in vocab.items()}, f)
    print(f"Saved: {emb_path}  (shape: [{V}, {args.dim}])")
    print(f"Saved: {vocab_path}")


if __name__ == "__main__":
    main()
