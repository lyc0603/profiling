#!/usr/bin/env python3
"""
Node2Vec-style SGNS (float32) — NO SUBSAMPLING.
- Uses all nodes in walks (with --min_count).
- Sparse embeddings + SparseAdam (memory efficient).
- Buffered line shuffling; LR warmup+decay; grad clipping.
- Forces training on selected GPU (default: GPU 1).

Example:
  CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python scripts/train.py \
    --walks /home/yichen/profiling/processed_data/random_walk/weighted_walks.txt \
    --dim 32 --window 8 --neg 5 --epochs 3 --batch_size 4096 --min_count 1 \
    --out_dir /home/yichen/profiling/processed_data/model
"""

import os, argparse, random, gzip, json
from collections import Counter, deque
from typing import Dict, List, Iterator, Tuple
import numpy as np
from tqdm import tqdm


# -------------------- helpers --------------------
def open_any(path: str, mode="rt"):
    return gzip.open(path, mode) if path.endswith(".gz") else open(path, mode)


def build_vocab_and_freqs(path: str, min_count: int):
    counts = Counter()
    with open_any(path, "rt") as f:
        for line in tqdm(f, desc="Counting tokens"):
            for t in line.strip().split():
                try:
                    nid = int(t)
                except ValueError:
                    continue
                counts[nid] += 1
    kept = [(nid, c) for nid, c in counts.items() if c >= min_count]
    kept.sort(key=lambda x: -x[1])
    vocab = {nid: i for i, (nid, _) in enumerate(kept)}
    freqs = np.array([c for _, c in kept], dtype=np.int64)
    if len(freqs) == 0:
        raise RuntimeError("No tokens met min_count; lower --min_count")
    return vocab, freqs


def build_alias_from_freqs(freqs: np.ndarray):
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


def line_iter_shuffled(path: str, buf_lines=200_000):
    buf = deque()
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        for line in f:
            buf.append(line)
            if len(buf) >= buf_lines:
                idx = np.random.permutation(len(buf))
                for i in idx:
                    yield buf[i]
                buf.clear()
        idx = np.random.permutation(len(buf))
        for i in idx:
            yield buf[i]


def pair_stream(
    path: str, vocab: Dict[int, int], window: int, epochs: int
) -> Iterator[Tuple[int, int]]:
    for _ in range(epochs):
        for line in line_iter_shuffled(path):
            idx_seq: List[int] = []
            for t in line.strip().split():
                try:
                    nid = int(t)
                except ValueError:
                    continue
                idx = vocab.get(nid)
                if idx is not None:
                    idx_seq.append(idx)
            L = len(idx_seq)
            if L < 2:
                continue
            for i in range(L):
                c = idx_seq[i]
                w = random.randint(1, window)
                left, right = max(0, i - w), min(L, i + w + 1)
                for j in range(left, right):
                    if j != i:
                        yield c, idx_seq[j]


# -------------------- main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--walks", required=True)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--neg", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--min_count", type=int, default=1)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--gpu_index", type=int, default=1)
    args = ap.parse_args()

    # GPU + allocator (set before torch import)
    if args.gpu_index >= 0:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import torch
    from torch import nn

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        torch.backends.cuda.matmul.allow_tf32 = True

    # vocab + freqs
    vocab, freqs = build_vocab_and_freqs(args.walks, args.min_count)
    V = len(vocab)
    print(f"Vocab size: {V:,}")

    # negative sampler
    J, q = build_alias_from_freqs(freqs)
    rng = np.random.default_rng(1234)

    # model (sparse embeddings)
    class SGNS(nn.Module):
        def __init__(self, V, D):
            super().__init__()
            self.in_emb = nn.Embedding(V, D, sparse=True)
            self.out_emb = nn.Embedding(V, D, sparse=True)
            nn.init.uniform_(self.in_emb.weight, -0.5 / D, 0.5 / D)
            nn.init.zeros_(self.out_emb.weight)
            self.logsigmoid = nn.LogSigmoid()

        def forward(self, c, pos, neg):
            cen = self.in_emb(c)
            pos_w = self.out_emb(pos)
            neg_w = self.out_emb(neg)
            pos_score = torch.sum(cen * pos_w, dim=1)
            neg_score = torch.bmm(neg_w, cen.unsqueeze(2)).squeeze(2)
            return -(
                self.logsigmoid(pos_score).mean()
                + self.logsigmoid(-neg_score).mean(dim=1).mean()
            )

    model = SGNS(V, args.dim).to(device)
    opt = torch.optim.SparseAdam(model.parameters(), lr=args.lr)

    # warmup + light decay
    warmup = 5_000
    decay_after = 200_000
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lr_lambda=lambda s: min(1.0, (s + 1) / warmup)
        * (1.0 if s < decay_after else 0.5),
    )

    # train
    stream = pair_stream(args.walks, vocab, args.window, args.epochs)
    B, K = args.batch_size, args.neg
    buf_c, buf_p = [], []
    pbar = tqdm(desc="Training", unit="batch")

    def flush():
        if not buf_c:
            return
        centers = torch.tensor(buf_c, dtype=torch.long, device=device)
        pos = torch.tensor(buf_p, dtype=torch.long, device=device)
        neg_np = np.stack(
            [[alias_draw(J, q, rng) for _ in range(K)] for _ in range(len(buf_c))],
            axis=0,
        )
        neg = torch.tensor(neg_np, dtype=torch.long, device=device)
        opt.zero_grad(set_to_none=True)
        loss = model(centers, pos, neg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        pbar.set_postfix(loss=float(loss.detach().cpu()))
        pbar.update(1)
        buf_c.clear()
        buf_p.clear()

    for c, p in stream:
        buf_c.append(c)
        buf_p.append(p)
        if len(buf_c) >= B:
            flush()
    flush()
    pbar.close()

    # save
    os.makedirs(args.out_dir, exist_ok=True)
    torch.save(
        model.in_emb.weight.detach().cpu(), os.path.join(args.out_dir, "embeddings.pt")
    )
    with open(os.path.join(args.out_dir, "vocab_allnodes.json"), "w") as f:
        json.dump({str(nid): int(i) for nid, i in vocab.items()}, f)
    print("Saved model + vocab.")


if __name__ == "__main__":
    main()
