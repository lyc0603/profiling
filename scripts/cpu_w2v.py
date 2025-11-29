#!/usr/bin/env python3
"""
CPU Word2Vec (Gensim) trainer for random-walk sequences.

- Streams lines from a (huge) walks file: one walk per line, space-separated node IDs.
- Uses multiple CPU cores (workers).
- Skip-gram + negative sampling (Node2Vec-style).
- Optional buffered shuffling to reduce ordering bias.
- Logs per-epoch incremental loss and average loss per token.
- Saves Gensim model and word vectors (.kv, optional .bin/.txt).

Usage:
  python scripts/cpu_w2v.py \
    --walks /home/yichen/profiling/processed_data/random_walk/weighted_walks.txt \
    --vector_size 64 --window 8 --min_count 1 --negative 15 --epochs 5 --alpha 0.02\
    --shuffle_buf 250000 --batch_words 300000 --sample 1e-5 --ns_exponent 0.5\
    --out_dir /home/yichen/profiling/processed_data/model \
    --save_model --save_kv --save_bin  # choose what to save
"""

import os
import time
import argparse
import gzip
import random
import multiprocessing as mp
from collections import deque
from pathlib import Path

from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec


# ---------- IO ----------
def open_any(path, mode="rt"):
    """Open plain text or .gz transparently."""
    p = str(path)
    return gzip.open(p, mode) if p.endswith(".gz") else open(p, mode)


# ---------- Iterable over sentences (with optional buffered shuffle) ----------
class WalksIterable:
    """
    Memory-efficient iterator over walk lines.
    If shuffle_buf > 0, it shuffles blocks of lines in-memory to reduce ordering bias.
    """

    def __init__(
        self, walks_path, shuffle_buf: int = 0, seed: int = 42, min_len: int = 2
    ):
        self.walks_path = Path(walks_path)
        self.shuffle_buf = int(shuffle_buf)
        self.seed = int(seed)
        self.min_len = int(min_len)

    def __iter__(self):
        rng = random.Random(self.seed)
        if self.shuffle_buf <= 0:
            with open_any(self.walks_path, "rt") as f:
                for line in f:
                    toks = line.strip().split()
                    if len(toks) >= self.min_len:
                        yield toks
        else:
            buf = deque()
            with open_any(self.walks_path, "rt") as f:
                for line in f:
                    buf.append(line)
                    if len(buf) >= self.shuffle_buf:
                        idxs = list(range(len(buf)))
                        rng.shuffle(idxs)
                        for i in idxs:
                            toks = buf[i].strip().split()
                            if len(toks) >= self.min_len:
                                yield toks
                        buf.clear()
                # drain remainder
                if buf:
                    idxs = list(range(len(buf)))
                    rng.shuffle(idxs)
                    for i in idxs:
                        toks = buf[i].strip().split()
                        if len(toks) >= self.min_len:
                            yield toks


# ---------- Progress callback ----------
class LossLogger(CallbackAny2Vec):
    """
    Prints incremental & average loss at each epoch end.
    Gensim's `compute_loss=True` returns a cumulative loss; we diff it per epoch.
    Average loss divides the incremental loss by corpus_total_words.
    """

    def __init__(self, corpus_total_words: int | None):
        self.prev_total = 0.0
        self.epoch = 0
        self.corpus_total_words = corpus_total_words
        self.start_time = None

    def on_epoch_begin(self, model):
        self.start_time = time.time()

    def on_epoch_end(self, model):
        total = model.get_latest_training_loss()
        inc = total - self.prev_total
        self.prev_total = total
        duration = time.time() - self.start_time

        if self.corpus_total_words and self.corpus_total_words > 0:
            avg = inc / float(self.corpus_total_words)
            print(
                f"epoch {self.epoch}: inc_loss={inc:.3f}, total_loss={total:.3f}, avg_loss/token={avg:.6f}, duration={duration:.2f}s"
            )
        else:
            print(
                f"epoch {self.epoch}: inc_loss={inc:.3f}, total_loss={total:.3f}, duration={duration:.2f}s"
            )

        self.epoch += 1


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--walks", required=True, help="Path to walks file (.txt or .gz).")
    ap.add_argument("--vector_size", type=int, default=64)
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--min_count", type=int, default=1)
    ap.add_argument("--negative", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--sg", type=int, default=1, help="1=skip-gram, 0=CBOW")
    ap.add_argument(
        "--sample", type=float, default=0.0, help="Subsampling threshold; 0 disables."
    )
    ap.add_argument("--alpha", type=float, default=0.025)
    ap.add_argument("--min_alpha", type=float, default=0.0001)
    ap.add_argument(
        "--shuffle_buf", type=int, default=200000, help="Buffered shuffle size (0=off)."
    )
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 2))
    ap.add_argument(
        "--batch_words",
        type=int,
        default=200000,
        help="Target words per worker batch during training.",
    )
    ap.add_argument(
        "--ns_exponent",
        type=float,
        default=0.75,
        help="Negative sampling distribution exponent (default 0.75).",
    )
    ap.add_argument("--out_dir", required=True)

    # Saving options
    ap.add_argument(
        "--save_model", action="store_true", help="Save full Word2Vec model (.model)."
    )
    ap.add_argument("--save_kv", action="store_true", help="Save KeyedVectors (.kv).")
    ap.add_argument(
        "--save_txt", action="store_true", help="Save word2vec text format (.txt)."
    )
    ap.add_argument(
        "--save_bin", action="store_true", help="Save word2vec binary format (.bin)."
    )
    ap.add_argument(
        "--precompute_norms",
        action="store_true",
        help="Precompute L2 norms before saving KeyedVectors (faster similarity).",
    )

    return ap.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Iterable corpus
    sentences = WalksIterable(
        args.walks, shuffle_buf=args.shuffle_buf, seed=args.seed, min_len=2
    )

    # Build + train
    model = Word2Vec(
        vector_size=args.vector_size,
        window=args.window,
        min_count=args.min_count,
        sg=args.sg,
        negative=args.negative,
        sample=args.sample,
        alpha=args.alpha,
        min_alpha=args.min_alpha,
        hs=0,
        workers=args.workers,
        compute_loss=True,
        seed=args.seed,
        batch_words=args.batch_words,
        ns_exponent=args.ns_exponent,
    )

    # Build vocab in a streaming pass
    print("Building vocab...")
    model.build_vocab(sentences, update=False)
    print(
        f"Vocab size: {len(model.wv)} | corpus_total_words: {model.corpus_total_words}"
    )

    # Logger with average loss per token
    logger = LossLogger(corpus_total_words=model.corpus_total_words)

    # Train (streaming again). Use batch_words to control worker chunk size.
    print("Training...")
    model.train(
        sentences,
        total_examples=model.corpus_count,  # inferred during build_vocab
        total_words=model.corpus_total_words,
        epochs=args.epochs,
        compute_loss=True,
        callbacks=[logger],
    )

    # Optional precompute norms (useful if you'll query similarities a lot)
    if args.precompute_norms:
        model.wv.fill_norms()

    # Save artifacts (cast Path -> str)
    # Choose what you need; .kv is typically enough if you're done training.
    if args.save_model:
        model_path = out_dir / "w2v.model"
        model.save(str(model_path))
        print(f"Saved full model: {model_path}")

    if args.save_kv:
        kv_path = out_dir / "w2v.kv"
        model.wv.save(str(kv_path))
        print(f"Saved KeyedVectors: {kv_path}")

    if args.save_bin:
        bin_path = out_dir / "w2v.bin"
        model.wv.save_word2vec_format(str(bin_path), binary=True)
        print(f"Saved word2vec binary: {bin_path}")

    if args.save_txt:
        txt_path = out_dir / "w2v.txt"
        model.wv.save_word2vec_format(str(txt_path), binary=False)
        print(f"Saved word2vec text: {txt_path}")

    print("Done.")


if __name__ == "__main__":
    main()
