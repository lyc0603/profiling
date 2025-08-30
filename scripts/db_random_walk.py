#!/usr/bin/env python3
"""
Step 2 (MP, even split): Weighted random walk generator over (neighbors_cdf, node_outdeg).

Features
- Even seed distribution across workers (no SEEDS_PER_CHUNK).
- One Postgres connection per worker.
- Dangling policies: restart | terminate | teleport.
- Sharded output per worker to avoid write contention; optional merge.
"""

import os
import random
import shutil
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from typing import List, Optional, Sequence, Tuple

import psycopg2
from tqdm import tqdm

# ---------- DB ----------
PG_HOST = "localhost"
PG_PORT = 5432
PG_DB = "profiling"
PG_USER = "yichen"
PG_PASS = "000603"
# ------------------------

# ---------- Tables ----------
NEIGH_TABLE = "neighbors_cdf"
OUTDEG_TABLE = "node_outdeg"
# ----------------------------

# ---------- Walk Params (edit as needed) ----------
WALKS_PER_SEED = 2
WALK_LENGTH = 40
DANGLING_POLICY = "restart"  # "restart" | "terminate" | "teleport"
RNG_SEED = 42  # set to None for nondeterministic
OUTPUT_BASENAME = "weighted_walks"  # written under PROCESSED_DATA_PATH
MERGE_OUTPUT = True
SPLIT_STRATEGY = "contiguous"  # "contiguous" | "roundrobin"
# --------------------------------------------------
from environ.constant import PROCESSED_DATA_PATH

# ---------- Parallelism ----------
N_WORKERS = 30  # leave 1 core for OS
# ----------------------------------


def _get_conn():
    conn = psycopg2.connect(
        host=PG_HOST, port=PG_PORT, dbname=PG_DB, user=PG_USER, password=PG_PASS
    )
    with conn.cursor() as cur:
        # light tuning
        cur.execute("SET jit = off;")
        cur.execute("SET random_page_cost = 1.1;")
        cur.execute("SET seq_page_cost = 1.0;")
        cur.execute("SET effective_io_concurrency = 256;")
        cur.execute("SET work_mem = '256MB';")
        # server-side prepared statements
        cur.execute(
            f"PREPARE get_wsum(bigint) AS SELECT wsum FROM {OUTDEG_TABLE} WHERE node = $1;"
        )
        cur.execute(
            f"""
            PREPARE pick_dst(bigint, double precision) AS
              SELECT dst
              FROM {NEIGH_TABLE}
              WHERE src = $1 AND cume >= $2
              ORDER BY cume
              LIMIT 1;
            """
        )
    return conn


class WeightedWalker:
    def __init__(self, seeds_pool: Sequence[int], policy: str, rng_seed: Optional[int]):
        if policy not in {"restart", "terminate", "teleport"}:
            raise ValueError("policy must be one of {'restart','terminate','teleport'}")
        self.conn = _get_conn()
        self.cur = self.conn.cursor()
        self.seeds_pool = list(seeds_pool)
        if not self.seeds_pool:
            raise ValueError("seeds_pool must be non-empty")
        self.policy = policy
        self.rng = random.Random(rng_seed) if rng_seed is not None else random.Random()

    def close(self):
        try:
            self.cur.close()
        except Exception:
            pass
        try:
            self.conn.close()
        except Exception:
            pass

    # --- primitives ---
    def _get_wsum(self, node: int) -> Optional[float]:
        self.cur.execute("EXECUTE get_wsum(%s)", (node,))
        row = self.cur.fetchone()
        if not row:
            return None
        wsum = row[0]
        return None if wsum is None or wsum <= 0 else float(wsum)

    def _pick_dst(self, src: int, r: float) -> Optional[int]:
        self.cur.execute("EXECUTE pick_dst(%s, %s)", (src, r))
        row = self.cur.fetchone()
        return None if not row else int(row[0])

    def next_hop(self, current: int, seed_for_walk: int) -> Optional[int]:
        wsum = self._get_wsum(current)
        if wsum is None:
            if self.policy == "terminate":
                return None
            elif self.policy == "restart":
                return seed_for_walk
            else:  # teleport
                return self.rng.choice(self.seeds_pool)
        r = self.rng.random() * wsum
        dst = self._pick_dst(current, r)
        if dst is None:
            if self.policy == "terminate":
                return None
            elif self.policy == "restart":
                return seed_for_walk
            else:
                return self.rng.choice(self.seeds_pool)
        return dst

    def walk(self, seed: int, length: int) -> List[int]:
        path = [seed]
        current = seed
        for _ in range(length):
            nxt = self.next_hop(current, seed_for_walk=seed)
            if nxt is None:
                break
            path.append(nxt)
            current = nxt
        return path


def _fetch_top_degree_seeds(k: int = 1_000_000) -> List[int]:
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT node FROM {OUTDEG_TABLE} ORDER BY outdeg DESC LIMIT %s;",
                (k,),
            )
            return [row[0] for row in cur]
    finally:
        conn.close()


def _fetch_random_seeds(k: int = 1_000_000) -> List[int]:
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                WITH sample AS (
                    SELECT node
                    FROM {OUTDEG_TABLE} TABLESAMPLE SYSTEM (0.5)
                    LIMIT %s
                )
                SELECT node FROM sample;
                """,
                (k * 2,),
            )
            seeds = [row[0] for row in cur]
            if len(seeds) < k:
                cur.execute(f"SELECT node FROM {OUTDEG_TABLE} LIMIT %s;", (k,))
                seeds = [row[0] for row in cur]
            return seeds[:k]
    finally:
        conn.close()


@dataclass
class WorkerConfig:
    seeds_pool: List[int]  # used for teleport policy
    walks_per_seed: int
    walk_length: int
    policy: str
    rng_seed: Optional[int]
    out_dir: str
    shard_idx: int


def _worker_run(args: Tuple[List[int], WorkerConfig]) -> Tuple[str, int]:
    """Process a worker's seed list, write a shard file, return (path, n_seeds_done)."""
    seed_chunk, cfg = args
    # Derive per-worker RNG for determinism
    base_seed = (
        None if cfg.rng_seed is None else (cfg.rng_seed + cfg.shard_idx * 1_000_003)
    )

    walker = WeightedWalker(cfg.seeds_pool, cfg.policy, base_seed)
    try:
        os.makedirs(cfg.out_dir, exist_ok=True)
        shard_path = os.path.join(
            cfg.out_dir, f"{OUTPUT_BASENAME}.part-{cfg.shard_idx:05d}.txt"
        )
        with open(shard_path, "w") as f:
            for s in seed_chunk:
                for _ in range(cfg.walks_per_seed):
                    w = walker.walk(s, cfg.walk_length)
                    f.write(" ".join(map(str, w)) + "\n")
        return shard_path, len(seed_chunk)
    finally:
        walker.close()


def split_even_contiguous(seeds: List[int], n_parts: int) -> List[List[int]]:
    """Split into n_parts contiguous chunks with sizes differing by at most 1."""
    n = len(seeds)
    n_parts = max(1, min(n_parts, n))  # cap to num seeds
    q, r = divmod(n, n_parts)
    parts, start = [], 0
    for i in range(n_parts):
        size = q + (1 if i < r else 0)
        end = start + size
        parts.append(seeds[start:end])
        start = end
    return parts


def split_even_roundrobin(seeds: List[int], n_parts: int) -> List[List[int]]:
    """Even round-robin split; perfect balance, scattered IDs."""
    n_parts = max(1, min(n_parts, len(seeds)))
    parts = [[] for _ in range(n_parts)]
    for i, s in enumerate(seeds):
        parts[i % n_parts].append(s)
    return parts


def merge_parts(out_dir: str, merged_path: str) -> None:
    part_paths = sorted(
        p
        for p in (os.path.join(out_dir, f) for f in os.listdir(out_dir))
        if os.path.basename(p).startswith(OUTPUT_BASENAME + ".part-")
    )
    with open(merged_path, "wb") as w:
        for p in part_paths:
            with open(p, "rb") as r:
                shutil.copyfileobj(r, w)


def main():
    # ----------------- Choose seeds -----------------
    seeds = _fetch_top_degree_seeds(k=1_000_000)
    # seeds = _fetch_random_seeds(k=1_000_000)
    # ------------------------------------------------

    out_dir = f"{PROCESSED_DATA_PATH}/random_walk"

    # Split evenly across workers (no chunk size)
    if SPLIT_STRATEGY == "roundrobin":
        chunks = split_even_roundrobin(seeds, N_WORKERS)
    else:
        chunks = split_even_contiguous(seeds, N_WORKERS)

    # Build tasks (one per worker/shard)
    cfg_base = {
        "seeds_pool": seeds,
        "walks_per_seed": WALKS_PER_SEED,
        "walk_length": WALK_LENGTH,
        "policy": DANGLING_POLICY,
        "rng_seed": RNG_SEED,
        "out_dir": out_dir,
    }
    tasks = [
        (chunk, WorkerConfig(shard_idx=i, **cfg_base)) for i, chunk in enumerate(chunks)
    ]
    total_seeds = sum(len(c) for c in chunks)

    # Run pool with progress bar
    with (
        Pool(processes=len(chunks) or 1) as pool,
        tqdm(total=total_seeds, desc="Generating walks", unit="seed") as pbar,
    ):
        for shard_path, n_done in pool.imap_unordered(_worker_run, tasks):
            pbar.update(n_done)

    if MERGE_OUTPUT:
        merged = os.path.join(out_dir, f"{OUTPUT_BASENAME}.txt")
        merge_parts(out_dir, merged)
        print(f"Merged output written to: {merged}")
    else:
        print(
            f"Sharded outputs written under: {out_dir} (files {OUTPUT_BASENAME}.part-*.txt)"
        )


if __name__ == "__main__":
    main()
