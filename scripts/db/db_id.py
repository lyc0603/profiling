#!/usr/bin/env python3
"""
Build integer-id mapping for eth_edges with only TWO commits:

  1) wallets fully built + committed   (UNION ALL + ON CONFLICT, dedup via UNIQUE)
  2) edges_int fully built with CTAS (ordered), indexes added, ANALYZE, committed

Heavy objects and temp files are placed in TABLESPACE `bigspace` for speed.
"""

import psycopg2

# ---------- DB ----------
PG_HOST = "localhost"
PG_PORT = 5432
PG_DB = "profiling"
PG_USER = "yichen"
PG_PASS = "000603"
# ------------------------

TABLESPACE = "bigspace"


def get_conn():
    conn = psycopg2.connect(
        host=PG_HOST, port=PG_PORT, dbname=PG_DB, user=PG_USER, password=PG_PASS
    )
    with conn.cursor() as cur:
        # Session tuning for this one job
        cur.execute("SET jit = off;")
        cur.execute("SET work_mem = '1GB';")  # 1–2GB if no swap
        cur.execute("SET maintenance_work_mem = '32GB';")  # large for index builds
        cur.execute("SET max_parallel_workers_per_gather = 16;")
        cur.execute("SET parallel_leader_participation = on;")
        cur.execute("SET synchronous_commit = off;")
        cur.execute("SET wal_compression = on;")
        cur.execute("SET temp_buffers = '256MB';")
        cur.execute("SET random_page_cost = 1.1;")
        cur.execute("SET seq_page_cost = 1.0;")
        cur.execute("SET effective_io_concurrency = 256;")
        cur.execute("SET log_temp_files = '100MB';")
        # Put temp files on the big/fast volume and avoid premature aborts
        cur.execute(f"SET temp_tablespaces = '{TABLESPACE}';")
        cur.execute(
            "SET temp_file_limit = '-1';"
        )  # or e.g. '200GB' if you want a guardrail
    return conn


def build_two_commit():
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            # ----------- Commit #1: wallets -----------
            print("[1/2] Build wallets (UNION ALL + ON CONFLICT)…")

            # DDL (in bigspace)
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS wallets (
                  node_id BIGSERIAL PRIMARY KEY,
                  address TEXT UNIQUE NOT NULL
                ) TABLESPACE {TABLESPACE};
            """
            )

            # Fast path: avoid global dedup; UNIQUE(address) filters duplicates
            cur.execute(
                """
                INSERT INTO wallets(address)
                SELECT from_address FROM eth_edges
                UNION ALL
                SELECT to_address   FROM eth_edges
                ON CONFLICT (address) DO NOTHING;
            """
            )

            # Stats for better join plans in step 2
            cur.execute("ANALYZE wallets;")

            # First and only commit for Step 1
            conn.commit()
            print("   wallets committed.")

            # ----------- Commit #2: edges_int via CTAS + indexes -----------
            print("[2/2] Build edges_int with integer ids (CTAS, UNLOGGED, ordered)…")

            # Clean rebuild
            cur.execute("DROP TABLE IF EXISTS edges_int;")

            # CTAS (fastest). Pairs are unique, so no GROUP BY. Order by src for locality.
            cur.execute(
                f"""
                CREATE UNLOGGED TABLE edges_int
                TABLESPACE {TABLESPACE} AS
                SELECT
                    wf.node_id AS src,
                    wt.node_id AS dst,
                    e.amount_usd AS weight
                FROM eth_edges e
                JOIN wallets wf ON e.from_address = wf.address
                JOIN wallets wt ON e.to_address   = wt.address
                ORDER BY 1;  -- physically cluster by src
            """
            )

            # Add PK and secondary index AFTER load (uses maintenance_work_mem)
            # Bulk PK build is faster than maintaining it during insert
            cur.execute("ALTER TABLE edges_int ADD PRIMARY KEY (src, dst);")

            # Build dst index; you can add parallel_workers to hint parallel index build
            cur.execute(
                f"CREATE INDEX idx_edges_int_dst ON edges_int(dst) TABLESPACE {TABLESPACE};"
            )

            # Gather stats
            cur.execute("ANALYZE edges_int;")

            # Second and final commit
            conn.commit()

    finally:
        conn.close()


if __name__ == "__main__":
    build_two_commit()
