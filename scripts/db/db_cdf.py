#!/usr/bin/env python3
"""
Step 1: Precompute per-node CDF for weighted next-hop sampling (handles negatives).
"""

import psycopg2
from psycopg2 import sql

# ---------- DB ----------
PG_HOST = "localhost"
PG_PORT = 5432
PG_DB = "profiling"
PG_USER = "yichen"
PG_PASS = "000603"
# ------------------------

# ---------- Settings ----------
TABLESPACE = "bigspace"
SOURCE_TABLE = "edges_int"  # (src BIGINT, dst BIGINT, weight DOUBLE PRECISION)
NEIGH_TABLE = "neighbors_cdf"
OUTDEG_TABLE = "node_outdeg"

DROP_EXISTING = True  # drop & rebuild
USE_LOG_WEIGHT = True  # True -> ln(1+max(w,0)); False -> max(w,0)
# -----------------------------


def get_conn():
    conn = psycopg2.connect(
        host=PG_HOST, port=PG_PORT, dbname=PG_DB, user=PG_USER, password=PG_PASS
    )
    with conn.cursor() as cur:
        cur.execute("SET jit = off;")
        cur.execute("SET work_mem = '1GB';")
        cur.execute("SET maintenance_work_mem = '32GB';")
        cur.execute("SET max_parallel_workers_per_gather = 16;")
        cur.execute("SET parallel_leader_participation = on;")
        cur.execute("SET synchronous_commit = off;")
        cur.execute("SET wal_compression = on;")
        cur.execute("SET temp_buffers = '256MB';")
        cur.execute("SET random_page_cost = 1.1;")
        cur.execute("SET seq_page_cost = 1.0;")
        cur.execute("SET effective_io_concurrency = 256;")
        cur.execute("SET log_temp_files = '100MB';")
        cur.execute("SET temp_tablespaces = %s;", (TABLESPACE,))
        cur.execute("SET temp_file_limit = '-1';")
    return conn


def build_neighbors_cdf():
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            if DROP_EXISTING:
                print(f"Dropping old {NEIGH_TABLE} / {OUTDEG_TABLE} (if any)…")
                cur.execute(
                    sql.SQL("DROP TABLE IF EXISTS {}").format(
                        sql.Identifier(OUTDEG_TABLE)
                    )
                )
                cur.execute(
                    sql.SQL("DROP TABLE IF EXISTS {}").format(
                        sql.Identifier(NEIGH_TABLE)
                    )
                )
                conn.commit()

            # Safe base weight: clamp negatives to zero
            weight_base = "GREATEST(e.weight, 0::double precision)"
            weight_expr = (
                weight_base if not USE_LOG_WEIGHT else f"LN(1 + {weight_base})"
            )

            print(f"Building {NEIGH_TABLE} (UNLOGGED, tablespace {TABLESPACE})…")
            cur.execute(
                sql.SQL(
                    """
                    CREATE UNLOGGED TABLE {neigh}
                    TABLESPACE {tsp}
                    AS
                    SELECT
                        e.src,
                        e.dst,
                        {wexpr} AS weight,
                        SUM({wexpr}) OVER (PARTITION BY e.src) AS wsum,
                        SUM({wexpr}) OVER (
                            PARTITION BY e.src
                            ORDER BY e.dst
                            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                        ) AS cume
                    FROM {edges} AS e
                    WHERE {wbase} > 0
                    ORDER BY e.src, cume;
                    """
                ).format(
                    neigh=sql.Identifier(NEIGH_TABLE),
                    tsp=sql.Identifier(TABLESPACE),
                    edges=sql.Identifier(SOURCE_TABLE),
                    wexpr=sql.SQL(weight_expr),
                    wbase=sql.SQL(weight_base),
                )
            )
            conn.commit()

            print(f"ANALYZE {NEIGH_TABLE} …")
            cur.execute(sql.SQL("ANALYZE {}").format(sql.Identifier(NEIGH_TABLE)))
            conn.commit()

        # CREATE INDEX CONCURRENTLY must be autocommit
        print(f"Creating CONCURRENT index on ({NEIGH_TABLE}.src, cume)…")
        conn.autocommit = True
        with conn.cursor() as cur:
            idx_neigh = sql.Identifier(f"idx_{NEIGH_TABLE}_src_cume")
            cur.execute(
                sql.SQL(
                    "CREATE INDEX CONCURRENTLY IF NOT EXISTS {idx} "
                    "ON {neigh}(src, cume) TABLESPACE {tsp};"
                ).format(
                    idx=idx_neigh,
                    neigh=sql.Identifier(NEIGH_TABLE),
                    tsp=sql.Identifier(TABLESPACE),
                )
            )
        conn.autocommit = False

        with conn.cursor() as cur:
            print(f"Building {OUTDEG_TABLE} …")
            cur.execute(
                sql.SQL(
                    """
                    CREATE UNLOGGED TABLE {outdeg}
                    TABLESPACE {tsp}
                    AS
                    SELECT src AS node, COUNT(*) AS outdeg, MAX(wsum) AS wsum
                    FROM {neigh}
                    GROUP BY src;
                    """
                ).format(
                    outdeg=sql.Identifier(OUTDEG_TABLE),
                    tsp=sql.Identifier(TABLESPACE),
                    neigh=sql.Identifier(NEIGH_TABLE),
                )
            )
            conn.commit()

            print(f"ANALYZE {OUTDEG_TABLE} …")
            cur.execute(sql.SQL("ANALYZE {}").format(sql.Identifier(OUTDEG_TABLE)))
            conn.commit()

        print(f"Creating CONCURRENT index on {OUTDEG_TABLE}(node)…")
        conn.autocommit = True
        with conn.cursor() as cur:
            idx_out = sql.Identifier(f"idx_{OUTDEG_TABLE}_node")
            cur.execute(
                sql.SQL(
                    "CREATE INDEX CONCURRENTLY IF NOT EXISTS {idx} "
                    "ON {out}(node) TABLESPACE {tsp};"
                ).format(
                    idx=idx_out,
                    out=sql.Identifier(OUTDEG_TABLE),
                    tsp=sql.Identifier(TABLESPACE),
                )
            )
        conn.autocommit = False

        print("Step 1 complete")
    finally:
        conn.close()


if __name__ == "__main__":
    build_neighbors_cdf()
