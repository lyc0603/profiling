#!/usr/bin/env python3
"""
Load TWO Parquet files (token/native) into PostgreSQL separately,
then aggregate into a single eth_edges table.

All tables are created in TABLESPACE bigspace:
  - eth_token_edges   (UNLOGGED, raw token edges)
  - eth_native_edges  (UNLOGGED, raw native edges)
  - eth_edges         (final aggregated, set LOGGED after build)
"""

import os
import io
from tqdm import tqdm
import pyarrow.parquet as pq
import psycopg2
from psycopg2.extensions import connection as PGConn
from environ.constant import PROCESSED_DATA_PATH

# ---------- Config ----------
PG_HOST = "localhost"
PG_PORT = 5432
PG_DB = "profiling"
PG_USER = "yichen"
PG_PASS = "000603"

TOKEN_FILE = f"{PROCESSED_DATA_PATH}/token_transfer_graph.parquet"
NATIVE_FILE = f"{PROCESSED_DATA_PATH}/native_transfer_graph.parquet"

TABLESPACE_NAME = "bigspace"
BATCH_ROWS = 100_000_000
# ----------------------------


DDL_CREATE_TOKEN = f"""
DROP TABLE IF EXISTS eth_token_edges;
CREATE UNLOGGED TABLE eth_token_edges (
  from_address  TEXT NOT NULL,
  to_address    TEXT NOT NULL,
  amount_usd    DOUBLE PRECISION NOT NULL
) TABLESPACE {TABLESPACE_NAME};
"""

DDL_CREATE_NATIVE = f"""
DROP TABLE IF EXISTS eth_native_edges;
CREATE UNLOGGED TABLE eth_native_edges (
  from_address  TEXT NOT NULL,
  to_address    TEXT NOT NULL,
  amount_usd    DOUBLE PRECISION NOT NULL
) TABLESPACE {TABLESPACE_NAME};
"""

INDEX_HINTS = [
    "CREATE INDEX IF NOT EXISTS idx_eth_edges_from ON eth_edges(from_address);",
    "CREATE INDEX IF NOT EXISTS idx_eth_edges_to   ON eth_edges(to_address);",
]


def get_conn(autocommit: bool = True) -> PGConn:
    conn = psycopg2.connect(
        host=PG_HOST, port=PG_PORT, dbname=PG_DB, user=PG_USER, password=PG_PASS
    )
    conn.autocommit = autocommit
    with conn.cursor() as cur:
        cur.execute("SET synchronous_commit = off;")
        cur.execute("SET work_mem = '256MB';")
        cur.execute("SET maintenance_work_mem = '2GB';")
    return conn


def ensure_raw_tables(conn: PGConn) -> None:
    with conn.cursor() as cur:
        cur.execute(DDL_CREATE_TOKEN)
        cur.execute(DDL_CREATE_NATIVE)


def copy_parquet_into_table(
    conn: PGConn, parquet_path: str, table: str, batch_rows: int
) -> None:
    pf = pq.ParquetFile(parquet_path)
    total_rows = pf.metadata.num_rows
    pbar = tqdm(
        total=total_rows,
        unit="rows",
        desc=f"Loading {os.path.basename(parquet_path)} -> {table}",
    )

    with conn.cursor() as cur:
        for batch in pf.iter_batches(
            batch_size=batch_rows, columns=["from_address", "to_address", "amount_usd"]
        ):
            buf = io.StringIO()
            fa, ta, usd = batch.column(0), batch.column(1), batch.column(2)
            for i in range(batch.num_rows):
                buf.write(
                    f"{fa[i].as_py()}\t{ta[i].as_py()}\t{float(usd[i].as_py())}\n"
                )
            buf.seek(0)
            cur.copy_expert(
                f"COPY {table} (from_address, to_address, amount_usd) "
                "FROM STDIN WITH (FORMAT text, DELIMITER E'\\t')",
                file=buf,
            )
            pbar.update(batch.num_rows)
    pbar.close()


def rebuild_eth_edges(conn: PGConn) -> None:
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS eth_edges_new;")
        cur.execute(
            f"""
            CREATE UNLOGGED TABLE eth_edges_new TABLESPACE {TABLESPACE_NAME} AS
            SELECT from_address, to_address, SUM(amount_usd)::double precision AS amount_usd
            FROM (
                SELECT from_address, to_address, amount_usd FROM eth_token_edges
                UNION ALL
                SELECT from_address, to_address, amount_usd FROM eth_native_edges
            ) t
            GROUP BY 1,2;
        """
        )
        cur.execute(
            "ALTER TABLE eth_edges_new ADD PRIMARY KEY (from_address, to_address);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_eth_edges_new_from ON eth_edges_new(from_address);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_eth_edges_new_to   ON eth_edges_new(to_address);"
        )
        cur.execute("DROP TABLE IF EXISTS eth_edges;")
        cur.execute("ALTER TABLE eth_edges_new RENAME TO eth_edges;")
        cur.execute("ALTER TABLE eth_edges SET LOGGED;")
        cur.execute("ANALYZE eth_edges;")


def main() -> None:
    conn = get_conn(autocommit=True)
    ensure_raw_tables(conn)

    # Load both raw tables (on bigspace)
    copy_parquet_into_table(conn, TOKEN_FILE, "eth_token_edges", BATCH_ROWS)
    copy_parquet_into_table(conn, NATIVE_FILE, "eth_native_edges", BATCH_ROWS)

    # Aggregate into final table (also on bigspace)
    rebuild_eth_edges(conn)

    with conn.cursor() as cur:
        for sql in INDEX_HINTS:
            cur.execute(sql)


if __name__ == "__main__":
    main()
