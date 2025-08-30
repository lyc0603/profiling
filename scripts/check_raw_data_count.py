"""
Validate total row numbers in token/native parquet files.

- Uses Parquet metadata (fast) to count rows per file.
- Prints per-category totals and grand total.
- Saves a CSV report at: {PROCESSED_DATA_PATH}/row_count_report.csv
"""

import os
import glob
import csv
from tqdm import tqdm

# fast parquet metadata read
try:
    from pyarrow.parquet import ParquetFile
except ImportError as e:
    raise ImportError("Please install pyarrow: pip install pyarrow") from e

from environ.constant import DATA_PATH, PROCESSED_DATA_PATH


def count_rows_parquet(path: str) -> int:
    """Return number of rows in a parquet file using its metadata."""
    pf = ParquetFile(path)
    md = pf.metadata
    # Fallback if md is None (rare)
    return (
        md.num_rows
        if md is not None
        else sum(pf.read_row_group(i).num_rows for i in range(pf.num_row_groups))
    )


def summarize_counts(file_glob: str, label: str):
    files = sorted(glob.glob(file_glob))
    total = 0
    per_file = []

    if not files:
        print(f"[WARN] No files found for {label}: {file_glob}")

    for f in tqdm(files, desc=f"Counting {label}", unit="file"):
        try:
            n = count_rows_parquet(f)
        except Exception as e:
            print(f"[ERROR] Failed to read {f}: {e}")
            n = 0
        total += n
        per_file.append((label, f, n))
    return total, per_file


def main():
    token_glob = os.path.join(DATA_PATH, "ethereum/token_transfer/*.snappy.parquet")
    native_glob = os.path.join(DATA_PATH, "ethereum/native_transfer/*.snappy.parquet")

    token_total, _ = summarize_counts(token_glob, "token")
    native_total, _ = summarize_counts(native_glob, "native")

    grand_total = token_total + native_total

    print("\n=== Row Count Summary ===")
    print(f"Token total rows : {token_total:,}")
    print(f"Native total rows: {native_total:,}")
    print(f"Grand total rows : {grand_total:,}")


if __name__ == "__main__":
    main()
