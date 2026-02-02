import pyarrow.parquet as pq
from environ.constant import PROCESSED_DATA_PATH

for prefix in ["token_transfer", "native_transfer"]:
    path = f"{PROCESSED_DATA_PATH}/{prefix}_graph.parquet"
    pf = pq.ParquetFile(path)

    print("Number of row groups:", pf.num_row_groups)
    print("Total rows:", pf.metadata.num_rows)
