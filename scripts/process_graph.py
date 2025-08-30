"""Aggregate each file"""

import os
import glob
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from environ.constant import (
    token_transfer_schema,
    native_transfer_schema,
    DATA_PATH,
    PROCESSED_DATA_PATH,
)


def process_and_save(path, schema, output_dir):
    """Process a single file and save grouped data to output_dir as CSV."""
    df = pd.read_parquet(path, engine="pyarrow")
    df.columns = schema
    df = df[["from_address", "to_address", "amount_usd"]]
    df = df.groupby(["from_address", "to_address"])["amount_usd"].sum().reset_index()

    # Remove zero-USD rows
    df = df[df["amount_usd"] != 0]

    # Unique filename using basename
    filename = os.path.basename(path).replace(".snappy.parquet", ".csv")
    output_path = os.path.join(output_dir, filename)
    df.to_csv(output_path, index=False)
    return output_path


def run_pipeline(file_paths, schema, output_subdir, num_workers=10):
    """Run the parallel processing pipeline and save temp CSVs only."""
    output_dir = os.path.join(PROCESSED_DATA_PATH, output_subdir)
    os.makedirs(output_dir, exist_ok=True)

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(process_and_save, path, schema, output_dir)
            for path in file_paths
        ]
        for _ in tqdm(futures, desc=f"Processing {output_subdir}"):
            _.result()  # Force completion


if __name__ == "__main__":
    token_files = glob.glob(f"{DATA_PATH}/ethereum/token_transfer/*.snappy.parquet")
    native_files = glob.glob(f"{DATA_PATH}/ethereum/native_transfer/*.snappy.parquet")

    print("Processing token transfers...")
    run_pipeline(token_files, token_transfer_schema, output_subdir="token_transfer")

    print("Processing native transfers...")
    run_pipeline(native_files, native_transfer_schema, output_subdir="native_transfer")
