"""
Download the out-degree of each node_id from the profiling DB.
"""

import gc
import psycopg2
import json
from environ.constant import PROCESSED_DATA_PATH
from tqdm import tqdm

conn = psycopg2.connect(
    host="localhost", port=5432, dbname="profiling", user="yichen", password="000603"
)
cur = conn.cursor()
cur.execute("SET work_mem = '1GB'")

print("Loading out-degrees into a dict (node_id -> out_degree)…")
cur.execute("SELECT node, outdeg FROM node_outdeg")  # no ORDER needed for dict
rows = cur.fetchall()  # pulls everything once

outdeg_dict = {node: outdeg for node, outdeg in tqdm(rows)}
cur.close()
conn.close()

with open(f"{PROCESSED_DATA_PATH}/stablecoin2id.json", "r") as f:
    stablecoin2id = json.load(f)

seed_set = set(outdeg_dict.keys())
seed2id = {
    addr: node_id
    for addr, node_id in tqdm(stablecoin2id.items())
    if node_id in seed_set
}

with open(f"{PROCESSED_DATA_PATH}/seed2id.json", "w") as f:
    json.dump(seed2id, f)
