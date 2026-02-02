"""
Download the mapping from wallet address to node_id from the profiling DB.
"""

import psycopg2
import json
from environ.constant import PROCESSED_DATA_PATH
from tqdm import tqdm

conn = psycopg2.connect(
    host="localhost", port=5432, dbname="profiling", user="yichen", password="000603"
)
cur = conn.cursor()
cur.execute("SET work_mem = '1GB'")

print("Loading wallets into a dict (address -> node_id)…")
cur.execute("SELECT address, node_id FROM wallets")  # no ORDER needed for dict
rows = cur.fetchall()  # pulls everything once
addr2id = {addr: nid for addr, nid in rows}

# the reverse map:
id2addr = {nid: addr for addr, nid in rows}

cur.close()
conn.close()
print("Done.", len(addr2id))

# Save to JSON files
with open(f"{PROCESSED_DATA_PATH}/addr2id.json", "w") as f:
    json.dump(addr2id, f)

with open(f"{PROCESSED_DATA_PATH}/id2addr.json", "w") as f:
    json.dump(id2addr, f)


# Load stablecoin addresses into a set
stablecoin_set = set()
with open(f"{PROCESSED_DATA_PATH}/stablecoin_address.txt", "r") as f:
    for line in f:
        stablecoin_set.add(line.strip())

stablecoin2id = {
    addr: addr2id[addr] for addr in tqdm(stablecoin_set) if addr in addr2id
}

with open(f"{PROCESSED_DATA_PATH}/stablecoin2id.json", "w") as f:
    json.dump(stablecoin2id, f)
