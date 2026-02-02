"""Script to process ENS"""

import json
import gzip
from collections import defaultdict

from tqdm import tqdm

from environ.constant import DATA_PATH, PROCESSED_DATA_PATH

with gzip.open(DATA_PATH / "ens_domains.jsonl.gz", "rt") as f:
    ens_data = [json.loads(line) for line in tqdm(f)]

domain_add_set = {}
for record in tqdm(ens_data):

    # Only consider records with a name and an owner
    if record["name"] and (record["owner"] or record["wrappedOwner"]):

        # Check if wrappedOwner exists
        if record["wrappedOwner"]:
            domain_add_set[record["name"]] = record["wrappedOwner"]["id"]
        else:
            domain_add_set[record["name"]] = record["owner"]["id"]

with gzip.open(PROCESSED_DATA_PATH / "ens_domain.json.gz", "wt") as f:
    json.dump(domain_add_set, f)
