"""Script to merge stablecoin relevant account"""

import pandas as pd
from environ.constant import DATA_PATH, PROCESSED_DATA_PATH
from tqdm import tqdm

add_set = set()

for token in ["usdc", "usdt"]:
    with open(f"{PROCESSED_DATA_PATH}/{token}_address.txt", "r") as f:
        for line in tqdm(f):
            add_set.add(line.strip())

cex = pd.read_csv(f"{DATA_PATH}/cex.csv")
cex_addresses = cex.loc[cex["network"] == "ETH", "address"].unique().tolist()

for cex_address in cex_addresses:
    add_set.add(cex_address.lower())

# save the address
with open(f"{PROCESSED_DATA_PATH}/stablecoin_address.txt", "w") as f:
    for address in add_set:
        f.write(f"{address}\n")
