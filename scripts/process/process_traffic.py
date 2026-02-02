"""Script to process the traffic data"""

import glob

import pandas as pd
import matplotlib.pyplot as plt

from environ.constant import DATA_PATH, PROCESSED_DATA_PATH, MIN_DATE, MAX_DATE
from tqdm import tqdm

df_traffic_all = []


for path in ["30/30", "archive", "Archive_old"]:
    traffic_roots = glob.glob(str(DATA_PATH / "traffic" / path))
    # three folders: 30/30, archive, Archive_old
    for traffic_root in traffic_roots:
        traffic_folders = glob.glob(str(traffic_root + "/*"))
        # exchanges folder
        for traffic_folder in tqdm(traffic_folders):
            all_files = glob.glob(str(traffic_folder + "/*.csv"))
            # each exchange files
            for idx, filename in enumerate(all_files):
                cex_name = traffic_folder.split("/")[-1]
                df = pd.read_csv(filename, encoding="latin1")
                df = df.drop([0, 1]).rename(columns={"Location": "date"})
                df = df[
                    [
                        c
                        for c in df.columns
                        if not any(s in c for s in [".1", ".2", ".3"])
                    ]
                ]
                for col in df.columns:
                    if col != "date":
                        df[col] = pd.to_numeric(df[col])
                        df.rename(columns={col: col.strip()}, inplace=True)
                df["date"] = pd.to_datetime(df["date"])
                df = df.loc[(df["date"] >= MIN_DATE) & (df["date"] <= MAX_DATE)]
                # check date
                if df["date"].min() != pd.to_datetime(MIN_DATE):
                    raise ValueError(
                        f"{filename} has unexpected min date {df['date'].min()}!"
                    )
                if df["date"].max() != pd.to_datetime(MAX_DATE):
                    raise ValueError(
                        f"{filename} has unexpected max date {df['date'].max()}!"
                    )

                if idx == 0:
                    df_traffic = df
                    # track country names to avoid overlap
                    before_merge_country = set([_ for _ in df.columns if _ != "date"])
                else:
                    # check country name overlap
                    this_file_country = set([_ for _ in df.columns if _ != "date"])
                    new_country = this_file_country - before_merge_country

                    # merge on date
                    df_traffic = pd.merge(
                        df_traffic,
                        df[list(new_country) + ["date"]],
                        on="date",
                        how="outer",
                    )
                    before_merge_country = before_merge_country.union(this_file_country)

            df_traffic = df_traffic.loc[df_traffic["date"] < "2025-08-01"]
            df_traffic = df_traffic.drop(columns=["date"]).mean()
            df_traffic = df_traffic.to_frame().T
            df_traffic["cex"] = cex_name
            df_traffic_all.append(df_traffic)

df_traffic_all = pd.concat(df_traffic_all)
df_traffic_all.fillna(0, inplace=True)
df_traffic_all = df_traffic_all[
    ["cex"] + [c for c in df_traffic_all.columns if c != "cex"]
]

# convert to percentage
df_traffic_all = df_traffic_all.reset_index(drop=True)
for idx, row in df_traffic_all.iterrows():
    total = row.drop(labels=["cex"]).sum()
    df_traffic_all.loc[idx, df_traffic_all.columns != "cex"] = (
        row.drop(labels=["cex"]) / total
    )

# correct some name
df_traffic_all["South Korea"] = df_traffic_all["South Korea"] + df_traffic_all["Korea"]
df_traffic_all.drop(columns=["Korea"], inplace=True)

df_traffic_all.to_csv(PROCESSED_DATA_PATH / "traffic.csv", index=False)
