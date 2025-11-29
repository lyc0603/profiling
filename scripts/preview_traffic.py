"""Script to process traffic data"""

import glob

import pandas as pd
import matplotlib.pyplot as plt

from environ.constant import DATA_PATH, PROCESSED_DATA_PATH

c_name = []

for path in [
    "Archive_old/coinbase/*.csv",
    "Archive_old/binance/*.csv",
    "Archive_old/okx/*.csv",
]:
    file_path = DATA_PATH / "traffic" / path
    all_files = glob.glob(str(file_path))

    for idx, filename in enumerate(all_files):
        df = pd.read_csv(filename)
        df = df.drop([0, 1]).rename(columns={"Location": "date"})
        df = df[[c for c in df.columns if not any(s in c for s in [".1", ".2", ".3"])]]
        for col in df.columns:
            if col != "date":
                df[col] = pd.to_numeric(df[col])
                df.rename(columns={col: col.strip()}, inplace=True)
        df["date"] = pd.to_datetime(df["date"])
        if idx == 0:
            df_traffic = df
        else:
            df_traffic = pd.merge(df_traffic, df, on="date", how="outer")

    # each column is a country, within a country, take average
    top5 = (
        df_traffic.drop(columns=["date"])
        .mean()
        .sort_values(ascending=False)
        .head(5)
        .index.tolist()
    )

    # plot each column
    for col in df_traffic.columns:
        if col != "date":
            if col in top5:
                plt.plot(df_traffic["date"], df_traffic[col], label=col)
            else:
                plt.plot(df_traffic["date"], df_traffic[col], color="grey", alpha=0.3)

    c_name.append(set([_ for _ in df.columns if _ != "date"]))

    plt.legend(title="Top 5 Countries by Average Traffic", frameon=False)
    plt.title(f"Traffic Data from {path.split('/')[1].capitalize()}")
    plt.show()
