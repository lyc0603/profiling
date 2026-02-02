# profiling

## 1. Setup

```
git clone https://github.com/lyc0603/profiling.git
cd profiling
```

### Give execute permission to your script and then run `setup_repo.sh`

```
chmod +x setup_repo.sh
./setup_repo.sh
. venv/bin/activate
```

## 2. Data Fetching

- Run the /scripts/snowflake_data.sql in snowflake project worksheets to unload flipside data

- Create Directory
```
!mkdir -p ./ethereun/token_transfer

!mkdir -p ./ethereun/native_transfer

!mkdir -p ./ethereun/label

!mkdir -p ./ethereun/contract
```
- Run the following command in Snowsql
```
GET @~/eth_token_transfer/ file://./data/ethereum/contract PATTERN='.*[.]parquet' PARALLEL=32;

GET @~/eth_native_transfer/ file://./data/ethereum/contract PATTERN='.*[.]parquet' PARALLEL=32;

GET @~/eth_contract/ file://./data/ethereum/contract PATTERN='.*[.]parquet' PARALLEL=32;

GET @~/eth_label/ file://./data/ethereum/label PATTERN='.*[.]parquet' PARALLEL=32;
```

- Run the /scripts/snowflake_schema.sql in snowflake project worksheets to get data schema


## 3. Data Processing

- Aggregate flows into subgraphs
```
python process_graph.py
```

- Merge subgraphs into graph
```
python merge_graph.py
```

- Upload graph to PostgreSQL
```
python db_merge.py
```

- Isolate wallets and build integer edges
```
python db_id.py
```

- Build cumulative density function
```
python db_cdf.py
```

- Get stablecoin users
```
python process_stablecoin_list.py
```

- Merge stablecoin users and anchors
```
python merge_stablecoin.py
```

- Process stablecoin users and anchors
```
python scripts/db_id_downloaded.py
```

- Generate the seed
```
python scripts/db_id_out_degree.py
```

- Train wallet embedding
```
python scripts/cpu_w2v.py
```

## 4. Evaluation

- Process the trend (the shrinkage of number of address is due to 0 value txn and out degree)
```
python scripts/process_trend.py
```

- Generate anchor
```
python eval_anchor_emb.py
```
