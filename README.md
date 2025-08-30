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

GET @~/eth_label/ file://./data/ethereum/contract PATTERN='.*[.]parquet' PARALLEL=32;

GET @~/eth_label/ file://./data/ethereum/label PATTERN='.*[.]parquet' PARALLEL=32;
```

- Run the /scripts/snowflake_schema.sql in snowflake project worksheets to get data schema