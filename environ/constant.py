"""Global Constant"""

from environ.settings import PROJECT_ROOT


# Path
DATA_PATH = PROJECT_ROOT / "data"
PROCESSED_DATA_PATH = PROJECT_ROOT / "processed_data"
FIGURE_PATH = PROJECT_ROOT / "figures"
TABLE_PATH = PROJECT_ROOT / "tables"

# Snowflake Data Schema
token_transfer_schema = [
    "block_number",
    "block_timestamp",
    "tx_hash",
    "tx_position",
    "event_index",
    "from_address",
    "to_address",
    "contract_address",
    "token_standard",
    "token_is_verified",
    "name",
    "symbol",
    "decimals",
    "raw_amount_precise",
    "raw_amount",
    "amount_precise",
    "amount",
    "amount_usd",
    "origin_function_signature",
    "origin_from_address",
    "origin_to_address",
    "ez_token_transfers_id",
    "inserted_timestamp",
    "modified_timestamp",
]

native_transfer_schema = [
    "block_number",
    "block_timestamp",
    "tx_hash",
    "tx_position",
    "trace_index",
    "trace_address",
    "type",
    "from_address",
    "to_address",
    "amount",
    "amount_precise_raw",
    "amount_precise",
    "amount_usd",
    "origin_from_address",
    "origin_to_address",
    "origin_function_signature",
    "ez_native_transfers_id",
    "inserted_timestamp",
    "modified_timestamp",
]

label_schema = [
    "blockchain",
    "creator",
    "address",
    "address_name",
    "label_type",
    "label_subtype",
    "label",
    "dim_labels_id",
    "inserted_timestamp",
    "modified_timestamp",
]

contract_schema = [
    "address",
    "symbol",
    "name",
    "decimals",
    "created_block_number",
    "created_block_timestamp",
    "created_tx_hash",
    "creator_address",
    "dim_contracts_id",
    "inserted_timestamp",
    "modified_timestamp",
]
