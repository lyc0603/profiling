-- Token Transfer 
SELECT 
    column_name,
    ordinal_position
FROM ETHEREUM_ONCHAIN_CORE_DATA.INFORMATION_SCHEMA.COLUMNS
WHERE table_schema = 'CORE'
  AND table_name = 'EZ_TOKEN_TRANSFERS'
ORDER BY ordinal_position;

-- Native Transfer
SELECT 
    column_name,
    ordinal_position
FROM ETHEREUM_ONCHAIN_CORE_DATA.INFORMATION_SCHEMA.COLUMNS
WHERE table_schema = 'CORE'
  AND table_name = 'EZ_NATIVE_TRANSFERS'
ORDER BY ordinal_position;

-- Label
SELECT 
    column_name,
    ordinal_position
FROM ETHEREUM_ONCHAIN_CORE_DATA.INFORMATION_SCHEMA.COLUMNS
WHERE table_schema = 'CORE'
  AND table_name = 'DIM_LABELS'
ORDER BY ordinal_position;

-- Contract
SELECT 
    column_name,
    ordinal_position
FROM ETHEREUM_ONCHAIN_CORE_DATA.INFORMATION_SCHEMA.COLUMNS
WHERE table_schema = 'CORE'
  AND table_name = 'DIM_CONTRACTS'
ORDER BY ordinal_position;