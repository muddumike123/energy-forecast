-- One row per series, region and UTC hour. Re-ingested chunks are deduplicated
-- by keeping the most recent download.
select
    cast(ts_utc as timestamp) as ts_utc,
    series,
    region,
    cast(value as double) as value,
    ingested_at
from read_parquet('{{ var("raw_dir") }}/smard/**/*.parquet')
qualify row_number() over (
    partition by series, region, ts_utc
    order by ingested_at desc
) = 1
