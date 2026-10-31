-- One row per holiday date and calendar (DE = national, BB, SN).
select
    cast(holiday_date as date) as holiday_date,
    holiday_name,
    region
from read_parquet('{{ var("raw_dir") }}/holidays/holidays.parquet')
