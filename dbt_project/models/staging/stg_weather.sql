-- One row per city and UTC hour (Bright Sky).
select
    cast(ts_utc as timestamp) as ts_utc,
    city,
    cast(temperature as double) as temperature,
    cast(wind_speed as double) as wind_speed,
    cast(cloud_cover as double) as cloud_cover,
    cast(sunshine_min as double) as sunshine_min,
    cast(solar_kwh_m2 as double) as solar_kwh_m2,
    cast(precipitation_mm as double) as precipitation_mm,
    cast(relative_humidity as double) as relative_humidity,
    cast(pressure_msl_hpa as double) as pressure_msl_hpa,
    ingested_at
from read_parquet('{{ var("raw_dir") }}/weather/**/*.parquet')
qualify row_number() over (
    partition by city, ts_utc
    order by ingested_at desc
) = 1
