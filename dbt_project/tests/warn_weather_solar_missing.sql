-- Warns when a city's solar irradiance is mostly missing over the last 90 days.
-- Severity is 'warn' so the build still succeeds: the Cottbus station has had no
-- radiation sensor data since Feb 2026, and an error would block the daily job.
{{ config(severity='warn') }}

with recent as (
    select city, solar_kwh_m2
    from {{ ref('stg_weather') }}
    where ts_utc >= (select max(ts_utc) from {{ ref('stg_weather') }}) - interval 90 day
)

select
    city,
    round(100.0 * count(*) filter (where solar_kwh_m2 is null) / count(*), 1) as solar_missing_pct
from recent
group by city
having 100.0 * count(*) filter (where solar_kwh_m2 is null) / count(*) > 5
