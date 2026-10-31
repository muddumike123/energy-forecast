-- Gap-free hourly table in UTC. SMARD timestamps are already UTC epochs, so
-- the spine is continuous across DST changes. ts_local (Europe/Berlin) is a
-- derived attribute for calendar features, never a key.
with bounds as (
    select
        min(ts_utc) as first_ts,
        max(ts_utc) as last_ts
    from {{ ref('stg_smard') }}
),

spine as (
    select unnest(generate_series(first_ts, last_ts, interval 1 hour)) as ts_utc
    from bounds
),

series_wide as (
    select
        ts_utc,
        max(value) filter (where region = '50Hertz' and series = 'load') as load_50hz_mw,
        max(value) filter (where region = '50Hertz' and series = 'residual_load') as residual_load_50hz_mw,
        max(value) filter (where region = '50Hertz' and series = 'solar') as solar_50hz_mw,
        max(value) filter (where region = '50Hertz' and series = 'wind_onshore') as wind_onshore_50hz_mw,
        max(value) filter (where region = '50Hertz' and series = 'wind_offshore') as wind_offshore_50hz_mw,
        max(value) filter (where region = 'DE' and series = 'load') as load_de_mw,
        max(value) filter (where region = 'DE' and series = 'residual_load') as residual_load_de_mw,
        max(value) filter (where region = 'DE' and series = 'solar') as solar_de_mw,
        max(value) filter (where region = 'DE' and series = 'wind_onshore') as wind_onshore_de_mw,
        max(value) filter (where region = 'DE' and series = 'wind_offshore') as wind_offshore_de_mw,
        max(value) filter (where region = 'DE-LU' and series = 'price') as price_eur_mwh
    from {{ ref('stg_smard') }}
    group by ts_utc
)

select
    s.ts_utc,
    (s.ts_utc at time zone 'UTC') at time zone 'Europe/Berlin' as ts_local,
    w.* exclude (ts_utc)
from spine s
left join series_wide w using (ts_utc)
