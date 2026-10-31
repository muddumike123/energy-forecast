-- Model-ready hourly features for a forecast issued at 12:00 on D-1 (horizon up
-- to 36 h). Load and residual-load lags are at least 48 h. The 24 h price lag is
-- also known at issue time: D-1 prices clear in the auction at ~12:45 on D-2.
--
-- Caveat: weather here is the observed value at the target hour. Training
-- (step 3) must swap it for the forecast issued at 12:00 D-1 to avoid leakage.
with hourly as (
    select * from {{ ref('hourly') }}
),

holidays_by_day as (
    select
        holiday_date as local_date,
        bool_or(region = 'DE') as is_holiday_de,
        bool_or(region = 'BB') as is_holiday_bb,
        bool_or(region = 'SN') as is_holiday_sn
    from {{ ref('stg_holidays') }}
    group by holiday_date
),

days as (
    select
        d.local_date,
        coalesce(hb.is_holiday_de, false) as is_holiday_de,
        coalesce(hb.is_holiday_bb, false) as is_holiday_bb,
        coalesce(hb.is_holiday_sn, false) as is_holiday_sn,
        dayofweek(d.local_date) in (0, 6) as is_weekend
    from (select distinct cast(ts_local as date) as local_date from hourly) d
    left join holidays_by_day hb using (local_date)
),

day_flags as (
    select
        *,
        -- Working day next to a national holiday (simple bridge-day proxy).
        (not is_weekend and not is_holiday_de) and (
            coalesce(lag(is_holiday_de) over (order by local_date), false)
            or coalesce(lead(is_holiday_de) over (order by local_date), false)
        ) as is_bridge_day
    from days
),

weather_hourly as (
    select
        ts_utc,
        avg(temperature) as weather_temp_avg_c,
        avg(wind_speed) as weather_wind_avg_kmh,
        avg(solar_kwh_m2) as weather_solar_avg_kwh_m2,
        avg(cloud_cover) as weather_cloud_avg_pct,
        avg(case when city in ('cottbus', 'dresden', 'goerlitz', 'berlin') then temperature end) as weather_temp_east_c,
        avg(case when city in ('cottbus', 'dresden', 'goerlitz', 'berlin') then wind_speed end) as weather_wind_east_kmh,
        avg(case when city in ('cottbus', 'dresden', 'goerlitz', 'berlin') then solar_kwh_m2 end) as weather_solar_east_kwh_m2
    from {{ ref('stg_weather') }}
    group by ts_utc
)

select
    h.ts_utc,
    h.ts_local,
    extract(hour from h.ts_local) as hour_local,
    dayofweek(h.ts_local) as day_of_week_local,
    extract(month from h.ts_local) as month_local,
    d.is_weekend,
    d.is_holiday_de,
    d.is_holiday_bb,
    d.is_holiday_sn,
    d.is_bridge_day,

    -- targets and same-hour observations
    h.price_eur_mwh,
    h.load_50hz_mw,
    h.residual_load_50hz_mw,
    h.solar_50hz_mw,
    h.wind_onshore_50hz_mw,
    h.wind_offshore_50hz_mw,
    h.load_de_mw,
    h.residual_load_de_mw,

    -- lags (known at 12:00 D-1, see header)
    lag(h.price_eur_mwh, 24) over (order by h.ts_utc) as price_lag_24h,
    lag(h.price_eur_mwh, 48) over (order by h.ts_utc) as price_lag_48h,
    lag(h.price_eur_mwh, 168) over (order by h.ts_utc) as price_lag_168h,
    lag(h.price_eur_mwh, 336) over (order by h.ts_utc) as price_lag_336h,
    lag(h.residual_load_50hz_mw, 48) over (order by h.ts_utc) as residual_load_lag_48h,
    lag(h.residual_load_50hz_mw, 168) over (order by h.ts_utc) as residual_load_lag_168h,
    lag(h.residual_load_50hz_mw, 336) over (order by h.ts_utc) as residual_load_lag_336h,
    lag(h.residual_load_de_mw, 48) over (order by h.ts_utc) as residual_load_de_lag_48h,
    lag(h.residual_load_de_mw, 168) over (order by h.ts_utc) as residual_load_de_lag_168h,
    lag(h.residual_load_de_mw, 336) over (order by h.ts_utc) as residual_load_de_lag_336h,
    lag(h.load_50hz_mw, 48) over (order by h.ts_utc) as load_lag_48h,
    lag(h.load_50hz_mw, 168) over (order by h.ts_utc) as load_lag_168h,
    lag(h.load_50hz_mw, 336) over (order by h.ts_utc) as load_lag_336h,

    -- weather (observed; see caveat above)
    w.weather_temp_avg_c,
    w.weather_wind_avg_kmh,
    w.weather_solar_avg_kwh_m2,
    w.weather_cloud_avg_pct,
    w.weather_temp_east_c,
    w.weather_wind_east_kmh,
    w.weather_solar_east_kwh_m2
from hourly h
join day_flags d on cast(h.ts_local as date) = d.local_date
left join weather_hourly w on w.ts_utc = h.ts_utc
