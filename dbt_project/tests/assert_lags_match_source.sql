-- Lag features must equal the source value exactly 24 h, 48 h and 168 h earlier.
-- Fails if a lag was computed over the wrong number of rows (e.g. after a gap).
select f.ts_utc
from {{ ref('features') }} f
join {{ ref('hourly') }} h24 on h24.ts_utc = f.ts_utc - interval 24 hour
join {{ ref('hourly') }} h48 on h48.ts_utc = f.ts_utc - interval 48 hour
join {{ ref('hourly') }} h168 on h168.ts_utc = f.ts_utc - interval 168 hour
where f.price_lag_24h is distinct from h24.price_eur_mwh
   or f.price_lag_48h is distinct from h48.price_eur_mwh
   or f.price_lag_168h is distinct from h168.price_eur_mwh
   or f.residual_load_de_lag_48h is distinct from h48.residual_load_de_mw
