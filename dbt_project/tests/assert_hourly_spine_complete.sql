-- Fails if the hourly spine has duplicate or missing UTC hours.
with stats as (
    select
        min(ts_utc) as first_ts,
        max(ts_utc) as last_ts,
        count(*) as n_rows,
        count(distinct ts_utc) as n_distinct
    from {{ ref('hourly') }}
)

select *
from stats
where n_rows <> n_distinct
   or n_rows <> date_diff('hour', first_ts, last_ts) + 1
