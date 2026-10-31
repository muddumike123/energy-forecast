-- Local (Europe/Berlin) calendar days must contain 24 hours. The only exceptions
-- are the DST switch days: 23 hours on the last Sunday of March and 25 hours on
-- the last Sunday of October. Partial first and last days are ignored.
with per_day as (
    select
        cast(ts_local as date) as local_date,
        count(*) as n_hours
    from {{ ref('hourly') }}
    group by 1
),

bounds as (
    select
        cast(min(ts_local) as date) as first_day,
        cast(max(ts_local) as date) as last_day
    from {{ ref('hourly') }}
)

select p.*
from per_day p
cross join bounds b
where p.local_date > b.first_day
  and p.local_date < b.last_day
  and not (
        p.n_hours = 24
     or (p.n_hours = 23 and month(p.local_date) = 3 and dayofweek(p.local_date) = 0 and day(p.local_date) >= 25)
     or (p.n_hours = 25 and month(p.local_date) = 10 and dayofweek(p.local_date) = 0 and day(p.local_date) >= 25)
  )
