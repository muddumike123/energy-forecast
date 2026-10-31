"""HTTP helpers shared by the ingestion jobs."""

import time
from functools import lru_cache
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


@lru_cache(maxsize=1)
def _session() -> requests.Session:
    retry = Retry(
        total=5,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers["User-Agent"] = "leag-energy-forecast/0.1"
    return session


def get_json(url: str, params: dict[str, Any] | None = None, pause: float = 0.1) -> Any | None:
    """GET a JSON document. Returns None on 404 so callers can skip missing resources."""
    response = _session().get(url, params=params, timeout=60)
    if response.status_code == 404:
        return None
    response.raise_for_status()
    time.sleep(pause)
    return response.json()
