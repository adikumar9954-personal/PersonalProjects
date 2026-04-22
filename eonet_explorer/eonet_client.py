"""
EONET API v3 client — reusable module for querying NASA's Earth Observatory Natural Event Tracker.
Base URL: https://eonet.gsfc.nasa.gov/api/v3
No API key required.
"""

import requests
from datetime import date, datetime
from typing import Optional, Union

BASE_URL = "https://eonet.gsfc.nasa.gov/api/v3"

# ---------------------------------------------------------------------------
# Category IDs (use these as the `category` parameter)
# ---------------------------------------------------------------------------
CATEGORIES = {
    "drought":       "drought",
    "dust_haze":     "dustHaze",
    "earthquakes":   "earthquakes",
    "floods":        "floods",
    "landslides":    "landslides",
    "manmade":       "manmade",
    "sea_lake_ice":  "seaLakeIce",
    "severe_storms": "severeStorms",
    "snow":          "snow",
    "temp_extremes": "tempExtremes",
    "volcanoes":     "volcanoes",
    "water_color":   "waterColor",
    "wildfires":     "wildfires",
}

# ---------------------------------------------------------------------------
# Source IDs (use these as the `source` parameter)
# ---------------------------------------------------------------------------
SOURCES = {
    "AVO", "ABFIRE", "AU_BOM", "BYU_ICE", "BCWILDFIRE", "CALFIRE",
    "CEMS", "EO", "Earthdata", "FEMA", "FloodList", "GDACS", "GLIDE",
    "InciWeb", "IRWIN", "IDC", "JTWC", "MRR", "MBFIRE", "NASA_ESRS",
    "NASA_DISP", "NASA_HURR", "NOAA_NHC", "NOAA_CPC", "PDC", "ReliefWeb",
    "SIVolcano", "NATICE", "UNISYS", "USGS_EHP", "USGS_CMT", "HDDS", "DFES_WA",
}

# ---------------------------------------------------------------------------
# Magnitude unit IDs (use these as the `magID` parameter)
# ---------------------------------------------------------------------------
MAGNITUDES = {
    "acres":          "ac",
    "hectares":       "ha",
    "wind_knots":     "mag_kts",
    "body_wave":      "mb",
    "p_wave":         "mi",
    "richter_local":  "ml",
    "moment_mw":      "mms",
    "long_period_bw": "mwb",
    "centroid_mt":    "mwc",
    "regional_mt":    "mwr",
    "sq_nautical_mi": "sq_NM",
}


def _build_params(
    status: Optional[str] = None,
    limit: Optional[int] = None,
    days: Optional[int] = None,
    start: Optional[Union[str, date, datetime]] = None,
    end: Optional[Union[str, date, datetime]] = None,
    category: Optional[str] = None,
    source: Optional[str] = None,
    mag_id: Optional[str] = None,
    mag_min: Optional[float] = None,
    mag_max: Optional[float] = None,
    bbox: Optional[tuple[float, float, float, float]] = None,
) -> dict:
    """Convert keyword args to API query params, dropping None values."""
    params = {}
    if status:   params["status"]  = status
    if limit:    params["limit"]   = limit
    if days:     params["days"]    = days
    if category: params["category"] = category
    if source:   params["source"]  = source
    if mag_id:   params["magID"]   = mag_id
    if mag_min is not None: params["magMin"] = mag_min
    if mag_max is not None: params["magMax"] = mag_max

    if start:
        params["start"] = start.isoformat()[:10] if isinstance(start, (date, datetime)) else start
    if end:
        params["end"] = end.isoformat()[:10] if isinstance(end, (date, datetime)) else end

    if bbox:
        # bbox = (min_lon, max_lat, max_lon, min_lat)
        params["bbox"] = ",".join(str(v) for v in bbox)

    return params


def _get(endpoint: str, params: dict = None) -> dict:
    url = f"{BASE_URL}/{endpoint}"
    response = requests.get(url, params=params or {}, timeout=30)
    response.raise_for_status()
    return response.json()


# ---------------------------------------------------------------------------
# Core endpoint functions
# ---------------------------------------------------------------------------

def get_events(
    status: str = "open",
    limit: int = None,
    days: int = None,
    start: Union[str, date, datetime] = None,
    end: Union[str, date, datetime] = None,
    category: str = None,
    source: str = None,
    mag_id: str = None,
    mag_min: float = None,
    mag_max: float = None,
    bbox: tuple[float, float, float, float] = None,
    geojson: bool = False,
) -> dict:
    """
    Fetch events. status: 'open' | 'closed' | 'all'.
    Use start/end (YYYY-MM-DD) or days for time filtering.
    bbox = (min_lon, max_lat, max_lon, min_lat).
    """
    endpoint = "events/geojson" if geojson else "events"
    params = _build_params(
        status=status, limit=limit, days=days, start=start, end=end,
        category=category, source=source, mag_id=mag_id,
        mag_min=mag_min, mag_max=mag_max, bbox=bbox,
    )
    return _get(endpoint, params)


def get_categories(
    category_id: str = None,
    status: str = None,
    limit: int = None,
    days: int = None,
    start: Union[str, date, datetime] = None,
    end: Union[str, date, datetime] = None,
) -> dict:
    """Fetch category metadata, optionally with associated events."""
    endpoint = f"categories/{category_id}" if category_id else "categories"
    params = _build_params(status=status, limit=limit, days=days, start=start, end=end)
    return _get(endpoint, params)


def get_sources() -> dict:
    """Fetch all event sources."""
    return _get("sources")


def get_layers(category_id: str = None) -> dict:
    """Fetch map layer info, optionally filtered by category."""
    endpoint = f"layers/{category_id}" if category_id else "layers"
    return _get(endpoint)


def get_magnitudes() -> dict:
    """Fetch all magnitude types and their units."""
    return _get("magnitudes")


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def get_all_events_in_range(
    start: Union[str, date, datetime],
    end: Union[str, date, datetime],
    category: str = None,
    source: str = None,
) -> list[dict]:
    """
    Fetch all events (open + closed) within a date range.
    Returns the flat list of event objects.
    """
    data = get_events(
        status="all",
        start=start,
        end=end,
        category=category,
        source=source,
        limit=None,
    )
    return data.get("events", [])


def get_recent_events(days: int = 30, category: str = None) -> list[dict]:
    """Fetch open + closed events from the last N days."""
    data = get_events(status="all", days=days, category=category)
    return data.get("events", [])


def events_to_records(events: list[dict]) -> list[dict]:
    """
    Flatten the event list into a list of one-row-per-geometry-point dicts,
    suitable for loading into a DataFrame or CSV.
    Each row = one location+date observation of an event.
    """
    rows = []
    for event in events:
        base = {
            "id":          event["id"],
            "title":       event["title"],
            "description": event.get("description"),
            "closed":      event.get("closed"),
            "categories":  ", ".join(c["id"] for c in event.get("categories", [])),
            "sources":     ", ".join(s["id"] for s in event.get("sources", [])),
        }
        for geom in event.get("geometry", []):
            coords = geom.get("coordinates", [])
            row = base.copy()
            row["date"]       = geom.get("date")
            row["mag_value"]  = geom.get("magnitudeValue")
            row["mag_unit"]   = geom.get("magnitudeUnit")
            row["geom_type"]  = geom.get("type")
            row["lon"]        = coords[0] if geom.get("type") == "Point" else None
            row["lat"]        = coords[1] if geom.get("type") == "Point" else None
            row["polygon"]    = coords    if geom.get("type") == "Polygon" else None
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Fetching open events (last 7 days)...")
    events = get_recent_events(days=7)
    print(f"  {len(events)} events returned")
    if events:
        e = events[0]
        print(f"  Sample: [{e['id']}] {e['title']} — {e['categories'][0]['title']}")

    print("\nFetching categories...")
    cats = get_categories()
    for c in cats.get("categories", []):
        print(f"  {c['id']:20s} {c['title']}")
