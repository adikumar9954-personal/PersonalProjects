# EONET Data Types — Plain English Explainer

This document explains what the data in the EONET API actually means, assuming no background in climate science, GIS, or geospatial data.

---

## What is an "Event"?

An **event** is one named natural occurrence — a specific wildfire, a specific flood, a specific hurricane. Each event has a unique ID (like `EONET_19792`) and gets updated over time as the situation evolves. An event stays **open** while it's ongoing and becomes **closed** once it ends (with a closing date recorded).

A single real-world disaster = a single event. But that event can have many **observations** recorded under it over days or weeks (see Geometry below).

---

## What is Geometry?

**Geometry** is just the location data for an event — where on Earth it is.

EONET uses two types of geometry:

### Point
A single location expressed as a pair of coordinates: **longitude and latitude**.

- **Longitude** (lon): how far east or west of the center of the Earth you are. Ranges from -180 (far west, Pacific Ocean) to +180 (far east). Negative = Western Hemisphere.
- **Latitude** (lat): how far north or south of the equator you are. Ranges from -90 (South Pole) to +90 (North Pole). Positive = Northern Hemisphere.

Example: `[-81.633, 29.886]` means 81.6° west longitude and 29.9° north latitude — which puts you in northern Florida.

**Important:** EONET stores coordinates as `[longitude, latitude]` — that's the opposite order from how most people say it ("lat/lon"). Don't mix these up.

A Point is used when the event has a single known center — a wildfire's reported origin, an earthquake's epicenter, a flood location reported by a news source.

### Polygon
A **polygon** is a shape — a list of coordinate pairs that trace out an area boundary. This is used when the event *covers* an area rather than a single point, like a large iceberg, a spreading algae bloom, or the estimated footprint of a storm system.

Each point in the polygon list is one corner/vertex of the shape, and the last point connects back to the first to close it.

### Why does one event have multiple geometry entries?
An event's `geometry` field is a **list**, not a single location. Each entry in that list is one **observation** — one snapshot of the event at a specific moment in time. As a wildfire spreads, IRWIN (the wildfire tracker) pushes new updates. As a hurricane moves, its new position gets added. Each update = a new geometry entry with a new date and new coordinates.

So to reconstruct a storm's path, you read all the geometry entries in order by date — each one is a step in the track.

---

## What is a Bounding Box (bbox)?

A **bounding box** is a rectangular region on the map used to filter events geographically. You define it with four numbers:

```
min_lon, max_lat, max_lon, min_lat
```

Think of it as drawing a rectangle on a map by specifying the left edge, top edge, right edge, and bottom edge. Any event whose location falls inside the rectangle is included.

Example — to get events only in the contiguous United States:
```
bbox = (-125.0, 49.5, -66.0, 24.0)
```
That's: left edge at 125°W, top at 49.5°N, right at 66°W, bottom at 24°N.

---

## What is Magnitude?

**Magnitude** in EONET means "how big or intense is this event" — but the unit completely changes depending on what kind of event it is. There is no universal magnitude scale across all event types.

### By event type:

#### Wildfires — acres or hectares
The area of land that has burned.
- **1 acre** ≈ the size of a football field (roughly 4,000 m²).
- **1 hectare** = 10,000 m² ≈ 2.47 acres. Used by non-US sources.
- A "small" wildfire might be a few hundred acres. A major one (like the California Camp Fire) can exceed 150,000 acres.

#### Severe Storms — knots (kts)
The **maximum sustained wind speed** of the storm.
- **1 knot** = 1 nautical mile per hour ≈ 1.15 mph or 1.85 km/h.
- Category 1 hurricane threshold: ~64 knots (~74 mph).
- Category 5 hurricane threshold: ~137 knots (~157 mph).

#### Earthquakes — several different scales (all report as Mw or similar)
All earthquake magnitudes are **logarithmic** — each whole-number step is roughly 32× more energy released.
- **Ml (Richter local):** The original 1935 scale. Works best for small, nearby earthquakes.
- **Mw (moment magnitude):** The modern standard. Works at all sizes globally. When people say "magnitude 7.5 earthquake," they mean Mw.
- **Mb (body wave):** Used for mid-range earthquakes (M4–5.5) based on the speed of P-waves (pressure waves that travel through rock).
- **Mwb, Mwc, Mwr:** Variants of moment magnitude computed using different mathematical methods; they converge to roughly the same number for large events.
- **Mi/Mwp:** Derived from P-wave displacement on broadband instruments; fast to compute, used for early warnings.

In practice: for most purposes treat them all as "the magnitude" on a 0–10 scale. Anything above 6.0 is considered major.

#### Sea and Lake Ice — nautical miles squared (NM²)
The **surface area** of an iceberg.
- **1 NM²** ≈ 3.43 km² ≈ 1.32 square miles.
- This unit is used because nautical miles are standard in marine navigation.

#### Floods, Drought, Landslides, Dust/Haze, Snow, Temperature Extremes, Manmade, Water Color
**No magnitude values.** These event types do not have a standardized numeric intensity measure in EONET. Their geometry entries will always have `magnitudeValue: null` and `magnitudeUnit: null`. The event is documented by location and date alone.

---

## What are Sources?

A **source** is the organization that reported or detected the event. EONET does not collect data itself — it aggregates from 33 external data feeds.

This matters for two reasons:
1. **Coverage:** different sources cover different geographies and event types. FloodList covers global floods; CALFIRE only covers California wildfires.
2. **Data quality:** sources vary in how quickly they report, how precise their coordinates are, and whether they provide magnitude values. IRWIN (wildfire tracker) updates frequently with precise acreage; FloodList (floods) often reports based on news articles with city-level precision only.

---

## What are Categories?

A **category** is the type of natural event. An event can technically belong to multiple categories (e.g., an earthquake that triggers a flood), but in practice most events have just one.

The 13 categories in plain terms:

| Category | What it actually is |
|---|---|
| **Drought** | A region that has gone a long time without meaningful rainfall, causing water/food shortages. Reported at regional scale, rare in this dataset. |
| **Dust and Haze** | Large dust storms or persistent air pollution plumes visible from satellite. |
| **Earthquakes** | Ground shaking from tectonic plate movement. Richest dataset — thousands of entries with precise coordinates and magnitude. |
| **Floods** | Water covering land it doesn't normally cover — rivers overflowing, storm surge, flash floods. Sparser dataset, city-level precision. |
| **Landslides** | Earth or debris sliding downhill, including mudslides and avalanches. Typically single-point, event-level data. |
| **Manmade** | Human-caused events of extreme scale — industrial accidents, oil spills, large explosions. |
| **Sea and Lake Ice** | Icebergs and large sea ice formations tracked over time. Rich polygon data from satellite observation. |
| **Severe Storms** | Hurricanes, cyclones, typhoons, and major tornadoes. Well-tracked with storm path geometry and wind speed. |
| **Snow** | Anomalous snowfall events — unusually early, late, or heavy. Low volume. |
| **Temperature Extremes** | Heatwaves and cold snaps that are well outside normal ranges for that region and time of year. |
| **Volcanoes** | Active volcanic eruptions. Well-tracked, includes both ground effects and ash plume reports. |
| **Water Color** | Satellite-detected changes in water appearance from algae blooms, red tide, sediment plumes, or other phenomena. |
| **Wildfires** | Active fires in wildland areas. The highest-volume and most frequently updated category in EONET. |

---

## What are Layers?

**Layers** are satellite imagery feeds that EONET links to each event. They're WMS/WMTS services — standardized protocols for streaming map tiles from a server. If you were building a map application, you'd use these to show the actual satellite image of a wildfire or flood alongside the event marker. For data analysis purposes, you can safely ignore them.

---

## Dates and Time Zones

All dates and times in EONET are in **UTC** (Coordinated Universal Time) — the global time standard with no daylight saving time. UTC is sometimes called "Zulu time," which is why timestamps end in `Z` (e.g., `2026-04-20T10:45:00Z`).

When you see `2026-04-20T00:00:00Z`, it means the event was reported on April 20, 2026, but no specific time was provided — the source only gave a date, and midnight UTC was used as a placeholder. This is the most common case for floods, droughts, and most non-storm events.

When you see a specific time like `T10:45:00Z`, the original source actually recorded that timestamp — typically from automated systems like IRWIN (wildfires) or USGS (earthquakes).

---

## Summary: What You Can and Cannot Measure

| Thing you might want to know | Available? | Notes |
|---|---|---|
| Where did this happen? | Always | Coordinates for every event |
| When did it start? | Always | First geometry entry date |
| When did it end? | If closed | The `closed` field; null if still open |
| How long did it last? | Computable | `closed` date minus first geometry date |
| How big/intense was it? | Wildfires, Storms, Earthquakes, Ice only | `magnitudeValue` + `magnitudeUnit` |
| How did it move or spread over time? | Wildfires, Storms, some Ice | Multiple geometry entries with sequential dates |
| What area did it cover? | Ice, some Dust/Algae events | Polygon geometry |
| Who reported it? | Always | `sources` field |
