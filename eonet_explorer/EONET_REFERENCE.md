# EONET API v3 — Reference

NASA's Earth Observatory Natural Event Tracker. No API key required.  
Base URL: `https://eonet.gsfc.nasa.gov/api/v3`

---

## Endpoints

| Endpoint | Description |
|---|---|
| `GET /events` | Event list (JSON) |
| `GET /events/geojson` | Event list (GeoJSON FeatureCollection) |
| `GET /events/rss` | Event feed (RSS) |
| `GET /events/atom` | Event feed (ATOM) |
| `GET /categories` | All category metadata |
| `GET /categories/{id}` | Single category + associated events |
| `GET /sources` | All data sources |
| `GET /layers` | Map layer info (WMS/WMTS) |
| `GET /magnitudes` | All magnitude types |

---

## Query Parameters (Events)

| Parameter | Type | Description |
|---|---|---|
| `status` | string | `open` (default), `closed`, or `all` |
| `limit` | int | Max events returned |
| `days` | int | Events from prior N days (inclusive) |
| `start` | YYYY-MM-DD | Range start (use with `end`) |
| `end` | YYYY-MM-DD | Range end (use with `start`) |
| `category` | string | Filter by category ID; comma-separate for OR |
| `source` | string | Filter by source ID; comma-separate for OR |
| `magID` | string | Magnitude type filter |
| `magMin` | float | Minimum magnitude value |
| `magMax` | float | Maximum magnitude value |
| `bbox` | string | Bounding box: `min_lon,max_lat,max_lon,min_lat` |

**Notes:**
- `days` and `start`/`end` are mutually exclusive — use one or the other.
- No `page` parameter exists. Use `limit` + `start`/`end` windows to paginate manually.
- No total count is returned in the response — you cannot know how many events exist without querying.

---

## Event Object Structure

```json
{
  "id": "EONET_19792",
  "title": "Crews Rd Wildfire",
  "description": null,
  "link": "https://eonet.gsfc.nasa.gov/api/v3/events/EONET_19792",
  "closed": null,
  "categories": [{ "id": "wildfires", "title": "Wildfires" }],
  "sources": [{ "id": "IRWIN", "url": "https://..." }],
  "geometry": [
    {
      "magnitudeValue": 1700,
      "magnitudeUnit": "acres",
      "date": "2026-04-20T10:45:00Z",
      "type": "Point",
      "coordinates": [-81.633, 29.886]
    }
  ]
}
```

**Key notes:**
- `closed` is `null` for active events; an ISO datetime string for closed ones.
- `geometry` is an **array** — one entry per observation/update. Multi-update events have multiple geometry entries, each with its own date and coordinates.
- `coordinates` for a `Point` = `[longitude, latitude]`.
- Polygons appear for area events (e.g., ice extents, large wildfires) as a nested array of `[lon, lat]` pairs.
- Time precision: most timestamps are `00:00:00Z` (day-level resolution) unless the source reported a specific time.

---

## Time Coverage & Granularity

| Aspect | Details |
|---|---|
| **Historical depth** | Not explicitly bounded in docs. In practice, records go back to ~2000 for some categories; varies by source. Query with `start=2000-01-01` to probe. |
| **Temporal resolution** | Per-observation (each geometry point has its own date). For actively monitored events, multiple daily updates are possible. |
| **Timestamp precision** | Typically day-level (`00:00:00Z`). Occasionally hour/minute precise when the source provides it (e.g., IRWIN wildfire updates). |
| **Event duration** | Derivable from first geometry date → `closed` date. Not a stored field; must be computed. |
| **No pagination** | API returns all matching events in one response. Very wide date ranges may be slow or time out. |

---

## Event Categories

| ID | Title | Description |
|---|---|---|
| `drought` | Drought | Long-lasting absence of precipitation affecting agriculture, livestock, and water availability. |
| `dustHaze` | Dust and Haze | Dust storms, air pollution, and non-volcanic aerosols. |
| `earthquakes` | Earthquakes | All manner of shaking and displacement. Aftermath may also appear under landslides or floods. |
| `floods` | Floods | Inundation, water extending beyond river and lake extents. |
| `landslides` | Landslides | Landslides, mudslides, avalanches. |
| `manmade` | Manmade | Human-induced events that are extreme in extent. |
| `seaLakeIce` | Sea and Lake Ice | Ice on oceans and lakes, including sea and lake ice. |
| `severeStorms` | Severe Storms | Hurricanes, cyclones, tornadoes, and other atmospheric storms. |
| `snow` | Snow | Extreme or anomalous snowfall in timing or extent. |
| `tempExtremes` | Temperature Extremes | Anomalous land temperatures — heat or cold. |
| `volcanoes` | Volcanoes | Physical eruption effects and atmospheric ash/gas plumes. |
| `waterColor` | Water Color | Events altering water appearance: algae, red tide, sediment, phytoplankton. |
| `wildfires` | Wildfires | Wildland fires in forests and plains, including urban/industrial spread. |

---

## Magnitude Types by Category

| Magnitude ID | Unit | Applies To |
|---|---|---|
| `ac` | acres | Wildfires |
| `ha` | hectares | Wildfires (non-US sources) |
| `mag_kts` | knots | Severe Storms (max sustained winds) |
| `mb` | Mb | Earthquakes (body wave, P-wave amplitude, M4.0–5.5) |
| `mi` | Mi/Mwp | Earthquakes (P-wave displacement) |
| `ml` | Ml | Earthquakes (Richter local magnitude) |
| `mms` | Mw/Mww | Earthquakes (moment magnitude, most common modern scale) |
| `mwb` | Mwb | Earthquakes (long-period body-wave, M5.5+) |
| `mwc` | Mwc | Earthquakes (centroid moment tensor) |
| `mwr` | Mwr | Earthquakes (regional, below M5.0) |
| `sq_NM` | NM² | Sea/Lake Ice (iceberg area in nautical miles²) |

**Note:** Floods, Drought, Landslides, Dust/Haze, Snow, Temperature Extremes, Manmade, and Water Color events **do not have magnitude values** — their geometry entries will have `magnitudeValue: null`.

---

## Data Sources

| ID | Organization |
|---|---|
| `AVO` | Alaska Volcano Observatory |
| `ABFIRE` | Alberta Wildfire |
| `AU_BOM` | Australia Bureau of Meteorology |
| `BYU_ICE` | Brigham Young University Antarctic Iceberg Tracking |
| `BCWILDFIRE` | British Columbia Wildfire Service |
| `CALFIRE` | California Dept. of Forestry and Fire Protection |
| `CEMS` | Copernicus Emergency Management Service |
| `EO` | NASA Earth Observatory |
| `Earthdata` | NASA Earthdata |
| `FEMA` | Federal Emergency Management Agency |
| `FloodList` | FloodList |
| `GDACS` | Global Disaster Alert and Coordination System |
| `GLIDE` | GLobal IDEntifier Number |
| `InciWeb` | InciWeb (US wildfire incidents) |
| `IRWIN` | Integrated Reporting of Wildfire Information |
| `IDC` | International Charter on Space and Major Disasters |
| `JTWC` | Joint Typhoon Warning Center |
| `MRR` | LANCE Rapid Response |
| `MBFIRE` | Manitoba Wildfire Program |
| `NASA_ESRS` | NASA Earth Science and Remote Sensing Unit |
| `NASA_DISP` | NASA Earth Science Disasters Program |
| `NASA_HURR` | NASA Hurricane and Typhoon Updates |
| `NOAA_NHC` | National Hurricane Center |
| `NOAA_CPC` | NOAA Center for Weather and Climate Prediction |
| `PDC` | Pacific Disaster Center |
| `ReliefWeb` | ReliefWeb |
| `SIVolcano` | Smithsonian Institution Global Volcanism Program |
| `NATICE` | U.S. National Ice Center |
| `UNISYS` | Unisys Weather |
| `USGS_EHP` | USGS Earthquake Hazards Program |
| `USGS_CMT` | USGS Emergency Operations Collection Management Tool |
| `HDDS` | USGS Hazards Data Distribution System |
| `DFES_WA` | Western Australia Dept. of Fire and Emergency Services |

**Flood-relevant sources:** `FloodList`, `GDACS`, `CEMS`, `ReliefWeb`, `IDC`, `FEMA`, `PDC`, `NASA_DISP`, `GLIDE`

---

## What Data Exists Per Category

| Category | Has Magnitude | Geometry Type | Multi-Geometry (tracks) | Notes |
|---|---|---|---|---|
| Wildfires | Yes (acres/ha) | Point | Yes | IRWIN updates frequently; multiple points per fire as it grows |
| Severe Storms | Yes (knots) | Point | Yes | Track follows storm path over time |
| Earthquakes | Yes (various Mw) | Point | No | Typically single-point; precise timestamps |
| Sea/Lake Ice | Yes (NM²) | Polygon | Yes | Area polygons; icebergs tracked over time |
| Floods | No | Point | Sometimes | Coarser — sourced from news/disaster aggregators |
| Volcanoes | No | Point | Yes | Updates as eruption evolves |
| Drought | No | Point/Polygon | Rarely | Low data volume |
| Landslides | No | Point | No | Typically single-point events |
| Dust/Haze | No | Point/Polygon | Sometimes | Area coverage varies |
| Snow | No | Point/Polygon | Rarely | Low data volume |
| Temp Extremes | No | Point | Rarely | Low data volume |
| Manmade | No | Point | Rarely | Sparse |
| Water Color | No | Point/Polygon | Sometimes | Algae blooms tracked over time |
