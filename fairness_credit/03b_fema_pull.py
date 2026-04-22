"""
FEMA NFHL flood zone coverage at census-tract level for Florida.
Queries the NFHL REST API county-by-county using bounding boxes.
"""

import os, time, requests
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import shape, box

OUT = os.path.dirname(os.path.abspath(__file__))
NFHL_URL = "https://hazards.fema.gov/arcgis/rest/services/public/NFHL/MapServer/28/query"

# ── Load Florida tracts (already downloaded) ──────────────────────────────────
tracts = gpd.read_file(os.path.join(OUT, 'tiger_fl', 'tl_2020_12_tract.shp'))
tracts = tracts.to_crs('EPSG:5070')
tracts['tract_area_m2'] = tracts.geometry.area
print(f"Florida tracts: {len(tracts):,}")

# ── Get unique counties + their WGS84 bounding boxes ─────────────────────────
tracts_wgs = tracts.to_crs('EPSG:4326')
counties = tracts_wgs.groupby('COUNTYFP').apply(
    lambda g: g.geometry.total_bounds  # [xmin, ymin, xmax, ymax]
).reset_index()
counties.columns = ['COUNTYFP', 'bounds']
print(f"Florida counties: {len(counties)}")

# ── Query NFHL county-by-county ───────────────────────────────────────────────
def query_sfha_bbox(xmin, ymin, xmax, ymax, county_fips):
    """Fetch all SFHA=T polygons in a bounding box via NFHL REST API."""
    all_features = []
    offset = 0

    while True:
        params = {
            'where':             "SFHA_TF='T'",
            'geometry':          f'{xmin},{ymin},{xmax},{ymax}',
            'geometryType':      'esriGeometryEnvelope',
            'spatialRel':        'esriSpatialRelIntersects',
            'inSR':              '4326',
            'outSR':             '4326',
            'outFields':         'FLD_ZONE,SFHA_TF',
            'returnGeometry':    'true',
            'f':                 'geojson',
            'resultOffset':      offset,
            'resultRecordCount': 2000,
        }
        try:
            r = requests.get(NFHL_URL, params=params, timeout=60)
            r.raise_for_status()
            data = r.json()

            if 'error' in data:
                print(f"    API error for county {county_fips}: {data['error']}")
                break

            features = data.get('features', [])
            all_features.extend(features)

            if len(features) < 2000:
                break
            offset += 2000
            time.sleep(0.15)

        except Exception as e:
            print(f"    Exception for county {county_fips}: {e}")
            break

    return all_features

# ── Process each county ───────────────────────────────────────────────────────
all_sfha = []
failed = []

for i, row in counties.iterrows():
    cfp = row['COUNTYFP']
    xmin, ymin, xmax, ymax = row['bounds']

    features = query_sfha_bbox(xmin, ymin, xmax, ymax, cfp)

    if features:
        gdf = gpd.GeoDataFrame.from_features(features, crs='EPSG:4326')
        gdf['COUNTYFP'] = cfp
        all_sfha.append(gdf)
        print(f"  County {cfp}: {len(features):,} SFHA polygons")
    else:
        failed.append(cfp)
        print(f"  County {cfp}: 0 polygons")

    time.sleep(0.2)

print(f"\nFailed counties ({len(failed)}): {failed}")

if not all_sfha:
    raise RuntimeError("No SFHA data pulled at all")

# ── Combine all SFHA polygons ─────────────────────────────────────────────────
sfha = pd.concat(all_sfha, ignore_index=True)
sfha = sfha.to_crs('EPSG:5070')

print(f"\nTotal SFHA polygons: {len(sfha):,}")

# ── Spatial intersection: tracts x SFHA ──────────────────────────────────────
print("Computing spatial intersection (tracts x SFHA)...")
intersection = gpd.overlay(
    tracts[['GEOID', 'COUNTYFP', 'tract_area_m2', 'geometry']],
    sfha[['geometry']],
    how='intersection',
    keep_geom_type=False
)
intersection['intersect_area_m2'] = intersection.geometry.area

sfha_by_tract = (
    intersection.groupby('GEOID')['intersect_area_m2']
    .sum()
    .reset_index()
    .rename(columns={'intersect_area_m2': 'sfha_area_m2'})
)

# ── Merge back ────────────────────────────────────────────────────────────────
out = tracts[['GEOID', 'COUNTYFP', 'tract_area_m2']].merge(
    sfha_by_tract, on='GEOID', how='left'
)
out['sfha_area_m2'] = out['sfha_area_m2'].fillna(0)
out['pct_in_sfha'] = (out['sfha_area_m2'] / out['tract_area_m2'] * 100).clip(0, 100)
out = out.rename(columns={'GEOID': 'census_tract_geo'})

print(f"\nTracts with any SFHA: {(out['pct_in_sfha'] > 0).sum():,} / {len(out):,}")
print(f"Tracts >50% in SFHA:  {(out['pct_in_sfha'] > 50).sum():,}")

print("\nSFHA coverage distribution:")
print(pd.cut(out['pct_in_sfha'],
             bins=[0, 0.01, 10, 25, 50, 75, 100],
             labels=['0%','0-10%','10-25%','25-50%','50-75%','75-100%']
             ).value_counts().sort_index())

# ── Save ──────────────────────────────────────────────────────────────────────
out_df = out[['census_tract_geo', 'COUNTYFP', 'pct_in_sfha']].copy()
out_path = os.path.join(OUT, 'fema_fl_tracts.parquet')
out_df.rename(columns={'census_tract_geo': 'census_tract'}).to_parquet(out_path)
print(f"\nSaved: {out_path}")
