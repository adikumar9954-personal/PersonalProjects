"""
Fix FEMA coverage: retry the 46 failed counties using bbox grid subdivision.
For counties with >2000 polygons the REST API 500s on pagination,
so we split the bbox into a grid of cells small enough that each returns <2000.
"""

import os, time, requests
import pandas as pd
import numpy as np
import geopandas as gpd

OUT   = os.path.dirname(os.path.abspath(__file__))
URL   = "https://hazards.fema.gov/arcgis/rest/services/public/NFHL/MapServer/28/query"

FAILED = ['003','005','009','013','015','019','021','023','029','031','033','035',
          '037','041','045','051','057','061','063','065','067','069','071','073',
          '075','077','079','083','085','086','087','089','093','095','097','101',
          '105','107','109','111','113','115','117','121','127','129']

def count_features(xmin, ymin, xmax, ymax):
    """Return feature count for a bbox (cheap query)."""
    try:
        r = requests.get(URL, params={
            'where': "SFHA_TF='T'",
            'geometry': f'{xmin},{ymin},{xmax},{ymax}',
            'geometryType': 'esriGeometryEnvelope',
            'spatialRel': 'esriSpatialRelIntersects',
            'inSR': '4326', 'f': 'json',
            'returnCountOnly': 'true',
        }, timeout=30)
        return r.json().get('count', 0)
    except:
        return -1

def fetch_bbox(xmin, ymin, xmax, ymax, max_retries=3):
    """Fetch all SFHA polygons in a bbox. Returns list of features."""
    features = []
    for attempt in range(max_retries):
        try:
            r = requests.get(URL, params={
                'where': "SFHA_TF='T'",
                'geometry': f'{xmin},{ymin},{xmax},{ymax}',
                'geometryType': 'esriGeometryEnvelope',
                'spatialRel': 'esriSpatialRelIntersects',
                'inSR': '4326', 'outSR': '4326',
                'outFields': 'FLD_ZONE,SFHA_TF',
                'returnGeometry': 'true',
                'f': 'geojson',
                'resultRecordCount': 2000,
                'resultOffset': 0,
            }, timeout=45)
            r.raise_for_status()
            data = r.json()
            if 'error' in data:
                time.sleep(2 ** attempt)
                continue
            return data.get('features', [])
        except Exception as e:
            time.sleep(2 ** attempt)
    return []

def fetch_county(xmin, ymin, xmax, ymax, county_fips, grid_n=1):
    """
    Fetch SFHA polygons for a county. If grid_n>1, subdivide into grid_n x grid_n
    cells and fetch each separately (avoids pagination 500s).
    """
    if grid_n == 1:
        return fetch_bbox(xmin, ymin, xmax, ymax)

    xs = np.linspace(xmin, xmax, grid_n + 1)
    ys = np.linspace(ymin, ymax, grid_n + 1)
    all_feats = []
    for i in range(grid_n):
        for j in range(grid_n):
            feats = fetch_bbox(xs[i], ys[j], xs[i+1], ys[j+1])
            all_feats.extend(feats)
            time.sleep(0.1)
    return all_feats

# ── Load existing FEMA tracts (from successful counties) ──────────────────────
existing = pd.read_parquet(os.path.join(OUT, 'fema_fl_tracts.parquet'))
print(f"Existing coverage: {(existing['pct_in_sfha']>0).sum():,} tracts with SFHA data")

# ── Load Florida tracts + county bboxes ───────────────────────────────────────
tracts = gpd.read_file(os.path.join(OUT, 'tiger_fl', 'tl_2020_12_tract.shp'))
tracts_ea = tracts.to_crs('EPSG:5070')
tracts_ea['tract_area_m2'] = tracts_ea.geometry.area

tracts_wgs = tracts.to_crs('EPSG:4326')
county_bounds = (tracts_wgs.groupby('COUNTYFP')
                 .apply(lambda g: g.geometry.total_bounds)
                 .reset_index())
county_bounds.columns = ['COUNTYFP', 'bounds']

failed_bounds = county_bounds[county_bounds['COUNTYFP'].isin(FAILED)]
print(f"Retrying {len(failed_bounds)} counties with grid subdivision\n")

# ── Retry each failed county ──────────────────────────────────────────────────
new_features = []

for _, row in failed_bounds.iterrows():
    cfp = row['COUNTYFP']
    xmin, ymin, xmax, ymax = row['bounds']

    # First probe: how many features in this county?
    n = count_features(xmin, ymin, xmax, ymax)
    time.sleep(0.2)

    if n == 0:
        print(f"  County {cfp}: truly empty (count=0)")
        continue
    elif n < 0:
        print(f"  County {cfp}: count query failed, trying 2x2 grid")
        grid = 2
    elif n < 1500:
        grid = 1
    elif n < 5000:
        grid = 2
    elif n < 12000:
        grid = 3
    else:
        grid = 4

    feats = fetch_county(xmin, ymin, xmax, ymax, cfp, grid_n=grid)

    if feats:
        gdf = gpd.GeoDataFrame.from_features(feats, crs='EPSG:4326')
        gdf['COUNTYFP'] = cfp
        new_features.append(gdf)
        print(f"  County {cfp}: {len(feats):,} polygons (grid={grid}x{grid})")
    else:
        print(f"  County {cfp}: still failed after grid={grid}x{grid}")

    time.sleep(0.3)

# ── Build combined SFHA GeoDataFrame ─────────────────────────────────────────
if not new_features:
    print("\nNo new SFHA data retrieved. Keeping existing coverage.")
else:
    sfha_new = pd.concat(new_features, ignore_index=True).to_crs('EPSG:5070')
    print(f"\nNew SFHA polygons from retried counties: {len(sfha_new):,}")

    # Spatial intersection with tracts that were previously missing SFHA data
    missing_tracts = tracts_ea[tracts_ea['COUNTYFP'].isin(FAILED)].copy()
    print(f"Tracts in failed counties: {len(missing_tracts):,}")

    intersection = gpd.overlay(
        missing_tracts[['GEOID','COUNTYFP','tract_area_m2','geometry']],
        sfha_new[['geometry']],
        how='intersection', keep_geom_type=False
    )
    intersection['intersect_area_m2'] = intersection.geometry.area

    sfha_by_tract = (intersection.groupby('GEOID')['intersect_area_m2']
                     .sum().reset_index()
                     .rename(columns={'intersect_area_m2':'sfha_area_m2'}))

    new_tracts = missing_tracts[['GEOID','COUNTYFP','tract_area_m2']].merge(
        sfha_by_tract, on='GEOID', how='left'
    )
    new_tracts['sfha_area_m2'] = new_tracts['sfha_area_m2'].fillna(0)
    new_tracts['pct_in_sfha']  = (new_tracts['sfha_area_m2'] /
                                   new_tracts['tract_area_m2'] * 100).clip(0, 100)
    new_tracts = new_tracts.rename(columns={'GEOID':'census_tract'})
    new_tracts = new_tracts[['census_tract','COUNTYFP','pct_in_sfha']]

    # Combine with existing
    combined = pd.concat([
        existing.rename(columns={'census_tract_geo':'census_tract'} if 'census_tract_geo' in existing.columns else {}),
        new_tracts
    ], ignore_index=True).drop_duplicates('census_tract', keep='first')

    print(f"\nCombined coverage: {len(combined):,} tracts")
    print(f"Tracts with SFHA > 0: {(combined['pct_in_sfha']>0).sum():,}")

    combined.to_parquet(os.path.join(OUT, 'fema_fl_tracts_full.parquet'), index=False)
    print(f"Saved fema_fl_tracts_full.parquet")

    # Quick distribution check
    print("\nFull SFHA coverage distribution:")
    print(pd.cut(combined['pct_in_sfha'],
                 bins=[0,0.01,10,25,50,75,100],
                 labels=['0%','0-10%','10-25%','25-50%','50-75%','75-100%']
                 ).value_counts().sort_index())
