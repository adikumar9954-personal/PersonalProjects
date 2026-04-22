"""
Data Pull: HMDA + FEMA Flood Zone
-----------------------------------
Pulls:
  1. HMDA loan-level data for Florida 2018–2022 (home purchase loans)
     via CFPB Data Browser API
  2. FEMA flood zone coverage at census-tract level
     via NFHL ArcGIS REST API (layer 28 = flood hazard areas)

Then merges and writes a clean analysis-ready file.

Florida chosen because:
  - Second-highest flood insurance exposure in the US
  - Diverse racial demographics
  - Multiple major FEMA remapping events post-Hurricane Irma (2017)
  - Large HMDA sample size
"""

import os, time, requests, zipfile, io
import pandas as pd
import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))

# ══════════════════════════════════════════════════════════════════════════════
# 1. HMDA DATA
#    Endpoint: https://ffiec.cfpb.gov/v2/data-browser-api/view/csv
#    Filters: Florida, home purchase (loan_purpose=1), 2018-2022
#    Key fields: census_tract, race, action_taken, interest_rate, rate_spread,
#                loan_amount, income
# ══════════════════════════════════════════════════════════════════════════════

HMDA_FIELDS = [
    'census_tract',
    'action_taken',              # 1=originated, 2=approved not accepted, 3=denied
    'loan_purpose',              # 1=home purchase
    'loan_amount',
    'income',                    # applicant income ($thousands)
    'interest_rate',
    'rate_spread',               # APR minus average prime offer rate
    'applicant_race-1',
    'applicant_race-2',
    'derived_race',              # CFPB-derived single race
    'applicant_sex',
    'denial_reason-1',
    'property_value',
    'loan_to_value_ratio',
    'debt_to_income_ratio',
    'derived_loan_product_type',
    'activity_year',
    'county_code',
    'state_code',
]

def pull_hmda(state='FL', years=None, loan_purpose='1'):
    """Pull HMDA data for one state, multiple years."""
    if years is None:
        years = [2018, 2019, 2020, 2021, 2022]

    base = 'https://ffiec.cfpb.gov/v2/data-browser-api/view/csv'
    dfs = []

    for year in years:
        print(f"  Pulling HMDA {state} {year}...")
        params = {
            'states':        state,
            'years':         str(year),
            'loan_purposes': loan_purpose,   # home purchase only
        }
        try:
            r = requests.get(base, params=params, timeout=120)
            r.raise_for_status()

            from io import StringIO
            df = pd.read_csv(StringIO(r.text), dtype=str, low_memory=False)

            # Keep only columns we care about (some may not exist in all years)
            keep = [c for c in HMDA_FIELDS if c in df.columns]
            df = df[keep].copy()
            df['activity_year'] = year

            dfs.append(df)
            print(f"    -> {len(df):,} rows, {df.columns.tolist()[:5]}...")
            time.sleep(0.5)   # be polite to the API

        except Exception as e:
            print(f"    ERROR: {e}")

    if not dfs:
        raise RuntimeError("No HMDA data pulled")

    hmda = pd.concat(dfs, ignore_index=True)
    return hmda


# ══════════════════════════════════════════════════════════════════════════════
# 2. FEMA FLOOD ZONE COVERAGE AT CENSUS-TRACT LEVEL
#    We use a pre-computed HUD crosswalk approach: HUD publishes a Census
#    Tract <-> FEMA Special Flood Hazard Area crosswalk derived from the NFHL.
#
#    Fallback: query NFHL REST API directly (slow but always current).
#    For this pull we use FEMA's public NFHL REST service to get county-level
#    flood zone data, then join to census tracts via TIGER geography.
# ══════════════════════════════════════════════════════════════════════════════

def pull_fema_tract_florida():
    """
    Get FEMA flood zone coverage (% of census tract in SFHA) for Florida.

    Strategy:
      - Download Florida census tract boundaries from Census TIGER (small file)
      - Query NFHL ArcGIS REST for Florida flood hazard polygons
      - Spatial intersection -> pct_in_sfha per tract

    Returns a DataFrame indexed by census_tract.
    """
    try:
        import geopandas as gpd
        from shapely.geometry import shape
        import json
    except ImportError:
        print("  geopandas not available, skipping spatial step")
        return None

    print("\nPulling Census tract boundaries for Florida...")

    # Census TIGER 2020 tract shapefile for Florida (state FIPS = 12)
    tiger_url = (
        "https://www2.census.gov/geo/tiger/TIGER2020/TRACT/"
        "tl_2020_12_tract.zip"
    )

    try:
        print("  Downloading TIGER tract shapefile...")
        r = requests.get(tiger_url, timeout=120)
        r.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
            zf.extractall(os.path.join(OUT, 'tiger_fl'))

        tracts = gpd.read_file(os.path.join(OUT, 'tiger_fl', 'tl_2020_12_tract.shp'))
        tracts = tracts.to_crs('EPSG:5070')   # Albers Equal Area — correct for area calc
        tracts['tract_area_m2'] = tracts.geometry.area
        print(f"  -> {len(tracts):,} census tracts")

    except Exception as e:
        print(f"  TIGER download failed: {e}")
        return None

    # Query NFHL REST API for Florida flood hazard polygons
    # Layer 28 = S_Fld_Haz_Ar (flood hazard areas), filter SFHA_TF = 'T'
    print("\nQuerying FEMA NFHL REST API for Florida SFHA polygons...")
    print("  (This may take a few minutes — large state)")

    nfhl_base = (
        "https://hazards.fema.gov/arcgis/rest/services/public/NFHL/MapServer/28/query"
    )

    # Florida bounding box (approx): xmin=-87.6, ymin=24.5, xmax=-80.0, ymax=31.0
    # NFHL uses EPSG:4326
    all_features = []
    offset = 0
    page_size = 1000

    while True:
        params = {
            'where':          "SFHA_TF='T' AND STATE_NAME='Florida'",
            'outFields':      'FLD_ZONE,SFHA_TF,ZONE_SUBTY',
            'geometryType':   'esriGeometryPolygon',
            'spatialRel':     'esriSpatialRelIntersects',
            'inSR':           '4326',
            'outSR':          '4326',
            'f':              'geojson',
            'resultOffset':    offset,
            'resultRecordCount': page_size,
            'returnGeometry': 'true',
        }
        try:
            r = requests.get(nfhl_base, params=params, timeout=120)
            r.raise_for_status()
            data = r.json()
            features = data.get('features', [])

            if not features:
                break

            all_features.extend(features)
            print(f"  -> Fetched {len(all_features):,} SFHA polygons so far...")

            if len(features) < page_size:
                break
            offset += page_size
            time.sleep(0.3)

        except Exception as e:
            print(f"  NFHL query error at offset {offset}: {e}")
            break

    if not all_features:
        print("  No SFHA features returned, skipping flood zone join")
        return None

    print(f"  Total SFHA polygons: {len(all_features):,}")

    # Build GeoDataFrame
    sfha = gpd.GeoDataFrame.from_features(all_features, crs='EPSG:4326')
    sfha = sfha.to_crs('EPSG:5070')
    sfha['sfha_area_m2'] = sfha.geometry.area

    # Spatial intersection: tracts x SFHA
    print("\nComputing tract-level SFHA coverage (spatial intersection)...")
    intersection = gpd.overlay(tracts[['GEOID','tract_area_m2','geometry']],
                               sfha[['geometry']],
                               how='intersection')
    intersection['intersect_area_m2'] = intersection.geometry.area

    # Aggregate: total SFHA area per tract
    sfha_by_tract = (
        intersection.groupby('GEOID')['intersect_area_m2']
        .sum()
        .reset_index()
        .rename(columns={'intersect_area_m2': 'sfha_area_m2'})
    )

    # Merge back to get pct_in_sfha
    tracts_out = tracts[['GEOID', 'COUNTYFP', 'TRACTCE', 'tract_area_m2']].merge(
        sfha_by_tract, on='GEOID', how='left'
    )
    tracts_out['sfha_area_m2'] = tracts_out['sfha_area_m2'].fillna(0)
    tracts_out['pct_in_sfha'] = (
        tracts_out['sfha_area_m2'] / tracts_out['tract_area_m2'] * 100
    ).clip(0, 100)

    # HMDA uses 11-digit census tract code: state(2) + county(3) + tract(6)
    # TIGER GEOID is already 11-digit
    tracts_out = tracts_out.rename(columns={'GEOID': 'census_tract_geo'})

    return tracts_out[['census_tract_geo', 'COUNTYFP', 'pct_in_sfha']]


# ══════════════════════════════════════════════════════════════════════════════
# 3. PULL, MERGE, CLEAN
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("STEP 1: HMDA Data Pull")
print("=" * 60)

hmda_path = os.path.join(OUT, 'hmda_fl_raw.parquet')

if os.path.exists(hmda_path):
    print(f"  Loading cached HMDA from {hmda_path}")
    hmda = pd.read_parquet(hmda_path)
else:
    hmda = pull_hmda(state='FL', years=[2018, 2019, 2020, 2021, 2022])
    hmda.to_parquet(hmda_path, index=False)
    print(f"  Saved to {hmda_path}")

print(f"\n  Total HMDA rows: {len(hmda):,}")
print(f"  Columns: {hmda.columns.tolist()}")

# ── Recode race to a clean variable ───────────────────────────────────────────
RACE_MAP = {
    '1': 'American Indian or Alaska Native',
    '2': 'Asian',
    '3': 'Black or African American',
    '4': 'Native Hawaiian or Pacific Islander',
    '5': 'White',
    '6': 'Info not provided',
    '7': 'Not applicable',
}

if 'derived_race' in hmda.columns:
    hmda['race_clean'] = hmda['derived_race']
elif 'applicant_race-1' in hmda.columns:
    hmda['race_clean'] = hmda['applicant_race-1'].map(RACE_MAP).fillna('Other')

# ── Recode action_taken ───────────────────────────────────────────────────────
ACTION_MAP = {'1': 'Originated', '2': 'Approved not accepted',
              '3': 'Denied', '4': 'Withdrawn', '5': 'Incomplete',
              '6': 'Purchased', '7': 'Preapproval denied', '8': 'Preapproval approved'}
hmda['action_label'] = hmda['action_taken'].map(ACTION_MAP)
hmda['approved'] = (hmda['action_taken'] == '1').astype(int)

# ── Numeric conversions ────────────────────────────────────────────────────────
for col in ['loan_amount', 'income', 'interest_rate', 'rate_spread',
            'property_value', 'loan_to_value_ratio']:
    if col in hmda.columns:
        hmda[col] = pd.to_numeric(hmda[col], errors='coerce')

# ── Drop rows without census tract ────────────────────────────────────────────
hmda = hmda.dropna(subset=['census_tract'])
hmda['census_tract'] = hmda['census_tract'].astype(str).str.zfill(11)

# ── Quick summary ─────────────────────────────────────────────────────────────
print(f"\n  After cleaning: {len(hmda):,} rows")
print(f"\n  Action taken breakdown:")
print(hmda['action_label'].value_counts().head(8).to_string())

print(f"\n  Race breakdown (top 6):")
print(hmda['race_clean'].value_counts().head(8).to_string())

print(f"\n  Approval rate by race:")
approval = (hmda[hmda['action_taken'].isin(['1','3'])]
            .groupby('race_clean')['approved']
            .agg(['mean', 'count'])
            .rename(columns={'mean': 'approval_rate', 'count': 'n'})
            .sort_values('approval_rate', ascending=False))
print(approval[approval['n'] > 500].to_string())

print(f"\n  Year distribution:")
print(hmda['activity_year'].value_counts().sort_index().to_string())

# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 2: FEMA Flood Zone Data")
print("=" * 60)

fema_path = os.path.join(OUT, 'fema_fl_tracts.parquet')

if os.path.exists(fema_path):
    print(f"  Loading cached FEMA data from {fema_path}")
    fema = pd.read_parquet(fema_path)
else:
    fema = pull_fema_tract_florida()
    if fema is not None:
        fema.to_parquet(fema_path, index=False)
        print(f"  Saved to {fema_path}")

if fema is not None:
    print(f"\n  Florida census tracts: {len(fema):,}")
    print(f"  Tracts with any SFHA coverage: {(fema['pct_in_sfha'] > 0).sum():,}")
    print(f"  Tracts >50% in SFHA: {(fema['pct_in_sfha'] > 50).sum():,}")
    print(f"\n  SFHA coverage distribution:")
    print(pd.cut(fema['pct_in_sfha'],
                 bins=[0, 0.01, 10, 25, 50, 75, 100],
                 labels=['0%', '0-10%', '10-25%', '25-50%', '50-75%', '75-100%']
                 ).value_counts().sort_index().to_string())

    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("STEP 3: Merge HMDA + FEMA")
    print("=" * 60)

    # FEMA GEOID is 11-digit, HMDA census_tract is 11-digit
    fema_merge = fema.rename(columns={'census_tract_geo': 'census_tract'})

    merged = hmda.merge(fema_merge[['census_tract', 'pct_in_sfha']],
                        on='census_tract', how='left')

    print(f"\n  Merged rows: {len(merged):,}")
    print(f"  Rows with flood zone data: {merged['pct_in_sfha'].notna().sum():,}")
    print(f"  Match rate: {merged['pct_in_sfha'].notna().mean():.1%}")

    # Binary treatment: in SFHA (>5% of tract in flood zone)
    merged['in_sfha'] = (merged['pct_in_sfha'] > 5).astype(float)
    merged.loc[merged['pct_in_sfha'].isna(), 'in_sfha'] = np.nan

    print(f"\n  Loans in SFHA tracts: {merged['in_sfha'].sum():,.0f} ({merged['in_sfha'].mean():.1%})")

    # ── Key cross-tab: approval rate by race x flood zone ──────────────────────
    print(f"\n  Approval rate by Race x Flood Zone:")
    key_races = ['White alone', 'Black or African American alone',
                 'White', 'Black or African American']
    key_races = [r for r in key_races if r in merged['race_clean'].unique()][:2]

    if len(key_races) >= 2:
        subset = merged[
            merged['race_clean'].isin(key_races) &
            merged['action_taken'].isin(['1', '3']) &
            merged['in_sfha'].notna()
        ]
        crosstab = (subset.groupby(['race_clean', 'in_sfha'])['approved']
                         .agg(['mean', 'count'])
                         .rename(columns={'mean': 'approval_rate', 'count': 'n'}))
        print(crosstab.to_string())

    # Save merged file
    merged_path = os.path.join(OUT, 'hmda_fema_fl.parquet')
    merged.to_parquet(merged_path, index=False)
    print(f"\n  Saved merged file: {merged_path}")
    print(f"  Shape: {merged.shape}")

else:
    print("  FEMA spatial step skipped — proceeding with HMDA only")
    merged_path = os.path.join(OUT, 'hmda_fl_raw.parquet')

print("\nDone. Files written to:", OUT)
