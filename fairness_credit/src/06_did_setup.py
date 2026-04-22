"""
DiD Setup: Pull FEMA LOMC remapping events for Florida.

The DiD treatment variable: a census tract that was remapped INTO a SFHA
between 2018 and 2022. Before the remapping, lenders treat it as low-risk.
After, it has mandatory flood insurance requirements.

Data source: FEMA LOMC (Letter of Map Change) batch files
  https://msc.fema.gov/portal/resources/lomc

LOMC types:
  LOMR (Letter of Map Revision)  — remaps communities, can add/remove SFHAs
  LOMA (Letter of Map Amendment) — removes specific properties from SFHA
  LOMR-F (LOMR based on fill)    — removes properties after fill

For our DiD we want LOMRs that ADD properties to SFHA (most restrictive).
These become effective 6 months after the Letter of Final Determination (LFD).

Alternative approach (faster): FEMA publishes the effective date of the
current FIRM for each flood zone polygon in the NFHL (EFF_DATE field).
We can compare two NFHL snapshots OR use the NFHL metadata to find panels
updated within our study window.

This script:
1. Downloads FEMA LOMC index files (ZIP) for 2017-2022
2. Parses Florida LOMRs (community FIPS starts with 12)
3. Identifies communities with map effective dates 2018-2022
4. Joins to census tract geography to create treatment timing variable
"""

import os, io, zipfile, requests, time
import pandas as pd
import numpy as np
import re

OUT = os.path.dirname(os.path.abspath(__file__))

# ══════════════════════════════════════════════════════════════════════════════
# 1. DOWNLOAD FEMA LOMC BATCH FILES
#    Index files contain metadata on all LOMCs by year
#    URL: https://msc.fema.gov/portal/resources/lomc
# ══════════════════════════════════════════════════════════════════════════════

LOMC_BASE = "https://hazards.fema.gov/femaportal/docs/searchresultsmap"

def try_lomc_index(year):
    """
    Try to get LOMC index data for a given year.
    FEMA doesn't expose a clean bulk API so we try several known URL patterns.
    """
    # Pattern 1: Annual summary CSV (sometimes published by FEMA)
    urls = [
        f"https://msc.fema.gov/portal/resources/lomc/LOMC_{year}_Index.zip",
        f"https://hazards.fema.gov/femaportal/wps/PA_WPSPortalWebApp/resources/lomc/lomc_index_{year}.zip",
    ]
    for url in urls:
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200:
                return r.content
        except:
            pass
    return None

# ── Alternative: Use NFHL EFF_DATE approach ───────────────────────────────────
# The NFHL REST API layer 28 has EFF_DATE for each polygon.
# We can query for Florida polygons with EFF_DATE between 2018-01-01 and 2022-12-31
# to find tracts that were newly mapped into SFHA during our study window.

print("Querying NFHL for recently remapped Florida SFHA polygons (2018-2022)...")
print("(Using EFF_DATE field to identify new map additions)\n")

NFHL_URL = "https://hazards.fema.gov/arcgis/rest/services/public/NFHL/MapServer/28/query"

# Check available fields first
r = requests.get(NFHL_URL.replace('/query', ''), params={'f': 'json'}, timeout=30)
fields = [f['name'] for f in r.json().get('fields', [])]
print(f"Available fields: {fields}\n")

# EFF_DATE might be available — let's check a sample record
test = requests.get(NFHL_URL, params={
    'where': "SFHA_TF='T'",
    'geometry': '-80.5,25.5,-80.0,25.9',
    'geometryType': 'esriGeometryEnvelope',
    'inSR': '4326', 'outSR': '4326',
    'outFields': '*',   # get all fields
    'f': 'geojson',
    'resultRecordCount': 2,
}, timeout=30)
sample = test.json().get('features', [])
if sample:
    print("Sample NFHL record fields:")
    for k, v in sample[0]['properties'].items():
        print(f"  {k}: {v}")
print()

# ══════════════════════════════════════════════════════════════════════════════
# 2. FEMA FIRM PANEL LOOKUP
#    Alternative: use FEMA's FIRM Panel lookup to find updated panels
#    Layer 3 in NFHL = FIRM panels with effective dates
# ══════════════════════════════════════════════════════════════════════════════

print("Checking NFHL Layer 3 (FIRM Panels) for effective dates...")
firm_url = "https://hazards.fema.gov/arcgis/rest/services/public/NFHL/MapServer/3/query"

# Test query on Miami-Dade
test2 = requests.get(firm_url, params={
    'where': '1=1',
    'geometry': '-80.5,25.5,-80.0,25.9',
    'geometryType': 'esriGeometryEnvelope',
    'inSR': '4326',
    'outFields': '*',
    'f': 'json',
    'resultRecordCount': 3,
}, timeout=30)
d2 = test2.json()
if d2.get('features'):
    print("FIRM Panel fields:")
    for k, v in d2['features'][0]['attributes'].items():
        print(f"  {k}: {v}")
else:
    print("No FIRM panel features returned:", d2.get('error','unknown'))

# ══════════════════════════════════════════════════════════════════════════════
# 3. FEMA NATIONAL FLOOD HAZARD LAYER — CHANGE LOOKUP via DFIRM_ID
#    Each NFHL polygon has a DFIRM_ID encoding state+county+panel number.
#    We use this to build a community-level remapping timeline.
# ══════════════════════════════════════════════════════════════════════════════

print("\nQuerying NFHL effective dates for all Florida SFHA polygons (sample)...")

# Get DFIRM_ID and SOURCE_CIT fields for Florida (bounding box approach, sample)
fl_bbox = '-87.65,24.4,-80.0,31.0'

test3 = requests.get(NFHL_URL, params={
    'where': "SFHA_TF='T'",
    'geometry': fl_bbox,
    'geometryType': 'esriGeometryEnvelope',
    'inSR': '4326',
    'outFields': 'DFIRM_ID,FLD_ZONE,SFHA_TF,SOURCE_CIT',
    'returnGeometry': 'false',
    'f': 'json',
    'resultRecordCount': 5,
}, timeout=30)
d3 = test3.json()
if d3.get('features'):
    print("Sample NFHL records with DFIRM_ID:")
    for feat in d3['features']:
        print(f"  {feat['attributes']}")
else:
    print("Error:", d3.get('error', 'no features'))

# ══════════════════════════════════════════════════════════════════════════════
# 4. USE FEMA FLOOD MAP CHANGES API (MSC)
#    https://msc.fema.gov/portal/advanceSearch — supports download by FIRM panel
#    We use the FEMA Resilience Analysis and Planning Tool (RAPT) data instead
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("ALTERNATIVE: Construct DiD from HMDA data itself")
print("="*60)
print("""
Since direct LOMC bulk data isn't easily accessible programmatically,
the most tractable DiD approach uses what we already have:

  TREATMENT PROXY: Tracts with high SFHA coverage (>25%)
  PRE/POST: Application years 2018-2019 (pre) vs. 2021-2022 (post)
  'EVENT': Hurricane Irma (September 2017) + subsequent FEMA remapping
            of Florida flood maps, which were widely updated 2019-2020

Hurricane Irma made landfall in Florida on September 10, 2017.
FEMA subsequently updated FIRMs across many Florida counties.
The new maps became effective in many counties in 2018-2020.

This gives us a staggered DiD:
  - Counties with post-Irma FIRM updates (treatment, various timing)
  - Counties without updates in this period (control)
  - Outcome: mortgage terms pre/post update

Let's check the MSC for known Florida FIRM update dates.
""")

# ── FEMA MSC Map Status API ───────────────────────────────────────────────────
# FEMA publishes community FIRM status including effective dates
print("Fetching Florida FIRM effective dates from FEMA National Flood Insurance Program...")

# FEMA NFIP community file — community-level FIRM dates
# Available as CSV from FEMA open data portal
community_url = "https://www.fema.gov/api/open/v2/datastore/query?entity=nfipCommunityDatabase&format=csv&limit=1000&filter=stateName%3AFLORIDA"

try:
    r = requests.get(community_url, timeout=30)
    if r.status_code == 200:
        from io import StringIO
        comm_df = pd.read_csv(StringIO(r.text))
        print(f"Community database columns: {comm_df.columns.tolist()[:10]}")
        print(f"Records: {len(comm_df)}")
        if len(comm_df) > 0:
            print(comm_df.head(3).to_string())
    else:
        print(f"Status {r.status_code}")
except Exception as e:
    print(f"Community API error: {e}")

# ── Try FEMA OpenFEMA datasets ────────────────────────────────────────────────
print("\nTrying OpenFEMA NFIP Flood Map Changes dataset...")
openfema_url = "https://www.fema.gov/api/open/v2/datastore/query"
params = {
    'entity':  'nfipFloodMapChanges',
    'format':  'csv',
    'limit':   1000,
    'filter':  'state:FL',
}
try:
    r2 = requests.get(openfema_url, params=params, timeout=30)
    print(f"Status: {r2.status_code}")
    if r2.status_code == 200:
        from io import StringIO
        fmc = pd.read_csv(StringIO(r2.text))
        print(f"Columns: {fmc.columns.tolist()}")
        print(f"Records: {len(fmc)}")
        if len(fmc) > 0:
            print(fmc.head(3).to_string())
except Exception as e:
    print(f"OpenFEMA error: {e}")

# ── Try OpenFEMA disaster declarations (Hurricane Irma = DR-4337) ─────────────
print("\nFetching Hurricane Irma (DR-4337) affected areas...")
irma_url = "https://www.fema.gov/api/open/v2/datastore/query"
try:
    r3 = requests.get(irma_url, params={
        'entity':  'disasterDeclarationsSummaries',
        'format':  'csv',
        'filter':  'disasterNumber:4337,state:FL',
        'limit':   500,
    }, timeout=30)
    if r3.status_code == 200:
        from io import StringIO
        irma = pd.read_csv(StringIO(r3.text))
        print(f"Irma FL counties: {len(irma)}")
        print(f"Columns: {irma.columns.tolist()}")
        if len(irma) > 0:
            # Get declared counties — these are our treatment group
            if 'designatedArea' in irma.columns:
                print("Designated areas:", irma['designatedArea'].unique()[:10])
            if 'fipsCountyCode' in irma.columns:
                print("FIPS codes:", irma['fipsCountyCode'].unique()[:10].tolist())
            irma_path = os.path.join(OUT, 'irma_fl_counties.csv')
            irma.to_csv(irma_path, index=False)
            print(f"Saved: {irma_path}")
    else:
        print(f"Status {r3.status_code}: {r3.text[:200]}")
except Exception as e:
    print(f"Irma query error: {e}")
