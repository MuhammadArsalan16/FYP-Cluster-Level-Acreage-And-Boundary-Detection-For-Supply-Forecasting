import os
import math
import cv2
import numpy as np
import pandas as pd
import geopandas as gpd
import requests
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt

BASE_DIR = "/home/muhammadarsalan/Desktop/FYP"
DATASET_DIR = os.path.join(BASE_DIR, "Dataset")
TEST_GPKG = os.path.join(DATASET_DIR, "namal_calibration_test.gpkg")
OUT_DIR = os.path.join(DATASET_DIR, "calibration_results")
os.makedirs(OUT_DIR, exist_ok=True)

if not os.path.exists(TEST_GPKG):
    raise FileNotFoundError(f"File not found: {TEST_GPKG}. Save your test polygons in QGIS first.")

gdf_wgs84 = gpd.read_file(TEST_GPKG).to_crs(epsg=4326)
gdf_3857 = gdf_wgs84.to_crs(epsg=3857)
utm_crs = gdf_wgs84.estimate_utm_crs()
gdf_metric = gdf_wgs84.to_crs(utm_crs)

print(f"Loaded {len(gdf_wgs84)} calibration polygons | Local Metric CRS: {utm_crs}")

ORIGIN_SHIFT = 20037508.342789244

def latlon_to_tile(lat, lon, zoom=18):
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    xtile = int((lon + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.log(math.tan(lat_rad) + (1.0 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
    return xtile, ytile

def tile_bounds_3857(xtile, ytile, zoom=18):
    tile_span = (2 * ORIGIN_SHIFT) / (2 ** zoom)
    min_x = -ORIGIN_SHIFT + xtile * tile_span
    max_x = min_x + tile_span
    max_y = ORIGIN_SHIFT - ytile * tile_span
    min_y = max_y - tile_span
    return min_x, min_y, max_x, max_y

def fetch_tile(xtile, ytile, zoom=18):
    url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{zoom}/{ytile}/{xtile}"
    res = requests.get(url, headers={"User-Agent": "VGreenCalibration/1.0"}, timeout=10)
    res.raise_for_status()
    return np.array(Image.open(BytesIO(res.content)).convert("RGB"))

def super_resolve_4x(img):
    h, w, _ = img.shape
    up = cv2.resize(img, (w * 4, h * 4), interpolation=cv2.INTER_CUBIC)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(up, -1, kernel)

results = []

for idx, row in gdf_wgs84.iterrows():
    f_id = row.get("field_id", f"TEST-{idx+1:02d}")
    exp_type = row.get("expected_type", "Ground")
    centroid = row["geometry"].centroid

    geom_m = gdf_metric.iloc[idx]["geometry"]
    area_sqm = geom_m.area
    perimeter_m = geom_m.length
    area_acres = area_sqm * 0.000247105
    area_kanals = area_sqm / 505.857

    xt, yt = latlon_to_tile(centroid.y, centroid.x, zoom=18)
    raw_tile = fetch_tile(xt, yt, zoom=18)
    sr_tile = super_resolve_4x(raw_tile)
    H, W, _ = sr_tile.shape

    min_x, min_y, max_x, max_y = tile_bounds_3857(xt, yt, zoom=18)
    poly_3857 = gdf_3857.iloc[idx]["geometry"]
    if poly_3857.geom_type == "MultiPolygon":
        poly_3857 = max(poly_3857.geoms, key=lambda a: a.area)
    exterior_coords = np.array(poly_3857.exterior.coords)

    px_x = ((exterior_coords[:, 0] - min_x) / (max_x - min_x) * W).astype(np.int32)
    px_y = ((max_y - exterior_coords[:, 1]) / (max_y - min_y) * H).astype(np.int32)
    pts = np.stack([px_x, px_y], axis=1)

    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    inside_pixels = sr_tile[mask == 255].astype(np.float32) + 1e-6

    if len(inside_pixels) > 0:
        R, G, B = inside_pixels[:, 0], inside_pixels[:, 1], inside_pixels[:, 2]
        gvi = float(np.mean((G - R) / (G + R)))
        exg = float(np.mean(2 * G - R - B))
    else:
        gvi, exg = 0.0, 0.0

    if gvi < 0.02 and exg < 5.0:
        detected_status = "Discarded (Built-up / Bare)"
        color = "#EF4444"
    else:
        detected_status = "Active Vegetated Turf"
        color = "#10B981"

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(sr_tile)
    ax.plot(px_x, px_y, color=color, linewidth=2.5, label="QGIS Traced Boundary")
    ax.set_title(
        f"{f_id} ({exp_type})\n"
        f"Area: {area_sqm:,.1f} m² ({area_kanals:.2f} Kanals | {area_acres:.2f} Ac)\n"
        f"Status: {detected_status} (GVI: {gvi:.4f})",
        fontsize=9, fontweight="bold"
    )
    ax.axis("off")
    img_out = os.path.join(OUT_DIR, f"{f_id}_calibration_check.png")
    plt.savefig(img_out, dpi=250, bbox_inches="tight")
    plt.close()

    results.append({
        "Field_ID": f_id,
        "Expected": exp_type,
        "Area_SqM": round(area_sqm, 1),
        "Perimeter_M": round(perimeter_m, 1),
        "Kanals": round(area_kanals, 2),
        "Acres": round(area_acres, 3),
        "Inside_GVI": round(gvi, 4),
        "Inside_ExG": round(exg, 2),
        "Pipeline_Verdict": detected_status
    })

df_res = pd.DataFrame(results)
print("\n" + "="*80)
print("NAMAL & CHAGDA BOUNDARY CALIBRATION REPORT")
print("="*80)
print(df_res.to_string(index=False))
print("="*80)
print(f"Visual Alignment Overlays saved to: {OUT_DIR}")
