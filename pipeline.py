import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pystac_client
import rasterio
from rasterio.windows import from_bounds
from shapely.geometry import box
import os

print("--- Starting Satellite Data Pipeline ---")

# 1. Define Area of Interest (San Francisco region)
minx, miny, maxx, maxy = -122.51, 37.70, -122.37, 37.81
bbox = [minx, miny, maxx, maxy]
polygon = box(minx, miny, maxx, maxy)
aoi_gdf = gpd.GeoDataFrame(geometry=[polygon], crs="EPSG:4326")

# 2. Connect to the Satellite Catalog
print("Connecting to AWS Satellite Catalog...")
STAC_URL = "https://earth-search.aws.element84.com/v1"
client = pystac_client.Client.open(STAC_URL)

search = client.search(
    collections=["sentinel-2-l2a"],
    bbox=bbox,
    datetime="2025-06-01/2025-09-30",      
    query={"eo:cloud_cover": {"lt": 10}},   # Clear days
    max_items=1
)

items = list(search.get_items())
if not items:
    raise ValueError("No clear satellite scenes found. Try widening dates.")

best_scene = items[0]
print(f"Found Scene: {best_scene.id} (Cloud Cover: {best_scene.properties['eo:cloud_cover']}%)")

# 3. Match map projections
red_url = best_scene.assets["red"].href
nir_url = best_scene.assets["nir"].href

with rasterio.open(red_url) as sample_src:
    native_crs = sample_src.crs

aoi_gdf_utm = aoi_gdf.to_crs(native_crs)
utm_bounds = aoi_gdf_utm.total_bounds

# 4. Stream data from the cloud
print("Streaming imagery bands (this takes a few seconds)...")
with rasterio.open(red_url) as red_src:
    red_window = from_bounds(*utm_bounds, transform=red_src.transform)
    red_matrix = red_src.read(1, window=red_window).astype(np.float32)

with rasterio.open(nir_url) as nir_src:
    nir_window = from_bounds(*utm_bounds, transform=nir_src.transform)
    nir_matrix = nir_src.read(1, window=nir_window).astype(np.float32)

# Align matrix sizes
min_rows, min_cols = min(red_matrix.shape[0], nir_matrix.shape[0]), min(red_matrix.shape[1], nir_matrix.shape[1])
red_matrix = red_matrix[:min_rows, :min_cols]
nir_matrix = nir_matrix[:min_rows, :min_cols]

# 5. Calculate Vegetation Index (NDVI)
print("Calculating NDVI features...")
np.seterr(divide='ignore', invalid='ignore')
ndvi = (nir_matrix - red_matrix) / (nir_matrix + red_matrix)
ndvi = np.nan_to_num(ndvi, nan=0.0, posinf=1.0, neginf=-1.0)

# 6. Save the outputs locally
print("Saving outputs to your project folder...")
plt.figure(figsize=(10, 10))
plt.imshow(ndvi, cmap="RdYlGn", vmin=-1, vmax=1)
plt.colorbar(label="NDVI Score")
plt.title("My First Geospatial Feature Matrix")
plt.axis("off")

# Save visual image
plt.savefig("vegetation_map.png", dpi=300, bbox_inches='tight')
# Save raw data matrix
np.save("ndvi_matrix.npy", ndvi)

print("🎉 SUCCESS! Check your folder for 'vegetation_map.png' and 'ndvi_matrix.npy'")