import requests
import xml.etree.ElementTree as ET
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.colors import ListedColormap, Normalize
import matplotlib
import datetime
import os

matplotlib.use('Agg')  # Headless mode

# --- Step 1: Find the latest model run ---
wfs_url = (
    "https://opendata.fmi.fi/wfs?service=WFS&version=2.0.0&request=getFeature&"
    "storedquery_id=fmi::forecast::harmonie::surface::grid"
)

response = requests.get(wfs_url, timeout=60)
response.raise_for_status()
tree = ET.fromstring(response.content)

ns = {
    'gml': 'http://www.opengis.net/gml/3.2',
    'omso': 'http://inspire.ec.europa.eu/schemas/omso/3.0',
}

origintimes = []
for elem in tree.findall('.//omso:phenomenonTime//gml:beginPosition', ns):
    text = elem.text
    if text:
        origintimes.append(text)

if not origintimes:
    for elem in tree.findall('.//gml:beginPosition', ns):
        text = elem.text
        if text:
            origintimes.append(text)

if not origintimes:
    raise Exception("No model runs found")

latest_origintime = max(origintimes)
run_time_str = datetime.datetime.strptime(latest_origintime, "%Y-%m-%dT%H:%M:%SZ").strftime("%Y-%m-%d %H:%M UTC")
print(f"Latest model run: {run_time_str}")

# --- Step 2: Download latest data in standard lat/lon ---
download_url = (
    "https://opendata.fmi.fi/download?"
    "producer=harmonie_scandinavia_surface&"
    "param=temperature&"
    "format=netcdf&"
    "bbox=19,56,30,61&"
    "projection=EPSG:4326"
)

print(f"Downloading: {download_url}")
response = requests.get(download_url, timeout=300)
response.raise_for_status()

nc_path = "harmonie.nc"
with open(nc_path, "wb") as f:
    f.write(response.content)

# --- Step 3: Load and convert to °C ---
ds = xr.open_dataset(nc_path)
temp_k = ds['air_temperature_4']
temp_c = temp_k - 273.15

# Calculate min/max for the analysis (time=0)
analysis_temp = temp_c.isel(time=0)
min_temp = float(analysis_temp.min())
max_temp = float(analysis_temp.max())

# --- Step 4: Parse your high-res color ramp from temperature_style.qml ---
qml_path = "temperature_color_table_high.qml"
if not os.path.exists(qml_path):
    raise FileNotFoundError("temperature_style.qml missing – add your new high-res QML file")

tree = ET.parse(qml_path)
root = tree.getroot()

items = []
for item in root.findall(".//item"):
    value = float(item.get('value'))
    color_hex = item.get('color').lstrip('#')
    alpha = int(item.get('alpha', 255)) / 255.0
    r = int(color_hex[0:2], 16) / 255.0
    g = int(color_hex[2:4], 16) / 255.0
    b = int(color_hex[4:6], 16) / 255.0
    items.append((value, (r, g, b, alpha)))

if not items:
    raise Exception("No color items parsed from .qml")

items.sort(key=lambda x: x[0])
values = [i[0] for i in items]
colors = [i[1] for i in items]

cmap = ListedColormap(colors)
norm = Normalize(vmin=-40, vmax=50)  # Matches your QML range

# --- Step 5: Plot with enhanced title ---
fig = plt.figure(figsize=(12, 10))
ax = plt.axes(projection=ccrs.PlateCarree())

temp_c.isel(time=0).plot(
    ax=ax,
    transform=ccrs.PlateCarree(),
    cmap=cmap,
    norm=norm,
    cbar_kwargs={'label': 'Temperature (°C)', 'shrink': 0.8}
)

ax.coastlines(resolution='10m')
ax.gridlines(draw_labels=True)
ax.set_extent([19, 30, 56, 61])

# Enhanced title with run time and min/max
plt.title(
    f"HARMONIE 2m Temperature (°C)\n"
    f"Model run: {run_time_str} | Analysis\n"
    f"Min: {min_temp:.1f}°C | Max: {max_temp:.1f}°C",
    fontsize=14
)

plt.savefig("map.png", dpi=200, bbox_inches='tight')
plt.close()

print(f"Map generated: Min {min_temp:.1f}°C, Max {max_temp:.1f}°C")
