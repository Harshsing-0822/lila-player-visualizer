import streamlit as st
st.set_page_config(layout="wide")

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

st.title("LILA Player Journey Visualization Tool")

DATA_PATH = Path("player_data")
MAP_PATH = DATA_PATH / "minimaps"

# -----------------------------
# Select Date
# -----------------------------
dates = [d.name for d in DATA_PATH.iterdir() if d.is_dir() and "February" in d.name]

selected_date = st.selectbox("Select Date", dates)

date_path = DATA_PATH / selected_date

# -----------------------------
# Load Gameplay Files
# -----------------------------
files = list(date_path.rglob("*.nakama-0"))

matches = sorted({f.name.split("_")[1] for f in files})

selected_match = st.selectbox("Select Match", matches)

match_files = [f for f in files if selected_match in f.name]

selected_files = st.multiselect(
    "Select Players",
    match_files,
    default=match_files[:5]
)

# -----------------------------
# Select Map
# -----------------------------
maps = list(MAP_PATH.glob("*.[pj][pn]g"))

selected_map = st.selectbox("Select Map", [m.name for m in maps])

# -----------------------------
# Load Map and Remove Black Border
# -----------------------------
img = cv2.imread(str(MAP_PATH / selected_map))

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

_, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

coords = np.column_stack(np.where(mask > 0))

y_min, x_min = coords.min(axis=0)
y_max, x_max = coords.max(axis=0)

img_crop = img[y_min:y_max, x_min:x_max]

map_height, map_width = img_crop.shape[:2]

map_image = Image.fromarray(cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB))

# -----------------------------
# Load Player Data
# -----------------------------
dfs = []

for f in selected_files:

    df = pd.read_parquet(f)

    df["player"] = f.name

    if "event" in df.columns:
        df["event"] = df["event"].astype(str)

    dfs.append(df)

if not dfs:
    st.warning("No players selected")
    st.stop()

df_all = pd.concat(dfs)

# -----------------------------
# Timeline Playback
# -----------------------------
if "ts" in df_all.columns:

    df_all["ts"] = pd.to_datetime(df_all["ts"])

    selected_time = st.slider(
        "Match Timeline",
        df_all["ts"].min().to_pydatetime(),
        df_all["ts"].max().to_pydatetime(),
        df_all["ts"].max().to_pydatetime()
    )

    df_all = df_all[df_all["ts"] <= selected_time]

# -----------------------------
# Convert World → Map Pixels
# -----------------------------
WORLD_MIN_X = df_all["x"].min()
WORLD_MAX_X = df_all["x"].max()

WORLD_MIN_Y = df_all["y"].min()
WORLD_MAX_Y = df_all["y"].max()

df_all["x_map"] = (df_all["x"] - WORLD_MIN_X) / (WORLD_MAX_X - WORLD_MIN_X) * map_width
df_all["y_map"] = map_height - (
    (df_all["y"] - WORLD_MIN_Y) /
    (WORLD_MAX_Y - WORLD_MIN_Y)
) * map_height

# -----------------------------
# Plot Player Movement
# -----------------------------
fig = go.Figure()

for player, d in df_all.groupby("player"):

    fig.add_trace(
        go.Scatter(
            x=d["x_map"],
            y=d["y_map"],
            mode="lines",
            name=player
        )
    )

# Event markers
fig.add_trace(
    go.Scatter(
        x=df_all["x_map"],
        y=df_all["y_map"],
        mode="markers",
        marker=dict(size=6, color="red"),
        text=df_all.get("event"),
        name="Events"
    )
)

# -----------------------------
# Add Map Background
# -----------------------------
fig.update_layout(

    images=[
        dict(
            source=map_image,
            xref="x",
            yref="y",
            x=0,
            y=map_height,     # anchor at top-left
            sizex=map_width,
            sizey=map_height,
            sizing="stretch",
            layer="below"
        )
    ],

    xaxis=dict(
        range=[0, map_width],
        showgrid=False,
        zeroline=False,
        constrain="domain",
        fixedrange=True
    ),

    yaxis=dict(
        range=[0, map_height],   # normal orientation
        showgrid=False,
        scaleanchor="x",
        zeroline=False,
        constrain="domain",
        fixedrange=True
    ),

    margin=dict(l=0, r=0, t=40, b=0)
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Heatmap
# -----------------------------
st.subheader("Player Activity Heatmap")

heatmap = px.density_heatmap(
    df_all,
    x="x_map",
    y="y_map",
    nbinsx=60,
    nbinsy=60
)

st.plotly_chart(heatmap, use_container_width=True)

# -----------------------------
# Data Preview
# -----------------------------
st.subheader("Raw Telemetry Data")

st.dataframe(df_all.head(20))