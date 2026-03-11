import streamlit as st
st.set_page_config(layout="wide")

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os
from PIL import Image

st.title("LILA Player Journey Visualization Tool")

DATA_PATH = "player_data"
MAP_PATH = os.path.join(DATA_PATH, "minimaps")

# -----------------------------
# Select Date
# -----------------------------
dates = [
    d for d in os.listdir(DATA_PATH)
    if os.path.isdir(os.path.join(DATA_PATH, d)) and "February" in d
]

selected_date = st.selectbox("Select Date", dates)

date_path = os.path.join(DATA_PATH, selected_date)

# -----------------------------
# Load Gameplay Files
# -----------------------------
files = []

for root, dirs, filenames in os.walk(date_path):
    for file in filenames:
        if file.endswith(".nakama-0"):
            files.append(os.path.join(root, file))

# -----------------------------
# Match Filter
# -----------------------------
matches = list(set([os.path.basename(f).split("_")[1] for f in files]))

selected_match = st.selectbox("Select Match", matches)

match_files = [f for f in files if selected_match in f]

selected_files = st.multiselect(
    "Select Players",
    match_files,
    default=match_files[:5] if len(match_files) > 5 else match_files
)

# -----------------------------
# Select Map
# -----------------------------
maps = [m for m in os.listdir(MAP_PATH) if m.endswith(("png","jpg"))]

selected_map = st.selectbox("Select Map", maps)

map_image = Image.open(os.path.join(MAP_PATH, selected_map))

# -----------------------------
# Load Player Data
# -----------------------------
dataframes = []

for file in selected_files:

    df = pd.read_parquet(file)

    if "event" in df.columns:
        df["event"] = df["event"].astype(str)

    df["player"] = os.path.basename(file)

    dataframes.append(df)

if len(dataframes) == 0:
    st.warning("No players selected")
    st.stop()

df_all = pd.concat(dataframes)

# -----------------------------
# Timeline Playback
# -----------------------------
if "ts" in df_all.columns:

    df_all["ts"] = pd.to_datetime(df_all["ts"])

    min_time = df_all["ts"].min().to_pydatetime()
    max_time = df_all["ts"].max().to_pydatetime()

    selected_time = st.slider(
        "Match Timeline",
        min_value=min_time,
        max_value=max_time,
        value=max_time
    )

    df_all = df_all[df_all["ts"] <= selected_time]

# -----------------------------
# Normalize Coordinates
# -----------------------------
min_x = df_all["x"].min()
max_x = df_all["x"].max()

min_y = df_all["y"].min()
max_y = df_all["y"].max()

df_all["x_norm"] = (df_all["x"] - min_x) / (max_x - min_x)
df_all["y_norm"] = (df_all["y"] - min_y) / (max_y - min_y)

# Flip Y axis
df_all["y_norm"] = 1 - df_all["y_norm"]

# Clamp coordinates inside the map
df_all["x_norm"] = df_all["x_norm"].clip(0, 1)
df_all["y_norm"] = df_all["y_norm"].clip(0, 1)

# -----------------------------
# Plot Player Movement
# -----------------------------
fig = go.Figure()

for player in df_all["player"].unique():

    df_player = df_all[df_all["player"] == player]

    fig.add_trace(
        go.Scatter(
            x=df_player["x_norm"],
            y=df_player["y_norm"],
            mode="lines",
            name=player
        )
    )

# Event markers
fig.add_trace(
    go.Scatter(
        x=df_all["x_norm"],
        y=df_all["y_norm"],
        mode="markers",
        marker=dict(size=6, color="red"),
        text=df_all["event"],
        name="Events"
    )
)

# -----------------------------
# Add Map Background
# -----------------------------
fig.update_layout(
    title="Player Movement Paths",

    images=[
        dict(
            source=map_image,
            xref="x",
            yref="y",
            x=0,
            y=1,
            sizex=1,
            sizey=1,
            sizing="stretch",
            opacity=0.7,
            layer="below"
        )
    ],

    xaxis=dict(
        range=[0,1],
        showgrid=False,
        scaleanchor="y",   # IMPORTANT: lock aspect ratio
    ),

    yaxis=dict(
        range=[0,1],
        showgrid=False
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
    x="x_norm",
    y="y_norm",
    nbinsx=50,
    nbinsy=50
)

st.plotly_chart(heatmap, use_container_width=True)

# -----------------------------
# Data Preview
# -----------------------------
st.subheader("Raw Telemetry Data")

st.dataframe(df_all.head(20))

