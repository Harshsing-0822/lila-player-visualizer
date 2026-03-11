import streamlit as st
st.set_page_config(layout="wide")

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

# ============================================
# TITLE
# ============================================

st.title("🎮 LILA Player Journey Visualization Tool")

# ============================================
# ASSUMPTIONS PANEL
# ============================================

with st.expander("📘 Assumptions & Data Interpretation"):

    st.markdown("""
### Data Assumptions Used in This Visualization Tool

1. **World Coordinates → Minimap Mapping**  
Player telemetry coordinates (`x`,`y`) represent world positions.  
These are mapped to minimap pixels using **min-max normalization**.

2. **Minimap Orientation**  
Game world origin assumed bottom-left while image origin is top-left.  
Therefore **Y-axis is inverted** during mapping.

3. **Minimap Cropping**  
Minimap images contain black borders.  
The tool automatically crops the playable region.

4. **Player Identification**  
Each telemetry file corresponds to **one player in a match**.

5. **Bot Detection**  
Players containing `"bot"` in their name are classified as bots.

6. **Event Encoding**  
Event fields are sometimes stored as **byte strings**, so they are decoded.

7. **Timeline Interpretation**  
Events are visualized chronologically using timestamp `ts`.

8. **Heatmap Meaning**  
Heatmap shows **density of player presence or events**.
""")

# ============================================
# PATHS
# ============================================

DATA_PATH = Path("player_data")
MAP_PATH = DATA_PATH / "minimaps"

# ============================================
# CACHE
# ============================================

@st.cache_data
def load_parquet(file):
    return pd.read_parquet(file)

# ============================================
# SIDEBAR CONTROLS
# ============================================

st.sidebar.header("Controls")

dates = sorted([d.name for d in DATA_PATH.iterdir() if d.is_dir()])
selected_date = st.sidebar.selectbox("Select Date", dates)

date_path = DATA_PATH / selected_date

# ============================================
# MATCH FILES
# ============================================

files = list(date_path.rglob("*.nakama-0"))

match_dict = {}

for f in files:

    parts = f.stem.split("_")

    if len(parts) > 1:

        match_id = parts[1]

        match_dict.setdefault(match_id, []).append(f)

matches = sorted(match_dict.keys())

selected_match = st.sidebar.selectbox("Select Match", matches)

match_files = match_dict[selected_match]

# ============================================
# PLAYER SELECTOR
# ============================================

player_map = {f.stem: f for f in match_files}

selected_players = st.sidebar.multiselect(
    "Select Players",
    list(player_map.keys()),
    default=list(player_map.keys())[:3]
)

selected_files = [player_map[p] for p in selected_players]

# ============================================
# MAP SELECT
# ============================================

maps = list(MAP_PATH.glob("*.[pj][pn]g"))

selected_map = st.sidebar.selectbox(
    "Select Map",
    [m.name for m in maps]
)

# ============================================
# LOAD MAP
# ============================================

def load_map(map_file):

    img = cv2.imread(str(map_file))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    coords = np.column_stack(np.where(mask > 0))

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    img_crop = img[y_min:y_max, x_min:x_max]

    map_height, map_width = img_crop.shape[:2]

    map_image = Image.fromarray(cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB))

    return map_image, map_width, map_height


map_image, map_width, map_height = load_map(MAP_PATH / selected_map)

# ============================================
# LOAD PLAYER DATA
# ============================================

dfs = []

for f in selected_files:

    df = load_parquet(f)

    df["player"] = f.stem

    # decode byte events
    if "event" in df.columns:

        df["event"] = df["event"].apply(
            lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
        )

    dfs.append(df)

if not dfs:
    st.warning("Select players")
    st.stop()

df_all = pd.concat(dfs)

# ============================================
# TIMELINE
# ============================================

if "ts" in df_all.columns:

    df_all["ts"] = pd.to_datetime(df_all["ts"])

    selected_time = st.sidebar.slider(
        "Timeline",
        df_all["ts"].min().to_pydatetime(),
        df_all["ts"].max().to_pydatetime(),
        df_all["ts"].max().to_pydatetime()
    )

    df_all = df_all[df_all["ts"] <= selected_time]

# ============================================
# WORLD → MAP
# ============================================

WORLD_MIN_X = df_all["x"].min()
WORLD_MAX_X = df_all["x"].max()

WORLD_MIN_Y = df_all["y"].min()
WORLD_MAX_Y = df_all["y"].max()

df_all["x_map"] = (df_all["x"] - WORLD_MIN_X) / (WORLD_MAX_X - WORLD_MIN_X) * map_width

df_all["y_map"] = map_height - (
    (df_all["y"] - WORLD_MIN_Y) /
    (WORLD_MAX_Y - WORLD_MIN_Y)
) * map_height

# ============================================
# EVENT COLORS (RESTORED GOOD FEATURE)
# ============================================

event_colors = {}

if "event" in df_all.columns:

    unique_events = df_all["event"].dropna().unique()

    palette = px.colors.qualitative.Bold

    event_colors = {
        event: palette[i % len(palette)]
        for i, event in enumerate(unique_events)
    }

# ============================================
# MAP VISUALIZATION
# ============================================

st.subheader("🗺 Player Journey Map")

fig = go.Figure()

player_colors = px.colors.qualitative.Plotly

for i, (player, d) in enumerate(df_all.groupby("player")):

    fig.add_trace(
        go.Scatter(
            x=d["x_map"],
            y=d["y_map"],
            mode="lines",
            name=player,
            line=dict(width=3, color=player_colors[i % len(player_colors)])
        )
    )

# EVENTS

if "event" in df_all.columns:

    for event_type, d in df_all.groupby("event"):

        fig.add_trace(
            go.Scatter(
                x=d["x_map"],
                y=d["y_map"],
                mode="markers",
                name=str(event_type),
                marker=dict(
                    size=8,
                    color=event_colors[event_type]
                )
            )
        )

fig.update_layout(

    images=[
        dict(
            source=map_image,
            xref="x",
            yref="y",
            x=0,
            y=map_height,
            sizex=map_width,
            sizey=map_height,
            sizing="stretch",
            layer="below"
        )
    ],

    xaxis=dict(range=[0, map_width], visible=False),
    yaxis=dict(range=[0, map_height], scaleanchor="x", visible=False),

    margin=dict(l=0, r=0, t=30, b=0)
)

st.plotly_chart(fig, use_container_width=True)

# ============================================
# HEATMAP
# ============================================

st.subheader("🔥 Player Activity Heatmap")

heatmap = px.density_heatmap(
    df_all,
    x="x_map",
    y="y_map",
    nbinsx=60,
    nbinsy=60
)

st.plotly_chart(heatmap, use_container_width=True)

# ============================================
# TELEMETRY TABLE
# ============================================

st.subheader("Telemetry Data")

st.dataframe(df_all.head(50))