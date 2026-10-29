import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import importlib.util
import sys
import math
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Streamlit Page Config
st.set_page_config(
    page_title="Urban Autonomous Driving & City Map Navigation",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent.parent

# Import urban map engine
spec = importlib.util.spec_from_file_location("urban_engine", BASE_DIR / "code" / "11_urban_map_engine.py")
urban_engine = importlib.util.module_from_spec(spec)
sys.modules["urban_engine"] = urban_engine
spec.loader.exec_module(urban_engine)

UrbanCityGraph = urban_engine.UrbanCityGraph
UrbanAdaptivePlanner = urban_engine.UrbanAdaptivePlanner
EKFLocalization = urban_engine.EKFLocalization

# Custom Dark Theme Styling
st.markdown("""
<style>
    .main { background-color: #12141C; color: #FFFFFF; }
    .header-box {
        background: linear-gradient(135deg, #1F2430 0%, #2D3446 100%);
        padding: 20px;
        border-radius: 12px;
        border-left: 6px solid #7C4DFF;
        margin-bottom: 20px;
    }
    .metric-box {
        background-color: #1A1D27;
        border: 1px solid #2D3446;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Header Title
st.markdown("""
<div class="header-box">
    <h1 style='margin:0; color:#B388FF;'>Real-World Urban Driving Region Navigation System</h1>
    <p style='margin:5px 0 0 0; color:#E0E0E0; font-size:1.1rem;'>
        Roundabouts | Traffic Signal Control | Weak GPS Dead-Reckoning | Adaptive Risk-Aware ML-A*
    </p>
</div>
""", unsafe_allow_html=True)

# Initialize Session State
if "urban_graph" not in st.session_state:
    st.session_state.urban_graph = UrbanCityGraph()
    st.session_state.urban_planner = UrbanAdaptivePlanner(st.session_state.urban_graph)
    st.session_state.ekf = EKFLocalization()

graph = st.session_state.urban_graph
planner = st.session_state.urban_planner
ekf = st.session_state.ekf

# Sidebar Options
st.sidebar.header("⚙️ Urban Environment Controls")

st.sidebar.subheader("🚥 Traffic Light Signal States")
tl_ader_state = st.sidebar.radio("Rue Clément Ader Traffic Light:", ["RED", "GREEN"], index=0)
tl_sadi_state = st.sidebar.radio("Avenue Sadi Lecointe Traffic Light:", ["RED", "GREEN"], index=1)

# Update traffic light states in graph
graph.edges[("R_ROUNDABOUT", "TL_ADER")]["signal_state"] = tl_ader_state
graph.edges[("TL_ADER", "TL_SADI")]["signal_state"] = tl_sadi_state

st.sidebar.subheader("📡 GPS Signal Conditions")
weak_gps_active = st.sidebar.checkbox("Enable Weak GPS Zone Interference", value=True)
graph.edges[("GPS_START", "GPS_MID")]["gps_noise_std"] = 15.0 if weak_gps_active else 0.5
graph.edges[("GPS_MID", "GPS_END")]["gps_noise_std"] = 15.0 if weak_gps_active else 0.5

st.sidebar.subheader("🚨 Urban Bottlenecks & Hazards")
roundabout_jam = st.sidebar.checkbox("Simulate Heavy Congestion at Roundabout", value=False)
graph.edges[("R_ROUNDABOUT", "TL_ADER")]["density"] = 0.85 if roundabout_jam else 0.35

st.sidebar.subheader("📍 Vehicle Journey")
nodes_keys = list(graph.nodes.keys())
start_node = st.sidebar.selectbox("Start Point:", nodes_keys, index=0)
goal_node = st.sidebar.selectbox("Destination:", nodes_keys, index=len(nodes_keys)-1)

# Compute Optimal Path
path = planner.run_urban_ml_astar(start_node, goal_node)
metrics = planner.evaluate_urban_path(path)

# Layout Columns
col_map, col_info = st.columns([7, 4])

with col_map:
    st.subheader("🏙️ Interactive City Map (Driving Region Visualizer)")

    fig, ax = plt.subplots(figsize=(11, 7.5), facecolor='#E8ECEF')
    ax.set_facecolor('#E8ECEF')

    # Draw Map Roads & Avenues
    for (u, v), edge in graph.edges.items():
        x1, y1 = graph.nodes[u]["x"], graph.nodes[u]["y"]
        x2, y2 = graph.nodes[v]["x"], graph.nodes[v]["y"]

        if edge["is_weak_gps"]:
            # Blue highlighted line for Weak GPS corridor (matches user reference image!)
            ax.plot([x1, x2], [y1, y2], color='#1E88E5', linewidth=8, alpha=0.8, zorder=2)
            ax.plot([x1, x2], [y1, y2], color='#90CAF9', linewidth=3, linestyle='--', zorder=3)
        else:
            # Pink / Cream Urban Road Style (matches user image!)
            ax.plot([x1, x2], [y1, y2], color='#EF9A9A', linewidth=7, alpha=0.9, zorder=2)
            ax.plot([x1, x2], [y1, y2], color='#FFFFFF', linewidth=2.5, linestyle='-', zorder=3)

    # Draw Highlighted Calculated Route
    if path and len(path) >= 2:
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            x1, y1 = graph.nodes[u]["x"], graph.nodes[u]["y"]
            x2, y2 = graph.nodes[v]["x"], graph.nodes[v]["y"]
            ax.plot([x1, x2], [y1, y2], color='#7C4DFF', linewidth=5.5, zorder=4, alpha=0.9)

    # Draw Roundabout (Purple Circle as shown in reference image!)
    r_x, r_y = graph.nodes["R_ROUNDABOUT"]["x"], graph.nodes["R_ROUNDABOUT"]["y"]
    roundabout_circle = patches.Circle((r_x, r_y), radius=22, facecolor='#D1C4E9', edgecolor='#4A148C', linewidth=3, zorder=5)
    ax.add_patch(roundabout_circle)
    ax.text(r_x - 70, r_y + 35, "Roundabout", color='#5E35B1', fontsize=12, fontweight='bold', zorder=7)
    ax.annotate("", xy=(r_x, r_y + 24), xytext=(r_x - 30, r_y + 40),
                arrowprops=dict(arrowstyle="->", color='#5E35B1', lw=2), zorder=7)

    # Draw Traffic Lights (White Circles as shown in reference image!)
    for tl_id, name in [("TL_ADER", "Rue Clément Ader"), ("TL_SADI", "Avenue Sadi Lecointe")]:
        tx, ty = graph.nodes[tl_id]["x"], graph.nodes[tl_id]["y"]
        sig_state = graph.edges[("R_ROUNDABOUT", "TL_ADER") if tl_id=="TL_ADER" else ("TL_ADER", "TL_SADI")]["signal_state"]
        light_color = '#FF1744' if sig_state == "RED" else '#00E676'
        
        # White outer circle
        tl_circle = patches.Circle((tx, ty), radius=12, facecolor='#FFFFFF', edgecolor='#333333', linewidth=2.5, zorder=6)
        ax.add_patch(tl_circle)
        # Inner active light indicator
        ax.scatter(tx, ty, color=light_color, s=80, zorder=7)

    # Annotate "Traffic Lights" text with arrows pointing to white circles
    ax.text(260, 420, "Traffic Lights", color='#111111', fontsize=13, fontweight='bold', ha='center', zorder=8)
    ax.annotate("", xy=(graph.nodes["TL_ADER"]["x"], graph.nodes["TL_ADER"]["y"] + 14), xytext=(240, 410),
                arrowprops=dict(arrowstyle="->", color='#111111', lw=1.8), zorder=8)
    ax.annotate("", xy=(graph.nodes["TL_SADI"]["x"], graph.nodes["TL_SADI"]["y"] + 14), xytext=(280, 410),
                arrowprops=dict(arrowstyle="->", color='#111111', lw=1.8), zorder=8)

    # Annotate "Driving Region" box (Top Left)
    rect = patches.Rectangle((20, 425), 180, 35, facecolor='#FFFFFF', edgecolor='#000000', linewidth=2, zorder=8)
    ax.add_patch(rect)
    ax.text(30, 435, "Driving Region", color='#000000', fontsize=12, fontweight='bold', zorder=9)

    # Annotate "Weak GPS" Blue arrow
    ax.text(580, 365, "Weak GPS", color='#1565C0', fontsize=12, fontweight='bold', zorder=8)
    ax.annotate("", xy=(graph.nodes["GPS_MID"]["x"] + 20, graph.nodes["GPS_MID"]["y"]), xytext=(575, 368),
                arrowprops=dict(arrowstyle="->", color='#1565C0', lw=2), zorder=8)

    # Draw Nodes & Labels
    for nid, data in graph.nodes.items():
        if nid not in ["R_ROUNDABOUT", "TL_ADER", "TL_SADI"]:
            ax.scatter(data["x"], data["y"], color='#37474F', s=70, zorder=5)

    ax.set_xlim(0, 780)
    ax.set_ylim(0, 480)
    ax.axis('off')
    st.pyplot(fig)

with col_info:
    st.subheader("📊 Optimal Route Metrics")

    st.markdown(f"""
    <div class="metric-box">
        <h4 style="color:#B388FF; margin:0;">Calculated Optimal Path</h4>
        <p style="font-size:1.1rem; margin:8px 0; color:#00E676;"><b>{" ➔ ".join(path) if path else "NO PATH FOUND"}</b></p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Total Route Distance", f"{metrics['total_distance']} m")
        st.metric("Estimated Travel Time", f"{metrics['total_time']} s")
    with c2:
        st.metric("Avg Collision Risk Score", f"{metrics['avg_risk']}")
        st.metric("GPS Noise Level", "15.0 m (High)" if weak_gps_active else "0.5 m (Low)")

    st.markdown("---")
    st.subheader("🛰️ Extended Kalman Filter (EKF)")
    
    # EKF Simulation Demo
    ekf.state = np.array([300.0, 360.0, 12.0, 0.1])
    ekf.predict(dt=0.5, accel=0.0)
    gps_std = 15.0 if weak_gps_active else 0.5
    z_gps = np.array([306.0 + np.random.normal(0, gps_std), 360.0 + np.random.normal(0, gps_std)])
    uncertainty = ekf.update_gps(z_gps, gps_noise_std=gps_std)

    st.markdown(f"""
    <div class="metric-box">
        <h5 style="color:#90CAF9; margin:0;">EKF Sensor Fusion Status</h5>
        <p style="margin:5px 0;"><b>GPS Measurement Noise:</b> {gps_std} m</p>
        <p style="margin:5px 0;"><b>Estimated Position Error:</b> {uncertainty:.3f} m</p>
        <p style="margin:5px 0; color:{'#FF1744' if gps_std > 5 else '#00E676'};">
            <b>Status:</b> {'⚠️ Dead-Reckoning Active (Weak GPS)' if gps_std > 5 else '✅ High Precision Satellite Lock'}
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("<p style='text-align:center; color:#888;'>Real-World Urban Driving Region System | Developed for Advanced Research</p>", unsafe_allow_html=True)
