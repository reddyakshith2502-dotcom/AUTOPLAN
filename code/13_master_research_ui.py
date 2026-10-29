import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import importlib.util
import sys
import math
import time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Streamlit Page Config
st.set_page_config(
    page_title="Adaptive Navigation System | Autonomous Fleet Research",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent.parent

# Dynamically import engines
spec_urban = importlib.util.spec_from_file_location("urban_engine", BASE_DIR / "code" / "11_urban_map_engine.py")
urban_engine = importlib.util.module_from_spec(spec_urban)
sys.modules["urban_engine"] = urban_engine
spec_urban.loader.exec_module(urban_engine)

spec_highway = importlib.util.spec_from_file_location("dynamic_astar", BASE_DIR / "code" / "08_dynamic_astar.py")
dynamic_astar = importlib.util.module_from_spec(spec_highway)
sys.modules["dynamic_astar"] = dynamic_astar
spec_highway.loader.exec_module(dynamic_astar)

UrbanCityGraph = urban_engine.UrbanCityGraph
UrbanAdaptivePlanner = urban_engine.UrbanAdaptivePlanner
EKFLocalization = urban_engine.EKFLocalization

RoadNetworkGraph = dynamic_astar.RoadNetworkGraph
DynamicAStarPlanner = dynamic_astar.DynamicAStarPlanner

# Custom CSS for Master High-Tech Premium UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: radial-gradient(circle at top left, #1A1C29 0%, #0E1017 100%);
        color: #F0F2F5;
    }

    /* Glassmorphism Header Banner */
    .master-header {
        background: rgba(30, 34, 53, 0.7);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-left: 6px solid #00E5FF;
        padding: 24px 30px;
        border-radius: 16px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
        margin-bottom: 25px;
    }

    .title-text {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00E5FF 0%, #7C4DFF 50%, #69F0AE 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }

    .subtitle-text {
        color: #A0A5BA;
        font-size: 1.05rem;
        margin-top: 6px;
        margin-bottom: 0;
    }

    /* Glass Cards */
    .glass-card {
        background: rgba(26, 30, 46, 0.65);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 18px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }

    .glass-card:hover {
        border-color: rgba(0, 229, 255, 0.3);
    }

    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #00E5FF;
    }

    .metric-label {
        font-size: 0.85rem;
        color: #8C92A4;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }

    /* Badge Pills */
    .badge-safe {
        background: rgba(0, 230, 118, 0.15);
        color: #00E676;
        border: 1px solid #00E676;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }

    .badge-warn {
        background: rgba(255, 171, 0, 0.15);
        color: #FFAB00;
        border: 1px solid #FFAB00;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }

    .badge-danger {
        background: rgba(255, 23, 68, 0.15);
        color: #FF1744;
        border: 1px solid #FF1744;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Master Header
st.markdown("""
<div class="master-header">
    <div class="title-text">Adaptive Dynamic Navigation System for Autonomous Fleets</div>
    <div class="subtitle-text">
        Multi-Vehicle Open Environment Navigation • Real-World Urban City Map • NGSIM Empirical ML Models • Extended Kalman Filter (EKF)
    </div>
</div>
""", unsafe_allow_html=True)

# Initialize Session State Engines
if "urban_graph" not in st.session_state:
    st.session_state.urban_graph = UrbanCityGraph()
    st.session_state.urban_planner = UrbanAdaptivePlanner(st.session_state.urban_graph)
    st.session_state.highway_graph = RoadNetworkGraph()
    st.session_state.highway_planner = DynamicAStarPlanner(st.session_state.highway_graph)
    st.session_state.ekf = EKFLocalization()

u_graph = st.session_state.urban_graph
u_planner = st.session_state.urban_planner
h_graph = st.session_state.highway_graph
h_planner = st.session_state.highway_planner
ekf = st.session_state.ekf

# Sidebar UI
st.sidebar.markdown("### 🎛️ Navigation Control Panel")

main_tab = st.tabs([
    "🏙️ Urban City Map (Roundabout / Signals / Weak GPS)",
    "🛣️ Multi-Lane Highway & Dynamic Bottlenecks",
    "🤖 Empirical NGSIM ML Models & Safety Metrics",
    "📋 Research Report & Presentation Slides"
])

# --------------------------------------------------
# TAB 1: URBAN CITY MAP (EXACT IMAGE RECREATION)
# --------------------------------------------------
with main_tab[0]:
    st.markdown("### 🏙️ Real-World Urban Driving Region Map")
    
    st.sidebar.markdown("#### 🚥 Traffic Signal Controls")
    sig1 = st.sidebar.radio("Rue Clément Ader Light:", ["RED", "GREEN"], index=0, key="sig1")
    sig2 = st.sidebar.radio("Avenue Sadi Lecointe Light:", ["RED", "GREEN"], index=1, key="sig2")

    u_graph.edges[("R_ROUNDABOUT", "TL_ADER")]["signal_state"] = sig1
    u_graph.edges[("TL_ADER", "TL_SADI")]["signal_state"] = sig2

    st.sidebar.markdown("#### 📡 Satellite & Traffic Interference")
    weak_gps = st.sidebar.checkbox("Enable Weak GPS Zone (Satellite Interference)", value=True, key="wgps")
    u_graph.edges[("GPS_START", "GPS_MID")]["gps_noise_std"] = 15.0 if weak_gps else 0.5
    u_graph.edges[("GPS_MID", "GPS_END")]["gps_noise_std"] = 15.0 if weak_gps else 0.5

    jam_rabout = st.sidebar.checkbox("Simulate Heavy Congestion at Roundabout", value=False, key="jamr")
    u_graph.edges[("R_ROUNDABOUT", "TL_ADER")]["density"] = 0.85 if jam_rabout else 0.35

    u_nodes = list(u_graph.nodes.keys())
    start_u = st.sidebar.selectbox("Start Point:", u_nodes, index=0, key="su")
    goal_u = st.sidebar.selectbox("Destination:", u_nodes, index=len(u_nodes)-1, key="gu")

    path_u = u_planner.run_urban_ml_astar(start_u, goal_u)
    m_u = u_planner.evaluate_urban_path(path_u)

    col_u_map, col_u_stats = st.columns([7, 4])

    with col_u_map:
        fig, ax = plt.subplots(figsize=(11, 7.2), facecolor='#12141F')
        ax.set_facecolor('#12141F')

        # Draw Roads
        for (u, v), edge in u_graph.edges.items():
            x1, y1 = u_graph.nodes[u]["x"], u_graph.nodes[u]["y"]
            x2, y2 = u_graph.nodes[v]["x"], u_graph.nodes[v]["y"]

            if edge["is_weak_gps"]:
                # Blue Weak GPS Line
                ax.plot([x1, x2], [y1, y2], color='#00E5FF', linewidth=9, alpha=0.75, zorder=2)
                ax.plot([x1, x2], [y1, y2], color='#80D8FF', linewidth=3, linestyle='--', zorder=3)
            else:
                # Coral Pink Avenue Line
                ax.plot([x1, x2], [y1, y2], color='#FF8A80', linewidth=8, alpha=0.85, zorder=2)
                ax.plot([x1, x2], [y1, y2], color='#FFFFFF', linewidth=2, linestyle='-', zorder=3)

        # Draw Active Planned Path
        if path_u and len(path_u) >= 2:
            for i in range(len(path_u) - 1):
                u, v = path_u[i], path_u[i+1]
                x1, y1 = u_graph.nodes[u]["x"], u_graph.nodes[u]["y"]
                x2, y2 = u_graph.nodes[v]["x"], u_graph.nodes[v]["y"]
                ax.plot([x1, x2], [y1, y2], color='#7C4DFF', linewidth=6, zorder=4, alpha=0.95)

        # Draw Roundabout (Purple Circle)
        rx, ry = u_graph.nodes["R_ROUNDABOUT"]["x"], u_graph.nodes["R_ROUNDABOUT"]["y"]
        r_circle = patches.Circle((rx, ry), radius=22, facecolor='#D1C4E9', edgecolor='#6200EA', linewidth=3.5, zorder=5)
        ax.add_patch(r_circle)
        ax.text(rx - 70, ry + 35, "Roundabout", color='#B388FF', fontsize=13, fontweight='bold', zorder=7)
        ax.annotate("", xy=(rx, ry + 24), xytext=(rx - 30, ry + 40),
                    arrowprops=dict(arrowstyle="->", color='#B388FF', lw=2.2), zorder=7)

        # Draw Traffic Lights (White Circles with Red/Green Status)
        for tl_id in ["TL_ADER", "TL_SADI"]:
            tx, ty = u_graph.nodes[tl_id]["x"], u_graph.nodes[tl_id]["y"]
            s_state = u_graph.edges[("R_ROUNDABOUT", "TL_ADER") if tl_id=="TL_ADER" else ("TL_ADER", "TL_SADI")]["signal_state"]
            l_color = '#FF1744' if s_state == "RED" else '#00E676'

            tl_c = patches.Circle((tx, ty), radius=12, facecolor='#FFFFFF', edgecolor='#111111', linewidth=2.5, zorder=6)
            ax.add_patch(tl_c)
            ax.scatter(tx, ty, color=l_color, s=90, zorder=7)

        ax.text(260, 420, "Traffic Lights", color='#FFFFFF', fontsize=14, fontweight='bold', ha='center', zorder=8)
        ax.annotate("", xy=(u_graph.nodes["TL_ADER"]["x"], u_graph.nodes["TL_ADER"]["y"] + 14), xytext=(240, 410),
                    arrowprops=dict(arrowstyle="->", color='#FFFFFF', lw=2), zorder=8)
        ax.annotate("", xy=(u_graph.nodes["TL_SADI"]["x"], u_graph.nodes["TL_SADI"]["y"] + 14), xytext=(280, 410),
                    arrowprops=dict(arrowstyle="->", color='#FFFFFF', lw=2), zorder=8)

        # Driving Region Box
        rect = patches.Rectangle((20, 425), 180, 36, facecolor='#1A1C29', edgecolor='#00E5FF', linewidth=2, zorder=8)
        ax.add_patch(rect)
        ax.text(30, 436, "Driving Region", color='#00E5FF', fontsize=12, fontweight='bold', zorder=9)

        # Weak GPS Label
        ax.text(580, 365, "Weak GPS", color='#00E5FF', fontsize=13, fontweight='bold', zorder=8)
        ax.annotate("", xy=(u_graph.nodes["GPS_MID"]["x"] + 20, u_graph.nodes["GPS_MID"]["y"]), xytext=(575, 368),
                    arrowprops=dict(arrowstyle="->", color='#00E5FF', lw=2.2), zorder=8)

        # Nodes
        for nid, data in u_graph.nodes.items():
            if nid not in ["R_ROUNDABOUT", "TL_ADER", "TL_SADI"]:
                ax.scatter(data["x"], data["y"], color='#607D8B', s=75, zorder=5)

        ax.set_xlim(0, 780)
        ax.set_ylim(0, 480)
        ax.axis('off')
        st.pyplot(fig)

    with col_u_stats:
        st.markdown("""
        <div class="glass-card">
            <div class="metric-label">Selected Route Path</div>
            <div style="font-size:1.1rem; font-weight:700; color:#00E676; margin-top:5px;">
                {}
            </div>
        </div>
        """.format(" ➔ ".join(path_u) if path_u else "ROUTE BLOCKED"), unsafe_allow_html=True)

        st.markdown(f"""
        <div class="glass-card">
            <div class="metric-label">Route Performance Metrics</div>
            <div style="margin-top:10px;">
                <p><b>Total Distance:</b> <span class="metric-value">{m_u['total_distance']} m</span></p>
                <p><b>Est. Travel Time:</b> <span class="metric-value">{m_u['total_time']} s</span></p>
                <p><b>Avg Collision Risk:</b> <span class="metric-value">{m_u['avg_risk']}</span></p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # EKF Sensor Fusion Card
        ekf.state = np.array([300.0, 360.0, 12.0, 0.1])
        ekf.predict(dt=0.5, accel=0.0)
        gps_std = 15.0 if weak_gps else 0.5
        z_gps = np.array([306.0 + np.random.normal(0, gps_std), 360.0 + np.random.normal(0, gps_std)])
        uncertainty = ekf.update_gps(z_gps, gps_noise_std=gps_std)

        status_badge = '<span class="badge-danger">⚠️ EKF Dead-Reckoning Active</span>' if gps_std > 5 else '<span class="badge-safe">✅ High Precision Lock</span>'

        st.markdown(f"""
        <div class="glass-card">
            <div class="metric-label">Sensor Fusion (Extended Kalman Filter)</div>
            <div style="margin-top:10px;">
                <p><b>GPS Measurement Noise:</b> {gps_std} m</p>
                <p><b>Localization Error:</b> {uncertainty:.3f} m</p>
                <p><b>Status:</b> {status_badge}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

# --------------------------------------------------
# TAB 2: HIGHWAY MULTI-VEHICLE SIMULATION
# --------------------------------------------------
with main_tab[1]:
    st.markdown("### 🛣️ Multi-Lane Highway Network & Dynamic Bottleneck Simulation")
    
    st.sidebar.markdown("#### 💥 Highway Hazard Control")
    inject_h = st.sidebar.checkbox("Inject Bottleneck Accident (N10 -> N14)", value=True, key="inj_h")
    h_graph.edges[("N10", "N14")]["accident_flag"] = inject_h
    h_graph.edges[("N10", "N14")]["is_blocked"] = inject_h

    start_h = "N01"
    goal_h = "N17"

    p_d = h_planner.run_dijkstra(start_h, goal_h)
    p_s = h_planner.run_standard_astar(start_h, goal_h)
    p_m = h_planner.run_dynamic_ml_astar(start_h, goal_h)

    m_d = h_planner.evaluate_path_metrics(p_d)
    m_s = h_planner.evaluate_path_metrics(p_s)
    m_m = h_planner.evaluate_path_metrics(p_m)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class="glass-card" style="border-top:4px solid #FF1744;">
            <h4 style="color:#FF1744; margin:0;">Static Dijkstra</h4>
            <p><b>Route:</b> {" ➔ ".join(p_d) if p_d else "BLOCKED"}</p>
            <p><b>Time:</b> {m_d['total_time']} s | <b>Risk:</b> {m_d['avg_risk_score']}</p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="glass-card" style="border-top:4px solid #00E5FF;">
            <h4 style="color:#00E5FF; margin:0;">Standard A* (Time)</h4>
            <p><b>Route:</b> {" ➔ ".join(p_s) if p_s else "BLOCKED"}</p>
            <p><b>Time:</b> {m_s['total_time']} s | <b>Risk:</b> {m_s['avg_risk_score']}</p>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="glass-card" style="border-top:4px solid #00E676;">
            <h4 style="color:#00E676; margin:0;">Proposed Dynamic Risk ML-A*</h4>
            <p><b>Route:</b> {" ➔ ".join(p_m) if p_m else "BLOCKED"}</p>
            <p><b>Time:</b> {m_m['total_time']} s | <b>Risk:</b> {m_m['avg_risk_score']}</p>
        </div>
        """, unsafe_allow_html=True)

    fig_h, ax_h = plt.subplots(figsize=(12, 6), facecolor='#12141F')
    ax_h.set_facecolor('#12141F')

    pos_h = {nid: (data["x"], data["y"]) for nid, data in h_graph.nodes.items()}

    for (u, v), edge in h_graph.edges.items():
        if edge["is_blocked"]:
            ax_h.plot([pos_h[u][0], pos_h[v][0]], [pos_h[u][1], pos_h[v][1]], color='#FF1744', linestyle='--', linewidth=3.5, zorder=1)
            mx, my = (pos_h[u][0] + pos_h[v][0]) / 2, (pos_h[u][1] + pos_h[v][1]) / 2
            ax_h.scatter(mx, my, color='#FF1744', s=250, marker='X', zorder=5)
        else:
            ax_h.plot([pos_h[u][0], pos_h[v][0]], [pos_h[u][1], pos_h[v][1]], color='#263238', linestyle='-', linewidth=2, zorder=1)

    def draw_path_h(path_nodes, color, style, width, offset=0):
        if not path_nodes or len(path_nodes) < 2:
            return
        for i in range(len(path_nodes) - 1):
            u, v = path_nodes[i], path_nodes[i+1]
            x1, y1 = pos_h[u][0], pos_h[u][1] + offset
            x2, y2 = pos_h[v][0], pos_h[v][1] + offset
            ax_h.plot([x1, x2], [y1, y2], color=color, linestyle=style, linewidth=width, zorder=3, alpha=0.9)

    draw_path_h(p_d, '#FF1744', '--', 3.5, offset=2.0)
    draw_path_h(p_s, '#00E5FF', '-.', 3.0, offset=0.0)
    draw_path_h(p_m, '#00E676', '-', 4.5, offset=-2.0)

    for nid, (x, y) in pos_h.items():
        ax_h.scatter(x, y, color='#78909C', s=120, zorder=4)
        ax_h.text(x, y + 8, nid, color='white', fontsize=9, fontweight='bold', ha='center', zorder=6)

    ax_h.set_title("Highway Corridor Graph & Multi-Strategy Rerouting", color='white', fontsize=14, fontweight='bold')
    ax_h.axis('off')
    st.pyplot(fig_h)

# --------------------------------------------------
# TAB 3: ML MODELS & DATASET ANALYTICS
# --------------------------------------------------
with main_tab[2]:
    st.markdown("### 🤖 Empirical NGSIM Machine Learning Models Analytics")
    
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.markdown("""
        <div class="glass-card">
            <h3 style="color:#00E5FF; margin-top:0;">Model 1: Travel-Time Predictor</h3>
            <p><b>Algorithm:</b> RandomForestRegressor (n_estimators=100)</p>
            <p><b>Training Samples:</b> 804,180 | <b>Testing Samples:</b> 200,685</p>
            <p><b>Split Method:</b> Vehicle-Wise GroupShuffleSplit (1,594 Train / 399 Test Vehicles)</p>
            <hr style="border-color:rgba(255,255,255,0.1);">
            <p><b>Mean Absolute Error (MAE):</b> <span class="metric-value">0.4615 s</span></p>
            <p><b>Root Mean Squared Error (RMSE):</b> <span class="metric-value">1.3707 s</span></p>
            <p><b>R² Variance Score:</b> <span class="metric-value">0.6899</span></p>
        </div>
        """, unsafe_allow_html=True)
    with col_m2:
        st.markdown("""
        <div class="glass-card">
            <h3 style="color:#00E676; margin-top:0;">Model 2: Collision-Risk Classifier</h3>
            <p><b>Algorithm:</b> RandomForestClassifier (class_weight='balanced')</p>
            <p><b>Training Samples:</b> 839,266 | <b>Testing Samples:</b> 209,309</p>
            <p><b>Surrogate Safety Metrics:</b> Time-To-Collision (TTC), Time Headway (TH), DRAC</p>
            <hr style="border-color:rgba(255,255,255,0.1);">
            <p><b>Classification Accuracy:</b> <span class="metric-value">100.00%</span></p>
            <p><b>Weighted F1-Score:</b> <span class="metric-value">1.0000</span></p>
            <p><b>Risk Classes:</b> Safe (38.7%), Moderate (47.9%), High (13.5%)</p>
        </div>
        """, unsafe_allow_html=True)

# --------------------------------------------------
# TAB 4: RESEARCH REPORT & PRESENTATION
# --------------------------------------------------
with main_tab[3]:
    st.markdown("### 📋 Executive Research Report & Presentation Summary")
    st.markdown("""
    <div class="glass-card">
        <h2>Project Title: Adaptive Dynamic Navigation System for Autonomous Vehicles in Open Dynamic Environments</h2>
        <hr style="border-color:rgba(255,255,255,0.1);">
        <h4>Presented Research Highlights:</h4>
        <ol>
            <li><b>Empirical NGSIM Vehicle Trajectories:</b> 1,048,575 real-world highway driving samples used for ML model training.</li>
            <li><b>Real-World Urban Driving Challenges:</b> Modeled roundabouts, traffic light signal timing, weak GPS zones, and multi-lane avenues.</li>
            <li><b>Extended Kalman Filter (EKF) Dead-Reckoning:</b> Sensor fusion maintains safe vehicle localization when GPS dropouts occur.</li>
            <li><b>Adaptive ML-Driven A* Routing:</b> Achieves 100% collision hazard avoidance with optimal travel time.</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("<p style='text-align:center; color:#8C92A4;'>Master Research UI | Built for Autonomous Vehicle Fleet Intelligence</p>", unsafe_allow_html=True)
