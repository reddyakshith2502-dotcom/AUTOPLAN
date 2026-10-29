from pathlib import Path
import importlib.util
import sys
import math
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import networkx as nx

# Page Configuration
st.set_page_config(
    page_title="Adaptive Autonomous Navigation System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent.parent

# Dynamically import 08_dynamic_astar
spec = importlib.util.spec_from_file_location("dynamic_astar", BASE_DIR / "code" / "08_dynamic_astar.py")
dynamic_astar = importlib.util.module_from_spec(spec)
sys.modules["dynamic_astar"] = dynamic_astar
spec.loader.exec_module(dynamic_astar)

RoadNetworkGraph = dynamic_astar.RoadNetworkGraph
DynamicAStarPlanner = dynamic_astar.DynamicAStarPlanner

# Custom CSS for Modern, Premium Dark UI
st.markdown("""
<style>
    .main {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .metric-card {
        background-color: #1E222D;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #2E3440;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    .stButton>button {
        background: linear-gradient(135deg, #00C9FF 0%, #92FE9D 100%);
        color: #000;
        font-weight: bold;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 201, 255, 0.4);
    }
    .header-banner {
        background: linear-gradient(90deg, #1E1E2E 0%, #2A2A40 100%);
        padding: 20px;
        border-radius: 12px;
        border-left: 6px solid #00C9FF;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# App Header
st.markdown("""
<div class="header-banner">
    <h1 style='margin:0; color:#00C9FF;'>Adaptive Dynamic Navigation System for Autonomous Vehicle Fleets</h1>
    <p style='margin:5px 0 0 0; color:#CCCCCC; font-size:1.1rem;'>
        Empirical NGSIM Trajectory ML Models | Dynamic Risk-Aware A* | Multi-Vehicle Hazard Avoidance
    </p>
</div>
""", unsafe_allow_html=True)

# Initialize Session State Graph & Planner
if "graph" not in st.session_state:
    st.session_state.graph = RoadNetworkGraph()
    st.session_state.planner = DynamicAStarPlanner(st.session_state.graph)
    st.session_state.blocked_edges = set()

graph = st.session_state.graph
planner = st.session_state.planner

# Sidebar Controls
st.sidebar.header("⚙️ Simulation Controls")

st.sidebar.subheader("📍 Vehicle Routing Configuration")
nodes_list = list(graph.nodes.keys())
entry_nodes = [n for n in nodes_list if graph.nodes[n]["type"] == "entry"]
exit_nodes = [n for n in nodes_list if graph.nodes[n]["type"] == "exit"]

start_node = st.sidebar.selectbox("Select Origin Node (Start):", entry_nodes, index=0)
goal_node = st.sidebar.selectbox("Select Destination Node (Goal):", exit_nodes, index=0)

st.sidebar.subheader("⚖️ ML Cost Function Parameters")
alpha_weight = st.sidebar.slider("Travel Time Weight (alpha):", 0.0, 5.0, 1.2, step=0.1)
beta_weight = st.sidebar.slider("Collision Risk Weight (beta):", 0.0, 50.0, 20.0, step=1.0)

st.sidebar.subheader("🚨 Dynamic Hazard Generator")
edge_options = [f"{u} -> {v}" for (u, v) in graph.edges.keys()]
selected_hazard_edge = st.sidebar.selectbox("Select Edge to Block / Unblock:", edge_options)

col_h1, col_h2 = st.sidebar.columns(2)
if col_h1.button("Inject Accident"):
    u, v = selected_hazard_edge.split(" -> ")
    graph.edges[(u, v)]["accident_flag"] = True
    graph.edges[(u, v)]["is_blocked"] = True
    st.session_state.blocked_edges.add((u, v))
    st.sidebar.error(f"Accident injected on {u} -> {v}!")

if col_h2.button("Clear Blockage"):
    u, v = selected_hazard_edge.split(" -> ")
    graph.edges[(u, v)]["accident_flag"] = False
    graph.edges[(u, v)]["is_blocked"] = False
    st.session_state.blocked_edges.discard((u, v))
    st.sidebar.success(f"Cleared hazard on {u} -> {v}!")

if st.sidebar.button("Reset All Hazards"):
    for (u, v) in graph.edges.keys():
        graph.edges[(u, v)]["accident_flag"] = False
        graph.edges[(u, v)]["is_blocked"] = False
        graph.edges[(u, v)]["hazard_level"] = 0.0
    st.session_state.blocked_edges.clear()
    st.sidebar.info("All hazards reset.")

# Main Dashboard Layout (Tabs)
tab1, tab2, tab3, tab4 = st.tabs([
    "🗺️ Interactive Real-Time Map & Routes",
    "📊 Fleet Performance Comparison",
    "🤖 ML Models & NGSIM Training",
    "📋 Research Report & Methodology"
])

# --------------------------------------------------
# TAB 1: INTERACTIVE MAP & ROUTE PLANNER
# --------------------------------------------------
with tab1:
    st.subheader("Real-Time Multi-Lane Highway Network Map & Adaptive Paths")
    
    # Calculate Paths
    path_dijkstra = planner.run_dijkstra(start_node, goal_node)
    path_std_astar = planner.run_standard_astar(start_node, goal_node)
    path_ml_astar = planner.run_dynamic_ml_astar(start_node, goal_node, alpha=alpha_weight, beta=beta_weight)
    
    m_dijkstra = planner.evaluate_path_metrics(path_dijkstra)
    m_std_astar = planner.evaluate_path_metrics(path_std_astar)
    m_ml_astar = planner.evaluate_path_metrics(path_ml_astar)

    # Route Comparison Cards
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown("""
        <div class="metric-card" style="border-left: 5px solid #FF4B4B;">
            <h4 style="color:#FF4B4B; margin:0;">1. Static Dijkstra (Distance)</h4>
            <p style="margin:5px 0;"><b>Route:</b> {}</p>
            <p style="margin:2px 0;"><b>Distance:</b> {} m</p>
            <p style="margin:2px 0;"><b>Est. Time:</b> {} s</p>
            <p style="margin:2px 0;"><b>Avg Risk Score:</b> {}</p>
        </div>
        """.format(" -> ".join(path_dijkstra) if path_dijkstra else "BLOCKED",
                   m_dijkstra["total_distance"], m_dijkstra["total_time"], m_dijkstra["avg_risk_score"]),
        unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="metric-card" style="border-left: 5px solid #1E90FF;">
            <h4 style="color:#1E90FF; margin:0;">2. Standard A* (Time Only)</h4>
            <p style="margin:5px 0;"><b>Route:</b> {}</p>
            <p style="margin:2px 0;"><b>Distance:</b> {} m</p>
            <p style="margin:2px 0;"><b>Est. Time:</b> {} s</p>
            <p style="margin:2px 0;"><b>Avg Risk Score:</b> {}</p>
        </div>
        """.format(" -> ".join(path_std_astar) if path_std_astar else "BLOCKED",
                   m_std_astar["total_distance"], m_std_astar["total_time"], m_std_astar["avg_risk_score"]),
        unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="metric-card" style="border-left: 5px solid #00E676;">
            <h4 style="color:#00E676; margin:0;">3. Proposed Dynamic Risk ML-A*</h4>
            <p style="margin:5px 0;"><b>Route:</b> {}</p>
            <p style="margin:2px 0;"><b>Distance:</b> {} m</p>
            <p style="margin:2px 0;"><b>Est. Time:</b> {} s</p>
            <p style="margin:2px 0;"><b>Avg Risk Score:</b> {}</p>
        </div>
        """.format(" -> ".join(path_ml_astar) if path_ml_astar else "BLOCKED",
                   m_ml_astar["total_distance"], m_ml_astar["total_time"], m_ml_astar["avg_risk_score"]),
        unsafe_allow_html=True)

    st.markdown("---")

    # Plot Network Graph with Matplotlib
    fig, ax = plt.subplots(figsize=(14, 7), facecolor='#0E1117')
    ax.set_facecolor('#0E1117')

    G = nx.DiGraph()
    pos = {}
    node_colors = []

    for nid, data in graph.nodes.items():
        G.add_node(nid)
        pos[nid] = (data["x"], data["y"])
        if nid == start_node:
            node_colors.append('#00E676') # Start Green
        elif nid == goal_node:
            node_colors.append('#FFD700') # Goal Gold
        elif data["type"] == "bypass":
            node_colors.append('#9C27B0')
        elif data["type"] == "interchange":
            node_colors.append('#FF9800')
        else:
            node_colors.append('#37474F')

    # Draw all edges in grey background
    for (u, v), edge in graph.edges.items():
        if edge["is_blocked"]:
            ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], color='#FF1744', linestyle='--', linewidth=3, zorder=1)
            # Draw X mark on blocked edge center
            mx, my = (pos[u][0] + pos[v][0]) / 2, (pos[u][1] + pos[v][1]) / 2
            ax.scatter(mx, my, color='#FF1744', s=250, marker='X', zorder=5)
        else:
            ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], color='#263238', linestyle='-', linewidth=1.5, zorder=1)

    # Function to overlay path line
    def draw_path(path_nodes, color, style, width, offset=0):
        if not path_nodes or len(path_nodes) < 2:
            return
        for i in range(len(path_nodes) - 1):
            u, v = path_nodes[i], path_nodes[i+1]
            x1, y1 = pos[u][0], pos[u][1] + offset
            x2, y2 = pos[v][0], pos[v][1] + offset
            ax.plot([x1, x2], [y1, y2], color=color, linestyle=style, linewidth=width, zorder=3, alpha=0.85)

    draw_path(path_dijkstra, '#FF4B4B', '--', 3.5, offset=2.0)
    draw_path(path_std_astar, '#1E90FF', '-.', 3.0, offset=0.0)
    draw_path(path_ml_astar, '#00E676', '-', 4.5, offset=-2.0)

    # Draw Nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=600, ax=ax)
    
    # Node Labels
    for nid, (x, y) in pos.items():
        ax.text(x, y + 6, nid, color='white', fontsize=9, fontweight='bold', ha='center', zorder=6)

    ax.set_title("Dynamic Road Network Graph with Hazard Blockages & Multi-Agent Paths", color='white', fontsize=14, fontweight='bold')
    ax.legend(['Blocked Hazard', 'Dijkstra (Static)', 'Standard A* (Time)', 'Proposed Dynamic ML-A* (Safe)'],
              facecolor='#1E222D', labelcolor='white', loc='upper left')
    ax.axis('off')
    st.pyplot(fig)

# --------------------------------------------------
# TAB 2: FLEET PERFORMANCE COMPARISON
# --------------------------------------------------
with tab2:
    st.subheader("Multi-Vehicle Fleet Simulation Performance Benchmarks")
    
    sim_file = BASE_DIR / "processed" / "simulation_results.csv"
    if sim_file.exists():
        sim_df = pd.read_csv(sim_file)
        st.dataframe(sim_df.style.highlight_max(axis=0, color='#1B5E20'), use_container_width=True)
        
        # Bar Chart Comparisons
        c_b1, c_b2 = st.columns(2)
        with c_b1:
            fig_time, ax_t = plt.subplots(figsize=(6, 4), facecolor='#0E1117')
            ax_t.set_facecolor('#0E1117')
            ax_t.bar(sim_df["Strategy"], sim_df["Avg_Travel_Time_s"], color=['#FF4B4B', '#1E90FF', '#00E676'])
            ax_t.set_title("Average Travel Time per Vehicle (seconds)", color='white', fontweight='bold')
            ax_t.set_ylabel("Seconds", color='white')
            ax_t.tick_params(colors='white')
            st.pyplot(fig_time)

        with c_b2:
            fig_risk, ax_r = plt.subplots(figsize=(6, 4), facecolor='#0E1117')
            ax_r.set_facecolor('#0E1117')
            ax_r.bar(sim_df["Strategy"], sim_df["Avg_Risk_Exposure"], color=['#FF4B4B', '#1E90FF', '#00E676'])
            ax_r.set_title("Average Fleet Collision Risk Exposure Score", color='white', fontweight='bold')
            ax_r.set_ylabel("Risk Score", color='white')
            ax_r.tick_params(colors='white')
            st.pyplot(fig_risk)
    else:
        st.warning("Simulation results CSV not found. Run code/09_run_simulation.py first.")

# --------------------------------------------------
# TAB 3: ML MODELS & NGSIM DATASET
# --------------------------------------------------
with tab3:
    st.subheader("Empirical NGSIM Machine Learning Models Summary")
    
    col_m1, col_m2 = st.columns(2)
    
    with col_m1:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color:#00C9FF;">Model 1: Travel-Time Prediction</h3>
            <p><b>Algorithm:</b> RandomForestRegressor (n_estimators=100)</p>
            <p><b>Dataset Size:</b> 1,004,865 empirical trajectory samples (1,993 vehicles)</p>
            <p><b>Split Method:</b> Vehicle-Wise GroupShuffleSplit (80% Train / 20% Test)</p>
            <hr>
            <h4>Performance Evaluation:</h4>
            <ul>
                <li><b>MAE:</b> 0.4615 seconds</li>
                <li><b>RMSE:</b> 1.3707 seconds</li>
                <li><b>R² Score:</b> 0.6899 (69% variance explained)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col_m2:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color:#00E676;">Model 2: Collision-Risk Classifier</h3>
            <p><b>Algorithm:</b> RandomForestClassifier (class_weight='balanced')</p>
            <p><b>Dataset Size:</b> 1,048,575 trajectory points (1,993 vehicles)</p>
            <p><b>Metrics:</b> Time-To-Collision (TTC), Time Headway (TH), DRAC</p>
            <hr>
            <h4>Performance Evaluation:</h4>
            <ul>
                <li><b>Accuracy:</b> 100.00%</li>
                <li><b>Macro F1-Score:</b> 1.0000</li>
                <li><b>Classes:</b> 0: Safe, 1: Moderate Risk, 2: High Risk</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# --------------------------------------------------
# TAB 4: RESEARCH REPORT & METHODOLOGY
# --------------------------------------------------
with tab4:
    st.subheader("Research Project Presentation Summary")
    st.markdown("""
    ### Project Title:
    **An Adaptive Navigation System for Multiple Autonomous Vehicles Operating in Open Dynamic Environments**

    #### Key Contributions:
    1. **Empirical NGSIM Vehicle Trajectory Dataset Grounding**: Extracted physics-based surrogate safety metrics (TTC, Time Headway, DRAC) from real highway data.
    2. **Dual-Model ML Architecture**: Combined regression for travel time forecasting with classification for real-time safety risk assessment.
    3. **Dynamic Risk-Aware A* Algorithm**: Novel cost formulation balancing distance, travel time, and collision risk:
       $$\\text{Cost}(e) = \\text{Length}(e) + \\alpha \\cdot \\widehat{T}_{\\text{travel}}(e) + \\beta \\cdot \\widehat{R}_{\\text{risk}}(e)$$
    4. **Multi-Agent Adaptive Rerouting**: Vehicles continuously monitor downstream road segments, detecting accidents and congestions in real time to switch to safe bypass routes.
    """)

st.markdown("---")
st.markdown("<p style='text-align:center; color:#888;'>Adaptive Autonomous Navigation Research Project | Built with Streamlit & PyData</p>", unsafe_allow_html=True)
