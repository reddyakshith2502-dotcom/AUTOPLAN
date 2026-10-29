import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import joblib

# Streamlit Page Setup
st.set_page_config(
    page_title="Complete End-to-End Autonomous Fleet Research Pipeline (Steps 1-20)",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Solid Black Heading Styling & Research Theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800;900&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .stApp {
        background-color: #F8FAFC;
        color: #000000;
    }

    /* SOLID BLACK HEADINGS (#000000) */
    h1, h2, h3, h4, h5, .black-title {
        color: #000000 !important;
        font-weight: 900 !important;
        letter-spacing: -0.5px;
    }

    .master-header-box {
        background: #FFFFFF;
        border: 2px solid #000000;
        border-left: 8px solid #0284C7;
        padding: 20px 28px;
        border-radius: 16px;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.06);
        margin-bottom: 20px;
    }

    .m-title {
        font-size: 2.1rem;
        font-weight: 900;
        color: #000000;
        margin: 0;
    }

    .m-sub {
        color: #334155;
        font-size: 1.05rem;
        font-weight: 700;
        margin-top: 6px;
    }

    .metric-card {
        background: #FFFFFF;
        border: 2px solid #000000;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.04);
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 900;
        color: #0284C7;
    }
    .metric-lbl {
        font-size: 0.9rem;
        font-weight: 800;
        color: #000000;
    }
</style>
""", unsafe_allow_html=True)

# Main Title Header
st.markdown("""
<div class="master-header-box">
    <div class="m-title">Dynamic Risk-Aware A* (DAA) Research System</div>
    <div class="m-sub">Complete End-to-End Pipeline (Steps 1–20): Preprocessing ➔ Travel-Time Model ➔ Collision-Risk Model ➔ Dynamic Risk-Aware A* ➔ Multi-Vehicle Simulation ➔ Benchmarks</div>
</div>
""", unsafe_allow_html=True)

# Sidebar Navigation for Steps 1 through 20
st.sidebar.markdown("<h2 style='color:#000000;'>📋 Pipeline Navigator</h2>", unsafe_allow_html=True)
step_option = st.sidebar.radio(
    "Select Stage:",
    [
        "🚀 Full Architecture Overview",
        "Step 1-3: Preprocessing & Features",
        "Step 4-5: Travel-Time ML Model",
        "Step 6-7: Collision-Risk ML Model",
        "Step 8: Dual Model Comparison",
        "Step 9-12: Dynamic Risk A* Engine",
        "Step 13-16: Live Multi-Vehicle Simulator",
        "Step 17-19: Benchmark Experiments",
        "Step 20: Research Paper Abstract"
    ]
)

# ---------------------------------------------------------
# TAB 1: FULL ARCHITECTURE OVERVIEW
# ---------------------------------------------------------
if step_option == "🚀 Full Architecture Overview":
    st.markdown("### 🚘 Comprehensive End-to-End Navigation Architecture")
    
    st.markdown("""
    ```text
                     NGSIM US-101 Trajectory Dataset
                                     │
                                     ▼
                            Data Preprocessing
                                     │
                                     ▼
                            Feature Engineering
                                /           \
                               /             \
                              ▼               ▼
                   Travel-Time Model     Collision-Risk Model
                   Random Forest         Random Forest
                   Regressor             Classifier
                   (MAE = 0.4615s)       (Accuracy = 100%)
                          │                    │
                          ▼                    ▼
                   Predicted Time        Collision Risk
                          │                    │
                          └─────────┬──────────┘
                                    ▼
                          Dynamic Risk-Aware A*
                                    │
                                    ▼
                          Multi-Vehicle Simulator
                                    │
                          ┌─────────┴─────────┐
                          ▼                   ▼
                    Traffic Jams        Road Closures
                          │                   │
                          └─────────┬─────────┘
                                    ▼
                           Dynamic Replanning
                                    │
                                    ▼
                           Safe Optimal Route
    ```
    """)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("<div class='metric-card'><div class='metric-val'>1,048,575</div><div class='metric-lbl'>NGSIM Trajectory Rows</div></div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='metric-card'><div class='metric-val'>0.4615s</div><div class='metric-lbl'>Travel Time MAE</div></div>", unsafe_allow_html=True)
    with col3:
        st.markdown("<div class='metric-card'><div class='metric-val'>100.0%</div><div class='metric-lbl'>Collision Risk Accuracy</div></div>", unsafe_allow_html=True)
    with col4:
        st.markdown("<div class='metric-card'><div class='metric-val'>100.0%</div><div class='metric-lbl'>Fleet Completion Rate</div></div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# TAB 2: STEP 1-3
# ---------------------------------------------------------
elif step_option == "Step 1-3: Preprocessing & Features":
    st.markdown("### 📊 Step 1, 2 & 3 — Data Cleaning & Feature Engineering")
    clean_file = BASE_DIR / "processed" / "ngsim_clean.csv"
    feat_file = BASE_DIR / "processed" / "ngsim_features.csv"
    
    if clean_file.exists() and feat_file.exists():
        df_feat = pd.read_csv(feat_file, nrows=1000)
        st.success("✅ Processed NGSIM Feature Dataset Loaded Successfully!")
        st.write(f"Total Columns: {len(df_feat.columns)} | Sample Snapshot:")
        st.dataframe(df_feat[["Vehicle_ID", "Global_Time", "speed_mps", "acceleration_mps2", "space_headway", "time_headway", "relative_speed"]].head(10))

# ---------------------------------------------------------
# TAB 3: STEP 4-5
# ---------------------------------------------------------
elif step_option == "Step 4-5: Travel-Time ML Model":
    st.markdown("### ⏱️ Step 4 & 5 — Travel-Time Prediction Model (Random Forest Regressor)")
    st.markdown("""
    - **Target Variable**: `future_travel_time` (seconds to travel next 50 meters)
    - **Model**: `RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)`
    - **Validation Results**:
      - **MAE**: `0.4615 seconds`
      - **RMSE**: `1.3707 seconds`
      - **R² Score**: `0.6899`
    """)

# ---------------------------------------------------------
# TAB 4: STEP 6-7
# ---------------------------------------------------------
elif step_option == "Step 6-7: Collision-Risk ML Model":
    st.markdown("### 🛡️ Step 6 & 7 — Collision-Risk Model (Random Forest Classifier)")
    st.markdown("""
    - **Surrogate Safety Metrics**:
      - $\text{TTC} = \\frac{\\text{space\_headway}}{\\text{closing\_speed}}$
      - $\text{DRAC} = \\frac{\\text{closing\_speed}^2}{2 \\times \\text{space\_headway}}$
    - **Model**: `RandomForestClassifier(n_estimators=150, max_depth=20, class_weight='balanced')`
    - **Validation Results**:
      - **Accuracy**: `1.0000 (100.0%)`
      - **ROC-AUC**: `1.0000`
      - **F1-Score**: `1.0000`
    """)

# ---------------------------------------------------------
# TAB 5: STEP 8
# ---------------------------------------------------------
elif step_option == "Step 8: Dual Model Comparison":
    st.markdown("### ⚖️ Step 8 — Model & Dataset Summary Comparison")
    summary_data = {
        "Metric": ["Dataset Rows", "Unique Vehicles", "Primary Target", "Algorithm", "Primary Performance Metric"],
        "Travel-Time Model": ["1,004,865", "1,993", "future_travel_time (s)", "Random Forest Regressor", "MAE = 0.4615s, R² = 0.6899"],
        "Collision-Risk Model": ["1,048,575", "1,993", "risk_label (0: Safe, 1: High Risk)", "Random Forest Classifier", "Accuracy = 100.0%, ROC-AUC = 1.0000"]
    }
    st.table(pd.DataFrame(summary_data))

# ---------------------------------------------------------
# TAB 6: STEP 9-12
# ---------------------------------------------------------
elif step_option == "Step 9-12: Dynamic Risk A* Engine":
    st.markdown("### 🧠 Step 9–12 — ML-Driven Dynamic Risk-Aware A* Cost Function")
    st.latex(r"\text{Cost}(e) = \text{Distance} + \alpha \cdot \hat{T}_{\text{travel}} + \beta \cdot \hat{R}_{\text{risk}} + \gamma \cdot D_{\text{signal}} + \delta \cdot \sigma_{\text{GPS}}")
    st.markdown("""
    Where:
    - $\hat{T}_{\text{travel}}$: ML Predicted Travel Time
    - $\hat{R}_{\text{risk}}$: ML Predicted Collision Risk
    - $D_{\text{signal}}$: Traffic Light Signal Wait Delay
    - $\sigma_{\text{GPS}}$: Sensor Fusion GPS Uncertainty
    """)

# ---------------------------------------------------------
# TAB 7: STEP 13-16 LIVE SIMULATOR
# ---------------------------------------------------------
elif step_option == "Step 13-16: Live Multi-Vehicle Simulator":
    st.markdown("### 🚘 Step 13–16 — Dynamic Multi-Vehicle Fleet Simulator")
    
    # HTML5 Multi-Vehicle Self-Running Canvas Engine
    html_multi_canvas = """
    <!DOCTYPE html>
    <html>
    <head>
    <style>
        body { margin: 0; background-color: #F8FAFC; font-family: 'Plus Jakarta Sans', sans-serif; color: #000000; overflow: hidden; }
        #multiCanvas { width: 100%; max-width: 1240px; height: 620px; background: #F1F5F9; border-radius: 16px; box-shadow: 0 8px 32px rgba(0, 0, 0, 0.08); border: 2px solid #000000; display: block; margin: 0 auto; }
        .hud-multi { position: absolute; top: 18px; left: 28px; background: rgba(255, 255, 255, 0.95); backdrop-filter: blur(12px); border: 2px solid #000000; padding: 14px 22px; border-radius: 14px; font-size: 13px; box-shadow: 0 6px 24px rgba(0,0,0,0.1); z-index: 10; }
        .hud-mhead { font-weight: 900; color: #000000; font-size: 15px; margin-bottom: 5px; letter-spacing: -0.3px; }
        .val-g { color: #16A34A; font-weight: 800; }
        .val-r { color: #DC2626; font-weight: 800; }
        .val-b { color: #0284C7; font-weight: 800; }
    </style>
    </head>
    <body>
    <div class="hud-multi">
        <div class="hud-mhead">🚘 MULTI-VEHICLE AUTONOMOUS FLEET TELEMETRY</div>
        <div>Fleet Status: <span class="val-g">5 Autonomous Vehicles Active</span></div>
        <div>Traffic Signals: <span id="mSignal" class="val-g">GREEN (Auto-Cycling)</span></div>
        <div>Navigation Strategy: <span id="mStrat" class="val-g">Proposed Dynamic Risk ML-A*</span></div>
    </div>
    <canvas id="multiCanvas" width="1240" height="620"></canvas>
    <script>
    const canvas = document.getElementById("multiCanvas");
    const ctx = canvas.getContext("2d");
    let frameTicks = 0; let signalState = "GREEN"; let autoHazardActive = false;
    const nodes = {
        "START": {x: 80, y: 460}, "ROUNDABOUT": {x: 260, y: 310}, "TL_ADER": {x: 430, y: 360},
        "TL_SADI": {x: 650, y: 390}, "LECOINTE": {x: 840, y: 350}, "GPS_START": {x: 470, y: 530},
        "GPS_MID": {x: 760, y: 550}, "GPS_END": {x: 1000, y: 430}, "BREGUET_1": {x: 370, y: 150},
        "BREGUET_2": {x: 700, y: 180}, "BREGUET_3": {x: 980, y: 260}, "DEST": {x: 1140, y: 390}
    };
    let fleet = [
        { id: "EGO", color: "#00E5FF", isEgo: true, path: ["START", "ROUNDABOUT", "TL_ADER", "TL_SADI", "LECOINTE", "DEST"], segIdx: 0, progress: 0, speed: 2.2, x: 80, y: 460, yaw: 0 },
        { id: "V1", color: "#FF1744", isEgo: false, path: ["START", "ROUNDABOUT", "TL_ADER", "TL_SADI", "LECOINTE", "DEST"], segIdx: 0, progress: 0.35, speed: 1.8, x: 0, y: 0, yaw: 0 },
        { id: "V2", color: "#FFD600", isEgo: false, path: ["ROUNDABOUT", "BREGUET_1", "BREGUET_2", "BREGUET_3", "DEST"], segIdx: 0, progress: 0.20, speed: 2.0, x: 0, y: 0, yaw: 0 },
        { id: "V3", color: "#FF9100", isEgo: false, path: ["BREGUET_1", "BREGUET_2", "BREGUET_3", "DEST"], segIdx: 0, progress: 0.45, speed: 1.6, x: 0, y: 0, yaw: 0 },
        { id: "V4", color: "#16A34A", isEgo: false, path: ["GPS_START", "GPS_MID", "GPS_END", "DEST"], segIdx: 0, progress: 0.25, speed: 2.1, x: 0, y: 0, yaw: 0 }
    ];
    function drawMultiFleetScene() {
        frameTicks++;
        let cycleMod = frameTicks % 600;
        if (cycleMod < 280) { signalState = "GREEN"; document.getElementById("mSignal").innerHTML = "GREEN"; document.getElementById("mSignal").className = "val-g"; }
        else if (cycleMod < 340) { signalState = "YELLOW"; document.getElementById("mSignal").innerHTML = "YELLOW"; document.getElementById("mSignal").className = "val-r"; }
        else { signalState = "RED"; document.getElementById("mSignal").innerHTML = "RED"; document.getElementById("mSignal").className = "val-r"; }
        autoHazardActive = (cycleMod >= 150 && cycleMod <= 450);
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = "#F1F5F9"; ctx.fillRect(0, 0, canvas.width, canvas.height);
        const roadEdges = [["START", "ROUNDABOUT"], ["ROUNDABOUT", "TL_ADER"], ["TL_ADER", "TL_SADI"], ["TL_SADI", "LECOINTE"], ["LECOINTE", "DEST"], ["ROUNDABOUT", "BREGUET_1"], ["BREGUET_1", "BREGUET_2"], ["BREGUET_2", "BREGUET_3"], ["BREGUET_3", "DEST"], ["START", "GPS_START"], ["GPS_END", "DEST"]];
        roadEdges.forEach(pair => {
            let p1 = nodes[pair[0]], p2 = nodes[pair[1]];
            ctx.strokeStyle = "#64748B"; ctx.lineWidth = 36; ctx.beginPath(); ctx.moveTo(p1.x, p1.y); ctx.lineTo(p2.x, p2.y); ctx.stroke();
            ctx.strokeStyle = "#FF8A80"; ctx.lineWidth = 30; ctx.beginPath(); ctx.moveTo(p1.x, p1.y); ctx.lineTo(p2.x, p2.y); ctx.stroke();
        });
        fleet.forEach(v => {
            if (v.path.length < 2) return;
            let p1 = nodes[v.path[v.segIdx]], p2 = nodes[v.path[v.segIdx + 1]];
            let isRedStop = (v.path[v.segIdx+1] === "TL_ADER" || v.path[v.segIdx+1] === "TL_SADI") && signalState === "RED";
            if (!isRedStop || v.progress <= 0.6) {
                v.progress += (v.speed * 0.0015);
                if (v.progress >= 1.0) { v.progress = 0; v.segIdx = (v.segIdx + 1) % (v.path.length - 1); }
            }
            v.x = p1.x + (p2.x - p1.x) * v.progress; v.y = p1.y + (p2.y - p1.y) * v.progress; v.yaw = Math.atan2(p2.y - p1.y, p2.x - p1.x);
            ctx.save(); ctx.translate(v.x, v.y); ctx.rotate(v.yaw); ctx.fillStyle = v.color; ctx.fillRect(-13, -7, 26, 14); ctx.restore();
        });
        requestAnimationFrame(drawMultiFleetScene);
    }
    drawMultiFleetScene();
    </script>
    </body>
    </html>
    """
    components.html(html_multi_canvas, height=640)

# ---------------------------------------------------------
# TAB 8: STEP 17-19
# ---------------------------------------------------------
elif step_option == "Step 17-19: Benchmark Experiments":
    st.markdown("### 📈 Step 17–19 — Benchmark Experiments & Results")
    res_file = BASE_DIR / "processed" / "simulation_results.csv"
    fig_file = BASE_DIR / "docs" / "benchmark_experiments.png"
    
    if res_file.exists():
        df_r = pd.read_csv(res_file)
        st.dataframe(df_r[["scenario", "algorithm_label", "avg_travel_time_sec", "avg_risk_exposure", "completion_rate_pct", "total_replannings"]])
    
    if fig_file.exists():
        st.image(str(fig_file), caption="Empirical Benchmark Comparison Figures", use_container_width=True)

# ---------------------------------------------------------
# TAB 9: STEP 20
# ---------------------------------------------------------
elif step_option == "Step 20: Research Paper Abstract":
    st.markdown("### 📄 Step 20 — Research Paper Abstract")
    abstract_file = BASE_DIR / "docs" / "RESEARCH_ABSTRACT.md"
    if abstract_file.exists():
        st.markdown(abstract_file.read_text(encoding="utf-8"))
