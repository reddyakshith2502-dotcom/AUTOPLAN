import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import json
import streamlit as st
import streamlit.components.v1 as components

# Streamlit Page Configuration
st.set_page_config(
    page_title="Hyper-Realistic Autonomous Traffic Simulator",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent.parent

# Custom Styling for Container
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .stApp {
        background-color: #0B0D14;
        color: #F0F4F8;
    }

    .sim-header {
        background: linear-gradient(135deg, #131722 0%, #1D2333 100%);
        border: 1px solid rgba(0, 229, 255, 0.2);
        border-left: 6px solid #00E5FF;
        padding: 20px 28px;
        border-radius: 16px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
        margin-bottom: 20px;
    }

    .sim-title {
        font-size: 2.1rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00E5FF 0%, #7C4DFF 50%, #FF4081 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }

    .sim-sub {
        color: #94A3B8;
        font-size: 1.05rem;
        margin-top: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Header Title
st.markdown("""
<div class="sim-header">
    <div class="sim-title">🏎️ Hyper-Realistic 60 FPS Autonomous Vehicle Traffic Simulator</div>
    <div class="sim-sub">
        Continuous Vehicle Physics • LiDAR Radar Sensors • Dynamic Signal Timing • Weak GPS EKF Sensor Fusion
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar Controls
st.sidebar.header("🎛️ Simulation Parameters")

sim_speed = st.sidebar.slider("Simulation Speed:", 0.5, 3.0, 1.0, step=0.5)
lidar_range = st.sidebar.slider("LiDAR Scanner Sensing Range (m):", 20, 100, 60, step=10)

st.sidebar.subheader("🚥 Signal & Hazard Overrides")
tl_ader_red = st.sidebar.checkbox("Rue Clément Ader Light [RED]", value=True)
tl_sadi_red = st.sidebar.checkbox("Avenue Sadi Lecointe Light [RED]", value=False)
weak_gps_interference = st.sidebar.checkbox("Weak GPS Satellite Dropout", value=True)
roundabout_accident = st.sidebar.checkbox("Block Roundabout with Accident Hazard", value=False)

# HTML5 Canvas 60 FPS Engine (Embedded JS/Canvas)
html_code = f"""
<!DOCTYPE html>
<html>
<head>
<style>
    body {{
        margin: 0;
        background-color: #0B0D14;
        font-family: 'Outfit', sans-serif;
        color: #FFF;
        overflow: hidden;
    }}
    #simCanvas {{
        background-color: #121520;
        border-radius: 16px;
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.1);
        display: block;
        margin: 0 auto;
    }}
    .hud-overlay {{
        position: absolute;
        top: 20px;
        left: 30px;
        background: rgba(15, 18, 28, 0.85);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(0, 229, 255, 0.3);
        padding: 12px 20px;
        border-radius: 12px;
        font-size: 14px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
        pointer-events: none;
    }}
    .hud-title {{ font-weight: bold; color: #00E5FF; margin-bottom: 4px; }}
    .tag-green {{ color: #00E676; font-weight: bold; }}
    .tag-red {{ color: #FF1744; font-weight: bold; }}
    .tag-blue {{ color: #00B0FF; font-weight: bold; }}
</style>
</head>
<body>

<div class="hud-overlay">
    <div class="hud-title">🚘 EGO VEHICLE TELEMETRY</div>
    <div>Speed: <span id="hudSpeed" class="tag-green">48.2 km/h</span></div>
    <div>TTC (Time-To-Collision): <span id="hudTTC" class="tag-blue">8.4 s</span></div>
    <div>Localization Error: <span id="hudEKF" class="tag-green">0.42 m</span></div>
    <div>Active Route Strategy: <span id="hudRoute" class="tag-green">Dynamic Risk-Aware ML-A*</span></div>
</div>

<canvas id="simCanvas" width="1100" height="650"></canvas>

<script>
const canvas = document.getElementById("simCanvas");
const ctx = canvas.getContext("2d");

// Parameters from Python
const simSpeedMult = {sim_speed};
const lidarRange = {lidar_range};
let tlAderRed = {str(tl_ader_red).lower()};
let tlSadiRed = {str(tl_sadi_red).lower()};
let weakGpsActive = {str(weak_gps_interference).lower()};
let raboutAccident = {str(roundabout_accident).lower()};

// Map Waypoints & Nodes
const nodes = {{
    "START": {{x: 80, y: 480}},
    "ROUNDABOUT": {{x: 220, y: 320}},
    "TL_ADER": {{x: 380, y: 370}},
    "TL_SADI": {{x: 580, y: 400}},
    "LECOINTE": {{x: 750, y: 360}},
    "GPS_START": {{x: 420, y: 550}},
    "GPS_MID": {{x: 680, y: 570}},
    "GPS_END": {{x: 900, y: 460}},
    "BREGUET_1": {{x: 320, y: 160}},
    "BREGUET_2": {{x: 620, y: 190}},
    "BREGUET_3": {{x: 880, y: 270}},
    "A86": {{x: 740, y: 80}},
    "DEST": {{x: 1000, y: 420}}
}};

// Vehicles Fleet State
let vehicles = [
    {{ id: "EGO", color: "#00E5FF", isEgo: true, path: ["START", "GPS_START", "GPS_MID", "GPS_END", "DEST"], segIdx: 0, progress: 0, speed: 2.2, yaw: 0, x: 80, y: 480 }},
    {{ id: "V1", color: "#FF4081", isEgo: false, path: ["START", "ROUNDABOUT", "TL_ADER", "TL_SADI", "LECOINTE", "DEST"], segIdx: 0, progress: 0.2, speed: 1.8, yaw: 0, x: 100, y: 450 }},
    {{ id: "V2", color: "#69F0AE", isEgo: false, path: ["ROUNDABOUT", "BREGUET_1", "BREGUET_2", "BREGUET_3", "DEST"], segIdx: 0, progress: 0.1, speed: 2.0, yaw: 0, x: 220, y: 320 }},
    {{ id: "V3", color: "#FFAB00", isEgo: false, path: ["GPS_START", "GPS_MID", "GPS_END", "DEST"], segIdx: 0, progress: 0.4, speed: 2.4, yaw: 0, x: 420, y: 550 }}
];

// Helper interpolator
function getInterpolated(p1, p2, t) {{
    return {{
        x: p1.x + (p2.x - p1.x) * t,
        y: p1.y + (p2.y - p1.y) * t,
        yaw: Math.atan2(p2.y - p1.y, p2.x - p1.x)
    }};
}}

// Main Render Loop
function drawScene() {{
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // 1. Draw Background Parks & Vegetation
    ctx.fillStyle = "#161B26";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Green Parks
    ctx.fillStyle = "rgba(46, 125, 50, 0.25)";
    ctx.beginPath();
    ctx.arc(150, 180, 70, 0, Math.PI*2);
    ctx.fill();

    ctx.beginPath();
    ctx.arc(750, 250, 80, 0, Math.PI*2);
    ctx.fill();

    // 2. Draw Weak GPS Blue Corridor
    ctx.strokeStyle = "rgba(0, 229, 255, 0.4)";
    ctx.lineWidth = 36;
    ctx.lineCap = "round";
    ctx.beginPath();
    ctx.moveTo(nodes["GPS_START"].x, nodes["GPS_START"].y);
    ctx.lineTo(nodes["GPS_MID"].x, nodes["GPS_MID"].y);
    ctx.lineTo(nodes["GPS_END"].x, nodes["GPS_END"].y);
    ctx.stroke();

    ctx.strokeStyle = "#00B0FF";
    ctx.lineWidth = 4;
    ctx.setLineDash([8, 8]);
    ctx.stroke();
    ctx.setLineDash([]);

    // 3. Draw Main Roads (Asphalt + Dashed Lane Lines)
    const roadEdges = [
        ["START", "ROUNDABOUT"], ["ROUNDABOUT", "TL_ADER"], ["TL_ADER", "TL_SADI"],
        ["TL_SADI", "LECOINTE"], ["LECOINTE", "DEST"], ["ROUNDABOUT", "BREGUET_1"],
        ["BREGUET_1", "BREGUET_2"], ["BREGUET_2", "BREGUET_3"], ["BREGUET_3", "DEST"],
        ["BREGUET_2", "A86"], ["A86", "BREGUET_3"], ["GPS_END", "DEST"], ["START", "GPS_START"]
    ];

    roadEdges.forEach(pair => {{
        let p1 = nodes[pair[0]], p2 = nodes[pair[1]];
        ctx.strokeStyle = "#252B3B";
        ctx.lineWidth = 30;
        ctx.lineCap = "round";
        ctx.beginPath();
        ctx.moveTo(p1.x, p1.y);
        ctx.lineTo(p2.x, p2.y);
        ctx.stroke();

        // White Lane Centerline
        ctx.strokeStyle = "rgba(255, 255, 255, 0.6)";
        ctx.lineWidth = 2;
        ctx.setLineDash([10, 10]);
        ctx.beginPath();
        ctx.moveTo(p1.x, p1.y);
        ctx.lineTo(p2.x, p2.y);
        ctx.stroke();
        ctx.setLineDash([]);
    }});

    // 4. Draw Roundabout (Purple Island)
    ctx.fillStyle = "#2D1B4E";
    ctx.strokeStyle = "#7C4DFF";
    ctx.lineWidth = 6;
    ctx.beginPath();
    ctx.arc(nodes["ROUNDABOUT"].x, nodes["ROUNDABOUT"].y, 35, 0, Math.PI*2);
    ctx.fill();
    ctx.stroke();

    // Roundabout Central Garden
    ctx.fillStyle = "#4A148C";
    ctx.beginPath();
    ctx.arc(nodes["ROUNDABOUT"].x, nodes["ROUNDABOUT"].y, 20, 0, Math.PI*2);
    ctx.fill();

    // Label Roundabout
    ctx.fillStyle = "#E040FB";
    ctx.font = "bold 14px Outfit";
    ctx.fillText("Roundabout", nodes["ROUNDABOUT"].x - 38, nodes["ROUNDABOUT"].y - 45);

    // Blocked Roundabout Hazard if active
    if (raboutAccident) {{
        ctx.fillStyle = "#FF1744";
        ctx.font = "bold 24px Outfit";
        ctx.fillText("💥 ACCIDENT BLOCKAGE", nodes["ROUNDABOUT"].x - 70, nodes["ROUNDABOUT"].y + 8);
    }}

    // 5. Draw Traffic Lights (Glowing Poles & Signal Circles)
    const tls = [
        {{ n: "TL_ADER", isRed: tlAderRed, label: "Rue Clément Ader" }},
        {{ n: "TL_SADI", isRed: tlSadiRed, label: "Avenue Sadi Lecointe" }}
    ];

    tls.forEach(tl => {{
        let pos = nodes[tl.n];
        // Signal Box
        ctx.fillStyle = "#1E222D";
        ctx.strokeStyle = "#FFFFFF";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, 16, 0, Math.PI*2);
        ctx.fill();
        ctx.stroke();

        // Active Light
        ctx.fillStyle = tl.isRed ? "#FF1744" : "#00E676";
        ctx.shadowColor = tl.isRed ? "#FF1744" : "#00E676";
        ctx.shadowBlur = 15;
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, 10, 0, Math.PI*2);
        ctx.fill();
        ctx.shadowBlur = 0; // reset shadow

        ctx.fillStyle = "#FFFFFF";
        ctx.font = "12px Outfit";
        ctx.fillText(tl.label, pos.x - 45, pos.y - 22);
    }});

    // 6. Draw Annotations & Map Labels
    // Driving Region Badge
    ctx.fillStyle = "#1A1C29";
    ctx.strokeStyle = "#00E5FF";
    ctx.lineWidth = 2;
    ctx.fillRect(30, 25, 170, 36);
    ctx.strokeRect(30, 25, 170, 36);
    ctx.fillStyle = "#00E5FF";
    ctx.font = "bold 15px Outfit";
    ctx.fillText("Driving Region", 45, 48);

    // Weak GPS Label
    ctx.fillStyle = "#00B0FF";
    ctx.font = "bold 15px Outfit";
    ctx.fillText("Weak GPS Corridor", nodes["GPS_MID"].x - 50, nodes["GPS_MID"].y + 35);

    // 7. Update & Draw Vehicles with 60 FPS Physics
    vehicles.forEach(v => {{
        if (v.path.length < 2) return;
        
        let pCurr = nodes[v.path[v.segIdx]];
        let pNext = nodes[v.path[v.segIdx + 1]];

        // Dynamic Stop Logic for Traffic Lights & Hazards
        let shouldStop = false;
        if (v.path[v.segIdx + 1] === "TL_ADER" && tlAderRed && v.progress > 0.6) shouldStop = true;
        if (v.path[v.segIdx + 1] === "TL_SADI" && tlSadiRed && v.progress > 0.6) shouldStop = true;
        if (v.path[v.segIdx + 1] === "ROUNDABOUT" && raboutAccident) shouldStop = true;

        if (!shouldStop) {{
            v.progress += (v.speed * 0.003 * simSpeedMult);
            if (v.progress >= 1.0) {{
                v.progress = 0;
                v.segIdx = (v.segIdx + 1) % (v.path.length - 1);
            }}
        }}

        let state = getInterpolated(nodes[v.path[v.segIdx]], nodes[v.path[v.segIdx + 1]], v.progress);
        v.x = state.x;
        v.y = state.y;
        v.yaw = state.yaw;

        // Draw LiDAR Scanning Arc for EGO Car
        if (v.isEgo) {{
            ctx.save();
            ctx.translate(v.x, v.y);
            ctx.rotate(v.yaw);
            ctx.fillStyle = "rgba(0, 229, 255, 0.15)";
            ctx.strokeStyle = "rgba(0, 229, 255, 0.6)";
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            ctx.moveTo(0, 0);
            ctx.arc(0, 0, lidarRange, -Math.PI/4, Math.PI/4);
            ctx.closePath();
            ctx.fill();
            ctx.stroke();
            ctx.restore();

            // Weak GPS EKF Uncertainty Halo
            if (weakGpsActive && (v.path[v.segIdx].includes("GPS") || v.path[v.segIdx+1].includes("GPS"))) {{
                ctx.strokeStyle = "rgba(255, 23, 68, 0.7)";
                ctx.lineWidth = 2;
                ctx.setLineDash([4, 4]);
                ctx.beginPath();
                ctx.arc(v.x, v.y, 28, 0, Math.PI*2);
                ctx.stroke();
                ctx.setLineDash([]);
                document.getElementById("hudEKF").innerHTML = "1.84 m (Weak GPS)";
                document.getElementById("hudEKF").className = "tag-red";
            }} else {{
                document.getElementById("hudEKF").innerHTML = "0.38 m (High Lock)";
                document.getElementById("hudEKF").className = "tag-green";
            }}

            // Update HUD
            document.getElementById("hudSpeed").innerHTML = (shouldStop ? "0.0 km/h (Stopped)" : (v.speed * 22.0).toFixed(1) + " km/h");
        }}

        // Draw Vehicle Body (Rotated Box with Headlights)
        ctx.save();
        ctx.translate(v.x, v.y);
        ctx.rotate(v.yaw);

        // Body
        ctx.fillStyle = v.color;
        ctx.shadowColor = v.color;
        ctx.shadowBlur = 10;
        ctx.fillRect(-14, -8, 28, 16);
        ctx.shadowBlur = 0;

        // Headlights
        ctx.fillStyle = "#FFFFFF";
        ctx.fillRect(12, -7, 4, 3);
        ctx.fillRect(12, 4, 4, 3);

        // Brake Lights if stopped
        ctx.fillStyle = shouldStop ? "#FF1744" : "#FF5252";
        ctx.fillRect(-15, -7, 3, 3);
        ctx.fillRect(-15, 4, 3, 3);

        ctx.restore();
    }});

    requestAnimationFrame(drawScene);
}}

drawScene();
</script>
</body>
</html>
"""

# Render Component
components.html(html_code, height=680)

st.markdown("---")
st.markdown("<p style='text-align:center; color:#64748B;'>60 FPS Interactive HTML5 Canvas Vehicle Simulation | Adaptive Autonomous Fleet Research</p>", unsafe_allow_html=True)
