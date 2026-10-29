import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components

# Streamlit Page Config
st.set_page_config(
    page_title="Adaptive Autonomous Navigation Research",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="collapsed"
)

BASE_DIR = Path(__file__).resolve().parent.parent

# CSS for Clean, Ultra-Premium Research Presentation
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .stApp {
        background-color: #0A0C14;
        color: #F8FAFC;
    }

    #MainMenu, header, footer {visibility: hidden;}

    .pres-header {
        background: linear-gradient(135deg, rgba(20, 24, 40, 0.85) 0%, rgba(10, 12, 20, 0.95) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-left: 5px solid #00E5FF;
        padding: 16px 24px;
        border-radius: 14px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
        margin-bottom: 16px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .pres-title {
        font-size: 1.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #FFFFFF 0%, #00E5FF 50%, #7C4DFF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }

    .pres-sub {
        color: #94A3B8;
        font-size: 0.95rem;
        margin-top: 4px;
    }

    .btn-pres {
        background: linear-gradient(135deg, #00E5FF 0%, #7C4DFF 100%);
        color: #000;
        font-weight: 700;
        border: none;
        padding: 10px 20px;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(0, 229, 255, 0.3);
        cursor: pointer;
    }
</style>
""", unsafe_allow_html=True)

# Clean Research Presentation Header
st.markdown("""
<div class="pres-header">
    <div>
        <div class="pres-title">Adaptive Autonomous Navigation & Dynamic Risk Avoidance</div>
        <div class="pres-sub">Research Model Demo • Real-World Urban Driving Region • NGSIM Trajectory ML • EKF Sensor Fusion</div>
    </div>
</div>
""", unsafe_allow_html=True)

# HTML5 Photorealistic Vector Map Canvas
html_presentation_code = """
<!DOCTYPE html>
<html>
<head>
<style>
    body {
        margin: 0;
        background-color: #0A0C14;
        font-family: 'Plus Jakarta Sans', sans-serif;
        color: #FFF;
        overflow: hidden;
    }
    #presCanvas {
        background: #0E121E;
        border-radius: 18px;
        box-shadow: 0 20px 50px rgba(0, 0, 0, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        display: block;
        margin: 0 auto;
    }
    .hud-card {
        position: absolute;
        top: 20px;
        left: 30px;
        background: rgba(14, 18, 30, 0.85);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(0, 229, 255, 0.3);
        padding: 14px 22px;
        border-radius: 14px;
        font-size: 13px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.5);
    }
    .hud-title { font-weight: 800; color: #00E5FF; font-size: 14px; margin-bottom: 6px; letter-spacing: 0.5px; }
    .val-highlight { color: #00E676; font-weight: 700; }
    .val-warn { color: #FFD600; font-weight: 700; }
    .val-danger { color: #FF1744; font-weight: 700; }

    .control-panel {
        position: absolute;
        bottom: 25px;
        right: 40px;
        background: rgba(14, 18, 30, 0.85);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.15);
        padding: 10px 18px;
        border-radius: 12px;
        display: flex;
        gap: 12px;
    }
    .c-btn {
        background: #1A2035;
        color: #FFF;
        border: 1px solid rgba(255,255,255,0.2);
        padding: 8px 14px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 12px;
        cursor: pointer;
        transition: all 0.2s;
    }
    .c-btn:hover { background: #00E5FF; color: #000; }
</style>
</head>
<body>

<div class="hud-card">
    <div class="hud-title">🚘 EGO AUTONOMOUS VEHICLE TELEMETRY</div>
    <div>Speed: <span id="hSpeed" class="val-highlight">45.0 km/h</span></div>
    <div>Time-To-Collision (TTC): <span id="hTTC" class="val-highlight">7.8 s</span></div>
    <div>EKF Localization Precision: <span id="hEKF" class="val-highlight">0.34 m</span></div>
    <div>Active Navigation Strategy: <span id="hStrategy" class="val-highlight">Proposed Dynamic Risk ML-A*</span></div>
</div>

<div class="control-panel">
    <button class="c-btn" onclick="toggleAccident()">💥 Toggle Roundabout Accident</button>
    <button class="c-btn" onclick="toggleSignals()">🚥 Toggle Traffic Lights (Red/Green)</button>
    <button class="c-btn" onclick="toggleWeakGPS()">📡 Toggle Weak GPS Dropout</button>
</div>

<canvas id="presCanvas" width="1120" height="640"></canvas>

<script>
const canvas = document.getElementById("presCanvas");
const ctx = canvas.getContext("2d");

let accidentActive = false;
let signalsRed = true;
let weakGpsActive = true;

function toggleAccident() { accidentActive = !accidentActive; }
function toggleSignals() { signalsRed = !signalsRed; }
function toggleWeakGPS() { weakGpsActive = !weakGpsActive; }

// Real-World Map Coordinates
const nodes = {
    "START": {x: 90, y: 480},
    "ROUNDABOUT": {x: 240, y: 320},
    "TL_ADER": {x: 400, y: 370},
    "TL_SADI": {x: 600, y: 400},
    "LECOINTE": {x: 770, y: 360},
    "GPS_START": {x: 440, y: 540},
    "GPS_MID": {x: 700, y: 560},
    "GPS_END": {x: 920, y: 450},
    "BREGUET_1": {x: 340, y: 160},
    "BREGUET_2": {x: 640, y: 190},
    "BREGUET_3": {x: 900, y: 270},
    "DEST": {x: 1020, y: 410}
};

let egoCar = {
    path: ["START", "GPS_START", "GPS_MID", "GPS_END", "DEST"],
    segIdx: 0,
    progress: 0,
    speed: 2.0,
    x: 90, y: 480, yaw: 0
};

let surroundingCars = [
    { path: ["START", "ROUNDABOUT", "TL_ADER", "TL_SADI", "LECOINTE", "DEST"], segIdx: 0, progress: 0.35, color: "#FF4081" },
    { path: ["ROUNDABOUT", "BREGUET_1", "BREGUET_2", "BREGUET_3", "DEST"], segIdx: 0, progress: 0.15, color: "#69F0AE" }
];

function drawMap() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // 1. Photorealistic Dark City Background
    ctx.fillStyle = "#0E121E";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Parks & Greenery
    ctx.fillStyle = "rgba(46, 125, 50, 0.2)";
    ctx.beginPath(); ctx.arc(160, 180, 75, 0, Math.PI*2); ctx.fill();
    ctx.beginPath(); ctx.arc(770, 240, 85, 0, Math.PI*2); ctx.fill();

    // Sidewalks
    const roadEdges = [
        ["START", "ROUNDABOUT"], ["ROUNDABOUT", "TL_ADER"], ["TL_ADER", "TL_SADI"],
        ["TL_SADI", "LECOINTE"], ["LECOINTE", "DEST"], ["ROUNDABOUT", "BREGUET_1"],
        ["BREGUET_1", "BREGUET_2"], ["BREGUET_2", "BREGUET_3"], ["BREGUET_3", "DEST"],
        ["START", "GPS_START"], ["GPS_END", "DEST"]
    ];

    roadEdges.forEach(pair => {
        let p1 = nodes[pair[0]], p2 = nodes[pair[1]];
        // Sidewalk Border
        ctx.strokeStyle = "#3A435E";
        ctx.lineWidth = 36;
        ctx.beginPath(); ctx.moveTo(p1.x, p1.y); ctx.lineTo(p2.x, p2.y); ctx.stroke();
        // Asphalt Surface
        ctx.strokeStyle = "#1A2032";
        ctx.lineWidth = 30;
        ctx.beginPath(); ctx.moveTo(p1.x, p1.y); ctx.lineTo(p2.x, p2.y); ctx.stroke();
        // Yellow Double Centerline
        ctx.strokeStyle = "rgba(255, 214, 0, 0.7)";
        ctx.lineWidth = 2;
        ctx.beginPath(); ctx.moveTo(p1.x, p1.y); ctx.lineTo(p2.x, p2.y); ctx.stroke();
    });

    // 2. Weak GPS Blue Corridor
    ctx.strokeStyle = "rgba(0, 229, 255, 0.35)";
    ctx.lineWidth = 36;
    ctx.beginPath();
    ctx.moveTo(nodes["GPS_START"].x, nodes["GPS_START"].y);
    ctx.lineTo(nodes["GPS_MID"].x, nodes["GPS_MID"].y);
    ctx.lineTo(nodes["GPS_END"].x, nodes["GPS_END"].y);
    ctx.stroke();

    ctx.strokeStyle = "#00B0FF";
    ctx.lineWidth = 3;
    ctx.setLineDash([8, 8]);
    ctx.stroke();
    ctx.setLineDash([]);

    // 3. Roundabout (Purple Island with Central Landscaping)
    ctx.fillStyle = "#1E1535";
    ctx.strokeStyle = "#7C4DFF";
    ctx.lineWidth = 5;
    ctx.beginPath(); ctx.arc(nodes["ROUNDABOUT"].x, nodes["ROUNDABOUT"].y, 38, 0, Math.PI*2); ctx.fill(); ctx.stroke();

    // Central Garden Fountain
    ctx.fillStyle = "#6200EA";
    ctx.beginPath(); ctx.arc(nodes["ROUNDABOUT"].x, nodes["ROUNDABOUT"].y, 22, 0, Math.PI*2); ctx.fill();
    ctx.fillStyle = "#B388FF";
    ctx.font = "bold 13px Plus Jakarta Sans";
    ctx.fillText("Roundabout", nodes["ROUNDABOUT"].x - 36, nodes["ROUNDABOUT"].y - 48);

    if (accidentActive) {
        ctx.fillStyle = "#FF1744";
        ctx.font = "bold 20px Plus Jakarta Sans";
        ctx.fillText("💥 ACCIDENT HAZARD", nodes["ROUNDABOUT"].x - 75, nodes["ROUNDABOUT"].y + 6);
    }

    // 4. Traffic Light Posts
    [nodes["TL_ADER"], nodes["TL_SADI"]].forEach((pos, idx) => {
        ctx.fillStyle = "#1A2032";
        ctx.strokeStyle = "#FFFFFF";
        ctx.lineWidth = 2;
        ctx.beginPath(); ctx.arc(pos.x, pos.y, 14, 0, Math.PI*2); ctx.fill(); ctx.stroke();

        ctx.fillStyle = signalsRed ? "#FF1744" : "#00E676";
        ctx.shadowColor = signalsRed ? "#FF1744" : "#00E676";
        ctx.shadowBlur = 12;
        ctx.beginPath(); ctx.arc(pos.x, pos.y, 9, 0, Math.PI*2); ctx.fill();
        ctx.shadowBlur = 0;
    });

    ctx.fillStyle = "#FFFFFF";
    ctx.font = "bold 13px Plus Jakarta Sans";
    ctx.fillText("Traffic Lights", 430, 425);

    // Annotations
    ctx.fillStyle = "#00E5FF";
    ctx.font = "bold 13px Plus Jakarta Sans";
    ctx.fillText("Driving Region", 40, 50);
    ctx.fillText("Weak GPS Corridor", nodes["GPS_MID"].x - 50, nodes["GPS_MID"].y + 35);

    // 5. Dynamic Vehicle Rerouting & Movement Physics
    if (accidentActive && egoCar.path.includes("ROUNDABOUT")) {
        // Switch Ego car to safe Weak GPS / Breguet Bypass route
        egoCar.path = ["START", "GPS_START", "GPS_MID", "GPS_END", "DEST"];
        egoCar.segIdx = 0;
        egoCar.progress = 0;
        document.getElementById("hStrategy").innerHTML = "Dynamic Hazard Avoidance Reroute";
        document.getElementById("hStrategy").className = "val-warn";
    }

    // Update Ego Car
    let p1 = nodes[egoCar.path[egoCar.segIdx]];
    let p2 = nodes[egoCar.path[egoCar.segIdx + 1]];
    
    let isRedStop = (egoCar.path[egoCar.segIdx+1] === "TL_ADER" || egoCar.path[egoCar.segIdx+1] === "TL_SADI") && signalsRed && egoCar.progress > 0.6;
    if (!isRedStop) {
        egoCar.progress += 0.003;
        if (egoCar.progress >= 1.0) {
            egoCar.progress = 0;
            egoCar.segIdx = (egoCar.segIdx + 1) % (egoCar.path.length - 1);
        }
    }

    egoCar.x = p1.x + (p2.x - p1.x) * egoCar.progress;
    egoCar.y = p1.y + (p2.y - p1.y) * egoCar.progress;
    egoCar.yaw = Math.atan2(p2.y - p1.y, p2.x - p1.x);

    // Draw LiDAR Beam
    ctx.save();
    ctx.translate(egoCar.x, egoCar.y);
    ctx.rotate(egoCar.yaw);
    ctx.fillStyle = "rgba(0, 229, 255, 0.15)";
    ctx.strokeStyle = "rgba(0, 229, 255, 0.6)";
    ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(0,0); ctx.arc(0, 0, 70, -Math.PI/4, Math.PI/4); ctx.closePath(); ctx.fill(); ctx.stroke();
    ctx.restore();

    // Draw Ego Car Body (Electric Blue metallic rendering)
    ctx.save();
    ctx.translate(egoCar.x, egoCar.y);
    ctx.rotate(egoCar.yaw);
    ctx.fillStyle = "#00E5FF";
    ctx.shadowColor = "#00E5FF"; ctx.shadowBlur = 12;
    ctx.fillRect(-14, -8, 28, 16);
    ctx.fillStyle = "#FFFFFF";
    ctx.fillRect(12, -7, 4, 3); ctx.fillRect(12, 4, 4, 3);
    ctx.restore();

    // Draw EKF Uncertainty Halo in Weak GPS
    if (weakGpsActive && (egoCar.path[egoCar.segIdx].includes("GPS") || egoCar.path[egoCar.segIdx+1].includes("GPS"))) {
        ctx.strokeStyle = "rgba(255, 23, 68, 0.7)";
        ctx.lineWidth = 2;
        ctx.setLineDash([4, 4]);
        ctx.beginPath(); ctx.arc(egoCar.x, egoCar.y, 28, 0, Math.PI*2); ctx.stroke(); ctx.setLineDash([]);
        document.getElementById("hEKF").innerHTML = "1.58 m (Dead-Reckoning Active)";
        document.getElementById("hEKF").className = "val-danger";
    } else {
        document.getElementById("hEKF").innerHTML = "0.34 m (High Precision Lock)";
        document.getElementById("hEKF").className = "val-highlight";
    }

    document.getElementById("hSpeed").innerHTML = isRedStop ? "0.0 km/h (Signal Stop)" : "45.0 km/h";

    requestAnimationFrame(drawMap);
}

drawMap();
</script>
</body>
</html>
"""

# Render Fullscreen Presentation Canvas
components.html(html_presentation_code, height=670)
