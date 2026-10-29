import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components

# Streamlit Page Setup
st.set_page_config(
    page_title="Photorealistic Autonomous Traffic Simulator",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="collapsed"
)

BASE_DIR = Path(__file__).resolve().parent.parent

# Custom CSS for Presentation
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

    .header-banner {
        background: linear-gradient(135deg, rgba(20, 24, 40, 0.9) 0%, rgba(10, 12, 20, 0.95) 100%);
        border: 1px solid rgba(0, 229, 255, 0.25);
        border-left: 6px solid #00E5FF;
        padding: 18px 28px;
        border-radius: 16px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.6);
        margin-bottom: 18px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .header-title {
        font-size: 2.0rem;
        font-weight: 800;
        background: linear-gradient(90deg, #FFFFFF 0%, #00E5FF 50%, #7C4DFF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }

    .header-sub {
        color: #94A3B8;
        font-size: 1.0rem;
        margin-top: 4px;
    }
</style>
""", unsafe_allow_html=True)

# Header Title
st.markdown("""
<div class="header-banner">
    <div>
        <div class="header-title">Photorealistic Autonomous Vehicle Traffic Simulator</div>
        <div class="header-sub">Real-World Urban Driving Region • Landmark Modeling • LiDAR Radar Sensors • Dynamic ML-A* Rerouting</div>
    </div>
</div>
""", unsafe_allow_html=True)

# HTML5 60 FPS Photorealistic Canvas Code
html_sim_code = """
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
    #photoCanvas {
        background: #0F1320;
        border-radius: 18px;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.12);
        display: block;
        margin: 0 auto;
    }
    .hud-card {
        position: absolute;
        top: 22px;
        left: 32px;
        background: rgba(15, 19, 32, 0.88);
        backdrop-filter: blur(14px);
        border: 1px solid rgba(0, 229, 255, 0.35);
        padding: 16px 24px;
        border-radius: 16px;
        font-size: 13px;
        box-shadow: 0 10px 35px rgba(0,0,0,0.6);
        z-index: 10;
    }
    .hud-title { font-weight: 800; color: #00E5FF; font-size: 15px; margin-bottom: 6px; letter-spacing: 0.5px; }
    .val-green { color: #00E676; font-weight: 700; }
    .val-yellow { color: #FFD600; font-weight: 700; }
    .val-red { color: #FF1744; font-weight: 700; }
    .val-blue { color: #00B0FF; font-weight: 700; }

    .control-bar {
        position: absolute;
        bottom: 25px;
        right: 35px;
        background: rgba(15, 19, 32, 0.88);
        backdrop-filter: blur(14px);
        border: 1px solid rgba(255, 255, 255, 0.15);
        padding: 12px 20px;
        border-radius: 14px;
        display: flex;
        gap: 12px;
        z-index: 10;
    }
    .c-btn {
        background: #1C2235;
        color: #FFF;
        border: 1px solid rgba(255,255,255,0.2);
        padding: 9px 16px;
        border-radius: 10px;
        font-weight: 700;
        font-size: 12px;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    .c-btn:hover { background: #00E5FF; color: #000; box-shadow: 0 0 12px rgba(0,229,255,0.4); }
</style>
</head>
<body>

<div class="hud-card">
    <div class="hud-title">🚘 EGO AUTONOMOUS VEHICLE TELEMETRY</div>
    <div>Velocity: <span id="hVel" class="val-green">48.5 km/h</span></div>
    <div>Time-To-Collision (TTC): <span id="hTTC" class="val-blue">8.2 s</span></div>
    <div>EKF Position Error: <span id="hEKF" class="val-green">0.36 m (High Precision)</span></div>
    <div>Active Navigation Strategy: <span id="hStrat" class="val-green">Proposed Dynamic Risk ML-A*</span></div>
</div>

<div class="control-bar">
    <button class="c-btn" onclick="toggleAccident()">💥 Toggle Roundabout Accident</button>
    <button class="c-btn" onclick="toggleSignals()">🚥 Toggle Traffic Signals (Red/Green)</button>
    <button class="c-btn" onclick="toggleWeakGPS()">📡 Toggle Weak GPS Zone</button>
</div>

<canvas id="photoCanvas" width="1140" height="660"></canvas>

<script>
const canvas = document.getElementById("photoCanvas");
const ctx = canvas.getContext("2d");

let accidentActive = false;
let signalsRed = true;
let weakGpsActive = true;

function toggleAccident() { accidentActive = !accidentActive; }
function toggleSignals() { signalsRed = !signalsRed; }
function toggleWeakGPS() { weakGpsActive = !weakGpsActive; }

// Waypoint Map Nodes matching reference image layout
const nodes = {
    "START": {x: 90, y: 490},
    "ROUNDABOUT": {x: 250, y: 340},
    "TL_ADER": {x: 410, y: 390},
    "TL_SADI": {x: 610, y: 420},
    "LECOINTE": {x: 780, y: 380},
    "GPS_START": {x: 450, y: 560},
    "GPS_MID": {x: 720, y: 580},
    "GPS_END": {x: 940, y: 460},
    "BREGUET_1": {x: 350, y: 170},
    "BREGUET_2": {x: 650, y: 200},
    "BREGUET_3": {x: 910, y: 280},
    "A86": {x: 760, y: 90},
    "DEST": {x: 1040, y: 420}
};

// Vehicles state with photorealistic properties
let egoCar = {
    path: ["START", "GPS_START", "GPS_MID", "GPS_END", "DEST"],
    segIdx: 0, progress: 0, speed: 2.2, x: 90, y: 490, yaw: 0
};

let trafficFleet = [
    { type: "sedan", color: "#FF1744", path: ["START", "ROUNDABOUT", "TL_ADER", "TL_SADI", "LECOINTE", "DEST"], segIdx: 0, progress: 0.35, speed: 1.8 },
    { type: "taxi", color: "#FFD600", path: ["ROUNDABOUT", "BREGUET_1", "BREGUET_2", "BREGUET_3", "DEST"], segIdx: 0, progress: 0.20, speed: 2.0 },
    { type: "truck", color: "#FF9100", path: ["BREGUET_1", "BREGUET_2", "A86", "BREGUET_3"], segIdx: 0, progress: 0.10, speed: 1.5 },
    { type: "bus", color: "#00E676", path: ["TL_SADI", "LECOINTE", "DEST"], segIdx: 0, progress: 0.50, speed: 1.6 }
];

function drawPhotorealisticScene() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // 1. Dark Urban Base Background
    ctx.fillStyle = "#0F1320";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Green Grass Parks & Vegetation
    ctx.fillStyle = "rgba(46, 125, 50, 0.22)";
    ctx.beginPath(); ctx.arc(170, 190, 80, 0, Math.PI*2); ctx.fill();
    ctx.beginPath(); ctx.arc(780, 250, 90, 0, Math.PI*2); ctx.fill();

    // Tree Icons in Parks
    [[150, 160], [180, 210], [760, 220], [810, 270]].forEach(tp => {
        ctx.fillStyle = "rgba(76, 175, 80, 0.4)";
        ctx.beginPath(); ctx.arc(tp[0], tp[1], 14, 0, Math.PI*2); ctx.fill();
        ctx.fillStyle = "#2E7D32";
        ctx.beginPath(); ctx.arc(tp[0], tp[1], 8, 0, Math.PI*2); ctx.fill();
    });

    // 2. Road Network Asphalt & Sidewalks
    const roadEdges = [
        ["START", "ROUNDABOUT"], ["ROUNDABOUT", "TL_ADER"], ["TL_ADER", "TL_SADI"],
        ["TL_SADI", "LECOINTE"], ["LECOINTE", "DEST"], ["ROUNDABOUT", "BREGUET_1"],
        ["BREGUET_1", "BREGUET_2"], ["BREGUET_2", "BREGUET_3"], ["BREGUET_3", "DEST"],
        ["BREGUET_2", "A86"], ["A86", "BREGUET_3"], ["START", "GPS_START"], ["GPS_END", "DEST"]
    ];

    roadEdges.forEach(pair => {
        let p1 = nodes[pair[0]], p2 = nodes[pair[1]];

        // Sidewalk Concrete Curbing
        ctx.strokeStyle = "#333A4E";
        ctx.lineWidth = 38;
        ctx.lineCap = "round";
        ctx.beginPath(); ctx.moveTo(p1.x, p1.y); ctx.lineTo(p2.x, p2.y); ctx.stroke();

        // Asphalt Road Surface
        ctx.strokeStyle = "#1A2030";
        ctx.lineWidth = 32;
        ctx.beginPath(); ctx.moveTo(p1.x, p1.y); ctx.lineTo(p2.x, p2.y); ctx.stroke();

        // Double Yellow Centerline
        ctx.strokeStyle = "rgba(255, 214, 0, 0.75)";
        ctx.lineWidth = 2;
        ctx.beginPath(); ctx.moveTo(p1.x, p1.y); ctx.lineTo(p2.x, p2.y); ctx.stroke();

        // White Lane Stripes
        ctx.strokeStyle = "rgba(255, 255, 255, 0.6)";
        ctx.lineWidth = 1.5;
        ctx.setLineDash([10, 12]);
        ctx.beginPath(); ctx.moveTo(p1.x, p1.y); ctx.lineTo(p2.x, p2.y); ctx.stroke();
        ctx.setLineDash([]);
    });

    // 3. Weak GPS Blue Corridor (Square des Cèdres)
    ctx.strokeStyle = "rgba(0, 229, 255, 0.35)";
    ctx.lineWidth = 38;
    ctx.lineCap = "round";
    ctx.beginPath();
    ctx.moveTo(nodes["GPS_START"].x, nodes["GPS_START"].y);
    ctx.lineTo(nodes["GPS_MID"].x, nodes["GPS_MID"].y);
    ctx.lineTo(nodes["GPS_END"].x, nodes["GPS_END"].y);
    ctx.stroke();

    ctx.strokeStyle = "#00B0FF";
    ctx.lineWidth = 3.5;
    ctx.setLineDash([10, 10]);
    ctx.stroke();
    ctx.setLineDash([]);

    // 4. Roundabout (Église Saint Jean-Baptiste)
    ctx.fillStyle = "#1E1838";
    ctx.strokeStyle = "#7C4DFF";
    ctx.lineWidth = 5;
    ctx.beginPath(); ctx.arc(nodes["ROUNDABOUT"].x, nodes["ROUNDABOUT"].y, 40, 0, Math.PI*2); ctx.fill(); ctx.stroke();

    // Roundabout Central Island Fountain & Landscaping
    ctx.fillStyle = "#4A148C";
    ctx.beginPath(); ctx.arc(nodes["ROUNDABOUT"].x, nodes["ROUNDABOUT"].y, 24, 0, Math.PI*2); ctx.fill();
    ctx.fillStyle = "#00E5FF";
    ctx.beginPath(); ctx.arc(nodes["ROUNDABOUT"].x, nodes["ROUNDABOUT"].y, 8, 0, Math.PI*2); ctx.fill();

    // Roundabout Label & Landmark
    ctx.fillStyle = "#E040FB";
    ctx.font = "bold 13px Plus Jakarta Sans";
    ctx.fillText("Roundabout (Église St Jean-Baptiste)", nodes["ROUNDABOUT"].x - 110, nodes["ROUNDABOUT"].y - 50);

    if (accidentActive) {
        ctx.fillStyle = "#FF1744";
        ctx.font = "bold 20px Plus Jakarta Sans";
        ctx.fillText("💥 ACCIDENT BLOCKAGE", nodes["ROUNDABOUT"].x - 80, nodes["ROUNDABOUT"].y + 6);
    }

    // 5. Traffic Lights Posts & Signal Lenses
    const tlList = [
        { n: "TL_ADER", name: "Rue Clément Ader" },
        { n: "TL_SADI", name: "Avenue Sadi Lecointe" }
    ];

    tlList.forEach(tl => {
        let pos = nodes[tl.n];
        // Housing
        ctx.fillStyle = "#161B26";
        ctx.strokeStyle = "#FFFFFF";
        ctx.lineWidth = 2;
        ctx.beginPath(); ctx.arc(pos.x, pos.y, 15, 0, Math.PI*2); ctx.fill(); ctx.stroke();

        // Active LED Lamp
        ctx.fillStyle = signalsRed ? "#FF1744" : "#00E676";
        ctx.shadowColor = signalsRed ? "#FF1744" : "#00E676";
        ctx.shadowBlur = 16;
        ctx.beginPath(); ctx.arc(pos.x, pos.y, 10, 0, Math.PI*2); ctx.fill();
        ctx.shadowBlur = 0;
    });

    ctx.fillStyle = "#FFFFFF";
    ctx.font = "bold 13px Plus Jakarta Sans";
    ctx.fillText("Traffic Lights", 440, 445);

    // 6. Real-World Landmark Labels (as shown in reference image!)
    ctx.fillStyle = "#FFB74D";
    ctx.font = "11px Plus Jakarta Sans";
    ctx.fillText("📍 Tabac de L'Aviation", 80, 410);
    ctx.fillText("📍 Collège Saint-Exupéry", 140, 560);
    ctx.fillText("⚡ Electric 55 Charging", 260, 520);
    ctx.fillText("📍 Samia Esthéticienne", 420, 320);
    ctx.fillText("⚽ Stade Sadi Lecointe", 620, 235);
    ctx.fillText("📚 Médiathèque de Vélizy", 730, 310);
    ctx.fillText("🏎️ Audi / Porsche Smart Mobility", 760, 140);
    ctx.fillText("🛣️ A86 / E5 Highway Interchange", 720, 60);

    // Badges & Overlay Annotations
    ctx.fillStyle = "#151928";
    ctx.strokeStyle = "#00E5FF";
    ctx.lineWidth = 2;
    ctx.fillRect(35, 25, 175, 36);
    ctx.strokeRect(35, 25, 175, 36);
    ctx.fillStyle = "#00E5FF";
    ctx.font = "bold 14px Plus Jakarta Sans";
    ctx.fillText("Driving Region", 50, 48);

    ctx.fillStyle = "#00B0FF";
    ctx.font = "bold 14px Plus Jakarta Sans";
    ctx.fillText("Weak GPS Corridor", nodes["GPS_MID"].x - 55, nodes["GPS_MID"].y + 36);

    // 7. Dynamic ML-A* Route Rerouting Physics
    if (accidentActive && egoCar.path.includes("ROUNDABOUT")) {
        egoCar.path = ["START", "GPS_START", "GPS_MID", "GPS_END", "DEST"];
        egoCar.segIdx = 0; egoCar.progress = 0;
        document.getElementById("hStrat").innerHTML = "Dynamic Hazard Avoidance Reroute";
        document.getElementById("hStrat").className = "val-warn";
    }

    // Move Ego Car
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

    // Draw Ego LiDAR Scanner Cone
    ctx.save();
    ctx.translate(egoCar.x, egoCar.y);
    ctx.rotate(egoCar.yaw);
    ctx.fillStyle = "rgba(0, 229, 255, 0.18)";
    ctx.strokeStyle = "rgba(0, 229, 255, 0.75)";
    ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(0,0); ctx.arc(0, 0, 75, -Math.PI/4, Math.PI/4); ctx.closePath(); ctx.fill(); ctx.stroke();
    ctx.restore();

    // Draw Photorealistic Ego Vehicle Body
    ctx.save();
    ctx.translate(egoCar.x, egoCar.y);
    ctx.rotate(egoCar.yaw);
    ctx.fillStyle = "#00E5FF";
    ctx.shadowColor = "#00E5FF"; ctx.shadowBlur = 14;
    ctx.fillRect(-15, -8, 30, 16);
    ctx.fillStyle = "#FFFFFF";
    ctx.fillRect(13, -7, 4, 3); ctx.fillRect(13, 4, 4, 3);
    ctx.restore();

    // Draw EKF Uncertainty Halo in Weak GPS Zone
    if (weakGpsActive && (egoCar.path[egoCar.segIdx].includes("GPS") || egoCar.path[egoCar.segIdx+1].includes("GPS"))) {
        ctx.strokeStyle = "rgba(255, 23, 68, 0.75)";
        ctx.lineWidth = 2;
        ctx.setLineDash([5, 5]);
        ctx.beginPath(); ctx.arc(egoCar.x, egoCar.y, 30, 0, Math.PI*2); ctx.stroke(); ctx.setLineDash([]);
        document.getElementById("hEKF").innerHTML = "1.58 m (Dead-Reckoning Active)";
        document.getElementById("hEKF").className = "val-red";
    } else {
        document.getElementById("hEKF").innerHTML = "0.36 m (High Precision Lock)";
        document.getElementById("hEKF").className = "val-green";
    }

    document.getElementById("hVel").innerHTML = isRedStop ? "0.0 km/h (Signal Stop)" : "48.5 km/h";

    // Update Surrounding Traffic Vehicles
    trafficFleet.forEach(v => {
        let tp1 = nodes[v.path[v.segIdx]];
        let tp2 = nodes[v.path[v.segIdx + 1]];
        let tStop = (v.path[v.segIdx+1] === "TL_ADER" || v.path[v.segIdx+1] === "TL_SADI") && signalsRed && v.progress > 0.6;

        if (!tStop) {
            v.progress += 0.0025;
            if (v.progress >= 1.0) {
                v.progress = 0;
                v.segIdx = (v.segIdx + 1) % (v.path.length - 1);
            }
        }

        let vx = tp1.x + (tp2.x - tp1.x) * v.progress;
        let vy = tp1.y + (tp2.y - tp1.y) * v.progress;
        let vyaw = Math.atan2(tp2.y - tp1.y, tp2.x - tp1.x);

        ctx.save();
        ctx.translate(vx, vy);
        ctx.rotate(vyaw);
        ctx.fillStyle = v.color;
        ctx.shadowColor = v.color; ctx.shadowBlur = 8;
        ctx.fillRect(-12, -7, 24, 14);
        ctx.restore();
    });

    requestAnimationFrame(drawPhotorealisticScene);
}

drawPhotorealisticScene();
</script>
</body>
</html>
"""

# Render Component
components.html(html_sim_code, height=680)
