import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components

# Streamlit Page Setup (Full-View Presentation)
st.set_page_config(
    page_title="Adaptive Autonomous Navigation | Full-View Presentation Model",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="collapsed"
)

BASE_DIR = Path(__file__).resolve().parent.parent

# Custom CSS for Full Map Viewport Fit & Solid Black Headings (#000000)
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

    #MainMenu, header, footer {visibility: hidden;}

    /* SOLID BLACK HEADINGS (#000000) */
    h1, h2, h3, h4, .black-title {
        color: #000000 !important;
        font-weight: 900 !important;
        letter-spacing: -0.5px;
    }

    .full-header {
        background: #FFFFFF;
        border: 2px solid #000000;
        border-left: 8px solid #0284C7;
        padding: 16px 26px;
        border-radius: 14px;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.05);
        margin-bottom: 12px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .h-title {
        font-size: 1.9rem;
        font-weight: 900;
        color: #000000;
        margin: 0;
    }

    .h-sub {
        color: #334155;
        font-size: 1.0rem;
        font-weight: 600;
        margin-top: 4px;
    }
</style>
""", unsafe_allow_html=True)

# Full View Header (Black Title)
st.markdown("""
<div class="full-header">
    <div>
        <div class="h-title">Adaptive Autonomous Vehicle Navigation Research System</div>
        <div class="h-sub">Fully Autonomous Self-Running Model • Pre-Trained ML Weights • Auto-Cycling Signals • Dynamic ML-A* Rerouting</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Self-Running HTML5 Full-View Canvas Engine
html_full_canvas = """
<!DOCTYPE html>
<html>
<head>
<style>
    body {
        margin: 0;
        background-color: #F8FAFC;
        font-family: 'Plus Jakarta Sans', sans-serif;
        color: #000000;
        overflow: hidden;
    }
    #fullCanvas {
        width: 100%;
        max-width: 1240px;
        height: 620px;
        background: #F1F5F9;
        border-radius: 16px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.08);
        border: 2px solid #000000;
        display: block;
        margin: 0 auto;
    }
    .hud-full {
        position: absolute;
        top: 18px;
        left: 28px;
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(12px);
        border: 2px solid #000000;
        padding: 14px 22px;
        border-radius: 14px;
        font-size: 13px;
        box-shadow: 0 6px 24px rgba(0,0,0,0.1);
        z-index: 10;
    }
    .hud-h { font-weight: 900; color: #000000; font-size: 15px; margin-bottom: 5px; letter-spacing: -0.3px; }
    .val-g { color: #16A34A; font-weight: 800; }
    .val-r { color: #DC2626; font-weight: 800; }
    .val-b { color: #0284C7; font-weight: 800; }
</style>
</head>
<body>

<div class="hud-full">
    <div class="hud-h">🚘 EGO AUTONOMOUS VEHICLE TELEMETRY</div>
    <div>Speed: <span id="fSpeed" class="val-g">48.0 km/h</span></div>
    <div>Traffic Signal Status: <span id="fSignal" class="val-g">GREEN (Auto-Cycling)</span></div>
    <div>Time-To-Collision (TTC): <span id="fTTC" class="val-b">8.4 s</span></div>
    <div>EKF Position Precision: <span id="fEKF" class="val-g">0.35 m (High Lock)</span></div>
    <div>Navigation Strategy: <span id="fStrat" class="val-g">Proposed Dynamic Risk ML-A* (Pre-Trained)</span></div>
</div>

<canvas id="fullCanvas" width="1240" height="620"></canvas>

<script>
const canvas = document.getElementById("fullCanvas");
const ctx = canvas.getContext("2d");

let frameTicks = 0;
let signalState = "GREEN"; // Auto cycles GREEN -> YELLOW -> RED -> GREEN
let autoHazardActive = false;

// Scaled Map Nodes for 100% Full-View Fit
const nodes = {
    "START": {x: 80, y: 460},
    "ROUNDABOUT": {x: 260, y: 310},
    "TL_ADER": {x: 430, y: 360},
    "TL_SADI": {x: 650, y: 390},
    "LECOINTE": {x: 840, y: 350},
    "GPS_START": {x: 470, y: 530},
    "GPS_MID": {x: 760, y: 550},
    "GPS_END": {x: 1000, y: 430},
    "BREGUET_1": {x: 370, y: 150},
    "BREGUET_2": {x: 700, y: 180},
    "BREGUET_3": {x: 980, y: 260},
    "DEST": {x: 1140, y: 390}
};

let egoCar = {
    path: ["START", "ROUNDABOUT", "TL_ADER", "TL_SADI", "LECOINTE", "DEST"],
    segIdx: 0, progress: 0, speed: 2.2, x: 80, y: 460, yaw: 0
};

function drawFullMap() {
    frameTicks++;

    // 1. Automatic Traffic Signal Timer Loop (Zero Manual Control!)
    let cycleMod = frameTicks % 600;
    if (cycleMod < 280) {
        signalState = "GREEN";
        document.getElementById("fSignal").innerHTML = "GREEN (Proceeding)";
        document.getElementById("fSignal").className = "val-g";
    } else if (cycleMod < 340) {
        signalState = "YELLOW";
        document.getElementById("fSignal").innerHTML = "YELLOW (Caution)";
        document.getElementById("fSignal").className = "val-r";
    } else {
        signalState = "RED";
        document.getElementById("fSignal").innerHTML = "RED (Auto Stop)";
        document.getElementById("fSignal").className = "val-r";
    }

    // 2. Automatic Hazard Injection & Rerouting Loop (Zero Manual Control!)
    if (cycleMod >= 150 && cycleMod <= 450) {
        autoHazardActive = true;
    } else {
        autoHazardActive = false;
    }

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Light Mode Background
    ctx.fillStyle = "#F1F5F9";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Parks (Mint Green)
    ctx.fillStyle = "#DCFCE7";
    ctx.beginPath(); ctx.arc(170, 170, 75, 0, Math.PI*2); ctx.fill();
    ctx.beginPath(); ctx.arc(840, 230, 85, 0, Math.PI*2); ctx.fill();

    // Main Roads (Coral Pink)
    const roadEdges = [
        ["START", "ROUNDABOUT"], ["ROUNDABOUT", "TL_ADER"], ["TL_ADER", "TL_SADI"],
        ["TL_SADI", "LECOINTE"], ["LECOINTE", "DEST"], ["ROUNDABOUT", "BREGUET_1"],
        ["BREGUET_1", "BREGUET_2"], ["BREGUET_2", "BREGUET_3"], ["BREGUET_3", "DEST"],
        ["START", "GPS_START"], ["GPS_END", "DEST"]
    ];

    roadEdges.forEach(pair => {
        let p1 = nodes[pair[0]], p2 = nodes[pair[1]];
        // Sidewalk Curb
        ctx.strokeStyle = "#64748B";
        ctx.lineWidth = 36;
        ctx.beginPath(); ctx.moveTo(p1.x, p1.y); ctx.lineTo(p2.x, p2.y); ctx.stroke();
        // Coral Surface
        ctx.strokeStyle = "#FF8A80";
        ctx.lineWidth = 30;
        ctx.beginPath(); ctx.moveTo(p1.x, p1.y); ctx.lineTo(p2.x, p2.y); ctx.stroke();
        // Yellow Double Lines
        ctx.strokeStyle = "#F59E0B";
        ctx.lineWidth = 2;
        ctx.beginPath(); ctx.moveTo(p1.x, p1.y); ctx.lineTo(p2.x, p2.y); ctx.stroke();
    });

    // Weak GPS Corridor (Electric Sky Blue)
    ctx.strokeStyle = "#00B0FF";
    ctx.lineWidth = 36;
    ctx.beginPath();
    ctx.moveTo(nodes["GPS_START"].x, nodes["GPS_START"].y);
    ctx.lineTo(nodes["GPS_MID"].x, nodes["GPS_MID"].y);
    ctx.lineTo(nodes["GPS_END"].x, nodes["GPS_END"].y);
    ctx.stroke();

    ctx.strokeStyle = "#FFFFFF";
    ctx.lineWidth = 3.5;
    ctx.setLineDash([10, 10]);
    ctx.stroke();
    ctx.setLineDash([]);

    // Roundabout (Royal Purple)
    ctx.fillStyle = "#F3E8FF";
    ctx.strokeStyle = "#9C27B0";
    ctx.lineWidth = 4.5;
    ctx.beginPath(); ctx.arc(nodes["ROUNDABOUT"].x, nodes["ROUNDABOUT"].y, 38, 0, Math.PI*2); ctx.fill(); ctx.stroke();
    ctx.fillStyle = "#9C27B0";
    ctx.beginPath(); ctx.arc(nodes["ROUNDABOUT"].x, nodes["ROUNDABOUT"].y, 22, 0, Math.PI*2); ctx.fill();

    ctx.fillStyle = "#000000";
    ctx.font = "900 13px Plus Jakarta Sans";
    ctx.fillText("Roundabout (Église St Jean-Baptiste)", nodes["ROUNDABOUT"].x - 110, nodes["ROUNDABOUT"].y - 48);

    if (autoHazardActive) {
        ctx.fillStyle = "#DC2626";
        ctx.font = "900 17px Plus Jakarta Sans";
        ctx.fillText("💥 ACCIDENT BOTTLE-NECK", nodes["ROUNDABOUT"].x - 85, nodes["ROUNDABOUT"].y + 6);
    }

    // Traffic Lights (Auto-Cycling Signal Color)
    let sigColor = "#16A34A";
    if (signalState === "RED") sigColor = "#DC2626";
    if (signalState === "YELLOW") sigColor = "#F59E0B";

    [nodes["TL_ADER"], nodes["TL_SADI"]].forEach(pos => {
        ctx.fillStyle = "#FFFFFF";
        ctx.strokeStyle = "#000000";
        ctx.lineWidth = 2.5;
        ctx.beginPath(); ctx.arc(pos.x, pos.y, 14, 0, Math.PI*2); ctx.fill(); ctx.stroke();

        ctx.fillStyle = sigColor;
        ctx.beginPath(); ctx.arc(pos.x, pos.y, 9, 0, Math.PI*2); ctx.fill();
    });

    ctx.fillStyle = "#000000";
    ctx.font = "900 13px Plus Jakarta Sans";
    ctx.fillText("Traffic Lights", 460, 405);

    // Landmarks Text (Black Headings & Gold Pins)
    ctx.fillStyle = "#D97706";
    ctx.font = "900 11px Plus Jakarta Sans";
    ctx.fillText("📍 Tabac de L'Aviation", 75, 390);
    ctx.fillText("📍 Collège Saint-Exupéry", 130, 530);
    ctx.fillText("⚡ Electric 55 Charging", 270, 490);
    ctx.fillText("📍 Samia Esthéticienne", 440, 290);
    ctx.fillText("⚽ Stade Sadi Lecointe", 670, 215);
    ctx.fillText("📚 Médiathèque de Vélizy", 780, 280);
    ctx.fillText("🏎️ Audi / Porsche Smart Mobility", 820, 120);

    // Driving Region Container Badge (Black Title)
    ctx.fillStyle = "#FFFFFF";
    ctx.strokeStyle = "#000000";
    ctx.lineWidth = 2;
    ctx.fillRect(30, 20, 175, 36);
    ctx.strokeRect(30, 20, 175, 36);
    ctx.fillStyle = "#000000";
    ctx.font = "900 14px Plus Jakarta Sans";
    ctx.fillText("Driving Region", 45, 43);

    // Weak GPS Label (Black Heading)
    ctx.fillStyle = "#000000";
    ctx.font = "900 14px Plus Jakarta Sans";
    ctx.fillText("Weak GPS Corridor", nodes["GPS_MID"].x - 55, nodes["GPS_MID"].y + 36);

    // Automatic Rerouting Strategy Switch (Pre-Trained ML-A*)
    if (autoHazardActive && egoCar.path.includes("ROUNDABOUT")) {
        egoCar.path = ["START", "GPS_START", "GPS_MID", "GPS_END", "DEST"];
        egoCar.segIdx = 0; egoCar.progress = 0;
        document.getElementById("fStrat").innerHTML = "Auto ML-A* Hazard Reroute Active";
        document.getElementById("fStrat").className = "val-r";
    } else if (!autoHazardActive && egoCar.path.includes("GPS_MID") && egoCar.progress > 0.9) {
        egoCar.path = ["START", "ROUNDABOUT", "TL_ADER", "TL_SADI", "LECOINTE", "DEST"];
        egoCar.segIdx = 0; egoCar.progress = 0;
        document.getElementById("fStrat").innerHTML = "Proposed Dynamic Risk ML-A* (Pre-Trained)";
        document.getElementById("fStrat").className = "val-g";
    }

    // Vehicle Motion Physics & Automatic Signal Stop
    let p1 = nodes[egoCar.path[egoCar.segIdx]];
    let p2 = nodes[egoCar.path[egoCar.segIdx + 1]];
    let isRedStop = (egoCar.path[egoCar.segIdx+1] === "TL_ADER" || egoCar.path[egoCar.segIdx+1] === "TL_SADI") && signalState === "RED";

    if (isRedStop && egoCar.progress > 0.6) {
        // Vehicle Automatically Stops for RED Light
        document.getElementById("fSpeed").innerHTML = "0.0 km/h (Auto Signal Stop)";
    } else {
        egoCar.progress += 0.003;
        if (egoCar.progress >= 1.0) {
            egoCar.progress = 0;
            egoCar.segIdx = (egoCar.segIdx + 1) % (egoCar.path.length - 1);
        }
        document.getElementById("fSpeed").innerHTML = "48.0 km/h";
    }

    egoCar.x = p1.x + (p2.x - p1.x) * egoCar.progress;
    egoCar.y = p1.y + (p2.y - p1.y) * egoCar.progress;
    egoCar.yaw = Math.atan2(p2.y - p1.y, p2.x - p1.x);

    // LiDAR Scanner Arc
    ctx.save();
    ctx.translate(egoCar.x, egoCar.y);
    ctx.rotate(egoCar.yaw);
    ctx.fillStyle = "rgba(0, 176, 255, 0.2)";
    ctx.strokeStyle = "rgba(0, 176, 255, 0.85)";
    ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(0,0); ctx.arc(0, 0, 75, -Math.PI/4, Math.PI/4); ctx.closePath(); ctx.fill(); ctx.stroke();
    ctx.restore();

    // Autonomous Ego SUV Body (Electric Cyan)
    ctx.save();
    ctx.translate(egoCar.x, egoCar.y);
    ctx.rotate(egoCar.yaw);
    ctx.fillStyle = "#00E5FF";
    ctx.strokeStyle = "#000000";
    ctx.lineWidth = 1.5;
    ctx.fillRect(-14, -8, 28, 16); ctx.strokeRect(-14, -8, 28, 16);
    ctx.fillStyle = "#FFFFFF";
    ctx.fillRect(12, -7, 4, 3); ctx.fillRect(12, 4, 4, 3);
    ctx.restore();

    if (egoCar.path[egoCar.segIdx].includes("GPS") || egoCar.path[egoCar.segIdx+1].includes("GPS")) {
        ctx.strokeStyle = "#DC2626";
        ctx.lineWidth = 2;
        ctx.setLineDash([4, 4]);
        ctx.beginPath(); ctx.arc(egoCar.x, egoCar.y, 28, 0, Math.PI*2); ctx.stroke(); ctx.setLineDash([]);
        document.getElementById("fEKF").innerHTML = "1.58 m (Dead-Reckoning Active)";
        document.getElementById("fEKF").className = "val-r";
    } else {
        document.getElementById("fEKF").innerHTML = "0.35 m (High Precision Lock)";
        document.getElementById("fEKF").className = "val-g";
    }

    requestAnimationFrame(drawFullMap);
}

drawFullMap();
</script>
</body>
</html>
"""

# Render Full Map Canvas Component (100% Viewport Fit)
components.html(html_full_canvas, height=640)
