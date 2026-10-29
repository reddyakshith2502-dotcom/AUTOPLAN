import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components

# Streamlit Page Setup (Multi-Vehicle Full View)
st.set_page_config(
    page_title="Adaptive Autonomous Navigation | Multi-Vehicle Fleet Research",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="collapsed"
)

BASE_DIR = Path(__file__).resolve().parent.parent

# Custom CSS for Solid Black Headings (#000000) & Rich Palette
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

    .master-header-box {
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

    .m-title {
        font-size: 1.9rem;
        font-weight: 900;
        color: #000000;
        margin: 0;
    }

    .m-sub {
        color: #334155;
        font-size: 1.0rem;
        font-weight: 600;
        margin-top: 4px;
    }
</style>
""", unsafe_allow_html=True)

# Main Header (Black Title)
st.markdown("""
<div class="master-header-box">
    <div>
        <div class="m-title">Adaptive Autonomous Vehicle Fleet Navigation Research System</div>
        <div class="m-sub">Multi-Vehicle Fleet Traffic • Full Map Fit • Auto-Cycling Signals • Pre-Trained ML Weights • EKF Sensor Fusion</div>
    </div>
</div>
""", unsafe_allow_html=True)

# HTML5 Multi-Vehicle Self-Running Canvas Engine
html_multi_canvas = """
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
    #multiCanvas {
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
    .hud-multi {
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
    .hud-mhead { font-weight: 900; color: #000000; font-size: 15px; margin-bottom: 5px; letter-spacing: -0.3px; }
    .val-g { color: #16A34A; font-weight: 800; }
    .val-r { color: #DC2626; font-weight: 800; }
    .val-b { color: #0284C7; font-weight: 800; }
</style>
</head>
<body>

<div class="hud-multi">
    <div class="hud-mhead">🚘 MULTI-VEHICLE AUTONOMOUS FLEET TELEMETRY</div>
    <div>Fleet Status: <span class="val-g">5 Autonomous & Traffic Vehicles Active</span></div>
    <div>Traffic Signals: <span id="mSignal" class="val-g">GREEN (Auto-Cycling)</span></div>
    <div>Time-To-Collision (TTC): <span id="mTTC" class="val-b">8.4 s</span></div>
    <div>EKF Position Precision: <span id="mEKF" class="val-g">0.35 m (High Lock)</span></div>
    <div>Navigation Strategy: <span id="mStrat" class="val-g">Proposed Dynamic Risk ML-A* (Pre-Trained)</span></div>
</div>

<canvas id="multiCanvas" width="1240" height="620"></canvas>

<script>
const canvas = document.getElementById("multiCanvas");
const ctx = canvas.getContext("2d");

let frameTicks = 0;
let signalState = "GREEN";
let autoHazardActive = false;

// Map Waypoint Nodes
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

// Multi-Vehicle Autonomous Fleet
let fleet = [
    { id: "EGO", color: "#00E5FF", isEgo: true, path: ["START", "ROUNDABOUT", "TL_ADER", "TL_SADI", "LECOINTE", "DEST"], segIdx: 0, progress: 0, speed: 2.2, x: 80, y: 460, yaw: 0 },
    { id: "V1", color: "#FF1744", isEgo: false, path: ["START", "ROUNDABOUT", "TL_ADER", "TL_SADI", "LECOINTE", "DEST"], segIdx: 0, progress: 0.35, speed: 1.8, x: 0, y: 0, yaw: 0 },
    { id: "V2", color: "#FFD600", isEgo: false, path: ["ROUNDABOUT", "BREGUET_1", "BREGUET_2", "BREGUET_3", "DEST"], segIdx: 0, progress: 0.20, speed: 2.0, x: 0, y: 0, yaw: 0 },
    { id: "V3", color: "#FF9100", isEgo: false, path: ["BREGUET_1", "BREGUET_2", "BREGUET_3", "DEST"], segIdx: 0, progress: 0.45, speed: 1.6, x: 0, y: 0, yaw: 0 },
    { id: "V4", color: "#16A34A", isEgo: false, path: ["GPS_START", "GPS_MID", "GPS_END", "DEST"], segIdx: 0, progress: 0.25, speed: 2.1, x: 0, y: 0, yaw: 0 }
];

function drawMultiFleetScene() {
    frameTicks++;

    // Automatic Signal Cycle Loop
    let cycleMod = frameTicks % 600;
    if (cycleMod < 280) {
        signalState = "GREEN";
        document.getElementById("mSignal").innerHTML = "GREEN (Proceeding)";
        document.getElementById("mSignal").className = "val-g";
    } else if (cycleMod < 340) {
        signalState = "YELLOW";
        document.getElementById("mSignal").innerHTML = "YELLOW (Caution)";
        document.getElementById("mSignal").className = "val-r";
    } else {
        signalState = "RED";
        document.getElementById("mSignal").innerHTML = "RED (Auto Stop)";
        document.getElementById("mSignal").className = "val-r";
    }

    // Automatic Dynamic Hazard Reroute Cycle
    if (cycleMod >= 150 && cycleMod <= 450) {
        autoHazardActive = true;
    } else {
        autoHazardActive = false;
    }

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Light Background
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
        ctx.strokeStyle = "#64748B";
        ctx.lineWidth = 36;
        ctx.beginPath(); ctx.moveTo(p1.x, p1.y); ctx.lineTo(p2.x, p2.y); ctx.stroke();
        ctx.strokeStyle = "#FF8A80";
        ctx.lineWidth = 30;
        ctx.beginPath(); ctx.moveTo(p1.x, p1.y); ctx.lineTo(p2.x, p2.y); ctx.stroke();
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

    // Roundabout (Royal Purple Island)
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

    // Traffic Lights (Auto-Cycling Colors)
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

    // Landmark Labels (Black Text & Gold Pins)
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

    // Weak GPS Label
    ctx.fillStyle = "#000000";
    ctx.font = "900 14px Plus Jakarta Sans";
    ctx.fillText("Weak GPS Corridor", nodes["GPS_MID"].x - 55, nodes["GPS_MID"].y + 36);

    // Ego Car Automatic Rerouting Strategy Switch
    let ego = fleet[0];
    if (autoHazardActive && ego.path.includes("ROUNDABOUT")) {
        ego.path = ["START", "GPS_START", "GPS_MID", "GPS_END", "DEST"];
        ego.segIdx = 0; ego.progress = 0;
        document.getElementById("mStrat").innerHTML = "Auto ML-A* Hazard Reroute Active";
        document.getElementById("mStrat").className = "val-r";
    } else if (!autoHazardActive && ego.path.includes("GPS_MID") && ego.progress > 0.9) {
        ego.path = ["START", "ROUNDABOUT", "TL_ADER", "TL_SADI", "LECOINTE", "DEST"];
        ego.segIdx = 0; ego.progress = 0;
        document.getElementById("mStrat").innerHTML = "Proposed Dynamic Risk ML-A* (Pre-Trained)";
        document.getElementById("mStrat").className = "val-g";
    }

    // Update & Draw All 5 Vehicles in the Autonomous Fleet
    fleet.forEach(v => {
        if (v.path.length < 2) return;

        let p1 = nodes[v.path[v.segIdx]];
        let p2 = nodes[v.path[v.segIdx + 1]];
        let isRedStop = (v.path[v.segIdx+1] === "TL_ADER" || v.path[v.segIdx+1] === "TL_SADI") && signalState === "RED";

        if (isRedStop && v.progress > 0.6) {
            // Vehicles Automatically Stop for RED Signal
        } else {
            v.progress += (v.speed * 0.0015);
            if (v.progress >= 1.0) {
                v.progress = 0;
                v.segIdx = (v.segIdx + 1) % (v.path.length - 1);
            }
        }

        v.x = p1.x + (p2.x - p1.x) * v.progress;
        v.y = p1.y + (p2.y - p1.y) * v.progress;
        v.yaw = Math.atan2(p2.y - p1.y, p2.x - p1.x);

        // Draw LiDAR Beam for Ego Autonomous Vehicle
        if (v.isEgo) {
            ctx.save();
            ctx.translate(v.x, v.y);
            ctx.rotate(v.yaw);
            ctx.fillStyle = "rgba(0, 176, 255, 0.2)";
            ctx.strokeStyle = "rgba(0, 176, 255, 0.85)";
            ctx.lineWidth = 1.5;
            ctx.beginPath(); ctx.moveTo(0,0); ctx.arc(0, 0, 75, -Math.PI/4, Math.PI/4); ctx.closePath(); ctx.fill(); ctx.stroke();
            ctx.restore();

            if (v.path[v.segIdx].includes("GPS") || v.path[v.segIdx+1].includes("GPS")) {
                ctx.strokeStyle = "#DC2626";
                ctx.lineWidth = 2;
                ctx.setLineDash([4, 4]);
                ctx.beginPath(); ctx.arc(v.x, v.y, 28, 0, Math.PI*2); ctx.stroke(); ctx.setLineDash([]);
                document.getElementById("mEKF").innerHTML = "1.58 m (Dead-Reckoning Active)";
                document.getElementById("mEKF").className = "val-r";
            } else {
                document.getElementById("mEKF").innerHTML = "0.35 m (High Precision Lock)";
                document.getElementById("mEKF").className = "val-g";
            }
        }

        // Draw Vehicle Body
        ctx.save();
        ctx.translate(v.x, v.y);
        ctx.rotate(v.yaw);
        ctx.fillStyle = v.color;
        ctx.strokeStyle = "#000000";
        ctx.lineWidth = 1.5;
        ctx.fillRect(-13, -7, 26, 14); ctx.strokeRect(-13, -7, 26, 14);
        ctx.fillStyle = "#FFFFFF";
        ctx.fillRect(11, -6, 3, 3); ctx.fillRect(11, 3, 3, 3);
        ctx.restore();
    });

    requestAnimationFrame(drawMultiFleetScene);
}

drawMultiFleetScene();
</script>
</body>
</html>
"""

# Render Multi-Vehicle Canvas Component (Full Map View)
components.html(html_multi_canvas, height=640)
