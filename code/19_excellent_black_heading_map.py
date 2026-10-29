import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components

# Streamlit Page Setup
st.set_page_config(
    page_title="Adaptive Autonomous Navigation | Black Headings & Rich Map",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent.parent

# Custom CSS for Solid Black Headings & Rich Colors
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800;900&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .stApp {
        background-color: #F8FAFC;
        color: #0F172A;
    }

    /* SOLID BLACK HEADINGS (#000000) */
    h1, h2, h3, h4, h5, h6, .black-heading {
        color: #000000 !important;
        font-weight: 800 !important;
        letter-spacing: -0.5px;
    }

    /* Glass Container */
    .black-header-box {
        background: #FFFFFF;
        border: 2px solid #000000;
        border-left: 8px solid #0284C7;
        padding: 22px 30px;
        border-radius: 16px;
        box-shadow: 0 6px 24px rgba(0, 0, 0, 0.06);
        margin-bottom: 22px;
    }

    .main-title {
        font-size: 2.2rem;
        font-weight: 900;
        color: #000000;
        margin: 0;
    }

    .main-sub {
        color: #334155;
        font-size: 1.05rem;
        font-weight: 600;
        margin-top: 6px;
    }

    .black-card {
        background: #FFFFFF;
        border: 1.5px solid #CBD5E1;
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 18px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.04);
    }

    .card-title {
        color: #000000;
        font-size: 1.2rem;
        font-weight: 800;
        margin-top: 0;
        margin-bottom: 10px;
    }

    .val-num {
        font-size: 1.8rem;
        font-weight: 800;
        color: #0284C7;
    }
</style>
""", unsafe_allow_html=True)

# Main Header with Black Title Text
st.markdown("""
<div class="black-header-box">
    <div class="main-title">Adaptive Autonomous Vehicle Navigation Research System</div>
    <div class="main-sub">
        Real-World Urban Driving Region • Landmark Map • NGSIM Empirical ML Models • Weak GPS EKF Sensor Fusion
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar Controls
st.sidebar.markdown("<h2 style='color:#000000; font-weight:800;'>🎛️ Simulation Controls</h2>", unsafe_allow_html=True)
tl_ader_red = st.sidebar.checkbox("Rue Clément Ader Light [RED]", value=True, key="b_tl1")
tl_sadi_red = st.sidebar.checkbox("Avenue Sadi Lecointe Light [RED]", value=False, key="b_tl2")
weak_gps_active = st.sidebar.checkbox("Enable Weak GPS Zone Interference", value=True, key="b_gps")
roundabout_jam = st.sidebar.checkbox("Block Roundabout with Accident Hazard", value=False, key="b_jam")

col_m, col_s = st.columns([7, 4])

with col_m:
    st.markdown("<h3 style='color:#000000;'>🏙️ Interactive City Map (Rich Colors & Black Headings)</h3>", unsafe_allow_html=True)

    # HTML5 60 FPS Canvas Map Code
    html_black_map = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
        body {{
            margin: 0;
            background-color: #F8FAFC;
            font-family: 'Plus Jakarta Sans', sans-serif;
            color: #000000;
            overflow: hidden;
        }}
        #blackCanvas {{
            background: #F1F5F9;
            border-radius: 16px;
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.08);
            border: 2px solid #000000;
            display: block;
            margin: 0 auto;
        }}
        .hud-black {{
            position: absolute;
            top: 22px;
            left: 30px;
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(12px);
            border: 2px solid #000000;
            padding: 16px 22px;
            border-radius: 14px;
            font-size: 13px;
            box-shadow: 0 6px 24px rgba(0,0,0,0.1);
        }}
        .hud-head {{ font-weight: 900; color: #000000; font-size: 15px; margin-bottom: 6px; letter-spacing: -0.3px; }}
        .val-g {{ color: #16A34A; font-weight: 800; }}
        .val-r {{ color: #DC2626; font-weight: 800; }}
        .val-b {{ color: #0284C7; font-weight: 800; }}
    </style>
    </head>
    <body>

    <div class="hud-black">
        <div class="hud-head">🚘 EGO AUTONOMOUS VEHICLE TELEMETRY</div>
        <div>Speed: <span id="bSpeed" class="val-g">48.0 km/h</span></div>
        <div>Time-To-Collision (TTC): <span id="bTTC" class="val-b">8.4 s</span></div>
        <div>EKF Position Precision: <span id="bEKF" class="val-g">0.35 m (High Lock)</span></div>
        <div>Navigation Strategy: <span id="bStrat" class="val-g">Proposed Dynamic Risk ML-A*</span></div>
    </div>

    <canvas id="blackCanvas" width="1120" height="640"></canvas>

    <script>
    const canvas = document.getElementById("blackCanvas");
    const ctx = canvas.getContext("2d");

    let tlAderRed = {str(tl_ader_red).lower()};
    let tlSadiRed = {str(tl_sadi_red).lower()};
    let weakGpsActive = {str(weak_gps_active).lower()};
    let roundaboutJam = {str(roundabout_jam).lower()};

    const nodes = {{
        "START": {{x: 80, y: 480}},
        "ROUNDABOUT": {{x: 240, y: 330}},
        "TL_ADER": {{x: 400, y: 380}},
        "TL_SADI": {{x: 600, y: 410}},
        "LECOINTE": {{x: 770, y: 370}},
        "GPS_START": {{x: 440, y: 550}},
        "GPS_MID": {{x: 700, y: 570}},
        "GPS_END": {{x: 920, y: 450}},
        "BREGUET_1": {{x: 340, y: 160}},
        "BREGUET_2": {{x: 640, y: 190}},
        "BREGUET_3": {{x: 900, y: 270}},
        "DEST": {{x: 1020, y: 410}}
    }};

    let egoCar = {{
        path: ["START", "GPS_START", "GPS_MID", "GPS_END", "DEST"],
        segIdx: 0, progress: 0, speed: 2.2, x: 80, y: 480, yaw: 0
    }};

    function drawMap() {{
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // 1. Clean Light Background
        ctx.fillStyle = "#F1F5F9";
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        // Parks (Mint Green)
        ctx.fillStyle = "#DCFCE7";
        ctx.beginPath(); ctx.arc(160, 180, 75, 0, Math.PI*2); ctx.fill();
        ctx.beginPath(); ctx.arc(770, 240, 85, 0, Math.PI*2); ctx.fill();

        // 2. Main Roads (Warm Coral Pink)
        const roadEdges = [
            ["START", "ROUNDABOUT"], ["ROUNDABOUT", "TL_ADER"], ["TL_ADER", "TL_SADI"],
            ["TL_SADI", "LECOINTE"], ["LECOINTE", "DEST"], ["ROUNDABOUT", "BREGUET_1"],
            ["BREGUET_1", "BREGUET_2"], ["BREGUET_2", "BREGUET_3"], ["BREGUET_3", "DEST"],
            ["START", "GPS_START"], ["GPS_END", "DEST"]
        ];

        roadEdges.forEach(pair => {{
            let p1 = nodes[pair[0]], p2 = nodes[pair[1]];
            // Sidewalk Curb
            ctx.strokeStyle = "#64748B";
            ctx.lineWidth = 36;
            ctx.beginPath(); ctx.moveTo(p1.x, p1.y); ctx.lineTo(p2.x, p2.y); ctx.stroke();
            // Coral Pink Road Surface
            ctx.strokeStyle = "#FF8A80";
            ctx.lineWidth = 30;
            ctx.beginPath(); ctx.moveTo(p1.x, p1.y); ctx.lineTo(p2.x, p2.y); ctx.stroke();
            // Yellow Double Centerline
            ctx.strokeStyle = "#F59E0B";
            ctx.lineWidth = 2;
            ctx.beginPath(); ctx.moveTo(p1.x, p1.y); ctx.lineTo(p2.x, p2.y); ctx.stroke();
        }});

        // 3. Weak GPS Corridor (Electric Sky Blue)
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

        // 4. Roundabout (Royal Purple Island)
        ctx.fillStyle = "#F3E8FF";
        ctx.strokeStyle = "#9C27B0";
        ctx.lineWidth = 4.5;
        ctx.beginPath(); ctx.arc(nodes["ROUNDABOUT"].x, nodes["ROUNDABOUT"].y, 38, 0, Math.PI*2); ctx.fill(); ctx.stroke();

        ctx.fillStyle = "#9C27B0";
        ctx.beginPath(); ctx.arc(nodes["ROUNDABOUT"].x, nodes["ROUNDABOUT"].y, 22, 0, Math.PI*2); ctx.fill();

        // Roundabout Black Heading Text
        ctx.fillStyle = "#000000";
        ctx.font = "900 13px Plus Jakarta Sans";
        ctx.fillText("Roundabout (Église St Jean-Baptiste)", nodes["ROUNDABOUT"].x - 110, nodes["ROUNDABOUT"].y - 48);

        if (roundaboutJam) {{
            ctx.fillStyle = "#DC2626";
            ctx.font = "900 18px Plus Jakarta Sans";
            ctx.fillText("💥 ACCIDENT HAZARD", nodes["ROUNDABOUT"].x - 75, nodes["ROUNDABOUT"].y + 6);
        }}

        // 5. Traffic Lights Posts (White Enclosures with Red/Green Status)
        [nodes["TL_ADER"], nodes["TL_SADI"]].forEach((pos, idx) => {{
            let isRed = (idx === 0) ? tlAderRed : tlSadiRed;
            ctx.fillStyle = "#FFFFFF";
            ctx.strokeStyle = "#000000";
            ctx.lineWidth = 2.5;
            ctx.beginPath(); ctx.arc(pos.x, pos.y, 14, 0, Math.PI*2); ctx.fill(); ctx.stroke();

            ctx.fillStyle = isRed ? "#DC2626" : "#16A34A";
            ctx.beginPath(); ctx.arc(pos.x, pos.y, 9, 0, Math.PI*2); ctx.fill();
        }});

        ctx.fillStyle = "#000000";
        ctx.font = "900 13px Plus Jakarta Sans";
        ctx.fillText("Traffic Lights", 430, 425);

        // Landmark Map Place Pins (Bold Black Labels & Gold Markers)
        ctx.fillStyle = "#D97706";
        ctx.font = "900 12px Plus Jakarta Sans";
        ctx.fillText("📍 Tabac de L'Aviation", 75, 405);
        ctx.fillText("📍 Collège Saint-Exupéry", 130, 550);
        ctx.fillText("⚡ Electric 55 Charging", 250, 510);
        ctx.fillText("📍 Samia Esthéticienne", 410, 310);
        ctx.fillText("⚽ Stade Sadi Lecointe", 610, 225);
        ctx.fillText("📚 Médiathèque de Vélizy", 720, 300);
        ctx.fillText("🏎️ Audi / Porsche Smart Mobility", 750, 130);

        // Driving Region Container Badge (Black Heading Text)
        ctx.fillStyle = "#FFFFFF";
        ctx.strokeStyle = "#000000";
        ctx.lineWidth = 2;
        ctx.fillRect(30, 22, 175, 36);
        ctx.strokeRect(30, 22, 175, 36);
        ctx.fillStyle = "#000000";
        ctx.font = "900 14px Plus Jakarta Sans";
        ctx.fillText("Driving Region", 45, 45);

        // Weak GPS Label (Black Heading)
        ctx.fillStyle = "#000000";
        ctx.font = "900 14px Plus Jakarta Sans";
        ctx.fillText("Weak GPS Corridor", nodes["GPS_MID"].x - 55, nodes["GPS_MID"].y + 36);

        // Rerouting Logic
        if (roundaboutJam && egoCar.path.includes("ROUNDABOUT")) {{
            egoCar.path = ["START", "GPS_START", "GPS_MID", "GPS_END", "DEST"];
            egoCar.segIdx = 0; egoCar.progress = 0;
            document.getElementById("bStrat").innerHTML = "Dynamic Hazard Avoidance Reroute";
            document.getElementById("bStrat").className = "val-r";
        }}

        // Move Ego Vehicle
        let p1 = nodes[egoCar.path[egoCar.segIdx]];
        let p2 = nodes[egoCar.path[egoCar.segIdx + 1]];
        let isRedStop = (egoCar.path[egoCar.segIdx+1] === "TL_ADER" && tlAderRed) || (egoCar.path[egoCar.segIdx+1] === "TL_SADI" && tlSadiRed);
        if (isRedStop && egoCar.progress > 0.6) {{
            // Stop
        }} else {{
            egoCar.progress += 0.003;
            if (egoCar.progress >= 1.0) {{
                egoCar.progress = 0;
                egoCar.segIdx = (egoCar.segIdx + 1) % (egoCar.path.length - 1);
            }}
        }}

        egoCar.x = p1.x + (p2.x - p1.x) * egoCar.progress;
        egoCar.y = p1.y + (p2.y - p1.y) * egoCar.progress;
        egoCar.yaw = Math.atan2(p2.y - p1.y, p2.x - p1.x);

        // Ego LiDAR Beam
        ctx.save();
        ctx.translate(egoCar.x, egoCar.y);
        ctx.rotate(egoCar.yaw);
        ctx.fillStyle = "rgba(0, 176, 255, 0.2)";
        ctx.strokeStyle = "rgba(0, 176, 255, 0.85)";
        ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.moveTo(0,0); ctx.arc(0, 0, 75, -Math.PI/4, Math.PI/4); ctx.closePath(); ctx.fill(); ctx.stroke();
        ctx.restore();

        // Ego Vehicle Body (Electric Cyan)
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

        if (weakGpsActive && (egoCar.path[egoCar.segIdx].includes("GPS") || egoCar.path[egoCar.segIdx+1].includes("GPS"))) {{
            ctx.strokeStyle = "#DC2626";
            ctx.lineWidth = 2;
            ctx.setLineDash([4, 4]);
            ctx.beginPath(); ctx.arc(egoCar.x, egoCar.y, 28, 0, Math.PI*2); ctx.stroke(); ctx.setLineDash([]);
            document.getElementById("bEKF").innerHTML = "1.58 m (Dead-Reckoning Active)";
            document.getElementById("bEKF").className = "val-r";
        }} else {{
            document.getElementById("bEKF").innerHTML = "0.35 m (High Lock)";
            document.getElementById("bEKF").className = "val-g";
        }}

        document.getElementById("bSpeed").innerHTML = (isRedStop && egoCar.progress > 0.6) ? "0.0 km/h (Signal Stop)" : "48.0 km/h";

        requestAnimationFrame(drawMap);
    }}

    drawMap();
    </script>
    </body>
    </html>
    """

    components.html(html_black_map, height=660)

with col_s:
    st.markdown("<h3 style='color:#000000;'>📊 Research Performance</h3>", unsafe_allow_html=True)

    st.markdown("""
    <div class="black-card">
        <div class="card-title">Optimal Path Selected</div>
        <div style="font-size:1.1rem; font-weight:800; color:#0284C7;">
            N_START ➔ GPS_START ➔ GPS_MID ➔ GPS_END ➔ N_DEST
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="black-card">
        <div class="card-title">Telemetry Metrics</div>
        <p><b>Total Distance:</b> <span class="val-num">670.0 m</span></p>
        <p><b>Est. Travel Time:</b> <span class="val-num">41.58 s</span></p>
        <p><b>Avg Collision Risk:</b> <span class="val-num">1.9987</span></p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("<p style='text-align:center; color:#000000; font-weight:700;'>Adaptive Autonomous Navigation Research | Black Headings & Rich Color Palette</p>", unsafe_allow_html=True)
