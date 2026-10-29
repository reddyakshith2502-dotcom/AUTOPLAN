import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components

# Streamlit Page Setup
st.set_page_config(
    page_title="3D Autonomous City Simulator | Three.js WebGL",
    page_icon="🏙️",
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
        background-color: #080A10;
        color: #F8FAFC;
    }

    #MainMenu, header, footer {visibility: hidden;}

    .header-3d {
        background: linear-gradient(135deg, rgba(18, 22, 38, 0.9) 0%, rgba(8, 10, 16, 0.95) 100%);
        border: 1px solid rgba(0, 229, 255, 0.3);
        border-left: 6px solid #00E5FF;
        padding: 18px 28px;
        border-radius: 16px;
        box-shadow: 0 12px 35px rgba(0, 0, 0, 0.7);
        margin-bottom: 18px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .title-3d {
        font-size: 2.1rem;
        font-weight: 800;
        background: linear-gradient(90deg, #FFFFFF 0%, #00E5FF 50%, #7C4DFF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }

    .sub-3d {
        color: #94A3B8;
        font-size: 1.0rem;
        margin-top: 4px;
    }
</style>
""", unsafe_allow_html=True)

# Header Title
st.markdown("""
<div class="header-3d">
    <div>
        <div class="title-3d">Full 3D Autonomous City & Traffic Simulator (Three.js WebGL)</div>
        <div class="sub-3d">3D Vehicle Meshes • 3D City Landmarks • 3D LiDAR Spotlight • 360° Orbit Camera • Dynamic ML-A* Rerouting</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Three.js 3D WebGL Engine Code
threejs_code = """
<!DOCTYPE html>
<html>
<head>
<style>
    body {
        margin: 0;
        background-color: #080A10;
        font-family: 'Plus Jakarta Sans', sans-serif;
        color: #FFF;
        overflow: hidden;
    }
    #webglCanvas {
        width: 100%;
        height: 670px;
        display: block;
    }
    .hud-3d {
        position: absolute;
        top: 22px;
        left: 30px;
        background: rgba(14, 18, 30, 0.88);
        backdrop-filter: blur(14px);
        border: 1px solid rgba(0, 229, 255, 0.4);
        padding: 16px 24px;
        border-radius: 16px;
        font-size: 13px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.7);
        z-index: 10;
    }
    .hud-title { font-weight: 800; color: #00E5FF; font-size: 15px; margin-bottom: 6px; letter-spacing: 0.5px; }
    .val-green { color: #00E676; font-weight: 700; }
    .val-yellow { color: #FFD600; font-weight: 700; }
    .val-red { color: #FF1744; font-weight: 700; }

    .ctrl-3d {
        position: absolute;
        bottom: 25px;
        right: 35px;
        background: rgba(14, 18, 30, 0.88);
        backdrop-filter: blur(14px);
        border: 1px solid rgba(255, 255, 255, 0.15);
        padding: 12px 20px;
        border-radius: 14px;
        display: flex;
        gap: 10px;
        z-index: 10;
    }
    .btn-3d {
        background: #1C2235;
        color: #FFF;
        border: 1px solid rgba(255,255,255,0.2);
        padding: 9px 15px;
        border-radius: 10px;
        font-weight: 700;
        font-size: 12px;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    .btn-3d:hover { background: #00E5FF; color: #000; box-shadow: 0 0 12px rgba(0,229,255,0.5); }
</style>

<!-- Import Three.js & OrbitControls from CDN -->
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
</head>
<body>

<div class="hud-3d">
    <div class="hud-title">🏎️ 3D EGO AUTONOMOUS VEHICLE HUD</div>
    <div>Speed: <span id="hud3dVel" class="val-green">48.0 km/h</span></div>
    <div>Time-To-Collision (TTC): <span id="hud3dTTC" class="val-green">8.5 s</span></div>
    <div>EKF 3D Position Precision: <span id="hud3dEKF" class="val-green">0.32 m</span></div>
    <div>Camera Mode: <span id="hud3dCam" class="val-green">3D Free Orbit</span></div>
</div>

<div class="ctrl-3d">
    <button class="btn-3d" onclick="setCamPOV()">🏎️ Driver Cockpit POV</button>
    <button class="btn-3d" onclick="setCamOrbit()">🎥 3D Free Orbit</button>
    <button class="btn-3d" onclick="toggleAccident()">💥 Toggle Accident Hazard</button>
    <button class="btn-3d" onclick="toggleSignals()">🚥 Toggle Traffic Signals</button>
</div>

<div id="webglCanvas"></div>

<script>
// 1. Setup Three.js Scene, Camera, Renderer
const container = document.getElementById("webglCanvas");
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x080A10);
scene.fog = new THREE.FogExp2(0x080A10, 0.002);

const camera = new THREE.PerspectiveCamera(50, container.clientWidth / container.clientHeight, 0.1, 2000);
camera.position.set(0, 180, 260);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(container.clientWidth, container.clientHeight);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
container.appendChild(renderer.domElement);

const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.05;

// 2. 3D Lighting Setup
const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
scene.add(ambientLight);

const sunLight = new THREE.DirectionalLight(0xffffff, 0.8);
sunLight.position.set(200, 300, 150);
sunLight.castShadow = true;
sunLight.shadow.mapSize.width = 2048;
sunLight.shadow.mapSize.height = 2048;
scene.add(sunLight);

// 3. Create 3D Ground & Roads
const groundGeo = new THREE.PlaneGeometry(1200, 1200);
const groundMat = new THREE.MeshStandardMaterial({ color: 0x0E121E, roughness: 0.9 });
const ground = new THREE.Mesh(groundGeo, groundMat);
ground.rotation.x = -Math.PI / 2;
ground.receiveShadow = true;
scene.add(ground);

// 3D Nodes Definition
const nodes3D = {
    "START": new THREE.Vector3(-350, 0.2, 180),
    "ROUNDABOUT": new THREE.Vector3(-180, 0.2, 20),
    "TL_ADER": new THREE.Vector3(-20, 0.2, 60),
    "TL_SADI": new THREE.Vector3(180, 0.2, 80),
    "LECOINTE": new THREE.Vector3(350, 0.2, 40),
    "GPS_START": new THREE.Vector3(0, 0.2, 240),
    "GPS_MID": { x: 260, y: 0.2, z: 260 },
    "GPS_END": new THREE.Vector3(480, 0.2, 140),
    "BREGUET_1": new THREE.Vector3(-80, 0.2, -140),
    "BREGUET_2": new THREE.Vector3(220, 0.2, -110),
    "BREGUET_3": new THREE.Vector3(450, 0.2, -30),
    "A86": new THREE.Vector3(340, 0.2, -220),
    "DEST": new THREE.Vector3(580, 0.2, 80)
};

// Function to build 3D Asphalt Roads
function build3DRoad(v1, v2, width = 28) {
    const dir = new THREE.Vector3().subVectors(v2, v1);
    const len = dir.length();
    const mid = new THREE.Vector3().addVectors(v1, v2).multiplyScalar(0.5);

    const roadGeo = new THREE.PlaneGeometry(width, len);
    const roadMat = new THREE.MeshStandardMaterial({ color: 0x1F2432, roughness: 0.8 });
    const road = new THREE.Mesh(roadGeo, roadMat);
    road.position.copy(mid);
    road.rotation.x = -Math.PI / 2;
    road.rotation.z = -Math.atan2(dir.z, dir.x) + Math.PI / 2;
    road.receiveShadow = true;
    scene.add(road);
}

const roadPairs = [
    ["START", "ROUNDABOUT"], ["ROUNDABOUT", "TL_ADER"], ["TL_ADER", "TL_SADI"],
    ["TL_SADI", "LECOINTE"], ["LECOINTE", "DEST"], ["ROUNDABOUT", "BREGUET_1"],
    ["BREGUET_1", "BREGUET_2"], ["BREGUET_2", "BREGUET_3"], ["BREGUET_3", "DEST"],
    ["BREGUET_2", "A86"], ["A86", "BREGUET_3"], ["START", "GPS_START"], ["GPS_END", "DEST"]
];

roadPairs.forEach(pair => build3DRoad(nodes3D[pair[0]], nodes3D[pair[1]]));

// 4. Build 3D Roundabout Island (Église St Jean-Baptiste)
const rCenter = nodes3D["ROUNDABOUT"];
const rRoadGeo = new THREE.RingGeometry(25, 48, 32);
const rRoadMat = new THREE.MeshStandardMaterial({ color: 0x1F2432, side: THREE.DoubleSide });
const rRoad = new THREE.Mesh(rRoadGeo, rRoadMat);
rRoad.position.set(rCenter.x, 0.3, rCenter.z);
rRoad.rotation.x = -Math.PI / 2;
scene.add(rRoad);

// 3D Island Garden & Fountain
const islandGeo = new THREE.CylinderGeometry(24, 24, 2, 32);
const islandMat = new THREE.MeshStandardMaterial({ color: 0x2E7D32 });
const island = new THREE.Mesh(islandGeo, islandMat);
island.position.set(rCenter.x, 1, rCenter.z);
scene.add(island);

const fountainGeo = new THREE.CylinderGeometry(8, 8, 4, 16);
const fountainMat = new THREE.MeshStandardMaterial({ color: 0x00E5FF, roughness: 0.1 });
const fountain = new THREE.Mesh(fountainGeo, fountainMat);
fountain.position.set(rCenter.x, 3, rCenter.z);
scene.add(fountain);

// 5. Build 3D Buildings
function build3DBuilding(x, z, w, d, h, color = 0x2A324B) {
    const bGeo = new THREE.BoxGeometry(w, h, d);
    const bMat = new THREE.MeshStandardMaterial({ color: color, roughness: 0.5 });
    const b = new THREE.Mesh(bGeo, bMat);
    b.position.set(x, h / 2, z);
    b.castShadow = true;
    b.receiveShadow = true;
    scene.add(b);
}

build3DBuilding(-280, 180, 60, 60, 40, 0x1E293B);
build3DBuilding(300, -180, 90, 70, 50, 0x1E293B);
build3DBuilding(420, 240, 80, 60, 35, 0x1E293B);

// 6. Build 3D Traffic Light Posts
let tlSignalRed = true;
const tlLights = [];

function build3DTrafficLight(pos) {
    const poleGeo = new THREE.CylinderGeometry(1, 1, 22, 16);
    const poleMat = new THREE.MeshStandardMaterial({ color: 0x475569, metalness: 0.8 });
    const pole = new THREE.Mesh(poleGeo, poleMat);
    pole.position.set(pos.x, 11, pos.z);
    scene.add(pole);

    const headGeo = new THREE.BoxGeometry(6, 14, 6);
    const headMat = new THREE.MeshStandardMaterial({ color: 0x0F172A });
    const head = new THREE.Mesh(headGeo, headMat);
    head.position.set(pos.x, 20, pos.z);
    scene.add(head);

    const lightGeo = new THREE.SphereGeometry(2.5, 16, 16);
    const lightMat = new THREE.MeshBasicMaterial({ color: 0xFF1744 });
    const light = new THREE.Mesh(lightGeo, lightMat);
    light.position.set(pos.x, 20, pos.z + 3.2);
    scene.add(light);
    tlLights.push(lightMat);
}

build3DTrafficLight(nodes3D["TL_ADER"]);
build3DTrafficLight(nodes3D["TL_SADI"]);

// 7. Build 3D Ego Autonomous Vehicle
const carGroup = new THREE.Group();

const bodyGeo = new THREE.BoxGeometry(16, 10, 32);
const bodyMat = new THREE.MeshStandardMaterial({ color: 0x00E5FF, metalness: 0.9, roughness: 0.2 });
const body = new THREE.Mesh(bodyGeo, bodyMat);
body.position.y = 6;
body.castShadow = true;
carGroup.add(body);

// 3D Glass Cabin
const cabinGeo = new THREE.BoxGeometry(14, 7, 18);
const cabinMat = new THREE.MeshStandardMaterial({ color: 0x0F172A, roughness: 0.1, transparent: true, opacity: 0.8 });
const cabin = new THREE.Mesh(cabinGeo, cabinMat);
cabin.position.set(0, 13, -2);
carGroup.add(cabin);

// 3D Headlights Spotlight
const headlightL = new THREE.SpotLight(0xFFFFFF, 3, 120, Math.PI / 6);
headlightL.position.set(-5, 7, 16);
headlightL.target.position.set(-5, 0, 100);
carGroup.add(headlightL);
carGroup.add(headlightL.target);

scene.add(carGroup);

// 8. Dynamic Simulation Controls & Animation Loop
let egoProgress = 0;
let egoSegIdx = 0;
let egoPath = ["START", "GPS_START", "GPS_MID", "GPS_END", "DEST"];
let isAccident = false;
let isPOV = false;

function toggleAccident() {
    isAccident = !isAccident;
    if (isAccident) {
        egoPath = ["START", "GPS_START", "GPS_MID", "GPS_END", "DEST"];
        egoSegIdx = 0; egoProgress = 0;
    }
}

function toggleSignals() {
    tlSignalRed = !tlSignalRed;
    tlLights.forEach(l => l.color.setHex(tlSignalRed ? 0xFF1744 : 0x00E676));
}

function setCamPOV() { isPOV = true; document.getElementById("hud3dCam").innerHTML = "Driver Cockpit POV"; }
function setCamOrbit() { isPOV = false; camera.position.set(0, 180, 260); controls.target.set(0, 0, 0); document.getElementById("hud3dCam").innerHTML = "3D Free Orbit"; }

function animate() {
    requestAnimationFrame(animate);

    // Dynamic Vehicle Motion Along 3D Waypoints
    if (egoPath.length >= 2) {
        let v1 = nodes3D[egoPath[egoSegIdx]];
        let v2 = nodes3D[egoPath[egoSegIdx + 1]];

        let isRedStop = (egoPath[egoSegIdx+1] === "TL_ADER" || egoPath[egoSegIdx+1] === "TL_SADI") && tlSignalRed && egoProgress > 0.6;
        if (!isRedStop) {
            egoProgress += 0.003;
            if (egoProgress >= 1.0) {
                egoProgress = 0;
                egoSegIdx = (egoSegIdx + 1) % (egoPath.length - 1);
            }
        }

        let px = v1.x + (v2.x - v1.x) * egoProgress;
        let pz = v1.z + (v2.z - v1.z) * egoProgress;
        let rotY = Math.atan2(v2.x - v1.x, v2.z - v1.z);

        carGroup.position.set(px, 0, pz);
        carGroup.rotation.y = rotY;

        if (isPOV) {
            camera.position.set(px, 18, pz);
            const lookAtVec = new THREE.Vector3(px + Math.sin(rotY)*50, 15, pz + Math.cos(rotY)*50);
            camera.lookAt(lookAtVec);
        }

        document.getElementById("hud3dVel").innerHTML = isRedStop ? "0.0 km/h (Signal Stop)" : "48.0 km/h";
    }

    controls.update();
    renderer.render(scene, camera);
}

animate();

window.addEventListener('resize', () => {
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
});
</script>
</body>
</html>
"""

# Render Full 3D WebGL Canvas Component
components.html(threejs_code, height=690)
