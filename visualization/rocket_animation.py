# Real-time Telemetry Gauges and Visual Rocket Launch Animation Viewport
import json
import plotly.graph_objects as go
import pandas as pd
from typing import Dict, Any

def create_gauge_indicator(
    value: float,
    title: str,
    max_range: float,
    unit: str,
    color: str = "#00e5ff",
    thresholds: list = None
) -> go.Figure:
    """Creates a modern circular telemetry cockpit gauge."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': f"<b>{title}</b><br><span style='font-size:0.8em;color:gray'>{unit}</span>", 'font': {'size': 14, 'color': 'white'}},
        number={'suffix': f" {unit}", 'font': {'size': 20, 'color': 'white'}},
        gauge={
            'axis': {'range': [0, max_range], 'tickcolor': "white"},
            'bar': {'color': color},
            'bgcolor': "#1e222d",
            'borderwidth': 1,
            'bordercolor': "#333",
            'steps': thresholds or [
                {'range': [0, max_range * 0.7], 'color': '#111827'},
                {'range': [max_range * 0.7, max_range * 0.9], 'color': '#1e3a5f'},
                {'range': [max_range * 0.9, max_range], 'color': '#3b1d2e'}
            ]
        }
    ))
    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=15, r=15, t=30, b=15),
        height=180,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig


def create_playback_snapshot(df: pd.DataFrame, time_sec: float) -> Dict[str, Any]:
    """Retrieves vehicle state at exact simulation elapsed time."""
    idx = (df["time_s"] - time_sec).abs().idxmin()
    row = df.loc[idx]
    return row.to_dict()


def generate_animated_rocket_html(df: pd.DataFrame, rocket_name: str = "Falcon 9") -> str:
    """
    Generates a full-featured HTML5/CSS3/JavaScript aerospace rocket launch viewport
    with animated liftoff, thrust particle flames, atmospheric sky transitions,
    staging separation, and HUD telemetry readout.
    """
    # Downsample trajectory points for smooth browser rendering
    step = max(1, len(df) // 120)
    sampled = df.iloc[::step].copy()
    if sampled.iloc[-1]["time_s"] != df.iloc[-1]["time_s"]:
        sampled = pd.concat([sampled, df.iloc[[-1]]])

    telemetry_data = []
    for _, r in sampled.iterrows():
        telemetry_data.append({
            "t": round(float(r["time_s"]), 1),
            "alt_km": round(float(r["altitude_m"]) / 1000.0, 2),
            "alt_m": round(float(r["altitude_m"]), 0),
            "vel_mps": round(float(r["velocity_mps"]), 1),
            "vel_kmh": round(float(r["velocity_kmh"]), 0),
            "mach": round(float(r["mach"]), 2),
            "g": round(float(r["g_load"]), 2),
            "fuel_pct": round(float(r["fuel_percentage"]), 1),
            "thrust": round(float(r["thrust_n"]), 0),
            "phase": str(r["phase"]),
            "stage": str(r["stage_name"])
        })

    json_str = json.dumps(telemetry_data)

    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="utf-8">
    <style>
        body {{
            margin: 0;
            padding: 0;
            background: #090c10;
            color: #e6edf3;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace;
            overflow: hidden;
            user-select: none;
        }}
        #viewport {{
            position: relative;
            width: 100%;
            height: 480px;
            overflow: hidden;
            border-radius: 8px;
            border: 1px solid #30363d;
            background: linear-gradient(to top, #1e3a5f 0%, #0d1b2a 40%, #050811 100%);
            transition: background 0.5s ease;
        }}
        /* Starfield in Space */
        .star {{
            position: absolute;
            background: #ffffff;
            border-radius: 50%;
            opacity: 0.8;
        }}
        /* Cloud Layers in Troposphere */
        .cloud {{
            position: absolute;
            background: rgba(255, 255, 255, 0.18);
            border-radius: 50px;
            filter: blur(8px);
        }}
        /* Launch Pad Ground Base */
        #ground {{
            position: absolute;
            bottom: 0;
            left: 0;
            width: 100%;
            height: 70px;
            background: linear-gradient(to top, #161b22, #21262d);
            border-top: 2px solid #388bfd;
            z-index: 5;
        }}
        #pad-tower {{
            position: absolute;
            bottom: 60px;
            left: 48%;
            width: 8px;
            height: 120px;
            background: repeating-linear-gradient(45deg, #8b949e, #8b949e 5px, #30363d 5px, #30363d 10px);
            z-index: 6;
        }}
        /* Earth Curve in High Altitude */
        #earth-horizon {{
            position: absolute;
            bottom: -320px;
            left: -10%;
            width: 120%;
            height: 400px;
            border-radius: 50%;
            background: radial-gradient(circle, #1f6feb 0%, #0d419d 60%, transparent 100%);
            box-shadow: 0 0 40px #58a6ff;
            opacity: 0;
            transition: opacity 1s ease;
            z-index: 4;
        }}
        /* Visual Rocket Container */
        #rocket-vehicle {{
            position: absolute;
            left: 50%;
            bottom: 75px;
            transform: translateX(-50%);
            width: 32px;
            height: 110px;
            z-index: 10;
            transition: bottom 0.08s linear, transform 0.08s ease;
        }}
        /* Rocket SVG & Components */
        .rocket-svg {{
            width: 100%;
            height: 100%;
            filter: drop-shadow(0 0 10px rgba(0, 229, 255, 0.4));
        }}
        /* Engine Exhaust Flame */
        #exhaust-plume {{
            position: absolute;
            top: 96px;
            left: 50%;
            transform: translateX(-50%);
            width: 16px;
            height: 45px;
            background: linear-gradient(to bottom, #ffffff 0%, #ffeb3b 25%, #ff5722 65%, transparent 100%);
            border-radius: 50% 50% 70% 70%;
            filter: blur(1px);
            animation: flicker 0.06s infinite alternate;
            z-index: 9;
        }}
        /* Stage 1 Booster Falling during Staging */
        #booster-spent {{
            position: absolute;
            left: 50%;
            width: 18px;
            height: 48px;
            background: #8b949e;
            border-radius: 2px;
            display: none;
            z-index: 8;
        }}
        @keyframes flicker {{
            0% {{ transform: translateX(-50%) scaleY(0.9) scaleX(0.95); opacity: 0.9; }}
            100% {{ transform: translateX(-50%) scaleY(1.2) scaleX(1.05); opacity: 1.0; }}
        }}
        /* HUD Telemetry Overlay */
        .hud-panel {{
            position: absolute;
            top: 15px;
            left: 15px;
            background: rgba(13, 17, 23, 0.85);
            backdrop-filter: blur(6px);
            border: 1px solid #30363d;
            border-radius: 6px;
            padding: 10px 14px;
            font-size: 13px;
            z-index: 20;
            box-shadow: 0 4px 12px rgba(0,0,0,0.5);
        }}
        .hud-val {{
            color: #58a6ff;
            font-weight: bold;
            font-family: monospace;
        }}
        .hud-status {{
            position: absolute;
            top: 15px;
            right: 15px;
            background: rgba(13, 17, 23, 0.85);
            backdrop-filter: blur(6px);
            border: 1px solid #238636;
            color: #3fb950;
            border-radius: 6px;
            padding: 8px 14px;
            font-size: 13px;
            font-weight: bold;
            z-index: 20;
        }}
        /* Audio/Countdown Display */
        #countdown-overlay {{
            position: absolute;
            top: 40%;
            left: 50%;
            transform: translate(-50%, -50%);
            font-size: 54px;
            font-weight: 900;
            color: #ffeb3b;
            text-shadow: 0 0 25px rgba(255, 235, 59, 0.8);
            z-index: 30;
            display: none;
        }}
        /* Control Bar */
        #controls {{
            position: absolute;
            bottom: 12px;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(13, 17, 23, 0.9);
            border: 1px solid #30363d;
            border-radius: 30px;
            padding: 6px 18px;
            display: flex;
            align-items: center;
            gap: 12px;
            z-index: 25;
        }}
        .btn {{
            background: #238636;
            color: white;
            border: none;
            padding: 6px 14px;
            border-radius: 20px;
            font-weight: bold;
            cursor: pointer;
            font-size: 12px;
            transition: all 0.2s;
        }}
        .btn:hover {{
            background: #2ea043;
            transform: scale(1.05);
        }}
        .btn-sec {{
            background: #21262d;
            border: 1px solid #30363d;
            color: #c9d1d9;
        }}
        .btn-sec:hover {{
            background: #30363d;
        }}
        #slider {{
            width: 180px;
            accent-color: #58a6ff;
        }}
    </style>
    </head>
    <body>
    <div id="viewport">
        <!-- Celestial / Atmospheric Layers -->
        <div id="stars-layer"></div>
        <div id="clouds-layer"></div>
        <div id="earth-horizon"></div>
        
        <!-- Ground / Pad -->
        <div id="ground"></div>
        <div id="pad-tower"></div>

        <!-- Visual Animated Rocket -->
        <div id="rocket-vehicle">
            <svg class="rocket-svg" viewBox="0 0 32 110">
                <!-- Nose Cone -->
                <path d="M16 2 Q22 20 22 32 L10 32 Q10 20 16 2 Z" fill="#f0f6fc"/>
                <path d="M16 2 L22 32 L16 32 Z" fill="#c9d1d9" opacity="0.4"/>
                <!-- Upper Stage (Stage 2) -->
                <rect x="10" y="32" width="12" height="24" rx="1" fill="#e6edf3"/>
                <line x1="10" y1="56" x2="22" y2="56" stroke="#00e5ff" stroke-width="2"/>
                <!-- Interstage / Stage 1 Booster -->
                <rect x="10" y="58" width="12" height="40" rx="1" fill="#f0f6fc"/>
                <!-- Engine Nozzle -->
                <polygon points="12,98 20,98 22,104 10,104" fill="#6e7681"/>
                <!-- Aerodynamic Fins -->
                <polygon points="10,85 2,102 10,100" fill="#30363d"/>
                <polygon points="22,85 30,102 22,100" fill="#30363d"/>
            </svg>
            <div id="exhaust-plume"></div>
        </div>

        <div id="booster-spent"></div>

        <!-- HUD Telemetry -->
        <div class="hud-panel">
            <div>MET: <span id="hud-time" class="hud-val">T+0.0s</span></div>
            <div>ALT: <span id="hud-alt" class="hud-val">0.00 km</span></div>
            <div>VEL: <span id="hud-vel" class="hud-val">0 km/h</span> (<span id="hud-mach" class="hud-val">M 0.00</span>)</div>
            <div>G-FORCE: <span id="hud-g" class="hud-val">1.00 G</span></div>
            <div>FUEL: <span id="hud-fuel" class="hud-val">100%</span></div>
        </div>

        <div id="hud-phase" class="hud-status">PRE-LAUNCH READY</div>
        <div id="countdown-overlay">T-5</div>

        <!-- Interactive Controls -->
        <div id="controls">
            <button id="btn-launch" class="btn">🚀 LAUNCH</button>
            <button id="btn-pause" class="btn btn-sec">⏸ PAUSE</button>
            <button id="btn-reset" class="btn btn-sec">🔄 RESET</button>
            <input type="range" id="slider" min="0" max="100" value="0">
            <span id="speed-label" style="font-size:11px;color:#8b949e;">1x</span>
        </div>
    </div>

    <script>
        const telemetry = {json_str};
        const totalSteps = telemetry.length;
        let currentIdx = 0;
        let isRunning = false;
        let animTimer = null;
        let playSpeed = 1;

        // Viewport elements
        const viewport = document.getElementById('viewport');
        const rocket = document.getElementById('rocket-vehicle');
        const exhaust = document.getElementById('exhaust-plume');
        const ground = document.getElementById('ground');
        const padTower = document.getElementById('pad-tower');
        const earthHorizon = document.getElementById('earth-horizon');
        const countdownEl = document.getElementById('countdown-overlay');

        const hudTime = document.getElementById('hud-time');
        const hudAlt = document.getElementById('hud-alt');
        const hudVel = document.getElementById('hud-vel');
        const hudMach = document.getElementById('hud-mach');
        const hudG = document.getElementById('hud-g');
        const hudFuel = document.getElementById('hud-fuel');
        const hudPhase = document.getElementById('hud-phase');
        const slider = document.getElementById('slider');

        // Generate ambient stars
        const starsLayer = document.getElementById('stars-layer');
        for(let i=0; i<60; i++) {{
            const star = document.createElement('div');
            star.className = 'star';
            star.style.width = (Math.random()*2.5 + 1) + 'px';
            star.style.height = star.style.width;
            star.style.left = (Math.random()*100) + '%';
            star.style.top = (Math.random()*85) + '%';
            star.style.opacity = Math.random()*0.9 + 0.1;
            starsLayer.appendChild(star);
        }}

        function updateFrame(idx) {{
            if(idx < 0) idx = 0;
            if(idx >= totalSteps) idx = totalSteps - 1;
            currentIdx = idx;
            slider.value = Math.floor((currentIdx / (totalSteps - 1)) * 100);

            const pt = telemetry[idx];
            hudTime.textContent = 'T+' + pt.t + 's';
            hudAlt.textContent = pt.alt_km + ' km';
            hudVel.textContent = pt.vel_kmh.toLocaleString() + ' km/h';
            hudMach.textContent = 'M ' + pt.mach;
            hudG.textContent = pt.g + ' G';
            hudFuel.textContent = pt.fuel_pct + '%';
            hudPhase.textContent = pt.phase;

            const altFraction = Math.min(1.0, pt.alt_km / 120.0);

            // Dynamic sky gradient transition based on altitude
            if (pt.alt_km < 12) {{
                // Troposphere
                viewport.style.background = 'linear-gradient(to top, #1f4068 0%, #162447 50%, #0d1b2a 100%)';
                earthHorizon.style.opacity = '0';
            }} else if (pt.alt_km < 50) {{
                // Stratosphere
                viewport.style.background = 'linear-gradient(to top, #111a2e 0%, #080d1a 60%, #03060c 100%)';
                earthHorizon.style.opacity = '0.3';
            }} else {{
                // Space / Low Earth Orbit
                viewport.style.background = 'linear-gradient(to top, #020408 0%, #010204 100%)';
                earthHorizon.style.opacity = '1';
            }}

            // Visual rocket position in viewport
            let bottomPx = 75 + (altFraction * 260);
            rocket.style.bottom = bottomPx + 'px';

            // Ground moves downward as rocket ascends
            if (pt.alt_km > 0.5) {{
                ground.style.transform = `translateY(${{Math.min(100, pt.alt_km * 40)}}px)`;
                padTower.style.transform = `translateY(${{Math.min(150, pt.alt_km * 40)}}px)`;
            }} else {{
                ground.style.transform = 'translateY(0px)';
                padTower.style.transform = 'translateY(0px)';
            }}

            // Engine thrust plume
            if (pt.thrust > 1000 && pt.fuel_pct > 0.1) {{
                exhaust.style.display = 'block';
                const flameScale = Math.min(1.5, Math.max(0.6, pt.thrust / 2500000));
                exhaust.style.height = (40 * flameScale) + 'px';
            }} else {{
                exhaust.style.display = 'none';
            }}

            // Pitch tilt as rocket conducts gravity turn
            let tiltDeg = Math.min(65, altFraction * 55);
            rocket.style.transform = `translateX(-50%) rotate(${{tiltDeg}}deg)`;
        }}

        function startCountdownAndLaunch() {{
            if (isRunning) return;
            let count = 3;
            countdownEl.style.display = 'block';
            countdownEl.textContent = 'T-' + count;

            const cTimer = setInterval(() => {{
                count--;
                if (count > 0) {{
                    countdownEl.textContent = 'T-' + count;
                }} else if (count === 0) {{
                    countdownEl.textContent = 'IGNITION & LIFTOFF!';
                }} else {{
                    clearInterval(cTimer);
                    countdownEl.style.display = 'none';
                    playSimulation();
                }}
            }}, 700);
        }}

        function playSimulation() {{
            if (isRunning) return;
            isRunning = true;
            document.getElementById('btn-launch').textContent = '▶ RUNNING';

            animTimer = setInterval(() => {{
                if (currentIdx < totalSteps - 1) {{
                    currentIdx++;
                    updateFrame(currentIdx);
                }} else {{
                    pauseSimulation();
                }}
            }}, 70 / playSpeed);
        }}

        function pauseSimulation() {{
            isRunning = false;
            clearInterval(animTimer);
            document.getElementById('btn-launch').textContent = '🚀 RESUME';
        }}

        function resetSimulation() {{
            pauseSimulation();
            document.getElementById('btn-launch').textContent = '🚀 LAUNCH';
            updateFrame(0);
        }}

        document.getElementById('btn-launch').addEventListener('click', () => {{
            if (!isRunning) {{
                if (currentIdx === 0) {{
                    startCountdownAndLaunch();
                }} else {{
                    playSimulation();
                }}
            }} else {{
                pauseSimulation();
            }}
        }});

        document.getElementById('btn-pause').addEventListener('click', pauseSimulation);
        document.getElementById('btn-reset').addEventListener('click', resetSimulation);

        slider.addEventListener('input', (e) => {{
            pauseSimulation();
            const pct = parseInt(e.target.value);
            const idx = Math.floor((pct / 100) * (totalSteps - 1));
            updateFrame(idx);
        }});

        // Initialize at T+0s
        updateFrame(0);
    </script>
    </body>
    </html>
    """
    return html_code
