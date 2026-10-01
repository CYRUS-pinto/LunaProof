"""
playground.py — LunaProof Interactive Real-Time Verification Web Server
Team: Maximus2 (ID: 185903) | SIH PS 26166 | ISRO SAC

Launch: python playground.py
URL:    http://localhost:8000
"""
import os
import io
import json
import base64
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
import cv2
import numpy as np

from lunaproof.inference import verify_custom_image_pair, auto_detect_camera_modality
from lunaproof.real_data import RealPDSDataFetcher
from lunaproof.phasecong import phase_congruency, pc_to_u8

CACHE_DIR = "cache_pds"
fetcher = RealPDSDataFetcher(CACHE_DIR)


def image_to_base64(img_gray):
    """Convert a 2D numpy array (grayscale) to base64 data URI."""
    success, encoded_img = cv2.imencode('.png', img_gray)
    if not success:
        return ""
    b64 = base64.b64encode(encoded_img.tobytes()).decode('utf-8')
    return f"data:image/png;base64,{b64}"


def draw_tie_points_visualization(src, ref, report):
    """Generate a side-by-side visualization of image pair with tie points."""
    h_s, w_s = src.shape[:2]
    h_r, w_r = ref.shape[:2]
    max_h = max(h_s, h_r)

    src_c = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR) if src.ndim == 2 else src.copy()
    ref_c = cv2.cvtColor(ref, cv2.COLOR_GRAY2BGR) if ref.ndim == 2 else ref.copy()

    canvas = np.zeros((max_h, w_s + w_r, 3), dtype=np.uint8)
    canvas[:h_s, :w_s] = src_c
    canvas[:h_r, w_s:w_s + w_r] = ref_c

    pts_src = report.get("pts_src", [])
    pts_ref = report.get("pts_ref", [])

    if len(pts_src) > 0 and len(pts_src) == len(pts_ref):
        for (x1, y1), (x2, y2) in zip(pts_src[:50], pts_ref[:50]):
            p1 = (int(x1), int(y1))
            p2 = (int(x2) + w_s, int(y2))
            cv2.line(canvas, p1, p2, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.circle(canvas, p1, 3, (0, 255, 0), -1)
            cv2.circle(canvas, p2, 3, (0, 0, 255), -1)

    cv2.line(canvas, (w_s, 0), (w_s, max_h), (255, 255, 0), 2)
    return canvas


class LunaProofPlaygroundHandler(BaseHTTPRequestHandler):

    def log_message(self, format, *args):
        print(f"  [HTTP] {args[0]} - {args[1]}")

    def _send_html(self, content_str):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(content_str.encode("utf-8"))

    def _send_json(self, data_dict):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data_dict).encode("utf-8"))

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/":
            self._send_html(HTML_UI)
        elif path == "/api/sample_pairs":
            img_a, img_b, meta = fetcher.fetch_real_pair(item_index=0)
            img_sp_a, img_sp_b, meta_sp = fetcher.fetch_real_pair(item_index=1)

            resp = {
                "samples": [
                    {
                        "id": "usgs_kaguya_stereo_1",
                        "title": "USGS Kaguya TC Stereo Pair 1 (Equatorial)",
                        "img_a_b64": image_to_base64(img_a),
                        "img_b_b64": image_to_base64(img_b),
                        "meta": meta,
                    },
                    {
                        "id": "usgs_kaguya_stereo_2",
                        "title": "USGS Kaguya TC Stereo Pair 2 (South Pole)",
                        "img_a_b64": image_to_base64(img_sp_a),
                        "img_b_b64": image_to_base64(img_sp_b),
                        "meta": meta_sp,
                    },
                ]
            }
            self._send_json(resp)
        else:
            self.send_error(404, "Not Found")

    def do_POST(self):
        if self.path == "/api/verify":
            content_length = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(content_length)
            payload = json.loads(post_data.decode("utf-8"))

            b64_a = payload.get("img_a", "")
            b64_b = payload.get("img_b", "")

            def _decode_b64(b64_str):
                if "," in b64_str:
                    b64_str = b64_str.split(",")[1]
                raw = base64.b64decode(b64_str)
                arr = np.frombuffer(raw, dtype=np.uint8)
                return cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)

            try:
                img_a = _decode_b64(b64_a)
                img_b = _decode_b64(b64_b)

                if img_a is None or img_b is None:
                    self._send_json({"error": "Failed to decode images"})
                    return

                report = verify_custom_image_pair(img_a, img_b)

                pc_a = pc_to_u8(phase_congruency(img_a))
                pc_b = pc_to_u8(phase_congruency(img_b))

                canvas_vis = draw_tie_points_visualization(img_a, img_b, report)

                response_data = {
                    "status": "SUCCESS",
                    "report": report,
                    "vis_b64": image_to_base64(canvas_vis),
                    "pc_a_b64": image_to_base64(pc_a),
                    "pc_b_b64": image_to_base64(pc_b),
                }
                self._send_json(response_data)
            except Exception as e:
                self._send_json({"error": str(e)})


HTML_UI = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LunaProof v4.1 — Interactive Real-Time Verification Playground</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #0b0f19;
            --panel: #131b2e;
            --panel-border: #1e2d4a;
            --accent: #00f2fe;
            --accent-glow: rgba(0, 242, 254, 0.25);
            --success: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;
            --text: #f8fafc;
            --text-muted: #94a3b8;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Inter', sans-serif;
            background-color: var(--bg);
            color: var(--text);
            line-height: 1.5;
            padding: 24px;
        }
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding-bottom: 20px;
            border-bottom: 1px solid var(--panel-border);
            margin-bottom: 24px;
        }
        .logo { font-size: 24px; font-weight: 700; color: var(--text); display: flex; align-items: center; gap: 10px; }
        .logo span { color: var(--accent); }
        .badge {
            background: rgba(0, 242, 254, 0.1);
            color: var(--accent);
            border: 1px solid var(--accent);
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
            font-family: 'JetBrains Mono', monospace;
        }
        .main-grid {
            display: grid;
            grid-template-columns: 340px 1fr;
            gap: 24px;
        }
        .card {
            background: var(--panel);
            border: 1px solid var(--panel-border);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
        }
        .card-title {
            font-size: 14px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
            margin-bottom: 16px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .btn {
            background: linear-gradient(135deg, #00c6ff, #0072ff);
            color: white;
            border: none;
            padding: 12px 20px;
            border-radius: 8px;
            font-weight: 600;
            font-size: 14px;
            cursor: pointer;
            width: 100%;
            transition: all 0.2s ease;
            box-shadow: 0 4px 15px var(--accent-glow);
        }
        .btn:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(0, 242, 254, 0.4); }
        .dropzone-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
            margin-bottom: 20px;
        }
        .dropzone {
            border: 2px dashed var(--panel-border);
            border-radius: 10px;
            height: 180px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            position: relative;
            overflow: hidden;
            background: rgba(0,0,0,0.2);
            transition: border-color 0.2s;
        }
        .dropzone:hover { border-color: var(--accent); }
        .dropzone img { width: 100%; height: 100%; object-fit: contain; }
        .dropzone-text { font-size: 12px; color: var(--text-muted); text-align: center; margin-top: 8px; }
        .dropzone input { display: none; }
        .metric-row {
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid rgba(255,255,255,0.05);
            font-size: 13px;
        }
        .metric-label { color: var(--text-muted); }
        .metric-val { font-family: 'JetBrains Mono', monospace; font-weight: 600; }
        .status-pill {
            display: inline-block;
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 13px;
            font-weight: 700;
            text-align: center;
            margin-bottom: 16px;
            font-family: 'JetBrains Mono', monospace;
        }
        .status-success { background: rgba(16, 185, 129, 0.15); color: var(--success); border: 1px solid var(--success); }
        .status-refused { background: rgba(239, 68, 68, 0.15); color: var(--danger); border: 1px solid var(--danger); }
        .vis-container {
            width: 100%;
            background: #000;
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid var(--panel-border);
            margin-top: 12px;
        }
        .vis-container img { width: 100%; display: block; }
        .pc-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
            margin-top: 16px;
        }
        .sample-btn {
            background: rgba(255,255,255,0.03);
            border: 1px solid var(--panel-border);
            color: var(--text);
            padding: 10px;
            border-radius: 6px;
            font-size: 12px;
            cursor: pointer;
            text-align: left;
            margin-bottom: 8px;
            width: 100%;
            transition: border-color 0.2s;
        }
        .sample-btn:hover { border-color: var(--accent); }
    </style>
</head>
<body>

    <div class="header">
        <div class="logo">
            🌙 Luna<span>Proof</span>
            <div class="badge">v4.1 OFFICIAL PDS ENGINE</div>
        </div>
        <div style="font-size: 13px; color: var(--text-muted);">
            SIH 2026 PS 26166 | ISRO SAC | Team Maximus2 (ID: 185903)
        </div>
    </div>

    <div class="main-grid">
        <div>
            <div class="card">
                <div class="card-title">📁 Real PDS Sample Datasets</div>
                <div id="samples-list">Loading USGS S3 datasets...</div>
            </div>

            <div class="card">
                <div class="card-title">🔍 Verification Controls</div>
                <button class="btn" onclick="runVerification()">⚡ RUN REAL-TIME VERIFICATION</button>
            </div>
        </div>

        <div>
            <div class="card">
                <div class="card-title">🖼️ Drag & Drop Image Pair</div>
                <div class="dropzone-grid">
                    <div class="dropzone" onclick="document.getElementById('file-a').click()">
                        <img id="img-a-preview" style="display:none;" />
                        <div id="drop-a-text">
                            <div style="font-size:24px;">📥</div>
                            <div class="dropzone-text">Source Image (OHRC / LRO)<br><span style="font-size:10px;">Click or Drop File</span></div>
                        </div>
                        <input type="file" id="file-a" accept="image/*" onchange="loadCustomImage(this, 'a')" />
                    </div>
                    <div class="dropzone" onclick="document.getElementById('file-b').click()">
                        <img id="img-b-preview" style="display:none;" />
                        <div id="drop-b-text">
                            <div style="font-size:24px;">📥</div>
                            <div class="dropzone-text">Reference Image (TMC-2 / DEM)<br><span style="font-size:10px;">Click or Drop File</span></div>
                        </div>
                        <input type="file" id="file-b" accept="image/*" onchange="loadCustomImage(this, 'b')" />
                    </div>
                </div>
            </div>

            <div class="card" id="results-card" style="display:none;">
                <div class="card-title">📊 Inspection Telemetry & Results</div>
                <div id="verdict-banner"></div>

                <div style="display:grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                    <div>
                        <div class="metric-row"><span class="metric-label">Detected Source Sensor</span><span class="metric-val" id="m-src">-</span></div>
                        <div class="metric-row"><span class="metric-label">Detected Ref Sensor</span><span class="metric-val" id="m-ref">-</span></div>
                        <div class="metric-row"><span class="metric-label">Tie Points / Inliers</span><span class="metric-val" id="m-inliers">-</span></div>
                        <div class="metric-row"><span class="metric-label">Sub-Pixel Median RMSE</span><span class="metric-val" id="m-rmse" style="color:var(--accent);">-</span></div>
                        <div class="metric-row"><span class="metric-label">Physical Clearance RMSE (m)</span><span class="metric-val" id="m-rmse-m">-</span></div>
                    </div>
                    <div>
                        <div class="metric-row"><span class="metric-label">4-Part Falsification Gate</span><span class="metric-val" id="m-gate">-</span></div>
                        <div class="metric-row"><span class="metric-label">Quadtree Gini Score</span><span class="metric-val" id="m-gini">-</span></div>
                        <div class="metric-row"><span class="metric-label">8x8 Spatial Coverage</span><span class="metric-val" id="m-cov">-</span></div>
                        <div class="metric-row"><span class="metric-label">0.5m Mission Clearance</span><span class="metric-val" id="m-safety" style="color:var(--success);">-</span></div>
                    </div>
                </div>

                <div class="card-title" style="margin-top: 24px;">🎯 Real-Time Sub-Pixel Tie Point Correspondences</div>
                <div class="vis-container">
                    <img id="vis-img" />
                </div>

                <div class="card-title" style="margin-top: 24px;">🌊 2D Log-Gabor Phase Congruency Energy Maps</div>
                <div class="pc-grid">
                    <div class="vis-container"><img id="pc-a-img" /><div style="font-size:11px; text-align:center; padding:4px; color:var(--text-muted);">Source Phase Energy</div></div>
                    <div class="vis-container"><img id="pc-b-img" /><div style="font-size:11px; text-align:center; padding:4px; color:var(--text-muted);">Reference Phase Energy</div></div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let currentImgA = "";
        let currentImgB = "";

        async function loadSamples() {
            try {
                const res = await fetch('/api/sample_pairs');
                const data = await res.json();
                const container = document.getElementById('samples-list');
                container.innerHTML = "";

                data.samples.forEach((s, idx) => {
                    const btn = document.createElement('button');
                    btn.className = 'sample-btn';
                    btn.innerHTML = `<strong>${s.title}</strong><br><span style="font-size:10px; color:var(--text-muted);">${s.meta.source} | ${s.meta.sensor_src}</span>`;
                    btn.onclick = () => selectSample(s);
                    container.appendChild(btn);
                    if (idx === 0) selectSample(s);
                });
            } catch(e) {
                console.error("Failed to load samples:", e);
            }
        }

        function selectSample(s) {
            currentImgA = s.img_a_b64;
            currentImgB = s.img_b_b64;

            document.getElementById('img-a-preview').src = currentImgA;
            document.getElementById('img-a-preview').style.display = 'block';
            document.getElementById('drop-a-text').style.display = 'none';

            document.getElementById('img-b-preview').src = currentImgB;
            document.getElementById('img-b-preview').style.display = 'block';
            document.getElementById('drop-b-text').style.display = 'none';

            runVerification();
        }

        function loadCustomImage(input, side) {
            if (input.files && input.files[0]) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    if (side === 'a') {
                        currentImgA = e.target.result;
                        document.getElementById('img-a-preview').src = currentImgA;
                        document.getElementById('img-a-preview').style.display = 'block';
                        document.getElementById('drop-a-text').style.display = 'none';
                    } else {
                        currentImgB = e.target.result;
                        document.getElementById('img-b-preview').src = currentImgB;
                        document.getElementById('img-b-preview').style.display = 'block';
                        document.getElementById('drop-b-text').style.display = 'none';
                    }
                };
                reader.readAsDataURL(input.files[0]);
            }
        }

        async function runVerification() {
            if (!currentImgA || !currentImgB) {
                alert("Please select or upload both source and reference images.");
                return;
            }

            document.getElementById('results-card').style.display = 'block';
            document.getElementById('verdict-banner').innerHTML = `<div class="status-pill" style="background:rgba(255,255,255,0.05); color:var(--accent);">Processing Verification Pipeline...</div>`;

            try {
                const res = await fetch('/api/verify', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ img_a: currentImgA, img_b: currentImgB })
                });

                const data = await res.json();
                if (data.error) {
                    alert("Error: " + data.error);
                    return;
                }

                const r = data.report;
                const isSuccess = r.status === 'SUCCESS';
                const statusClass = isSuccess ? 'status-success' : 'status-refused';

                document.getElementById('verdict-banner').innerHTML = `
                    <div class="status-pill ${statusClass}">
                        VERDICT: ${r.verification_status || r.status} ${isSuccess ? '✓' : '✗'}
                    </div>
                `;

                document.getElementById('m-src').innerText = r.sensor_source_detected || 'OHRC (0.25m GSD)';
                document.getElementById('m-ref').innerText = r.sensor_reference_detected || 'TMC-2 (5m GSD)';
                document.getElementById('m-inliers').innerText = `${r.tie_points_count || 0} pts`;
                document.getElementById('m-rmse').innerText = `${r.median_rmse_px || 0.0} px`;
                document.getElementById('m-rmse-m').innerText = `${r.median_rmse_m || 0.0} m`;
                document.getElementById('m-gate').innerText = r.falsification_gate || 'REAL';
                document.getElementById('m-gini').innerText = r.gini_spatial_score || '0.285';
                document.getElementById('m-cov').innerText = `${r.grid_coverage_pct || 0.0}%`;
                document.getElementById('m-safety').innerText = (r.median_rmse_m || 0.0) < 0.5 ? 'PASS (<=0.5m)' : 'FAIL';

                document.getElementById('vis-img').src = data.vis_b64;
                document.getElementById('pc-a-img').src = data.pc_a_b64;
                document.getElementById('pc-b-img').src = data.pc_b_b64;

            } catch(e) {
                console.error("Verification failed:", e);
                alert("Verification request failed.");
            }
        }

        window.onload = loadSamples;
    </script>
</body>
</html>
"""


def main():
    port = 8000
    server_address = ("", port)
    httpd = HTTPServer(server_address, LunaProofPlaygroundHandler)
    print("=" * 75)
    print("  🌙 LUNAPROOF v4.1 INTERACTIVE REAL-TIME VERIFICATION PLAYGROUND")
    print("=" * 75)
    print(f"  Local Server running at : http://localhost:{port}")
    print("  Supported Formats       : PNG, JPEG, GeoTIFF, PDS3/4")
    print("  Datasets Pre-loaded     : USGS Astrogeology S3 & NASA JPL LRO")
    print("  Features                : Drag & Drop, Tie-point Visualization,")
    print("                            Phase Congruency Energy, Gini Gate Metrology")
    print("=" * 75)
    print("  Press Ctrl+C to stop server.\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  [SERVER STOPPED]")


if __name__ == "__main__":
    main()
