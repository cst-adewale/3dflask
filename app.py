import os
import io
import base64
import cv2
import threading
import time
import requests
from flask import Flask, render_template, request, send_file, jsonify
from converter import ImageTo3DConverter

app = Flask(__name__)
converter = ImageTo3DConverter()

# Cache generated files in memory for fast retrieval
CACHE = {}

def start_self_pinger():
    def ping_loop():
        time.sleep(15)  # Initial grace period on startup
        while True:
            try:
                # Render provides RENDER_EXTERNAL_URL automatically, or use custom SELF_PING_URL / local fallback
                base_url = os.getenv("RENDER_EXTERNAL_URL") or os.getenv("SELF_PING_URL")
                if base_url:
                    ping_url = f"{base_url.rstrip('/')}/ping"
                    res = requests.get(ping_url, timeout=15)
                    print(f"[*] [KeepAlive Pinger] Pinged {ping_url} -> Status {res.status_code}")
                else:
                    port = int(os.environ.get('PORT', 5000))
                    ping_url = f"http://127.0.0.1:{port}/ping"
                    res = requests.get(ping_url, timeout=15)
                    print(f"[*] [KeepAlive Pinger] Local Ping -> Status {res.status_code}")
            except Exception as err:
                print(f"[*] [KeepAlive Pinger] Ping note: {err}")
            
            # Ping every 14 minutes (840 seconds) to prevent Render free tier sleep (15 min limit)
            time.sleep(14 * 60)

    pinger_thread = threading.Thread(target=ping_loop, daemon=True)
    pinger_thread.start()

# Start background pinger daemon
start_self_pinger()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ping')
def ping():
    return jsonify({'status': 'alive', 'message': 'Keep-alive ping successful'}), 200

@app.route('/favicon.ico')
def favicon():
    return send_file('static/favicon.svg', mimetype='image/svg+xml')

@app.route('/api/convert', methods=['POST'])
def convert():
    file = request.files['image']
    depth_scale = float(request.form.get('depth_scale', 0.25))
    resolution = int(request.form.get('resolution', 256))
    isolate_subject = request.form.get('isolate_subject', 'true').lower() == 'true'
    engine = request.form.get('engine', 'auto')
    
    img_bytes = file.read()
    try:
        glb_bytes, zip_bytes, depth_map, category, strategy, model_label = converter.convert(
            img_bytes, depth_scale, resolution, isolate_subject, engine
        )
    except Exception as e:
        error_msg = str(e)
        return jsonify({
            'status': 'error',
            'message': error_msg
        }), 400

    CACHE['model.glb'] = glb_bytes
    CACHE['model.zip'] = zip_bytes

    _, depth_encoded = cv2.imencode('.png', (depth_map * 255).astype('uint8'))
    depth_b64 = base64.b64encode(depth_encoded).decode('utf-8')

    return jsonify({
        'status': 'success',
        'depth_map': f'data:image/png;base64,{depth_b64}',
        'glb_url': '/api/download/model.glb',
        'zip_url': '/api/download/model.zip',
        'category': category,
        'strategy': strategy,
        'model_label': model_label
    })

@app.route('/api/download/<filename>')
def download(filename):
    if filename not in CACHE:
        return 'File not found', 404
    
    mimetype = 'model/gltf-binary' if filename.endswith('.glb') else 'application/zip'
    return send_file(
        io.BytesIO(CACHE[filename]),
        mimetype=mimetype,
        as_attachment=True,
        download_name=filename
    )

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"[*] Server running instantly on http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)
