import os
import io
import base64
import cv2
from flask import Flask, render_template, request, send_file, jsonify
from converter import ImageTo3DConverter

app = Flask(__name__)
converter = ImageTo3DConverter()

# Cache generated files in memory for fast retrieval
CACHE = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/convert', methods=['POST'])
def convert():
    file = request.files['image']
    depth_scale = float(request.form.get('depth_scale', 0.25))
    resolution = int(request.form.get('resolution', 256))
    isolate_subject = request.form.get('isolate_subject', 'true').lower() == 'true'
    
    img_bytes = file.read()
    glb_bytes, zip_bytes, depth_map, category, strategy, model_label = converter.convert(
        img_bytes, depth_scale, resolution, isolate_subject
    )

    CACHE['model.glb'] = glb_bytes
    CACHE['model.zip'] = zip_bytes

    # Depth map preview as base64 png
    _, depth_encoded = cv2.imencode('.png', (depth_map * 255).astype('uint8'))
    depth_b64 = base64.b64encode(depth_encoded).decode('utf-8')

    return jsonify({
        'status': 'success',
        'depth_map': f'data:image/png;base64,{depth_b64}',
        'glb_url': '/api/download/model.glb',
        'zip_url': '/api/download/model.zip',
        'category': category,
        'strategy': strategy,
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
    print(f"🚀 Server running instantly on http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)
