"""
Flask Web Application for Chart Pattern Recognition.
Upload a chart image and get candlestick + chart pattern predictions.
"""
import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

from model.predict import PatternPredictor

# ─── Configuration ───────────────────────────────────────────────────────────
UPLOAD_FOLDER = os.path.join(project_root, 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load predictor
print("\n  Loading ML models...")
predictor = PatternPredictor()
print("  Models loaded successfully!\n")


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle image upload and return predictions."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Use PNG, JPG, JPEG, GIF, BMP, or WEBP'}), 400

    try:
        # Read image bytes for prediction
        image_bytes = file.read()

        # Run prediction
        results = predictor.predict(image_bytes=image_bytes, top_k=3, threshold=0.10)

        return jsonify(results)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok',
        'candlestick_model': predictor.candlestick_model is not None,
        'chart_model': predictor.chart_model is not None,
    })


import os

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("   CHART PATTERN RECOGNITION — WEB APP")
    print("=" * 60)
    port = int(os.environ.get('PORT', 5001))
    print(f"   Running on port: {port}")
    print("=" * 60 + "\n")
    app.run(host='0.0.0.0', port=port, debug=False)
