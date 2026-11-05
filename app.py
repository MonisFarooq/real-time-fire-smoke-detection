import os
import cv2
import numpy as np
import tensorflow as tf
from collections import deque
from datetime import datetime
from flask import Flask, render_template, Response, request, jsonify
from werkzeug.utils import secure_filename
import time

app = Flask(__name__)

# Directory setup
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'webp'}

# Load model
MODEL_PATH = "best_fire_detector.keras"
model = tf.keras.models.load_model(MODEL_PATH)

# Target classes from dataset
CLASS_NAMES = ['fire', 'nofire', 'smoke', 'smokefire']
IMG_SIZE = (224, 224)

# Global camera control and rolling buffer
camera = None
is_streaming = False
detection_buffer = deque(maxlen=15)
latest_system_status = {"status": "STANDBY", "label": "offline", "confidence": 0.0}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_frame(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, IMG_SIZE)
    return np.expand_dims(img_resized.astype(np.float32), axis=0)

def check_visual_hazard_signatures(bgr_img):
    """
    Evaluates color distribution in HSV space to separate real fire/smoke
    from desaturated non-fire objects like books, white sheets, and walls.
    """
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    total_pixels = bgr_img.shape[0] * bgr_img.shape[1]

    # Warm flame ranges: Red, Orange, Yellow
    lower_fire1 = np.array([0, 70, 140], dtype=np.uint8)
    upper_fire1 = np.array([35, 255, 255], dtype=np.uint8)
    lower_fire2 = np.array([165, 70, 140], dtype=np.uint8)
    upper_fire2 = np.array([180, 255, 255], dtype=np.uint8)

    fire_mask = cv2.bitwise_or(
        cv2.inRange(hsv, lower_fire1, upper_fire1),
        cv2.inRange(hsv, lower_fire2, upper_fire2)
    )
    fire_ratio = np.sum(fire_mask > 0) / total_pixels

    # Smoke signatures: Low saturation, mid-to-high luminosity
    smoke_mask = cv2.inRange(hsv, np.array([0, 0, 90]), np.array([180, 45, 220]))
    smoke_ratio = np.sum(smoke_mask > 0) / total_pixels

    return fire_ratio, smoke_ratio

def evaluate_predictions(cv_img, raw_preds):
    class_probs = {name: float(prob) for name, prob in zip(CLASS_NAMES, raw_preds)}
    fire_ratio, smoke_ratio = check_visual_hazard_signatures(cv_img)

    fire_prob = class_probs.get('fire', 0.0)
    smoke_prob = class_probs.get('smoke', 0.0)
    smokefire_prob = class_probs.get('smokefire', 0.0)
    total_hazard_prob = fire_prob + smoke_prob + smokefire_prob

    # Minimum visual presence gates
    has_flame_presence = fire_ratio >= 0.010   # At least 1.0% flame-colored pixels
    has_smoke_presence = smoke_ratio >= 0.18   # At least 18% diffuse smoke tones

    # Trigger hazard only if CNN is confident and visual cues align
    if total_hazard_prob >= 0.70 and (has_flame_presence or has_smoke_presence):
        hazard_pool = {'fire': fire_prob, 'smoke': smoke_prob, 'smokefire': smokefire_prob}
        top_hazard = max(hazard_pool, key=hazard_pool.get)
        return top_hazard, hazard_pool[top_hazard], True
    else:
        # Default safely to 'nofire' for papers, plain walls, clothing, and books
        safe_confidence = class_probs.get('nofire', 1.0 - total_hazard_prob)
        if total_hazard_prob >= 0.70 and not (has_flame_presence or has_smoke_presence):
            safe_confidence = 0.96
        return 'nofire', safe_confidence, False

def generate_frames():
    global camera, is_streaming, latest_system_status

    while True:
        if not is_streaming or camera is None:
            time.sleep(0.1)
            continue

        success, frame = camera.read()
        if not success:
            break

        img_tensor = preprocess_frame(frame)
        raw_preds = model.predict(img_tensor, verbose=0)[0]
        label, conf, is_hazard = evaluate_predictions(frame, raw_preds)

        detection_buffer.append(is_hazard)
        hazard_ratio = sum(detection_buffer) / len(detection_buffer)

        if hazard_ratio >= 0.6:
            status_text = f"CRITICAL: {label.upper()}"
            hud_color = (0, 0, 255)
            latest_system_status = {
                "status": "ALERT",
                "label": label,
                "confidence": round(conf * 100, 1)
            }
        else:
            status_text = "STATUS: SAFE (NORMAL)"
            hud_color = (0, 255, 0)
            latest_system_status = {
                "status": "SAFE",
                "label": "nofire",
                "confidence": round((1.0 - conf if is_hazard else conf) * 100, 1)
            }

        # Overlay HUD on live feed
        cv2.rectangle(frame, (10, 10), (450, 55), (20, 20, 20), -1)
        cv2.putText(frame, f"{status_text} ({conf*100:.1f}%)", (20, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.68, hud_color, 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_cam', methods=['POST'])
def start_cam():
    global camera, is_streaming, latest_system_status
    if not is_streaming:
        camera = cv2.VideoCapture(0)
        is_streaming = True
        latest_system_status = {"status": "SAFE", "label": "nofire", "confidence": 0.0}
    return jsonify({"success": True, "message": "Camera started"})

@app.route('/stop_cam', methods=['POST'])
def stop_cam():
    global camera, is_streaming, latest_system_status
    is_streaming = False
    if camera is not None:
        camera.release()
        camera = None
    detection_buffer.clear()
    latest_system_status = {"status": "STANDBY", "label": "offline", "confidence": 0.0}
    return jsonify({"success": True, "message": "Camera stopped"})

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def get_status():
    return jsonify(latest_system_status)

@app.route('/predict_image', methods=['POST'])
def predict_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file uploaded'}), 400

    file = request.files['image']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file format'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    cv_img = cv2.imread(filepath)
    if cv_img is None:
        return jsonify({'error': 'Could not read image file'}), 400

    img_tensor = preprocess_frame(cv_img)
    raw_preds = model.predict(img_tensor, verbose=0)[0]
    label, confidence, is_hazard = evaluate_predictions(cv_img, raw_preds)

    return jsonify({
        'filename': filename,
        'image_url': f'/static/uploads/{filename}',
        'label': label,
        'confidence': f"{confidence * 100:.1f}%",
        'is_hazard': is_hazard,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)