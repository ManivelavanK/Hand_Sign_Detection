import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import tensorflow as tf
from cvzone.HandTrackingModule import HandDetector
import math
from collections import deque, Counter
import base64
import time

app = Flask(__name__)

# Vercel serverless function
from flask import Flask
app = Flask(__name__)

# Config
MODEL_PATH = "sign_language_model.h5"
DATA_DIR = r"C:\Users\adith\OneDrive\Desktop\SignLanguageDetection\Data"
IMG_SIZE = 224
OFFSET = 20
WHITE_BG_SIZE = 300
CONFIDENCE_THRESHOLD = 0.7

# Load model and labels
labels = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
model = tf.keras.models.load_model(MODEL_PATH)
detector = HandDetector(maxHands=1, detectionCon=0.8)
prediction_buffer = deque(maxlen=5)

class VideoCamera:
    def __init__(self):
        self.video = None
        self.current_prediction = "Camera Off"
        self.confidence = 0.0
        self.is_active = False
        
    def start_camera(self):
        if not self.is_active:
            self.video = cv2.VideoCapture(0)
            self.is_active = True
            self.current_prediction = "No Hand"
            return True
        return False
        
    def stop_camera(self):
        if self.is_active and self.video:
            self.video.release()
            self.video = None
            self.is_active = False
            self.current_prediction = "Camera Off"
            return True
        return False
        
    def __del__(self):
        if self.video:
            self.video.release()
        
    def get_frame(self):
        if not self.is_active or not self.video:
            return None, "Camera Off", 0.0
            
        success, frame = self.video.read()
        if not success:
            return None, "No Hand", 0.0
            
        frame = cv2.flip(frame, 1)
        hands, frame = detector.findHands(frame, draw=True)
        
        prediction = "No Hand"
        confidence = 0.0
        
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            y1 = max(0, y - OFFSET)
            y2 = min(frame.shape[0], y + h + OFFSET)
            x1 = max(0, x - OFFSET)
            x2 = min(frame.shape[1], x + w + OFFSET)
            
            img_crop = frame[y1:y2, x1:x2]
            if img_crop.size > 0:
                img_white = np.ones((WHITE_BG_SIZE, WHITE_BG_SIZE, 3), np.uint8) * 255
                aspect_ratio = h / w
                
                if aspect_ratio > 1:
                    k = WHITE_BG_SIZE / h
                    new_w = math.ceil(k * w)
                    resized = cv2.resize(img_crop, (new_w, WHITE_BG_SIZE))
                    w_gap = (WHITE_BG_SIZE - new_w) // 2
                    img_white[:, w_gap:w_gap + new_w] = resized
                else:
                    k = WHITE_BG_SIZE / w
                    new_h = math.ceil(k * h)
                    resized = cv2.resize(img_crop, (WHITE_BG_SIZE, new_h))
                    h_gap = (WHITE_BG_SIZE - new_h) // 2
                    img_white[h_gap:h_gap + new_h, :] = resized
                
                model_input = cv2.resize(img_white, (IMG_SIZE, IMG_SIZE))
                model_input = model_input.astype(np.float32) / 255.0
                model_input = np.expand_dims(model_input, axis=0)
                
                preds = model.predict(model_input, verbose=0)
                confidence = float(np.max(preds))
                class_idx = np.argmax(preds)
                
                if confidence > CONFIDENCE_THRESHOLD:
                    prediction_buffer.append(labels[class_idx])
                else:
                    prediction_buffer.append("Unknown")
        
        if prediction_buffer:
            prediction = Counter(prediction_buffer).most_common(1)[0][0]
        
        self.current_prediction = prediction
        self.confidence = confidence
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        return frame, prediction, confidence

camera = VideoCamera()

def generate_frames():
    while True:
        if camera.is_active:
            frame, _, _ = camera.get_frame()
            if frame is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            # Send placeholder frame when camera is off
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(placeholder, 'Camera Off - Click Start Camera', (120, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            ret, buffer = cv2.imencode('.jpg', placeholder)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.1)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/prediction')
def get_prediction():
    return jsonify({
        'prediction': camera.current_prediction,
        'confidence': round(camera.confidence, 2),
        'timestamp': int(time.time() * 1000),
        'camera_active': camera.is_active
    })

@app.route('/stats')
def get_stats():
    return jsonify({
        'total_detections': getattr(camera, 'total_detections', 0),
        'session_duration': int(time.time() - getattr(camera, 'session_start', time.time())),
        'avg_confidence': getattr(camera, 'avg_confidence', 0)
    })

@app.route('/camera/start', methods=['POST'])
def start_camera():
    success = camera.start_camera()
    return jsonify({'success': success, 'status': 'Camera started' if success else 'Camera already running'})

@app.route('/camera/stop', methods=['POST'])
def stop_camera():
    success = camera.stop_camera()
    return jsonify({'success': success, 'status': 'Camera stopped' if success else 'Camera already stopped'})

@app.route('/camera/status')
def camera_status():
    return jsonify({'active': camera.is_active})

# Vercel handler
def handler(request):
    return app(request.environ, lambda *args: None)

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)