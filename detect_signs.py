# detect_signs.py
import os
# 🔥 CRITICAL FIX: Disable JAX to avoid NumPy compatibility issues
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Import TensorFlow AFTER setting environment variables
import cv2
import numpy as np
import tensorflow as tf
from cvzone.HandTrackingModule import HandDetector
import math
from collections import deque, Counter

# === CONFIG ===
MODEL_PATH = "sign_language_model.h5"
DATA_DIR = r"C:\Users\adith\OneDrive\Desktop\SignLanguageDetection\Data"
IMG_SIZE = 224
OFFSET = 20
WHITE_BG_SIZE = 300
CONFIDENCE_THRESHOLD = 0.7

# Load labels
labels = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
print("CLASSES:", labels)

# Load model
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded")

# Setup
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1, detectionCon=0.8)
prediction_buffer = deque(maxlen=5)  # For smoothing

if not cap.isOpened():
    print("❌ Webcam error")
    exit()

print("🚀 Press 'q' to quit")

while True:
    success, img = cap.read()
    if not success:
        break
    img = cv2.flip(img, 1)
    img_display = img.copy()
    hands, img_display = detector.findHands(img_display, draw=True)

    prediction_text = "No Hand"
    confidence = 0.0

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        y1 = max(0, y - OFFSET)
        y2 = min(img.shape[0], y + h + OFFSET)
        x1 = max(0, x - OFFSET)
        x2 = min(img.shape[1], x + w + OFFSET)

        img_crop = img[y1:y2, x1:x2]
        if img_crop.size == 0:
            continue

        # Preprocess exactly like data collection
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

        # Prepare for model
        model_input = cv2.resize(img_white, (IMG_SIZE, IMG_SIZE))
        model_input = model_input.astype(np.float32) / 255.0
        model_input = np.expand_dims(model_input, axis=0)

        # Predict
        preds = model.predict(model_input, verbose=0)
        confidence = np.max(preds)
        class_idx = np.argmax(preds)

        if confidence > CONFIDENCE_THRESHOLD:
            prediction_buffer.append(labels[class_idx])
        else:
            prediction_buffer.append("Unknown")

    # Smoothing
    if prediction_buffer:
        smoothed_pred = Counter(prediction_buffer).most_common(1)[0][0]
    else:
        smoothed_pred = "No Hand"

    # Display
    cv2.rectangle(img_display, (0, 0), (640, 60), (0, 0, 0), -1)
    cv2.putText(img_display, f"Sign: {smoothed_pred} ({confidence:.2f})", 
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("Hand Sign Detection", img_display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("👋 Done")