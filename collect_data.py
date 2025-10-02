# collect_data.py
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import os

# === CONFIG ===
GESTURE = "yes"  # 👈 CHANGE THIS FOR EACH GESTURE!
DATA_DIR = r"C:\Users\adith\OneDrive\Desktop\SignLanguageDetection\Data"
FOLDER = os.path.join(DATA_DIR, GESTURE)
os.makedirs(FOLDER, exist_ok=True)

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1, detectionCon=0.8)
offset = 20
imgSize = 300
counter = 0

print(f"Collecting data for: '{GESTURE}'")
print("Show gesture → Press 's' to save. Press 'q' to quit.")
print("💡 Tip: Vary distance, angle, lighting while collecting!")

while True:
    success, img = cap.read()
    if not success:
        break
    img = cv2.flip(img, 1)  # Mirror view
    hands, img = detector.findHands(img, draw=True)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        y1, y2 = max(0, y - offset), min(img.shape[0], y + h + offset)
        x1, x2 = max(0, x - offset), min(img.shape[1], x + w + offset)
        
        imgCrop = img[y1:y2, x1:x2]
        if imgCrop.size == 0:
            continue
            
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = (imgSize - wCal) // 2
            imgWhite[:, wGap:wGap + wCal] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = (imgSize - hCal) // 2
            imgWhite[hGap:hGap + hCal, :] = imgResize

        cv2.imshow('Captured Hand', imgWhite)

    cv2.putText(img, f"Gesture: {GESTURE} | Saved: {counter}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    cv2.imshow("Data Collection", img)

    key = cv2.waitKey(1)
    if key == ord('s'):
        counter += 1
        cv2.imwrite(f'{FOLDER}/Image_{int(time.time()*1000)}.jpg', imgWhite)
        print(f"✅ Saved {counter}")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"✅ Done! Collected {counter} images for '{GESTURE}'")