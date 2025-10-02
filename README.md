# Hand Sign Detection 🖐️

Real-time hand gesture recognition using OpenCV and TensorFlow.

## Features
- Detects 5 signs: Hello, Thank You, I Love You, Yes, No
- Real-time webcam detection
- Trained with transfer learning (MobileNetV2)

## How to Use
1. Clone this repo
2. Install dependencies: `pip install -r requirements.txt`
3. **Collect your own data**: `python collect_data.py`
4. Train model: `python train_model.py`
5. Run detection: `python detect_signs.py`

## Dataset
Due to size, the dataset is not included. Use `collect_data.py` to create your own!

## Model
The trained model (`sign_language_model.h5`) is not included. Train your own using `train_model.py`.

## 📁 Important Notes

- **Dataset**: The `Data/` folder is not included in this repo due to size. 
  Use `collect_data.py` to create your own dataset.
  
- **Pre-trained Model**: `sign_language_model.h5` is not included. 
  Train your own model by running:
  ```bash
  python train_model.py

  
---

## ✅ Final Verification

After pushing to GitHub:
1. Go to your repo on GitHub.com
2. Verify you see **only code files** (no `Data/` folder)
3. Click "Add file" → "Upload files" → try uploading `sign_language_model.h5`
   - You should see: **"File too large"** → confirms `.gitignore` worked!

---

## 🎯 Summary

| Action | Command/Location |
|--------|------------------|
| **Code** | ✅ On GitHub |
| **Data (`Data/`)** | 🔒 Local backup only |
| **Model (`.h5`)** | 🔒 Local backup only |
| **How others use it** | Clone repo → collect data → train → detect |

You now have a **clean, professional GitHub repo** that follows best practices! 🌟

Let me know when your repo is live — I'd love to see it! 😊