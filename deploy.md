# 🚀 Deployment Guide

## Option 1: Heroku (Recommended)

### Prerequisites
- Heroku account (free)
- Git installed
- Heroku CLI installed

### Steps
1. **Install Heroku CLI**
   ```bash
   # Download from: https://devcenter.heroku.com/articles/heroku-cli
   ```

2. **Login to Heroku**
   ```bash
   heroku login
   ```

3. **Create Heroku App**
   ```bash
   heroku create your-sign-language-app
   ```

4. **Deploy**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git push heroku main
   ```

5. **Open App**
   ```bash
   heroku open
   ```

## Option 2: Railway

1. **Go to Railway.app**
2. **Connect GitHub repo**
3. **Deploy automatically**

## Option 3: Render

1. **Go to Render.com**
2. **Connect GitHub repo**
3. **Set build command**: `pip install -r requirements_deploy.txt`
4. **Set start command**: `gunicorn app:app`

## Option 4: Local Network

```bash
python app.py
# Access via: http://YOUR_IP:5000
```

## 📝 Important Notes

- **Camera Access**: Web deployment won't access local camera
- **Model File**: Upload `sign_language_model.h5` to deployment
- **Data Folder**: Not needed for deployment
- **HTTPS Required**: For camera access in browsers

## 🔧 Quick Deploy Commands

```bash
# Rename requirements for deployment
cp requirements_deploy.txt requirements.txt

# Git setup
git init
git add .
git commit -m "Deploy sign language detection"

# Heroku deploy
heroku create your-app-name
git push heroku main
```