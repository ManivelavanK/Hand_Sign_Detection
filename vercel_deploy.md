# 🚀 Vercel Deployment Guide

## Quick Deploy Steps

### 1. Install Vercel CLI
```bash
npm i -g vercel
```

### 2. Login to Vercel
```bash
vercel login
```

### 3. Deploy
```bash
vercel --prod
```

## Alternative: GitHub Integration

### 1. Push to GitHub
```bash
git init
git add .
git commit -m "Deploy to Vercel"
git branch -M main
git remote add origin https://github.com/yourusername/sign-language-detection.git
git push -u origin main
```

### 2. Connect to Vercel
1. Go to [vercel.com](https://vercel.com)
2. Click "New Project"
3. Import your GitHub repo
4. Deploy automatically

## 📝 Important Notes

- **Camera Limitation**: Vercel serverless functions can't access camera
- **Demo Mode**: Deployed version shows UI without camera functionality
- **Local Development**: Use `python app.py` for full camera features
- **File Size**: Model file might be too large for Vercel (500MB limit)

## 🔧 Files Created
- `api/index.py` - Vercel serverless function
- `vercel.json` - Vercel configuration
- `requirements.txt` - Python dependencies

## ⚡ One-Click Deploy

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/yourusername/sign-language-detection)

## 🌐 Access Your App
After deployment: `https://your-app-name.vercel.app`