@echo off
echo 🚀 Quick Deploy Script for Sign Language Detection

echo.
echo Choose deployment option:
echo 1. Heroku
echo 2. Local Network
echo 3. Docker
echo.

set /p choice="Enter choice (1-3): "

if "%choice%"=="1" (
    echo 📦 Preparing for Heroku...
    copy requirements_deploy.txt requirements.txt
    git init
    git add .
    git commit -m "Deploy sign language detection"
    echo.
    echo ⚠️  Now run: heroku create your-app-name
    echo ⚠️  Then run: git push heroku main
    pause
)

if "%choice%"=="2" (
    echo 🌐 Starting local network server...
    python app.py
)

if "%choice%"=="3" (
    echo 🐳 Building Docker image...
    docker build -f docker/Dockerfile -t sign-language-app .
    echo 🚀 Running Docker container...
    docker run -p 5000:5000 sign-language-app
)