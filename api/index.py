from flask import Flask, render_template_string, jsonify

app = Flask(__name__)

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Sign Language Detection 🖐️</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; }
        .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 20px; box-shadow: 0 20px 40px rgba(0,0,0,0.1); }
        h1 { text-align: center; color: #333; margin-bottom: 30px; }
        .demo-box { background: #f8f9fa; padding: 20px; border-radius: 15px; text-align: center; margin: 20px 0; }
        .signs { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; margin-top: 20px; }
        .sign-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 15px; text-align: center; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🖐️ Sign Language Detection</h1>
        <div class="demo-box">
            <h3>Demo Mode - Vercel Deployment</h3>
            <p>Camera functionality requires local deployment.</p>
            <p>Clone repo and run: <code>python app.py</code></p>
        </div>
        
        <div class="demo-box">
            <h3>📋 Supported Signs</h3>
            <div class="signs">
                <div class="sign-card">👋 Hello</div>
                <div class="sign-card">🙏 Thank You</div>
                <div class="sign-card">❤️ I Love You</div>
                <div class="sign-card">✅ Yes</div>
                <div class="sign-card">❌ No</div>
            </div>
        </div>
    </div>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/prediction')
def get_prediction():
    return jsonify({
        'prediction': "Demo Mode",
        'confidence': 0.0,
        'camera_active': False
    })