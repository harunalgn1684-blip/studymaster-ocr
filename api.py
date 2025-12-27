import os
import uuid
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS
from scanner import AnswerKeyScanner

app = Flask(__name__)
# Enable CORS to allow mobile app to communicate
CORS(app) 

# Initialize Scanner
scanner = AnswerKeyScanner()

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "service": "studymaster-ocr-api"}), 200

@app.route('/scan', methods=['POST'])
def scan_image():
    # 1. Handle Multipart File Upload
    if 'image' in request.files:
        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        
        try:
            ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else 'jpg'
            filename = f"{uuid.uuid4()}.{ext}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            print(f"Processing multipart image: {filepath}")
            result = scanner.run(filepath)
            
            try: os.remove(filepath)
            except: pass
            
            return jsonify(result), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # 2. Handle Base64 JSON Body
    data = request.get_json(silent=True)
    if data and 'image' in data:
        try:
            image_data = data['image']
            # Strip header if present (data:image/jpeg;base64,...)
            if ',' in image_data:
                image_data = image_data.split(',')[1]
                
            filename = f"{uuid.uuid4()}.jpg"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            with open(filepath, "wb") as fh:
                fh.write(base64.b64decode(image_data))
                
            print(f"Processing base64 image: {filepath}")
            result = scanner.run(filepath)
            
            try: os.remove(filepath)
            except: pass
            
            return jsonify(result), 200
        except Exception as e:
            print(f"Base64 error: {e}")
            return jsonify({"error": f"Invalid base64: {str(e)}"}), 500

    return jsonify({"error": "No image provided (file or base64)"}), 400

if __name__ == '__main__':
    # Run on 0.0.0.0 to be accessible from other devices in network
    app.run(host='0.0.0.0', port=5000, debug=True)
