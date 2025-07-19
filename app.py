from flask import Flask, request, jsonify, send_from_directory
from ultralytics import YOLO
import cv2
import numpy as np
import os
from datetime import datetime

app = Flask(__name__)
model = YOLO('yolov8n.pt')  # автоматически скачает веса при первом запуске

UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'static'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/process', methods=['POST'])
def process_image():
    file = request.files['image']
    filename = datetime.now().strftime("%Y%m%d%H%M%S") + '.jpg'
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    img = cv2.imread(filepath)
    results = model(img)
    output_img = results[0].plot()
    result_path = os.path.join(RESULT_FOLDER, 'result.jpg')
    cv2.imwrite(result_path, output_img)

    # Считаем только объекты класса 'table' или 'dining table'
    count = 0
    for box in results[0].boxes:
        class_id = int(box.cls[0])
        if model.model.names[class_id] == 'dining table' or model.model.names[class_id] == 'table':
            count += 1

    return jsonify(count=count)

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(RESULT_FOLDER, filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)