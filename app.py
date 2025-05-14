import cv2
import numpy as np
import mediapipe as mp
from flask import Flask, request, jsonify
from flask_cors import CORS
import base64

from constant import *


app = Flask(__name__)
CORS(app)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def extract_pose(img):
    results = pose.process(img)
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(img, 
                                  results.pose_landmarks, 
                                  mp_pose.POSE_CONNECTIONS)
    
    return results.pose_landmarks.landmark, img

def get_angle(a, b, c):
    ab = np.array([a.x - b.x, a.y - b.y])
    cb = np.array([c.x - b.x, c.y - b.y])

    dot = np.dot(ab, cb)
    
    magA = np.hypot(ab[0], ab[1])
    magB = np.hypot(cb[0], cb[1])

    angle = np.arccos(dot / (magA * magB)) * 180 / np.pi

    return angle

def calculate_strap_length(angle, tilt, height, weight, bag_weight):
    L_max = height * 0.40
    L_min = height * 0.25
    
    k_weight = height * 0.01
    k_posture = height * 0.005

    posture_penalty = angle / 30.0
    tilt_penalty = tilt / 15.0
    total_penalty = (bag_weight * k_weight) + (posture_penalty + tilt_penalty) * k_posture

    strap_length = max(L_min, min(L_max, L_max - total_penalty))
    
    return strap_length

@app.route('/process', methods=['POST'])
def process():
    global posture

    data = request.json
    if 'image' not in data:
        return jsonify({'error': 'No image provided'}), 400

    image_data = data['image'].split(',')[1]
    image_bytes = base64.b64decode(image_data)
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    posture, processed_img = extract_pose(img)

    _, buffer = cv2.imencode('.jpg', processed_img)
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
    
    return jsonify({"image": f"data:image/jpeg;base64,{jpg_as_text}"})

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    height = data.get('height')
    weight = data.get('weight')
    bag_weight = data.get('bagWeight')

    a = posture[LEFT_SHOULDER]
    b = posture[LEFT_HIP]
    c = posture[LEFT_KNEE]

    angle = get_angle(a, b, c)
    tilt = np.abs(posture[LEFT_SHOULDER].y - posture[RIGHT_SHOULDER].y) * 100
    
    strap_length = calculate_strap_length(angle, tilt, height, weight, bag_weight)

    return jsonify({"strap_length": strap_length})

if __name__ == '__main__':
    app.run(debug=True)