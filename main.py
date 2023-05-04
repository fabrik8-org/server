from PIL import Image
import cv2
import numpy as np
import requests
from io import BytesIO
from flask import Flask, jsonify, request
from flask_cors import CORS
import random
import base64

import requests

from model_funcs import draw_bounding_boxes

app = Flask(__name__)
CORS(app)
app.config['JSON_SORT_KEYS'] = False


def verify_image(encoded_image):
    try:
        image_bytes = base64.b64decode(encoded_image)

        # Open the image using PIL
        img = Image.open(BytesIO(image_bytes))

        # Verify the image
        img.verify()
        return True
    except:
        return False


@app.route('/health')
def health_check():
    return jsonify({"message": "Healthy"}), 200


@app.route('/predict', methods=['POST'])
def predict_defect():
    data = request.get_json()
    encoded_image = data.get('image')
    img = base64.b64decode(encoded_image)
    npimg = np.frombuffer(img, dtype=np.uint8)
    result = draw_bounding_boxes(npimg=npimg)
    image_base64 = base64.b64encode(result).decode('utf-8')
    prediction = random.choice([True, False])
    return jsonify({"prediction": prediction, "image": image_base64}), 200


if __name__ == '__main__':
    app.run(port=5001, debug=True)