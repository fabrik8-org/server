from PIL import Image
import requests
from io import BytesIO
from flask import Flask, jsonify, request
from flask_cors import CORS
import random

import requests

app = Flask(__name__)
CORS(app)
app.config['JSON_SORT_KEYS'] = False


def verify_image(url):
    try:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
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
    print(data)
    image_url = data.get('image')
    if not image_url:
        return jsonify({"message": "Invalid request, image URL missing"}), 400
    if not verify_image(image_url):
        return jsonify({"message": "Invalid image file"}), 400
    prediction = random.choice([True, False])
    return jsonify({"prediction": prediction}), 200


if __name__ == '__main__':
    app.run(port=5001, debug=True)
