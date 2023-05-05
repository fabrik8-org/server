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
    #---------------------------------------
    image_base64 = base64.b64encode(result['image']).decode('utf-8')
    prediction = result['defective']
    defect_percentage = result['percentage']
    return jsonify({"prediction": prediction, "image": image_base64, 'percentage': defect_percentage}), 200
    #---------------------------------------

    pred, result, all_bounding_boxes, output = draw_bounding_boxes(npimg=npimg)
    #this is the final image
    image_base64 = base64.b64encode(result).decode('utf-8')
    #this is the image with all the bounding boxes
    image_base64_all = base64.b64encode(all_bounding_boxes).decode('utf-8')
    prediction = pred
    print(output)
    return jsonify({"prediction": prediction, "image": image_base64}), 200
    #---------------------------------------



if __name__ == '__main__':
    app.run(port=5001, debug=True)
