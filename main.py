import time
import io
import os
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

# Neptune for monitoring
import neptune
from dotenv import load_dotenv
load_dotenv()

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


def init_monitoring():
    return neptune.init_run(
        capture_stdout=True,
        capture_stderr=True,
        capture_hardware_metrics=True,
    )


@app.route('/health')
def health_check():
    return jsonify({"message": "Healthy"}), 200


@app.route('/adjust', methods=['POST'])
def adjust_image():
    # Get the base64 string from the request
    data = request.get_json()
    base64_string = data.get('image')

    # Decode the base64 string into bytes
    image_bytes = base64.b64decode(base64_string)

    # Open the image using PIL
    image = Image.open(io.BytesIO(image_bytes))

    # Resize the image to 512x512
    resized_image = image.resize((512, 512))

    # Create an in-memory buffer to store the resized image
    output_buffer = io.BytesIO()

    # Save the resized image to the buffer in JPEG format
    resized_image.save(output_buffer, format='PNG')

    # Get the base64 string of the resized image
    resized_base64_string = base64.b64encode(
        output_buffer.getvalue()).decode('utf-8')

    # Return the resized base64 string
    return jsonify({"image": resized_base64_string}), 200


@app.route('/predict', methods=['POST'])
def predict_defect():
    run = init_monitoring()
    data = request.get_json()
    encoded_image = data.get('image')
    img = base64.b64decode(encoded_image)

    image = Image.open(io.BytesIO(img))
    run['image_input'].upload(image)

    npimg = np.frombuffer(img, dtype=np.uint8)
    time_start = time.time()
    result = draw_bounding_boxes(npimg=npimg)
    time_end = time.time()
    run['inference_time'] = time_end - time_start
    output = result['output']
    run['boundingBoxes/bounding_boxes'] = str(output[0]['boxes'])
    run['boundingBoxes/labels'] = str(output[0]['labels'])
    run['boundingBoxes/scores'] = str(output[0]['scores'])
    # This is the image with only the bounding boxes of the defects
    image_base64 = base64.b64encode(result['image']).decode('utf-8')
    img_output = base64.b64decode(image_base64)
    image_output = Image.open(io.BytesIO(img_output))
    run['image_output'].upload(image_output)
    # This is for the image with all the bounding boxes
    image_base64_all = base64.b64encode(
        result['image_all_boxes']).decode('utf-8')
    img_output_all = base64.b64decode(image_base64_all)
    image_output_all = Image.open(io.BytesIO(img_output_all))
    run['image_output_all_bounding_boxes'].upload(image_output_all)
    prediction = result['defective']
    defect_percentage = result['percentage']
    run['prediction'] = prediction
    run['defect_percentage'] = defect_percentage
    run.stop()
    return jsonify({"prediction": prediction, "image": image_base64, 'percentage': defect_percentage}), 200


if __name__ == '__main__':
    app.run(port=5001, debug=True)
