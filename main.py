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
import os
import io
import time
import random
import pandas as pd


app = Flask(__name__)
CORS(app)
app.config['JSON_SORT_KEYS'] = False

no_of_feedback_requests = 0
no_of_feedback_replied = 0
no_of_yes = 0
no_of_no = 0

def get_feedback_status():
    global no_of_feedback_requests
    global no_of_feedback_replied
    global no_of_yes
    global no_of_no

    feedback_replied_request_ratio = 1
    no_yes_ratio = 1

    if no_of_feedback_requests != 0 and no_of_feedback_replied != 0:
        feedback_replied_request_ratio = no_of_feedback_replied / no_of_feedback_requests
    
    if no_of_no != 0 and no_of_yes != 0:
        no_yes_ratio = no_of_no / no_of_yes
    
    threshold = ( random.randint(0,10) * feedback_replied_request_ratio * no_yes_ratio ) / 10

    if threshold >= 0.5:
        no_of_feedback_requests += 1
        return True
    
    return False



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


def adjust_image(base64_string):

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
    return  resized_base64_string


def init_monitoring():
    return neptune.init_run(
        capture_stdout=True,
        capture_stderr=True,
        capture_hardware_metrics=True,
    )


# @app.route('/datatrain')
# def create_train_data():
#     # Initialize the Neptune client
#     project = neptune.init_project(
#         mode="read-only",
#     )

#     # Get all the experiments in the project
#     experiments = project.fetch_runs_table().to_pandas()

#     print('\n\n\n-----------------------------------------------\n\n\n')

#     print(experiments.__dict__)


#     print('\n\n\n-----------------------------------------------\n\n\n')

#     # for experiment in experiments:

#     #     print(experiment['boundingBoxes/scores'])
#     #     break

#         # print(experiment.get_property('feedback/feedback'))
#         # print(experiment.get_property('feedback/filename'))
#         # print(experiment.get_property('feedback/defective'))
#         # print(experiment.get_property('feedback/x_min'))
#         # print(experiment.get_property('feedback/y_min'))
#         # print(experiment.get_property('feedback/x_max'))
#         # print(experiment.get_property('feedback/y_max'))
    
#     csv_doc = 'this variable points to your doc'

#     run = run = neptune.init_run(
#         project="farzan-frost/fabric8-dataset",
#         capture_stdout=True,
#         capture_stderr=True,
#         capture_hardware_metrics=True,
#     )

#     run['dataset'].upload(csv_doc)

#     run.stop()
        

#     project.stop()

#     return 200



@app.route('/health')
def health_check():
    return jsonify({"message": "Healthy"}), 200


@app.route('/predict', methods=['POST'])
def predict_defect():
    global x
    x += 1
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
    run_id = run["sys/id"].fetch()
    run.stop()
    return jsonify(
        {
            "prediction": prediction,
            "image_output": image_base64,
            "image_input": adjust_image(encoded_image),
            'percentage': defect_percentage, 
            'run_id': run_id,
            'request_feedback': get_feedback_status()
        }
    ), 200


@app.route('/feedback', methods=['POST'])
def feedback():
    global no_of_feedback_replied
    global no_of_yes
    global no_of_no

    no_of_feedback_replied += 1

    data = request.get_json()
    run_id = data.get('run_id')
    run = neptune.init_run(with_id=str(run_id))
    feedback = data.get('feedback')
    run['feedback/feedback'] = feedback
    if feedback == 'no':
        no_of_no += 1
        run['feedback/filename'] = run_id + 'png'
        run['feedback/defective'] = 1
        run['feedback/x_min'] = data.get('x_min')
        run['feedback/y_min'] = data.get('y_min')
        run['feedback/x_max'] = data.get('x_max')
        run['feedback/y_max'] = data.get('y_max')
        encoded_image = data.get('image')
        img = base64.b64decode(encoded_image)
        image = Image.open(io.BytesIO(img))
        run['feedback/image_input'].upload(image)
        encoded_image_ground_truth = data.get('image_ground_truth')
        img_ground_truth = base64.b64decode(encoded_image_ground_truth)
        image_ground_truth = Image.open(io.BytesIO(img_ground_truth))
        run['feedback/image_input_ground_truth'].upload(image_ground_truth)
    else:
        no_of_yes += 1
    run.stop()
    


if __name__ == '__main__':
    app.run(port=5001, debug=True)
