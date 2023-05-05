from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import matplotlib.patches as patches
from collections import Counter
import cv2
import torchvision
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODEL_PATH = './model.pth'
device = torch.device('cpu')


# load a model pre-trained on COCO
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
    pretrained=False, pretrained_backbone=False)


num_classes = 2  #

# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features

# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Load the trained weights
weights = MODEL_PATH  # '/kaggle/working/' + 'bestmodel_method_new.pth'

checkpoint = torch.load(weights, map_location=device)
model.load_state_dict(checkpoint['state_dict'])
model.eval()

x = model.to(device)


def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    """
    Calculates intersection over union

    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct Labels of Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)

    Returns:
        tensor: Intersection over union for all examples
    """

    # Slicing idx:idx+1 in order to keep tensor dimensionality
    # Doing ... in indexing if there would be additional dimensions
    # Like for Yolo algorithm which would have (N, S, S, 4) in shape
    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    elif box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # Need clamp(0) in case they do not intersect, then we want intersection to be 0
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)


def mean_average_precision(
    pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=20
):
    """
    Calculates mean average precision 

    Parameters:
        pred_boxes (list): list of lists containing all bboxes with each bboxes
        specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
        true_boxes (list): Similar as pred_boxes except all the correct ones 
        iou_threshold (float): threshold where predicted bboxes is correct
        box_format (str): "midpoint" or "corners" used to specify bboxes
        num_classes (int): number of classes

    Returns:
        float: mAP value across all classes given a specific IoU threshold 
    """

    # list storing all AP for respective classes
    average_precisions = []

    # used for numerical stability later on
    epsilon = 1e-6

    for c in range(num_classes):
        detections = []
        ground_truths = []

        # Go through all predictions and targets,
        # and only add the ones that belong to the
        # current class c
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        # find the amount of bboxes for each training example
        # Counter here finds how many ground truth bboxes we get
        # for each training example, so let's say img 0 has 3,
        # img 1 has 5 then we will obtain a dictionary with:
        # amount_bboxes = {0:3, 1:5}
        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        # We then go through each key, val in this dictionary
        # and convert to the following (w.r.t same example):
        # ammount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        # sort by box probabilities which is index 2
        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)

        # If none exists for this class then we can safely skip
        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            # Only take out the ground_truths that have the same
            # training idx as detection
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]

            num_gts = len(ground_truth_img)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    box_format=box_format,
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                # only detect ground truth detection once
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    # true positive and add this bounding box to seen
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1

            # if IOU is lower then the detection is a false positive
            else:
                FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        # torch.trapz for numerical integration
        average_precisions.append(torch.trapz(precisions, recalls))

    return sum(average_precisions) / len(average_precisions)


def nms(bboxes, iou_threshold, threshold, box_format="corners"):
    """
    Does Non Max Suppression given bboxes

    Parameters:
        bboxes (list): list of lists containing all bboxes with each bboxes
        specified as [class_pred, prob_score, x1, y1, x2, y2]
        iou_threshold (float): threshold where predicted bboxes is correct
        threshold (float): threshold to remove predicted bboxes (independent of IoU) 
        box_format (str): "midpoint" or "corners" used to specify bboxes

    Returns:
        list: bboxes after performing NMS given a specific IoU threshold
    """

    assert type(bboxes) == list

    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]
            or intersection_over_union(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
                box_format=box_format,
            )
            < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms


def get_dimensions(xmin, ymin, xmax, ymax):
    width = xmax - xmin
    height = ymax - ymin
    return xmin, ymin, width, height


def applyNMS(bboxes, scores, pred_labels, iou_threshold=0.2, threshold=0.7, box_format='corners'):
    formatted_bboxes = list(zip(pred_labels, scores, bboxes))
    formatted_input = [[x[0], x[1], *x[2]] for x in formatted_bboxes]
    return nms(formatted_input, iou_threshold, threshold, box_format)

#this function to get all the bounding boxes
def applyNMS_all(bboxes, scores, pred_labels, iou_threshold=1.0, threshold=0.0, box_format='corners'):
    formatted_bboxes = list(zip(pred_labels, scores, bboxes))
    formatted_input = [[x[0], x[1], *x[2]] for x in formatted_bboxes]
    return nms(formatted_input, iou_threshold, threshold, box_format)



def bbLabelFormat(xmin, ymin, rect_width):
    if ymin > 50:
        return (xmin, ymin)
    else:
        if xmin + rect_width > 462:
            return (xmin - rect_width - 10, ymin+20)
        return (xmin + 10, ymin + 20)


# This function is for reference only
def draw_bounding_boxes_with_plt(npimg, model=model):
    image = cv2.imdecode(npimg, flags=cv2.IMREAD_COLOR)
    orig_image = image.copy()

    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB).astype(np.float64)
    image /= 255.0
    image = np.transpose(image, (2, 0, 1)).astype(np.float64)
    image = torch.tensor(image, dtype=torch.float)
    image = torch.unsqueeze(image, 0)

    with torch.no_grad():
        outputs = model(image)

    outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]

    if len(outputs[0]['boxes']) != 0:
        boxes = outputs[0]['boxes'].data.numpy()
        scores = outputs[0]['scores'].data.numpy()
        pred_classes = [i for i in outputs[0]['labels'].numpy()]

        non_suppressed_boxes = applyNMS(boxes, scores, pred_classes)
        print(non_suppressed_boxes)

        fig, ax = plt.subplots()
        ax.imshow(orig_image, interpolation='nearest', aspect='equal')

        for j in range(len(non_suppressed_boxes)):
            box = non_suppressed_boxes[j]
            xmin, ymin, width, height = get_dimensions(
                int(box[2]), int(box[3]), int(box[4]), int(box[5]))
            rect = patches.Rectangle(
                (xmin, ymin), width, height, linewidth=2, edgecolor='r', facecolor="none")
            ax.add_patch(rect)

            rx, ry = rect.get_xy()
            cx = rx + rect.get_width() / 2.0
            cy = ry + rect.get_height() / 2.0
            ax.annotate('Defect ' + str(np.around(box[1] * 100, decimals=1)) + '%', (bbLabelFormat(xmin, ymin, rect.get_width())),
                        color='white', fontsize=12,
                        bbox=dict(facecolor='red', edgecolor='black', linewidth=0.8, alpha=0.7))

        ax.set_axis_off()
        plt.subplots_adjust(wspace=None, hspace=None)
        plt.show()

    print(f"Image processed...")
    return ax


def resize_image(npimg):
    # Load the image using imread
    image = cv2.imdecode(npimg, flags=cv2.IMREAD_COLOR)

    # Get the dimensions of the image
    height, width = image.shape[:2]

    # Determine the scale factor to resize the image
    max_dimension = 512
    scale = max_dimension / max(height, width)

    # Resize the image
    resized_image = cv2.resize(image, None, fx=scale, fy=scale)

    return resized_image


def make_image_square(image):
    height, width = image.shape[:2]

    # Check if the image is already square
    if height == width:
        return image

    # Calculate the desired size for the square image
    target_size = max(height, width)

    # Create a new blank square image with the desired size
    square_image = np.zeros((target_size, target_size, 3), np.uint8)
    square_image.fill(128)  # Fill the image with gray color

    # Calculate the offset to center the original image in the square image
    x_offset = (target_size - width) // 2
    y_offset = (target_size - height) // 2

    # Copy the original image to the center of the square image
    square_image[y_offset:y_offset+height, x_offset:x_offset+width] = image

    return square_image


def calculate_defective_percentage(bounding_boxes, image_shape):
    """
    Calculates the total area covered by the defective bounding boxes.

    Parameters:
        bounding_boxes (list): List of bounding boxes in the format [xmin, ymin, xmax, ymax].
        image_shape (tuple): Tuple representing the shape of the image (height, width).

    Returns:
        float: The percentage of the image covered by the bounding boxes.
    """
    defective_area = 0

    for box in bounding_boxes:
        _, _, width, height = get_dimensions(
            int(box[2]), int(box[3]), int(box[4]), int(box[5]))
        defective_area += width * height

    total_area = image_shape[0] * image_shape[1]
    defective_area_percentage = defective_area / total_area * 100
    return defective_area_percentage


def draw_bounding_boxes(npimg, model=model):
    """
    Draws bounding boxes on an image and returns the image with the bounding boxes.

    Parameters:
        npimg (numpy.ndarray): The image represented as a NumPy array.
        model (object): The model used for object detection (optional).

    Returns:
        bytes or None: The image with bounding boxes encoded as bytes in PNG format if bounding boxes are detected,
                    None if no bounding boxes are detected.

    """
    defective = False
    defective_area_percentage = 0
    image = cv2.imdecode(npimg, flags=cv2.IMREAD_COLOR)
    height, width, _ = image.shape
    if height > 512 or width > 512:
        image = resize_image(npimg)
    orig_image = image.copy()
    orig_image_all = image.copy()

    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB).astype(np.float64)
    image /= 255.0
    image = np.transpose(image, (2, 0, 1)).astype(np.float64)
    image = torch.tensor(image, dtype=torch.float)
    image = torch.unsqueeze(image, 0)

    with torch.no_grad():
        outputs = model(image)

    outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]

    if len(outputs[0]['boxes']) != 0:
        defective = True
        boxes = outputs[0]['boxes'].data.numpy()
        scores = outputs[0]['scores'].data.numpy()
        pred_classes = [i for i in outputs[0]['labels'].numpy()]

        non_suppressed_boxes = applyNMS(boxes, scores, pred_classes)

        # This get all the bounding boxes
        all_boxes = applyNMS_all(boxes, scores, pred_classes)
        print(all_boxes)

        for j in range(len(non_suppressed_boxes)):
            box = non_suppressed_boxes[j]
            xmin, ymin, width, height = get_dimensions(
                int(box[2]), int(box[3]), int(box[4]), int(box[5]))
            cv2.rectangle(orig_image, (xmin, ymin),
                          (xmin + width, ymin + height), (0, 0, 255), 2)
    #---------------------------------------

        defective_area_percentage = calculate_defective_percentage(
            non_suppressed_boxes, orig_image.shape)

    # Return the image with bounding boxes
    square_image = make_image_square(orig_image)
    _, output_image = cv2.imencode('.png', square_image)
    result_image = output_image.tobytes()
    return {'defective': defective, 'image': result_image, 'percentage': defective_area_percentage}
        # Return the image with bounding boxes
        _, output_image = cv2.imencode('.png', orig_image)
    #---------------------------------------

        # All bounding boxes

        for j in range(len(all_boxes)):
            box = all_boxes[j]
            xmin, ymin, width, height = get_dimensions(
                int(box[2]), int(box[3]), int(box[4]), int(box[5]))
            cv2.rectangle(orig_image_all, (xmin, ymin),
                          (xmin + width, ymin + height), (0, 0, 255), 2)

        # Return the image with all the bounding boxes
        _, output_image_all = cv2.imencode('.png', orig_image_all)

    #---------------------------------------
    
        return True, output_image.tobytes(), output_image_all.tobytes(), outputs
    else:
        return False
