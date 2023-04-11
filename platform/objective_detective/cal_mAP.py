import json
import os
from collections import defaultdict

import numpy as np
from PIL import Image

from noise_array import *

from objective_detective.pikachu import detect_image


def calculate_ap(recalls, precisions):
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))

    for i in range(precisions.size - 1, 0, -1):
        precisions[i - 1] = np.maximum(precisions[i - 1], precisions[i])

    indices = np.where(recalls[1:] != recalls[:-1])[0] + 1
    ap = np.sum((recalls[indices] - recalls[indices - 1]) * precisions[indices])
    return ap


def calculate_iou(box1, box2):
    xmin1, ymin1, xmax1, ymax1 = box1
    xmin2, ymin2, xmax2, ymax2 = box2

    inter_xmin = np.maximum(xmin1, xmin2)
    inter_ymin = np.maximum(ymin1, ymin2)
    inter_xmax = np.minimum(xmax1, xmax2)
    inter_ymax = np.minimum(ymax1, ymax2)

    inter_area = np.maximum(inter_xmax - inter_xmin, 0) * \
                 np.maximum(inter_ymax - inter_ymin, 0)
    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)
    union_area = area1 + area2 - inter_area

    iou = inter_area / union_area
    return iou


def calculate_map(label_file, image_folder, num_classes, iou_threshold=0.5):
    gn_attack = GaussianNoiseAttack()

    with open(label_file, 'r') as f:
        labels = json.load(f)

    ground_truths = defaultdict(list)
    detections = defaultdict(list)

    for image_file, label_data in labels.items():
        image_path = os.path.join(image_folder, image_file)
        image_array = read_image(image_path)
        image_array = image_array[:, :, :3]
        _, image_array = gn_attack(image_array, unpack=True)
        image_array = image_array.astype(np.uint8)

        detected_objects = detect_image(image_array)

        for obj in detected_objects:
            detections[int(obj[0])].append([image_file, obj[1], *obj[2:]])

        ground_truths[int(label_data['class'])].append([image_file, *label_data['loc']])

    aps = []
    for class_id in range(num_classes):
        sorted_detections = sorted(detections[class_id], key=lambda x: -x[1])

        tp = np.zeros(len(sorted_detections))
        fp = np.zeros(len(sorted_detections))
        gt_count = defaultdict(int)

        for gt in ground_truths[class_id]:
            gt_count[gt[0]] += 1

        used_gt = defaultdict(set)

        for idx, det in enumerate(sorted_detections):
            max_iou = iou_threshold
            best_gt_idx = -1

            for gt_idx, gt in enumerate(ground_truths[class_id]):
                if gt[0] == det[0] and gt_idx not in used_gt[det[0]]:
                    iou = calculate_iou(det[2:], gt[1:])
                    if iou > max_iou:
                        max_iou = iou
                        best_gt_idx = gt_idx

            if best_gt_idx >= 0:
                tp[idx] = 1
                used_gt[det[0]].add(best_gt_idx)
            else:
                fp[idx] = 1

        cumsum_tp = np.cumsum(tp)
        cumsum_fp = np.cumsum(fp)

        recalls = cumsum_tp / sum(gt_count.values())
        precisions = cumsum_tp / (cumsum_tp + cumsum_fp)

        ap = calculate_ap(recalls, precisions)
        aps.append(ap)

    mAP = np.mean(aps)
    return mAP


def read_image(image_path):
    with Image.open(image_path) as image:
        image_array = np.array(image)
        return image_array