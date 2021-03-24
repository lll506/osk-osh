import logging
import tempfile
import os
import torch
from collections import OrderedDict
from tqdm import tqdm
import cv2
import numpy as np
import shapely.geometry as shgeo

from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.utils.poly_nms.poly_nms import poly_nms
from maskrcnn_benchmark.config import cfg
import shapely
from shapely.geometry import Polygon, MultiPoint
import math

def getCrossPoint(LineA, LineB):
    ka = (LineA[3] - LineA[1]) / (LineA[2] - LineA[0])
    kb = (LineB[3] - LineB[1]) / (LineB[2] - LineB[0])
    if abs(math.atan(ka) - math.atan(kb)) < (math.pi * 7 /18) or abs(math.atan(ka) - math.atan(kb)) > (math.pi * 11 /18):
        return -100, -100
    x = (ka * LineA[0] - LineA[1] - kb * LineB[0] + LineB[1]) / (ka - kb)
    y = (ka * kb * (LineA[0] - LineB[0]) + ka * LineB[1] - kb * LineA[1]) / (ka - kb)
    return x, y

def postprocess(keypoint):
    # rect = np.zeros((4, 3, 2))
    rect = np.zeros((8, 4, 2))
    keypoint = keypoint.reshape(8, 2)
    for i in range(8):
        poly = cv2.minAreaRect(np.float32(np.delete(keypoint, i, axis=0))) 
        rect[i] = cv2.boxPoints(poly).reshape(4, 2)

    poly = cv2.minAreaRect(np.float32(keypoint)) 
    poly = shgeo.Polygon(cv2.boxPoints(poly).reshape(4, 2))

    poly_area, min_area = poly.area, poly.area
    min_index = 0
    for  j in range(8):
        poly_1 = shgeo.Polygon(rect[j]).area
        if poly_1 > 0.7 * poly_area and poly_1 < min_area:
            min_index = j
            min_area = poly_1
    return rect[min_index].reshape(8)

def write( output_folder, pred_dict ):
    output_folder_txt = os.path.join( output_folder, "results" )
    if not os.path.exists( output_folder_txt ):
        os.mkdir( output_folder_txt )
    for key in pred_dict:
        detections = pred_dict[key]
        output_path = os.path.join( output_folder_txt, "Task1_" + key + ".txt")
        with open(output_path, "w") as f:
            for det in detections:
                row = '{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.format(
                det[0], det[1],
                det[2], det[3],
                det[4], det[5],
                det[6], det[7],
                det[8], det[9]
                )
                f.write(row)

def handle_ratio_prediction(prediction):
    hboxes = prediction.bbox.data.numpy()
    rboxes = prediction.get_field( "rboxes" ).data.numpy()
    ratios = prediction.get_field( "ratios" ).data.numpy()
    scores = prediction.get_field( "scores" ).data.numpy()
    labels = prediction.get_field( "labels" ).data.numpy()


    h_idx = np.where(ratios > 0.8)[0]
    h = hboxes[h_idx]
    hboxes_vtx = np.vstack( [h[:, 0], h[:, 1], h[:, 2], h[:, 1], h[:, 2], h[:, 3], h[:, 0], h[:, 3]] ).transpose((1,0))
    rboxes[h_idx] = hboxes_vtx
    keep = poly_nms( np.hstack( [rboxes, scores[:, np.newaxis]] ).astype( np.double ), 0.1 )

    rboxes = rboxes[keep].astype( np.int32 )
    scores = scores[keep]
    labels = labels[keep]

    if len( rboxes ) > 0:
        rboxes = np.vstack( rboxes )
        return rboxes, scores, labels
    else:
        return None, None, None


def handle_keypoint_prediction(prediction, ratio1, ratio2, ratio3, ratio4):
    hboxes = prediction.bbox.data.numpy()
    keypoints = prediction.get_field( "points" ).data.numpy()
    # ratios = prediction.get_field( "ratios" ).data.numpy()
    scores = prediction.get_field( "scores" ).data.numpy()
    labels = prediction.get_field( "labels" ).data.numpy()
    logits = prediction.get_field( "logits" ).data.numpy()
    keypoints = np.array(keypoints)
    keypoints = keypoints.reshape((-1, 8, 2))

    logits = np.array(logits)
    logits = logits.reshape((-1, 8))
    logits = np.max(logits, axis=1)


    rboxes = np.zeros((hboxes.shape[0], 8))

    for i, keypoint in enumerate(keypoints):
        # if labels[i] == 10 or labels[i] == 12:
        #     xmin = hboxes[i][0]
        #     ymin = hboxes[i][1]
        #     xmax = hboxes[i][2]
        #     ymax = hboxes[i][3]
        #     rboxes[i] = np.array([xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax])
        # else:
        rect = cv2.minAreaRect(np.float32(keypoint)) 
        rect = cv2.boxPoints(rect).reshape(8)
        poly = shgeo.Polygon(rect.reshape(4, 2))
        
        rect1 = cv2.minAreaRect(np.float32(keypoint[0: 4])) 
        rect1 = cv2.boxPoints(rect1).reshape(8)
        poly1 = shgeo.Polygon(rect1.reshape(4, 2))
        poly1 = poly1.buffer(0.01)
        poly1_1 = poly1.intersection(poly)

        rect2 = np.float32(keypoint[0: 4]).reshape(8)
        poly2 = shgeo.Polygon(rect2.reshape(4, 2))
        poly2 = poly2.buffer(0.01)
        poly2_1 = poly2.intersection(poly)

        h_min = np.min(hboxes[i])
        h_max = np.max(hboxes[i])
        # scores[i] *= logits[i]
        if h_min <=1 or h_max >=1024:
            scores[i] *= 0.9

        if poly2_1.area / (poly.area + poly2.area - poly2_1.area + 1e-10) < 0.30:
            scores[i] = 0
            ratio1 +=1
        if poly2_1.area / (poly.area + poly2.area - poly2_1.area + 1e-10) > 0.75:
            rboxes[i] = rect2
            ratio2 += 1
        elif poly1_1.area / (poly.area + poly1.area - poly1_1.area + 1e-10) > 0.75:
            rboxes[i] = rect1
            ratio3 += 1
        else:
            rboxes[i] = rect
            scores[i] *= 0.9
            ratio4 += 1

        # rboxes[i] = postprocess(keypoint)
    # for j, keypoint in enumerate(keypoints):
    #     weight = 0
    #     logit = logits[j]
    #     lines = []
    #     indexes = [[0, 5, 1],
    #             [1, 6, 2],
    #             [2, 7, 3],
    #             [3, 8, 0]]
    #     for index in indexes:
    #         fit = []
    #         for i in range(len(index)):
    #             if logit[index[i]] >= -4:
    #                 fit.append(keypoint[index[i]])
    #         if len(fit) >= 2:
    #             fit = np.array(fit)
    #             output = cv2.fitLine(fit, cv2.DIST_L2, 0, 0.01, 0.01)
    #             k = output[1] / output[0]
    #             b = output[3] - k * output[2]
    #             x1 = 1
    #             y1 = k * x1 + b
    #             x2 = 100
    #             y2 = k * x2 + b
    #             lines.append(np.array([x1, y1, x2, y2]))
    #         else:
    #             weight = 1
    #             break
    #     if weight == 0:
    #         cross_point = []
    #         cross_point.append(getCrossPoint(lines[0], lines[1]))
    #         cross_point.append(getCrossPoint(lines[1], lines[2]))
    #         cross_point.append(getCrossPoint(lines[2], lines[3]))
    #         cross_point.append(getCrossPoint(lines[3], lines[0]))
    #         cross_point = np.array(cross_point)
    #         if -100 in cross_point:
    #             rect = cv2.minAreaRect(keypoint) 
    #             rect = cv2.boxPoints(rect).reshape(8)
    #         else:  
    #             rect = cross_point.reshape(8)
            
    # # for keypoint in keypoints:
    #     else:
    #         rect = cv2.minAreaRect(keypoint) 
    #         rect = cv2.boxPoints(rect).reshape(8)

    #     rboxes[j] = rect    


    # h_idx = np.where(ratios > 0.8)[0]
    # h = hboxes[h_idx]
    # hboxes_vtx = np.vstack( [h[:, 0], h[:, 1], h[:, 2], h[:, 1], h[:, 2], h[:, 3], h[:, 0], h[:, 3]] ).transpose((1,0))
    # rboxes[h_idx] = hboxes_vtx
    keep = poly_nms( np.hstack( [rboxes, scores[:, np.newaxis]] ).astype( np.double ), 0.1 )

    rboxes = rboxes[keep].astype( np.int32 )
    scores = scores[keep]
    labels = labels[keep]

    if len( rboxes ) > 0:
        rboxes = np.vstack( rboxes )
        return rboxes, scores, labels, ratio1, ratio2, ratio3, ratio4
    else:
        return None, None, None, ratio1, ratio2, ratio3, ratio4

def do_dota_evaluation(
    dataset,
    predictions,
    box_only,
    output_folder,
    iou_types,
    expected_results,
    expected_results_sigma_tol,
):
    pred_dict = {label:[] for label in dataset.categories.values()}
    ratio1 = 0
    ratio2 = 0
    ratio3 = 0
    ratio4 = 0
    for image_id, prediction in tqdm( enumerate(predictions) ):
        original_id = dataset.id_to_img_map[image_id]
        img_info = dataset.get_img_info(image_id)
        image_width = img_info["width"]
        image_height = img_info["height"]
        prediction = prediction.resize((image_width, image_height))

        if cfg.MODEL.RATIO_ON:
            rboxes, scores, labels = handle_ratio_prediction(prediction)
        elif cfg.MODEL.KEYPOINT_ON:
            rboxes, scores, labels, ratio1, ratio2, ratio3, ratio4 = handle_keypoint_prediction(prediction, ratio1, ratio2, ratio3, ratio4)            
        else:
            raise NotImplementedError
        if rboxes is None:
            continue

        # img_name = img_info["file_name"].split( "/" )[-1].split( "." )[0]
        img_name = os.path.basename( img_info["file_name"] )[:-4]

        for rbox, score, label in zip(rboxes, scores, labels):
            json_label = dataset.contiguous_category_id_to_json_id[label]
            json_label = dataset.categories[json_label]
            object_row = rbox.tolist()
            object_row.insert(0, score)
            object_row.insert(0, img_name)
            pred_dict[json_label].append(object_row)
    # print(ratio1)
    # print(ratio2)
    # print(ratio3)
    # print(ratio4)
    write( output_folder, pred_dict )


