
import os
import numpy as np
import cv2
import math
import time
import copy


def sigmoid(x):
    # if x >= 0:
    #     return 1 / (1 + np.exp(-x))
    # else:
    return .5 * (1 + np.tanh(.5 * x))

def get_max_preds(batch_heatmaps):
    """Get predictions from score maps.
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    """
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]       # 1
    num_joints = batch_heatmaps.shape[1]       # 21
    width = batch_heatmaps.shape[3]            # 64
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))  # (1, 21, 4096)

    # idx = np.zeros((batch_size, num_joints, 2))
    # x_ = np.arange(0, 56, 1)
    # x_ = np.array(np.expand_dims(x_,0).repeat(56,axis=0))
    # for i in range(batch_size):
    #     for j in range(num_joints):
    #         batch_heatmaps[i, j] *= 100
    #         # idx[i, j, 0] = np.sum(x_ * (np.exp(batch_heatmaps[i, j]) / np.sum(np.exp(batch_heatmaps[i, j]))))
    #         # idx[i, j, 1] = np.sum(x_.T * (np.exp(batch_heatmaps[i, j]) / np.sum(np.exp(batch_heatmaps[i, j]))))

    #         idx[i, j, 0] = np.sum(x_ * batch_heatmaps[i, j]) / np.sum(batch_heatmaps[i, j])
    #         idx[i, j, 1] = np.sum(x_.T * batch_heatmaps[i, j]) / np.sum(batch_heatmaps[i, j])

    idx = np.argmax(heatmaps_reshaped, 2)       # max id of each joint heatmap
    maxvals = np.amax(heatmaps_reshaped, 2)     # max val of each id

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)
    # preds = idx
    # pred_mask = np.tile(np.greater(maxvals, -2), (1, 1, 2))
    # pred_mask = pred_mask.astype(np.float32)
    # preds *= pred_mask
    return preds, maxvals

def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs
    return src_result

def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)

def get_affine_transform(center,
                         scale,
                         scale_1,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale])
    if not isinstance(scale_1, np.ndarray) and not isinstance(scale_1, list):
        scale_1 = np.array([scale_1, scale_1])

    scale_tmp = scale * 200
    src_w = scale_tmp[0]

    scale_tmp_1 = scale_1 * 200
    src_h = scale_tmp_1[0]

    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    src_dir_1 = get_dir([src_h * -0.5, 0], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)
    dst_dir_1 = np.array([dst_w * -0.5, 0], np.float32)


    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = center + src_dir_1 + scale_tmp_1 * shift
    dst[2:, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir_1

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans

def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]

def transform_preds(coords, center, scale_y, scale_x, output_size, maxval):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale_y, scale_x, 0, output_size, inv=1)
    for p in range(coords.shape[0]):
        coord = affine_transform(coords[p, 0:2], trans)
        coord_min = np.min(coord)
        coord_max = np.max(coord)
        # if maxval[p] >= -4 or coord_min < 0 or coord_max > 1024:
        if maxval[p] >= -3.5:
            target_coords[p, 0:2] = coord
        else:
            target_coords[p, 0:2] = center
    return target_coords


def get_final_preds(batch_heatmaps, center, scale_y, scale_x):
    coords, maxvals = get_max_preds(batch_heatmaps)
    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]
    # Post-processing
    for n in range(coords.shape[0]):
        for p in range(coords.shape[1]):
            hm = batch_heatmaps[n][p]
            px = int(math.floor(coords[n][p][0] + 0.5))
            py = int(math.floor(coords[n][p][1] + 0.5))
            if 1 < px < heatmap_width-1 and 1 < py < heatmap_height-1:
                diff = np.array([hm[py][px+1] - hm[py][px-1],
                                 hm[py+1][px]-hm[py-1][px]])
                coords[n][p] += np.sign(diff) * .2
    preds = coords.copy()
    # Transform back
    for i in range(coords.shape[0]):
        preds[i] = transform_preds(coords[i], center[i], scale_y[i], scale_x[i], [heatmap_width, heatmap_height], maxvals[i])
    return preds, maxvals





