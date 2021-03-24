import torch
import math
import numpy as np
import cv2
import copy


# transpose
FLIP_LEFT_RIGHT = 0
FLIP_TOP_BOTTOM = 1

class Keypoints(object):
    def __init__(self, keypoints, size, mode=None):
        # FIXME remove check once we have better integration with device
        # in my version this would consistently return a CPU tensor
        device = keypoints.device if isinstance(keypoints, torch.Tensor) else torch.device('cpu')
        keypoints = torch.as_tensor(keypoints, dtype=torch.float32, device=device)
        num_keypoints = keypoints.shape[0]
        if num_keypoints:
            keypoints = keypoints.view(num_keypoints, -1, 3)
        
        # TODO should I split them?
        # self.visibility = keypoints[..., 2]
        self.keypoints = keypoints# [..., :2]

        self.size = size
        self.mode = mode
        self.extra_fields = {}

    def crop(self, box):
        raise NotImplementedError()

    def resize(self, size, *args, **kwargs):
        ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(size, self.size))
        ratio_w, ratio_h = ratios
        resized_data = self.keypoints.clone()
        resized_data[..., 0] *= ratio_w
        resized_data[..., 1] *= ratio_h
        keypoints = type(self)(resized_data, size, self.mode)
        for k, v in self.extra_fields.items():
            keypoints.add_field(k, v)
        return keypoints

    def transpose(self, method):
        if method not in (FLIP_LEFT_RIGHT,):
            raise NotImplementedError(
                    "Only FLIP_LEFT_RIGHT implemented")

        flip_inds = type(self).FLIP_INDS
        flipped_data = self.keypoints[:, flip_inds]
        width = self.size[0]
        TO_REMOVE = 1
        # Flip x coordinates
        flipped_data[..., 0] = width - flipped_data[..., 0] - TO_REMOVE

        # Maintain COCO convention that if visibility == 0, then x, y = 0
        inds = flipped_data[..., 2] == 0
        flipped_data[inds] = 0

        keypoints = type(self)(flipped_data, self.size, self.mode)
        for k, v in self.extra_fields.items():
            keypoints.add_field(k, v)
        return keypoints

    def to(self, *args, **kwargs):
        keypoints = type(self)(self.keypoints.to(*args, **kwargs), self.size, self.mode)
        for k, v in self.extra_fields.items():
            if hasattr(v, "to"):
                v = v.to(*args, **kwargs)
            keypoints.add_field(k, v)
        return keypoints

    def __getitem__(self, item):
        keypoints = type(self)(self.keypoints[item], self.size, self.mode)
        for k, v in self.extra_fields.items():
            keypoints.add_field(k, v[item])
        return keypoints

    def add_field(self, field, field_data):
        self.extra_fields[field] = field_data

    def get_field(self, field):
        return self.extra_fields[field]

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += 'num_instances={}, '.format(len(self.keypoints))
        s += 'image_width={}, '.format(self.size[0])
        s += 'image_height={})'.format(self.size[1])
        return s


def _create_flip_indices(names, flip_map):
    full_flip_map = flip_map.copy()
    full_flip_map.update({v: k for k, v in flip_map.items()})
    flipped_names = [i if i not in full_flip_map else full_flip_map[i] for i in names]
    flip_indices = [names.index(i) for i in flipped_names]
    return torch.tensor(flip_indices)


class PersonKeypoints(Keypoints):
    NAMES = [
        'nose',
        'left_eye',
        'right_eye',
        'left_ear',
        'right_ear',
        'left_shoulder',
        'right_shoulder',
        'left_elbow',
        'right_elbow',
        'left_wrist',
        'right_wrist',
        'left_hip',
        'right_hip',
        'left_knee',
        'right_knee',
        'left_ankle',
        'right_ankle'
    ]
    FLIP_MAP = {
        'left_eye': 'right_eye',
        'left_ear': 'right_ear',
        'left_shoulder': 'right_shoulder',
        'left_elbow': 'right_elbow',
        'left_wrist': 'right_wrist',
        'left_hip': 'right_hip',
        'left_knee': 'right_knee',
        'left_ankle': 'right_ankle'
    }


# TODO this doesn't look great
PersonKeypoints.FLIP_INDS = _create_flip_indices(PersonKeypoints.NAMES, PersonKeypoints.FLIP_MAP)
def kp_connections(keypoints):
    kp_lines = [
        [keypoints.index('left_eye'), keypoints.index('right_eye')],
        [keypoints.index('left_eye'), keypoints.index('nose')],
        [keypoints.index('right_eye'), keypoints.index('nose')],
        [keypoints.index('right_eye'), keypoints.index('right_ear')],
        [keypoints.index('left_eye'), keypoints.index('left_ear')],
        [keypoints.index('right_shoulder'), keypoints.index('right_elbow')],
        [keypoints.index('right_elbow'), keypoints.index('right_wrist')],
        [keypoints.index('left_shoulder'), keypoints.index('left_elbow')],
        [keypoints.index('left_elbow'), keypoints.index('left_wrist')],
        [keypoints.index('right_hip'), keypoints.index('right_knee')],
        [keypoints.index('right_knee'), keypoints.index('right_ankle')],
        [keypoints.index('left_hip'), keypoints.index('left_knee')],
        [keypoints.index('left_knee'), keypoints.index('left_ankle')],
        [keypoints.index('right_shoulder'), keypoints.index('left_shoulder')],
        [keypoints.index('right_hip'), keypoints.index('left_hip')],
    ]
    return kp_lines
PersonKeypoints.CONNECTIONS = kp_connections(PersonKeypoints.NAMES)


# TODO make this nicer, this is a direct translation from C2 (but removing the inner loop)
def keypoints_to_heat_map(keypoints, rois, heatmap_size):
    if rois.numel() == 0:
        return rois.new().long(), rois.new().long()
    offset_x = rois[:, 0]
    offset_y = rois[:, 1]
    scale_x = heatmap_size / (rois[:, 2] - rois[:, 0])
    scale_y = heatmap_size / (rois[:, 3] - rois[:, 1])

    offset_x = offset_x[:, None]
    offset_y = offset_y[:, None]
    scale_x = scale_x[:, None]
    scale_y = scale_y[:, None]

    x = keypoints[..., 0]
    y = keypoints[..., 1]

    x_boundary_inds = x == rois[:, 2][:, None]
    y_boundary_inds = y == rois[:, 3][:, None]

    x = (x - offset_x) * scale_x
    x = x.floor().long()
    y = (y - offset_y) * scale_y
    y = y.floor().long()
    
    x[x_boundary_inds] = heatmap_size - 1
    y[y_boundary_inds] = heatmap_size - 1

    valid_loc = (x >= 0) & (y >= 0) & (x < heatmap_size) & (y < heatmap_size)
    

    # vis = keypoints[..., 2] > 0
    # valid = (valid_loc & vis).long()
    valid = valid_loc.long()

    lin_ind = y * heatmap_size + x
    heatmaps = lin_ind * valid

    return heatmaps, valid

#TODO 
def keypoints_to_heat_map_2x(keypoints, rois, heatmap_size):
    
    if rois.numel() == 0:
        return rois.new().long(), rois.new().long()    
    offset_x = rois[:, 0] - 0.5 * (rois[:, 2] - rois[:, 0])
    offset_y = rois[:, 1] - 0.5 * (rois[:, 3] - rois[:, 1])
    scale_x = heatmap_size / (2 * (rois[:, 2] - rois[:, 0]))
    scale_y = heatmap_size / (2 * (rois[:, 3] - rois[:, 1]))

    offset_x = offset_x[:, None]
    offset_y = offset_y[:, None]
    scale_x = scale_x[:, None]
    scale_y = scale_y[:, None]

    keypoints_ = keypoints.view((-1, 8))
    max_x_, max_x_idx = keypoints_[:,  ::2].max( 1 )
    min_x_, min_x_idx = keypoints_[:,  ::2].min( 1 )
    max_y_, max_y_idx = keypoints_[:, 1::2].max( 1 )
    min_y_, min_y_idx = keypoints_[:, 1::2].min( 1 )


    # print('miny:', min_y_idx[0])
    # print('minx:', min_x_idx[0])
    # print('maxx:', max_x_idx[0])
    # print('maxy:', max_y_idx[0])
    # print(min_x_idx[:, 0] + min_y_idx[:, 0] + max_x_idx[:, 0] + max_y_idx[:, 0])

    keypoints_ordered = torch.zeros_like( keypoints )
    keypoints_ordered[:, 0] = keypoints[range(len(keypoints)), min_y_idx]
    keypoints_ordered[:, 1] = keypoints[range(len(keypoints)), max_x_idx]
    keypoints_ordered[:, 2] = keypoints[range(len(keypoints)), max_y_idx]
    keypoints_ordered[:, 3] = keypoints[range(len(keypoints)), min_x_idx]

    hroi = min_x_idx + min_y_idx + max_x_idx + max_y_idx != 6
    keypoints_ordered[hroi, 0, 0] = min_x_[hroi] 
    keypoints_ordered[hroi, 0, 1] = min_y_[hroi]
    keypoints_ordered[hroi, 1, 0] = max_x_[hroi]
    keypoints_ordered[hroi, 1, 1] = min_y_[hroi]
    keypoints_ordered[hroi, 2, 0] = max_x_[hroi]
    keypoints_ordered[hroi, 2, 1] = max_y_[hroi]
    keypoints_ordered[hroi, 3, 0] = min_x_[hroi]
    keypoints_ordered[hroi, 3, 1] = max_y_[hroi]

    # x = keypoints[..., 0]
    # y = keypoints[..., 1]

    x = keypoints_ordered[..., 0]
    y = keypoints_ordered[..., 1]

    x_boundary_inds = x == (rois[:, 2][:, None] + 0.5 * (rois[:, 2][:, None] - rois[:, 0][:, None]))
    y_boundary_inds = y == (rois[:, 3][:, None] + 0.5 * (rois[:, 3][:, None] - rois[:, 1][:, None]))

    x = (x - offset_x) * scale_x
    x = x.floor().long()
    y = (y - offset_y) * scale_y
    y = y.floor().long()
    
    x[x_boundary_inds] = heatmap_size - 1
    y[y_boundary_inds] = heatmap_size - 1

    valid_loc = (x >= 0) & (y >= 0) & (x < heatmap_size) & (y < heatmap_size)
    
    valid = valid_loc.long()

    lin_ind = y * heatmap_size + x
    heatmaps = lin_ind * valid


    return heatmaps, valid


def get_3rd_point(a, b):
    direct = a - b
    m = np.array([-direct[:, 1], direct[:, 0]], dtype=np.float32)
    return b + np.swapaxes(m,0,1)

def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = np.zeros((src_point.shape[0], 2))
    src_result[:, 0] = src_point[:, 0] * cs - src_point[:, 1] * sn
    src_result[:, 1] = src_point[:, 0] * sn + src_point[:, 1] * cs
    return src_result


def get_dir_1(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = np.zeros((2))
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs
    return src_result

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

    box_num = center.shape[0]

    scale_tmp = scale
    src_w = scale_tmp[:, 0].reshape(box_num, -1) * -0.5
    src_0 = np.zeros((box_num, 1))
    src_w = np.hstack((src_0, src_w))

    scale_tmp_1 = scale_1
    src_h = scale_tmp_1[:, 0].reshape(box_num, -1) * -0.5
    src_1 = np.zeros((box_num, 1))
    src_h = np.hstack((src_h, src_1))

    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir(src_w, rot_rad)
    src_dir_1 = get_dir(src_h, rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)
    dst_dir_1 = np.array([dst_w * -0.5, 0], np.float32)

    src = np.zeros((box_num, 3, 2), dtype=np.float32)
    dst = np.zeros((box_num, 3, 2), dtype=np.float32)
    src[:, 0, :] = center + scale_tmp * shift
    src[:, 1, :] = center + src_dir + scale_tmp * shift
    dst[:, 0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[:, 1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[:, 2, :] = center + src_dir_1 + scale_tmp_1 * shift
    dst[:, 2, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir_1
    # print(src)
    # print(dst)
    trans = []
    for j in range(src.shape[0]):
        if inv:
            tran = cv2.getAffineTransform(np.float32(dst[j]), np.float32(src[j])).tolist()
        else:
            tran = cv2.getAffineTransform(np.float32(src[j]), np.float32(dst[j])).tolist()
        trans.append(tran)
    return trans

def get_affine_transform_1(center,
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
    src_w = scale_tmp[0] * -0.5
    src_0 = np.zeros((1))
    src_w = np.hstack((src_0, src_w))

    scale_tmp_1 = scale_1 * 200
    src_h = scale_tmp_1[0] * -0.5
    src_1 = np.zeros((1))
    src_h = np.hstack((src_h, src_1))

    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir_1(src_w, rot_rad)
    src_dir_1 = get_dir_1(src_h, rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)
    dst_dir_1 = np.array([dst_w * -0.5, 0], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2, :] = center + src_dir_1 + scale_tmp_1 * shift
    dst[2, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir_1
    # print(src)
    # print(dst)


    if inv:
        tran = cv2.getAffineTransform(np.float32(dst), np.float32(src)).tolist()
    else:
        tran = cv2.getAffineTransform(np.float32(src), np.float32(dst)).tolist()

    return tran


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]

def generate_target_line(joints, type=0):
    """
    :param joints:  [num_joints, 3]
    :param joints_vis: [num_joints, 3]
    :return: target, target_weight(1: visible, 0: invisible)
    """
    NUM_JOINTS = 4
    TARGET_TYPE = 'gaussian'
    HEATMAP_SIZE = [56, 56]
    IMAGE_SIZE = [224, 224]
    SIGMA = 1.5

    target_weight = np.ones((NUM_JOINTS), dtype=np.float32)

    
    assert TARGET_TYPE == 'gaussian', \
        'Only support gaussian map now!'

    if TARGET_TYPE == 'gaussian':
        target = np.zeros((NUM_JOINTS,
                           HEATMAP_SIZE[1],
                           HEATMAP_SIZE[0]),
                           dtype=np.float32)
        tmp_size = SIGMA * 3
        
        for joint_id in range(NUM_JOINTS):
            feat_stride = np.array(IMAGE_SIZE) / np.array(HEATMAP_SIZE)
            mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
            mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)

            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
            if mu_x >= HEATMAP_SIZE[0] or mu_y >= HEATMAP_SIZE[1] \
                    or mu_x < 0 or mu_y < 0:
                # If not, just return the image as is
                target_weight[joint_id] = 0
                continue

            # Generate gaussian
            size = 2 * tmp_size + 1
            x = np.arange(0, size, 1, np.float32)
            y = x[:, np.newaxis]
            x0 = y0 = size // 2
            if type == 0:
                g = np.exp(- ((x - x0) ** 2 / (2 * SIGMA ** 2) + (y - y0) ** 2 / (2 * SIGMA ** 2)))
            elif type == 1:
                g = np.exp(- ((x - x0) ** 2 / (4 * SIGMA ** 2) + (y - y0) ** 2 / (0.2 * SIGMA ** 2)))                

            g_x = max(0, -ul[0]), min(br[0], HEATMAP_SIZE[0]) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], HEATMAP_SIZE[1]) - ul[1]
            img_x = max(0, ul[0]), min(br[0], HEATMAP_SIZE[0])
            img_y = max(0, ul[1]), min(br[1], HEATMAP_SIZE[1])

            v = target_weight[joint_id]
            if v > 0.5:
                target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                    g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    return target, target_weight

def generate_target_corner(joints, type=0):
    """
    :param joints:  [num_joints, 3]
    :param joints_vis: [num_joints, 3]
    :return: target, target_weight(1: visible, 0: invisible)
    """
    NUM_JOINTS = 5
    TARGET_TYPE = 'gaussian'
    HEATMAP_SIZE = [56, 56]
    IMAGE_SIZE = [224, 224]
    SIGMA = 1.5

    target_weight = np.ones((NUM_JOINTS), dtype=np.float32)

    
    assert TARGET_TYPE == 'gaussian', \
        'Only support gaussian map now!'

    if TARGET_TYPE == 'gaussian':
        target = np.zeros((NUM_JOINTS,
                           HEATMAP_SIZE[1],
                           HEATMAP_SIZE[0]),
                           dtype=np.float32)
        tmp_size = SIGMA * 3
        
        for joint_id in range(NUM_JOINTS):
            feat_stride = np.array(IMAGE_SIZE) / np.array(HEATMAP_SIZE)
            mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
            mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)

            # Check that any part of the gaussian is in-bounds
            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
            if mu_x >= HEATMAP_SIZE[0] or mu_y >= HEATMAP_SIZE[1] \
                    or mu_x < 0 or mu_y < 0:
                target_weight[joint_id] = 0
                continue

            # Generate gaussian
            size = 2 * tmp_size + 1
            x = np.arange(0, size, 1, np.float32)
            y = x[:, np.newaxis]
            x0 = y0 = size // 2
            if type == 0:
                g = np.exp(- ((x - x0) ** 2 / (1 * SIGMA ** 2) + (y - y0) ** 2 / (1 * SIGMA ** 2)))
            elif type == 1:
                g = np.exp(- ((x - x0) ** 2 / (1 * SIGMA ** 2) + (y - y0) ** 2 / (0.2 * SIGMA ** 2)))                

            g_x = max(0, -ul[0]), min(br[0], HEATMAP_SIZE[0]) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], HEATMAP_SIZE[1]) - ul[1]
            img_x = max(0, ul[0]), min(br[0], HEATMAP_SIZE[0])
            img_y = max(0, ul[1]), min(br[1], HEATMAP_SIZE[1])

            v = target_weight[joint_id]
            if v > 0.5:
                target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                    g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    return target, target_weight 

def generate_target_line_unquant(joints, type=0):
    """
    :param joints:  [num_joints, 3]
    :param joints_vis: [num_joints, 3]
    :return: target, target_weight(1: visible, 0: invisible)
    """
    NUM_JOINTS = 4
    TARGET_TYPE = 'gaussian'
    HEATMAP_SIZE = [56, 56]
    SIGMA = 1.5

    
    
    assert TARGET_TYPE == 'gaussian', \
        'Only support gaussian map now!'

    if TARGET_TYPE == 'gaussian':
        target = np.zeros((NUM_JOINTS,
                           HEATMAP_SIZE[1],
                           HEATMAP_SIZE[0]),
                           dtype=np.float32)
        target_weight = np.ones((NUM_JOINTS), dtype=np.float32)
                  
        for joint_id in range(NUM_JOINTS):
            mu_x = joints[joint_id][0]
            mu_y = joints[joint_id][1]

            if mu_x >= HEATMAP_SIZE[0] or mu_y >= HEATMAP_SIZE[1] \
                    or mu_x < 0 or mu_y < 0:
                target_weight[joint_id] = 0
                continue

            # Generate gaussian
            x = np.arange(0, 56, 1, np.float32)
            y = x[:, np.newaxis]
            if type == 0:
                target[joint_id] = np.exp(- ((x - mu_x) ** 2 / (1 * SIGMA ** 2) + (y - mu_y) ** 2 / (1 * SIGMA ** 2)))
            elif type == 1:
                target[joint_id] = np.exp(- ((x - mu_x) ** 2 / (4 * SIGMA ** 2) + (y - mu_y) ** 2 / (0.2 * SIGMA ** 2)))     

    return target, target_weight

def generate_target_corner_unquant(joints, type=0):
    """
    :param joints:  [num_joints, 3]
    :param joints_vis: [num_joints, 3]
    :return: target, target_weight(1: visible, 0: invisible)
    """
    NUM_JOINTS = 4
    TARGET_TYPE = 'gaussian'
    HEATMAP_SIZE = [56, 56]
    SIGMA = 1.5
    
    assert TARGET_TYPE == 'gaussian', \
        'Only support gaussian map now!'

    if TARGET_TYPE == 'gaussian':
        
        target = np.zeros((NUM_JOINTS,
                           HEATMAP_SIZE[1],
                           HEATMAP_SIZE[0]),
                           dtype=np.float32)
        target_weight = np.ones((NUM_JOINTS), dtype=np.float32)

        for joint_id in range(NUM_JOINTS):
            mu_x = joints[joint_id][0]
            mu_y = joints[joint_id][1]

            # Check that any part of the gaussian is in-bounds
            if mu_x >= HEATMAP_SIZE[0] or mu_y >= HEATMAP_SIZE[1] \
                    or mu_x < 0 or mu_y < 0:
                target_weight[joint_id] = 0
                continue

            # Generate gaussian
            x = np.arange(0, 56, 1, np.float32)
            y = x[:, np.newaxis]
            if type == 0:
                target[joint_id] = np.exp(- ((x - mu_x) ** 2 / (1 * SIGMA ** 2) + (y - mu_y) ** 2 / (1 * SIGMA ** 2)))
            elif type == 1:
                target[joint_id] = np.exp(- ((x - mu_x) ** 2 / (2 * SIGMA ** 2) + (y - mu_y) ** 2 / (0.2 * SIGMA ** 2)))            

    return target, target_weight

def rotate_image(image, center, angle):
    image_center = center[0: 2]

    image_center = image_center.reshape(2)
    # image_center[0], image_center[1] = image_center[1], image_center[0]

    rot_mat = cv2.getRotationMatrix2D(tuple(image_center), angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)

    return result

def choose_best_pointorder_fit_another(poly1):
    x1 = poly1[0]
    y1 = poly1[1]
    x2 = poly1[2]
    y2 = poly1[3]
    x3 = poly1[4]
    y3 = poly1[5]
    x4 = poly1[6]
    y4 = poly1[7]

    xmin = min( x1, x2, x3, x4 )
    ymin = min( y1, y2, y3, y4 )
    xmax = max( x1, x2, x3, x4 )
    ymax = max( y1, y2, y3, y4 )
    poly2 = [xmin, ymin]

    combinate = [np.array([x1, y1, x2, y2, x3, y3, x4, y4]), np.array([x2, y2, x3, y3, x4, y4, x1, y1]),
                    np.array([x3, y3, x4, y4, x1, y1, x2, y2]), np.array([x4, y4, x1, y1, x2, y2, x3, y3])]
    combinate_first = [np.array([x1, y1]), np.array([x2, y2]),
                        np.array([x3, y3]), np.array([x4, y4])]
    dst_coordinate = np.array(poly2)
    distances = np.array([np.sum((coord - dst_coordinate)**2) for coord in combinate_first])
    sorted = distances.argsort()
    return combinate[sorted[0]]

def keypoints_to_heat_map_gaussian(keypoints, rois, heatmap_size):
    
    NUM_JOINTS = 9

    sf = 0.25
    if rois.numel() == 0:
        return rois.new().long(), rois.new().long()
    rois_ = rois.cuda().data.cpu().numpy()


    keypoints_ = keypoints.view((-1, 8))

    keypoints_ = keypoints_.cuda().data.cpu().numpy()

    # print(rois_)
    x1 = rois_[:, 0].reshape(rois_.shape[0], -1)
    y1 = rois_[:, 1].reshape(rois_.shape[0], -1)
    x2 = rois_[:, 2].reshape(rois_.shape[0], -1)
    y2 = rois_[:, 3].reshape(rois_.shape[0], -1)

    c = np.hstack(((x1 + x2) / 2.0, (y1 + y2) / 2.0))

    s = np.hstack((x2 - x1, y2 - y1))    
    s = (s.max(1) / 200.0).reshape(rois_.shape[0], -1)
    s = np.hstack((s, s))
    s = s * np.clip(sf + 1, 1 - sf, 1 + sf)
    r = 0

    keypoints_per_image = np.zeros((rois_.shape[0], NUM_JOINTS, heatmap_size, heatmap_size))
    keypoints_weight_per_image = np.zeros((rois_.shape[0], NUM_JOINTS))

    trans = get_affine_transform(c, s, r, [224, 224])
    for index, tran, keypoint in zip(range(0, rois_.shape[0]), trans, keypoints_):

        keypoint = keypoint.reshape(8)
        keypoint = choose_best_pointorder_fit_another(keypoint)
        keypoint = np.array(keypoint)
        kp_corner = keypoint
        kp_angle = np.zeros(4)
        kp_line = kp_corner.copy() 

        joints_corner = []
        for j in range(0, 8, 2):
            kp_corner[j], kp_corner[j + 1] = affine_transform([kp_corner[j], kp_corner[j+1]], tran)
            joints_corner.append([kp_corner[j], kp_corner[j + 1], 1.0])
        joints_corner.append([0.25 * (kp_corner[0] + kp_corner[2] + kp_corner[4] + kp_corner[6]), 0.25 * (kp_corner[1] + kp_corner[3] + kp_corner[5] + kp_corner[7]), 1.0])

        joints_line = []
        for j in range(0, 8, 2):
            if j == 6:
                kp_line[j] = 0.5 * (kp_corner[j] + kp_corner[0])
                kp_line[j+1] = 0.5 * (kp_corner[j+1] + kp_corner[1])
                kp_angle[int(j / 2)] = (kp_corner[1] - kp_corner[j+1]) / (kp_corner[0] - kp_corner[j] + 1e-10)
            else:
                kp_line[j] = 0.5 * (kp_corner[j] + kp_corner[j+2])
                kp_line[j+1] = 0.5 * (kp_corner[j+1] + kp_corner[j+3])
                kp_angle[int(j / 2)] = (kp_corner[j+3] - kp_corner[j+1]) / (kp_corner[j+2] - kp_corner[j] + 1e-10)
            joints_line.append([kp_line[j], kp_line[j + 1], 1.0])  
        
        joints_corner = np.array(joints_corner)
        joints_line = np.array(joints_line)

        target_corner, target_corner_weight = generate_target_corner_unquant(joints_corner, type=0)
        target_line, target_line_weight = generate_target_line_unquant(joints_line, type=0) 
        target = np.concatenate((target_corner, target_line), axis=0)
        target_weight = np.concatenate((target_corner_weight, target_line_weight), axis=0)
        # target_ = copy.deepcopy(target)
        # target_ = np.array(target_)
        # target_[0], target_[1], target_[2], target_[3], target_[4], target_[5], target_[6], target_[7], target_[8] = target[0], target[5], target[1], target[8], target[4], target[6], target[3], target[7], target[2]
        keypoints_per_image[index] = target
        keypoints_weight_per_image[index] = target_weight
    
    keypoints_per_image = torch.as_tensor(keypoints_per_image, dtype=torch.float32, device=keypoints.device)
    keypoints_weight_per_image = torch.as_tensor(keypoints_weight_per_image, dtype=torch.uint8, device=keypoints.device)


    return keypoints_per_image, keypoints_weight_per_image

def keypoints_to_heat_map_rotate(keypoints, rois, heatmap_size):
    
    NUM_JOINTS = 8

    sf = 0.25
    if rois.numel() == 0:
        return rois.new().long(), rois.new().long()
    rois_ = rois.cuda().data.cpu().numpy()


    keypoints_ = keypoints.view((-1, 8))
    keypoints_ = keypoints_.cuda().data.cpu().numpy()

    x1 = rois_[:, 0].reshape(rois_.shape[0], -1)
    y1 = rois_[:, 1].reshape(rois_.shape[0], -1)
    x2 = rois_[:, 2].reshape(rois_.shape[0], -1)
    y2 = rois_[:, 3].reshape(rois_.shape[0], -1)

    c = np.hstack(((x1 + x2) / 2.0, (y1 + y2) / 2.0))

    s_y = y2 - y1
    s_y = s_y.reshape(rois_.shape[0], -1)
    s_y = np.hstack((s_y, s_y))
    s_y = s_y * np.clip(sf + 1, 1 - sf, 1 + sf)

    s_x = x2 - x1
    s_x = s_x.reshape(rois_.shape[0], -1)
    s_x = np.hstack((s_x, s_x))
    s_x = s_x * np.clip(sf + 1, 1 - sf, 1 + sf)

    r = 0

    keypoints_per_image = np.zeros((rois_.shape[0], NUM_JOINTS, heatmap_size, heatmap_size))
    keypoints_weight_per_image = np.zeros((rois_.shape[0], NUM_JOINTS))

    trans = get_affine_transform(c, s_y, s_x, r, [56, 56])
    for index, tran, keypoint in zip(range(0, rois_.shape[0]), trans, keypoints_):

        keypoint = keypoint.reshape(8)
        keypoint = choose_best_pointorder_fit_another(keypoint)
        keypoint = np.array(keypoint)
        kp_corner = keypoint
        kp_angle = np.zeros(4)
        kp_line = kp_corner.copy() 

        joints_corner = []
        for j in range(0, 8, 2):
            kp_corner[j], kp_corner[j + 1] = affine_transform([kp_corner[j], kp_corner[j+1]], tran)
            joints_corner.append([kp_corner[j], kp_corner[j + 1], 1.0])

        joints_line = []
        for j in range(0, 8, 2):
            if j == 6:
                kp_line[j] = 0.5 * (kp_corner[j] + kp_corner[0])
                kp_line[j+1] = 0.5 * (kp_corner[j+1] + kp_corner[1])
                kp_angle[int(j / 2)] = (kp_corner[1] - kp_corner[j+1]) / (kp_corner[0] - kp_corner[j] + 1e-12)
            else:
                kp_line[j] = 0.5 * (kp_corner[j] + kp_corner[j+2])
                kp_line[j+1] = 0.5 * (kp_corner[j+1] + kp_corner[j+3])
                kp_angle[int(j / 2)] = (kp_corner[j+3] - kp_corner[j+1]) / (kp_corner[j+2] - kp_corner[j] + 1e-12)
            joints_line.append([kp_line[j], kp_line[j + 1], 1.0])  
        
        joints_corner = np.array(joints_corner)
        joints_line = np.array(joints_line)

        target_corner, target_corner_weight = generate_target_corner_unquant(joints_corner, type=1)
        target_line, target_line_weight = generate_target_line_unquant(joints_line, type=1) 

        target_corner_ = []
        for j in range(len(target_corner)):
            if target_corner_weight[j] != 0:
                target_corner_.append(np.maximum(rotate_image(target_corner[j], joints_corner[j], -math.atan(kp_angle[0]) / math.pi * 180), 
                    rotate_image(target_corner[j], joints_corner[j], -math.atan(kp_angle[1]) / math.pi * 180)))
            else:
                target_corner_.append(target_corner[j])

        target_line_ = []
        for j in range(len(target_line)):
            if target_line_weight[j] != 0:
                target_line_.append(rotate_image(target_line[j], joints_line[j], -math.atan(kp_angle[j]) / math.pi * 180))
            else:
                target_line_.append(target_line[j])

        target_corner_ = np.array(target_corner_)
        target_line_ = np.array(target_line_)

        target = np.concatenate((target_corner_, target_line_), axis=0)
        target_weight = np.concatenate((target_corner_weight, target_line_weight), axis=0)


        keypoints_per_image[index] = target
        keypoints_weight_per_image[index] = target_weight

    keypoints_per_image = torch.as_tensor(keypoints_per_image, dtype=torch.float32, device=keypoints.device)
    keypoints_weight_per_image = torch.as_tensor(keypoints_weight_per_image, dtype=torch.float32, device=keypoints.device)


    return keypoints_per_image