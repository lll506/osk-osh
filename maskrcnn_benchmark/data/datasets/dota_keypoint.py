# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torchvision

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
from PIL import Image
import os
import numpy as np
import cv2
import math
from maskrcnn_benchmark.config import cfg


def _has_only_empty_bbox(anno):
    return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)


def has_valid_annotation(anno):
    # remove ignored or crowd box
    anno = [obj for obj in anno if obj["iscrowd"] == 0 and obj["ignore"] == 0 ]
    # if it's empty, there is no annotation
    if len(anno) == 0:
        return False
    # if all boxes have close to zero area, there is no annotation
    if _has_only_empty_bbox(anno):
        return False
    # keypoints task have a slight different critera for considering
    # if an annotation is valid
    # if "keypoints" not in anno[0]:
    #    return True
    # for keypoint detection tasks, only consider valid images those
    # containing at least min_keypoints_per_image
    # if _count_visible_keypoints(anno) >= min_keypoints_per_image:
    #    return True
    return True


def poly_to_rect(polygons):
    """
    polygons:
        type=np.array.float32
        shape=(n_poly,8)
    return:
        type=np.array.float32
        shape=(n_poly,8)
    """
    rects_min_area = []
    for poly in polygons:
        x1,y1,x2,y2,x3,y3,x4,y4 = poly[0], poly[1], poly[2], poly[3], poly[4], poly[5], poly[6], poly[7]
        (r_x1,r_y1),(r_x2,r_y2),(r_x3,r_y3),(r_x4,r_y4) = \
            cv2.boxPoints(cv2.minAreaRect(np.array([[int(x1),int(y1)],[int(x2),int(y2)],[int(x3),int(y3)],[int(x4),int(y4)]])))
        rects_min_area.append([r_x1,r_y1,r_x2,r_y2,r_x3,r_y3,r_x4,r_y4])
    rects_min_area = np.array(rects_min_area)
    w_rects = np.sqrt((rects_min_area[:,0]-rects_min_area[:,2])**2+(rects_min_area[:,1]-rects_min_area[:,3])**2)
    h_rects = np.sqrt((rects_min_area[:,2]-rects_min_area[:,4])**2+(rects_min_area[:,3]-rects_min_area[:,5])**2)
    mask_x1_eq_x2 = rects_min_area[:,0]==rects_min_area[:,2]
    angle_w = np.where(mask_x1_eq_x2,0.5,np.arctan((rects_min_area[:,1]-rects_min_area[:,3])/
                                                  (np.where(mask_x1_eq_x2,0.0001,rects_min_area[:,2]-rects_min_area[:,0])))/math.pi)
    mask_x2_eq_x3 = rects_min_area[:,2]==rects_min_area[:,4]
    angle_h = np.where(mask_x2_eq_x3,0.5,np.arctan((rects_min_area[:,3]-rects_min_area[:,5])/
                                                  (np.where(mask_x2_eq_x3,0.0001,rects_min_area[:,4]-rects_min_area[:,2])))/math.pi)
    angle = np.where(w_rects>h_rects,angle_w,angle_h)
    angle = np.where(angle<0,angle+1,angle)
    mx_1,mx_2,mx_3,mx_4,my_1,my_2,my_3,my_4 = \
        (polygons[:,0]+polygons[:,2])/2,(polygons[:,2]+polygons[:,4])/2,(polygons[:,4]+polygons[:,6])/2,(polygons[:,6]+polygons[:,0])/2,\
        (polygons[:,1]+polygons[:,3])/2,(polygons[:,3]+polygons[:,5])/2,(polygons[:,5]+polygons[:,7])/2,(polygons[:,7]+polygons[:,1])/2
    wh = np.concatenate([np.sqrt((mx_1-mx_3)**2+(my_1-my_3)**2)[:,None],np.sqrt((mx_2-mx_4)**2+(my_2-my_4)**2)[:,None]],axis=-1)
    mask_mx1_eq_mx3 = mx_1==mx_3
    angle_mw = np.where(mask_mx1_eq_mx3,0.5,np.arctan((my_1-my_3)/
                                                  (np.where(mask_mx1_eq_mx3,0.0001,mx_3-mx_1)))/math.pi)
    mask_mx2_eq_mx4 = mx_2==mx_4
    angle_mh = np.where(mask_mx2_eq_mx4,0.5,np.arctan((my_2-my_4)/
                                                  (np.where(mask_mx2_eq_mx4,0.0001,mx_4-mx_2)))/math.pi)
    angle_m = np.where(wh[:,0]>wh[:,1],angle_mw,angle_mh)
    angle_m = np.where(angle_m<0,angle_m+1,angle_m)
    angle_err = np.sin(np.abs(angle-angle_m)*math.pi)
    angle_out = np.where(angle_err>math.sin(math.pi/4),angle-0.5,angle)
    angle_out = np.where(angle_out<0,angle_out+1,angle_out)
    angle_out = angle_out*math.pi
    cx_out = polygons[:,0::2].sum(axis=-1)/4
    cy_out = polygons[:,1::2].sum(axis=-1)/4
    w_out = np.max(wh,axis=-1)
    h_out = np.min(wh,axis=-1)
    half_dl = np.sqrt(w_out*w_out+h_out*h_out)/2
    a1 = math.pi-(math.pi-angle_out+np.arccos(w_out/2/half_dl))
    a2 = 2*np.arctan(h_out/w_out)+a1
    x1 = half_dl*np.cos(a2)
    y1 = -half_dl*np.sin(a2)
    x2 = half_dl*np.cos(a1)
    y2 = -half_dl*np.sin(a1)
    x3,y3,x4,y4 = -x1,-y1,-x2,-y2

    return np.stack([x1+cx_out,y1+cy_out,x2+cx_out,y2+cy_out,x3+cx_out,y3+cy_out,x4+cx_out,y4+cy_out]).transpose()


class DOTAkeypointDataset(torchvision.datasets.coco.CocoDetection):
    def __init__(
        self, ann_file, root, remove_images_without_annotations, transforms=None
    ):
        super(DOTAkeypointDataset, self).__init__(root, ann_file)
        # sort indices for reproducible results
        self.ids = sorted(self.ids)

        # filter images without detection annotations
        if remove_images_without_annotations:
            ids = []
            for img_id in self.ids:
                ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
                anno = self.coco.loadAnns(ann_ids)
                if has_valid_annotation(anno):
                    ids.append(img_id)
            self.ids = ids

        self.categories = {cat['id']: cat['name'] for cat in self.coco.cats.values()}

        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.coco.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        self._transforms = transforms

    def __getitem__(self, idx):

        coco = self.coco
        img_id = self.ids[idx]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anno = coco.loadAnns(ann_ids)

        loaded_img = coco.loadImgs(img_id)[0]
        path = loaded_img['file_name']

        img = Image.open(os.path.join(self.root, path)).convert('RGB')

        # if "angle" in loaded_img and loaded_img["angle"] is not 0:
        if 'angle' in loaded_img.keys() and loaded_img["angle"] is not 0:
            if loaded_img["angle"] == 90:
                img = img.rotate( 270, expand=True )
            elif loaded_img["angle"] == 180:
                img = img.rotate( 180, expand=True )
            elif loaded_img["angle"] == 270:
                img = img.rotate( 90, expand=True )
            else:
                raise ValueError()

        # filter crowd annotations
        # TODO might be better to add an extra field
        anno = [obj for obj in anno if obj["iscrowd"] == 0 and obj["ignore"] == 0]

        """
        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, img.size, mode="xywh").convert("xyxy")
        """

        def to_rrect( x ):
            x = cv2.minAreaRect( x )
            x = cv2.boxPoints( x )
            return x

        # masks = [obj["segmentation"] for obj in anno]

        keypoints = np.array( [obj["keypoint"] for obj in anno] )
        # keypoints = np.array(keypoints, dtype=np.float32).reshape(-1, 8)
        # keypoints = list( poly_to_rect(keypoints.reshape( (-1, 8) ) ) )
        keypoints = np.array( keypoints, dtype=np.float32 ).reshape( (-1, 8) )
        
        xmins = np.min( keypoints[:,  ::2], axis=1 )
        minx_idx = xmins < 1
        xmins[minx_idx] = 1
        ymins = np.min( keypoints[:, 1::2], axis=1 )
        miny_idx = ymins < 1
        ymins[miny_idx] = 1
        xmaxs = np.max( keypoints[:,  ::2], axis=1 )
        maxx_idx = xmaxs > 1024
        xmaxs[maxx_idx] = 1024
        ymaxs = np.max( keypoints[:, 1::2], axis=1 )
        maxy_idx = ymaxs > 1024
        ymaxs[maxy_idx] = 1024
        
        xyxy = np.vstack( [xmins, ymins, xmaxs, ymaxs] ).transpose()
        boxes = torch.from_numpy( xyxy ).reshape(-1, 4)  # guard against no boxes
        target = BoxList( boxes, img.size, mode="xyxy" )

        keypoints = SegmentationMask( keypoints.reshape( (-1, 1, 8)).tolist(), img.size, mode='poly' )
        target.add_field( "keypoints", keypoints )

        classes = [obj["category_id"] for obj in anno]
        classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
        classes = torch.tensor(classes)
        target.add_field("labels", classes)

        # NOTE Qimeng: close it for getting correct alpha
        #target = target.clip_to_image(remove_empty=True)

        if self._transforms is not None:
            img, target = self._transforms(img, target)

        return img, target, idx

    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data
