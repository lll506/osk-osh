import torch
from torch.nn import functional as F

from maskrcnn_benchmark.modeling.matcher import Matcher

from maskrcnn_benchmark.modeling.balanced_positive_negative_sampler import (
    BalancedPositiveNegativeSampler,
)
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.modeling.utils import cat
from maskrcnn_benchmark.layers import smooth_l1_loss
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist

from maskrcnn_benchmark.structures.keypoint import keypoints_to_heat_map, keypoints_to_heat_map_2x, keypoints_to_heat_map_rotate, keypoints_to_heat_map_rotate_1


def Keypoints(keypoints, device=None):
    polygons = list( map( lambda x: x.polygons[0], keypoints ) )
    rbox = torch.stack( polygons, axis=0 )
    keypoints = torch.as_tensor(rbox, dtype=torch.float32, device=device)
    num_keypoints = keypoints.shape[0]
    if num_keypoints:
        keypoints = keypoints.view(num_keypoints, -1, 2)
    return keypoints


def project_keypoints_to_heatmap(keypoints, proposals, discretization_size):
    proposals = proposals.convert("xyxy")
    return keypoints_to_heat_map_rotate(
        keypoints, proposals.bbox, discretization_size
        )


def centernet_loss(preds, targets):
    ''' Modified focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
        Arguments:
        preds (B x c x h x w)
        gt_regr (B x c x h x w)
    '''
    pos_inds = targets.eq(1).float()
    neg_inds = targets.lt(1).float()

    neg_weights = torch.pow(1 - targets, 4)

    loss = 0
    for pred in preds: 
        pred = torch.clamp(torch.sigmoid(pred), min=1e-4, max=1 - 1e-4)
        pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
        neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds
        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss 
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
    return loss / len(preds)


class KeypointRCNNLossComputation(object):
    def __init__(self, proposal_matcher, fg_bg_sampler, discretization_size):
        """
        Arguments:
            proposal_matcher (Matcher)
            fg_bg_sampler (BalancedPositiveNegativeSampler)
            discretization_size (int)
        """
        self.proposal_matcher = proposal_matcher
        self.fg_bg_sampler = fg_bg_sampler
        self.discretization_size = discretization_size

    def match_targets_to_proposals(self, proposal, target):
        match_quality_matrix = boxlist_iou(target, proposal)
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        del match_quality_matrix
        # Keypoint RCNN needs "labels" and "keypoints "fields for creating the targets
        target = target.copy_with_fields(["labels", "keypoints"])
        # get the targets corresponding GT for each proposal
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field("matched_idxs", matched_idxs)
        return matched_targets

    def prepare_targets(self, proposals, targets):
        labels = []
        keypoints = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            matched_targets = self.match_targets_to_proposals(
                proposals_per_image, targets_per_image
            )
            matched_idxs = matched_targets.get_field("matched_idxs")

            labels_per_image = matched_targets.get_field("labels")
            labels_per_image = labels_per_image.to(dtype=torch.int64)

            # this can probably be removed, but is left here for clarity
            # and completeness
            # TODO check if this is the right one, as BELOW_THRESHOLD
            neg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[neg_inds] = 0

            ignore_inds = matched_idxs == Matcher.BETWEEN_THRESHOLDS
            labels_per_image[ignore_inds] = -1  # -1 is ignored by sampler

            keypoints_per_image = matched_targets.get_field("keypoints")
            device = matched_targets.bbox.device
            keypoints_per_image = Keypoints(keypoints_per_image.instances.polygons, device=device)
            # within_box = _within_box(
            #     keypoints_per_image.keypoints, matched_targets.bbox
            # )

            # vis_kp = keypoints_per_image.keypoints[..., 0] > -1
            # is_visible = (within_box & vis_kp).sum(1) > 0
            # # is_visible = within_box > 0

            # labels_per_image[~is_visible] = -1

            labels.append(labels_per_image)
            keypoints.append(keypoints_per_image)

        return labels, keypoints

    def subsample(self, proposals, targets):
        """
        This method performs the positive/negative sampling, and return
        the sampled proposals.
        Note: this function keeps a state.

        Arguments:
            proposals (list[BoxList])
            targets (list[BoxList])
        """

        labels, keypoints = self.prepare_targets(proposals, targets)
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)

        proposals = list(proposals)
        # add corresponding label and regression_targets information to the bounding boxes
        for labels_per_image, keypoints_per_image, proposals_per_image in zip(
            labels, keypoints, proposals
        ):
            proposals_per_image.add_field("labels", labels_per_image)
            proposals_per_image.add_field("keypoints", keypoints_per_image)

        # distributed sampled proposals, that were obtained on all feature maps
        # concatenated via the fg_bg_sampler, into individual feature map levels
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(
            zip(sampled_pos_inds, sampled_neg_inds)
        ):
            img_sampled_inds = torch.nonzero(pos_inds_img).squeeze(1)
        
            proposals_per_image = proposals[img_idx][img_sampled_inds]
            proposals[img_idx] = proposals_per_image

        self._proposals = proposals
        return proposals

    def __call__(self, proposals, keypoint_logits):
        heatmaps = []
        weights = []
        # offsets = []
        # offsets_weights = []
        for proposals_per_image in proposals:
            kp = proposals_per_image.get_field("keypoints")

            heatmaps_per_image = project_keypoints_to_heatmap(
                kp, proposals_per_image, self.discretization_size
            )

            heatmaps.append(heatmaps_per_image)

        keypoint_targets = cat(heatmaps, dim=0)
        # keypoint_weights = cat(weights, dim=0)

        # valid = torch.nonzero(valid).squeeze(1)

        # torch.mean (in binary_cross_entropy_with_logits) does'nt
        # accept empty tensors, so handle it sepaartely
        if keypoint_targets.numel() == 0:
            return keypoint_logits.sum() * 0

        N, K, H, W = keypoint_logits.shape

        keypoint_logits = keypoint_logits.view(N, K, H * W)
        keypoint_targets = keypoint_targets.view(N, K, H * W)
        keypoint_loss = F.binary_cross_entropy_with_logits(keypoint_logits, keypoint_targets) * 100

        return keypoint_loss


def make_roi_keypoint_loss_evaluator(cfg):
    matcher = Matcher(
        cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD,
        cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD,
        allow_low_quality_matches=False,
    )
    fg_bg_sampler = BalancedPositiveNegativeSampler(
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE, cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION
    )
    resolution = cfg.MODEL.ROI_KEYPOINT_HEAD.RESOLUTION
    loss_evaluator = KeypointRCNNLossComputation(matcher, fg_bg_sampler, resolution)
    return loss_evaluator


