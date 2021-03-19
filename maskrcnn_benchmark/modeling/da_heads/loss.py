"""
This file contains specific functions for computing losses on the da_heads
file
"""

import torch
from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark.layers import consistency_loss
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
from maskrcnn_benchmark.modeling.poolers import Pooler
from ..utils import cat

device = torch.device("cuda")
class DALossComputation(object):
    """
    This class computes the DA loss.
    """

    def __init__(self, cfg):
        self.cfg = cfg.clone()
        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        self.center_loss = cfg.MODEL.CENTER_ON
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )

        self.pooler = pooler
        self.avgpool = nn.AvgPool2d(kernel_size=resolution, stride=resolution)
        
    def prepare_masks(self, targets):
        masks = []
        for targets_per_image in targets:
            is_source = targets_per_image.get_field('is_source')
            mask_per_image = is_source.new_ones(1, dtype=torch.bool) if is_source.any() else is_source.new_zeros(1, dtype=torch.bool)
            masks.append(mask_per_image)
        return masks

    def __call__(self, da_img, da_ins, da_img_consist, da_ins_consist, da_ins_labels, targets,centers, centers_inst, da_ins_center):
        """
        Arguments:
            da_img (list[Tensor])
            da_img_consist (list[Tensor])
            da_ins (Tensor)
            da_ins_consist (Tensor)
            da_ins_labels (Tensor)
            targets (list[BoxList])

        Returns:
            da_img_loss (Tensor)
            da_ins_loss (Tensor)
            da_consist_loss (Tensor)
        """

        masks = self.prepare_masks(targets)
        masks = torch.cat(masks, dim=0)
        # 2 , (bool)
#         print(masks.shape)
#         print(masks)
        da_img_flattened = []
        da_img_labels_flattened = []
        # for each feature level, permute the outputs to make them be in the
        # same format as the labels. Note that the labels are computed for
        # all feature levels concatenated, so we keep the same representation
        # for the image-level domain alignment
        for da_img_per_level in da_img:
            N, A, H, W = da_img_per_level.shape
            # 2,1,38,76
            #print(N, A, H, W)
            da_img_per_level = da_img_per_level.permute(0, 2, 3, 1)
            da_img_label_per_level = torch.zeros_like(da_img_per_level, dtype=torch.float32)
            da_img_label_per_level[masks, :] = 1

            da_img_per_level = da_img_per_level.reshape(N, -1)
            da_img_label_per_level = da_img_label_per_level.reshape(N, -1)
            
            da_img_flattened.append(da_img_per_level)
            da_img_labels_flattened.append(da_img_label_per_level)
            
        da_img_flattened = torch.cat(da_img_flattened, dim=0)
        da_img_labels_flattened = torch.cat(da_img_labels_flattened, dim=0)
#         print(da_img_labels_flattened.type(torch.cuda.BoolTensor))
        da_img_loss = F.binary_cross_entropy_with_logits(
            da_img_flattened, da_img_labels_flattened
        )
        
#         print(da_ins.shape,da_ins_labels.shape)
#         print(da_ins)
#         print(da_ins_labels)
        da_ins_loss = F.binary_cross_entropy_with_logits(
            torch.squeeze(da_ins), da_ins_labels.type(torch.cuda.FloatTensor)
        )

        da_consist_loss = consistency_loss(da_img_consist, da_ins_consist, da_ins_labels, size_average=True)
        if self.center_loss:
#             print("da_img_flattened.shape",da_img_flattened.shape) 2,2888
#             print("da_img_labels_flattened.shape",da_img_labels_flattened.shape) 2,2888
            te = da_img_labels_flattened.reshape(da_img_labels_flattened.shape[0]*da_img_labels_flattened.shape[1],)
#             print("te.shape",te.shape)
            targets_new = torch.zeros((2,1),dtype = torch.long) 
            targets_new[0] = da_img_labels_flattened[0][0]
            targets_new[1] = da_img_labels_flattened[1][0]
            cent_loss = compute_center_loss(da_img_flattened.reshape(-1,2888),centers,targets_new.type(torch.cuda.BoolTensor))
            
#             print("da_ins_center",da_ins_center.shape) #(512,1024)
#             print("da_ins_labels.shape",da_ins_labels.shape) #(512)
            cent_loss_inst = compute_center_loss(da_ins_center,centers_inst,da_ins_labels)
        
            center_deltas = get_center_delta(da_img_flattened.reshape(-1,2888).data, centers, targets_new.type(torch.cuda.BoolTensor), 0.5)
            center_inst_deltas = get_center_delta(da_ins_center.data, centers_inst, da_ins_labels, 0.5)
#             self.model.centers = centers - center_deltas
            return da_img_loss, da_ins_loss, da_consist_loss, cent_loss, center_deltas, cent_loss_inst, center_inst_deltas

        return da_img_loss, da_ins_loss, da_consist_loss, None, None, None, None

def make_da_heads_loss_evaluator(cfg):
    loss_evaluator = DALossComputation(cfg)
    return loss_evaluator

def compute_center_loss(features, centers, targets):
    indices = torch.zeros((targets.shape[0],1),dtype = torch.long)
#     print("indices shape",indices.shape)
    indices[targets] = 1
    indices = indices.reshape(indices.shape[0],)
#     print(indices.shape,centers.shape)
    features = features.view(features.size(0), -1)
    target_centers = centers[indices]
    criterion = torch.nn.MSELoss()
    center_loss = criterion(features, target_centers)
    return center_loss


def get_center_delta(features, centers, t, alpha):
    # implementation equation (4) in the center-loss paper
    features = features.view(features.size(0), -1)
    targets = torch.zeros((t.shape[0],1),dtype = torch.long)
    targets[t] = 1
    targets = targets.reshape(targets.shape[0],)
    targets, indices = torch.sort(targets)
    target_centers = centers[targets]
    features = features[indices]

    delta_centers = target_centers - features
    uni_targets, indices = torch.unique(
            targets.cpu(), sorted=True, return_inverse=True)
    
    uni_targets = uni_targets.to(device)
    indices = indices.to(device)

    delta_centers = torch.zeros(
        uni_targets.size(0), delta_centers.size(1)
    ).to(device).index_add_(0, indices, delta_centers)

    targets_repeat_num = uni_targets.size()[0]
    uni_targets_repeat_num = targets.size()[0]
    targets_repeat = targets.repeat(
            targets_repeat_num).view(targets_repeat_num, -1)
    uni_targets_repeat = uni_targets.unsqueeze(1).repeat(
            1, uni_targets_repeat_num)
    targets_repeat = targets_repeat.to(device)
    uni_targets_repeat = uni_targets_repeat.to(device)
    same_class_feature_count = torch.sum(
            targets_repeat == uni_targets_repeat, dim=1).float().unsqueeze(1)

    delta_centers = delta_centers / (same_class_feature_count + 1.0) * alpha
    result = torch.zeros_like(centers)
    result[uni_targets, :] = delta_centers
    return result
