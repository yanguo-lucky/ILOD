'''
Descripttion:
version:
Author: Jinlong Li CSU PhD
Date: 2022-01-04 23:51:49
LastEditors: Jinlong Li CSU PhD
LastEditTime: 2023-03-04 02:16:38
'''
"""
This file contains specific functions for computing losses on the da_heads
file
"""
import pdb
import torch
from torch import nn
from torch.nn import functional as F





from torch.nn import Module, Sequential, Conv2d, ReLU, AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding


class DALossComputation(object):
    """
    This class computes the DA loss.
    """

    def __init__(self, cfg):
        self.cfg = cfg.clone()
        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO


        self.avgpool = nn.AvgPool2d(kernel_size=resolution, stride=resolution)

    def prepare_masks(self, targets):
        masks = []
        for targets_per_image in targets:
            is_source = targets_per_image.get_field('is_source')
            mask_per_image = is_source.new_ones(1, dtype=torch.bool) if is_source.any() else is_source.new_zeros(1,
                                                                                                                 dtype=torch.bool)
            masks.append(mask_per_image)
        return masks

    def __call__(self, da_img, da_ins, da_img_consist, da_ins_consist, da_ins_labels, targets):
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

        da_img_flattened = []
        da_img_labels_flattened = []

        # for each feature level, permute the outputs to make them be in the
        # same format as the labels. Note that the labels are computed for
        # all feature levels concatenated, so we keep the same representation
        # for the image-level domain alignment
        for da_img_per_level in da_img:
            N, A, H, W = da_img_per_level.shape
            da_img_per_level = da_img_per_level.permute(0, 2, 3, 1)
            da_img_label_per_level = torch.zeros_like(da_img_per_level, dtype=torch.float32)
            da_img_label_per_level[masks, :] = 1

            da_img_per_level = da_img_per_level.reshape(N, -1)
            da_img_label_per_level = da_img_label_per_level.reshape(N, -1)

            da_img_flattened.append(da_img_per_level)
            da_img_labels_flattened.append(da_img_label_per_level)

        da_img_flattened = torch.cat(da_img_flattened, dim=0)
        da_img_labels_flattened = torch.cat(da_img_labels_flattened, dim=0)

        da_img_loss = F.binary_cross_entropy_with_logits(
            da_img_flattened, da_img_labels_flattened
        )
        da_ins_loss = F.binary_cross_entropy_with_logits(
            torch.squeeze(da_ins), da_ins_labels.type(torch.cuda.FloatTensor)
        )


        return da_img_loss, da_ins_loss


class DALossComputation_Component(object):
    """
    This class computes the DA loss.
    """

    def __init__(self, cfg):
        self.cfg = cfg.clone()
        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION


        self.avgpool = nn.AvgPool2d(kernel_size=resolution, stride=resolution)

        # for triplet loss
        self.margin_ins = 0.0
        self.margin_img = 0.0

    def prepare_masks(self, targets):
        masks = []
        for targets_per_image in targets:
            # pdb.set_trace()
            # print(type(targets_per_image))
            # print(targets_per_image)
            is_source = targets_per_image['is_source']
            is_source = torch.tensor(is_source, dtype=torch.bool)
            mask_per_image = is_source.new_ones(1, dtype=torch.bool) if is_source.any() else is_source.new_zeros(1,
                                                                                                                 dtype=torch.bool)
            masks.append(mask_per_image)
            # pdb.set_trace()
        return masks

    def da_img_loss(self, da_img, targets):
        da_img_flattened = []
        da_img_labels_flattened = []

        masks = self.prepare_masks(targets)
        masks = torch.cat(masks, dim=0)
        # for each feature level, permute the outputs to make them be in the
        # same format as the labels. Note that the labels are computed for
        # all feature levels concatenated, so we keep the same representation
        # for the image-level domain alignment
        i=0
        for da_img_per_level in da_img:
            N, A, H, W = da_img_per_level.shape
            da_img_per_level = da_img_per_level.permute(0, 2, 3, 1)
            da_img_label_per_level = torch.zeros_like(da_img_per_level, dtype=torch.float32)
            da_img_label_per_level[masks[i], :] = 1


            da_img_per_level = da_img_per_level.reshape(N, -1)
            da_img_label_per_level = da_img_label_per_level.reshape(N, -1)

            da_img_flattened.append(da_img_per_level)
            da_img_labels_flattened.append(da_img_label_per_level)
            i=i+1

        # da_img_flattened = torch.cat(da_img_flattened, dim=0)
        # da_img_labels_flattened = torch.cat(da_img_labels_flattened, dim=0)
        #
        # da_img_loss = F.binary_cross_entropy_with_logits(
        #     da_img_flattened, da_img_labels_flattened
        # )
        da_img_loss = 0.0

        for da_img_per_level, da_img_label_per_level in zip(da_img_flattened, da_img_labels_flattened):
            # 单独计算每层的损�?
            level_loss = F.binary_cross_entropy_with_logits(da_img_per_level, da_img_label_per_level)
            da_img_loss += level_loss

        return da_img_loss


def make_da_heads_loss_evaluator(cfg):
    loss_evaluator = DALossComputation_Component(cfg)
    return loss_evaluator


def make_da_heads_loss_evaluator_original(cfg):
    loss_evaluator = DALossComputation(cfg)
    return loss_evaluator