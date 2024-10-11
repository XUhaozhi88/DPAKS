# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmdet.registry import MODELS
from .anchor_head import AnchorHead
from .retina_head import RetinaHead

import copy
from abc import ABCMeta, abstractmethod
from inspect import signature
from typing import List, Optional, Tuple

import torch
from mmcv.ops import batched_nms
from mmengine.config import ConfigDict
from mmengine.model import BaseModule, constant_init
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.structures import SampleList
from mmdet.structures.bbox import (cat_boxes, get_box_tensor, get_box_wh,
                                   scale_boxes)
from mmdet.utils import InstanceList, OptMultiConfig
from ..test_time_augs import merge_aug_results
from ..utils import (filter_scores_and_topk, select_single_mlvl,
                     unpack_gt_instances)

import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.registry import MODELS, TASK_UTILS
from mmdet.structures.bbox import BaseBoxes, cat_boxes, get_box_tensor
from mmdet.utils import (ConfigType, InstanceList, OptConfigType,
                         OptInstanceList, OptMultiConfig)
from ..task_modules.prior_generators import (AnchorGenerator,
                                             anchor_inside_flags)
from ..task_modules.samplers import PseudoSampler
from ..utils import images_to_levels, multi_apply, unmap
from .base_dense_head import BaseDenseHead


@MODELS.register_module()
class RetinaSizeHead(RetinaHead):
    r"""An anchor-based head used in `RetinaNet
    <https://arxiv.org/pdf/1708.02002.pdf>`_.

    The head contains two subnetworks. The first classifies anchor boxes and
    the second regresses deltas for the anchors.

    Example:
        >>> import torch
        >>> self = RetinaHead(11, 7)
        >>> x = torch.rand(1, 7, 32, 32)
        >>> cls_score, bbox_pred = self.forward_single(x)
        >>> # Each anchor predicts a score for each class except background
        >>> cls_per_anchor = cls_score.shape[1] / self.num_anchors
        >>> box_per_anchor = bbox_pred.shape[1] / self.num_anchors
        >>> assert cls_per_anchor == (self.num_classes)
        >>> assert box_per_anchor == 4
    """

    def __init__(self,
                 loss_size,
                 num_size_classes=3,
                 **kwargs):
        super(RetinaSizeHead, self).__init__(**kwargs)
        self.num_size_classes = num_size_classes
        self.loss_size = MODELS.build(loss_size)        
        self.use_sigmoid_size = loss_size.get('use_sigmoid', False)
        if self.use_sigmoid_cls:
            self.size_out_channels = num_size_classes
        else:
            self.size_out_channels = num_size_classes + 1
        self._init_size_layers()

    def _init_size_layers(self):
        """Initialize layers of the head."""
        # self.size_convs = nn.ModuleList()
        # in_channels = self.in_channels
        # for i in range(self.stacked_convs):
        #     self.size_convs.append(
        #         ConvModule(
        #             in_channels,
        #             self.feat_channels,
        #             3,
        #             stride=1,
        #             padding=1,
        #             conv_cfg=self.conv_cfg,
        #             norm_cfg=self.norm_cfg))
        #     in_channels = self.feat_channels
        self.retina_size = nn.Conv2d(
            # in_channels,
            self.num_base_priors * self.cls_out_channels,
            self.num_base_priors * self.size_out_channels,
            3,
            padding=1)

    def forward_single(self, x):
        """Forward feature of a single scale level.

        Args:
            x (Tensor): Features of a single scale level.

        Returns:
            tuple:
                cls_score (Tensor): Cls scores for a single scale level
                    the channels number is num_anchors * num_classes.
                bbox_pred (Tensor): Box energies / deltas for a single scale
                    level, the channels number is num_anchors * 4.
        """
        cls_score, bbox_pred = super(RetinaSizeHead, self).forward_single(x)
        size_score = self.retina_size(cls_score)
        return cls_score, bbox_pred, size_score
    
    def loss(self, x: Tuple[Tensor], batch_data_samples: SampleList) -> dict:
        """Perform forward propagation and loss calculation of the detection
        head on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """
        outs = self(x)

        outputs = unpack_gt_instances(batch_data_samples)
        (batch_gt_instances, batch_gt_instances_ignore,
         batch_img_metas) = outputs

        loss_inputs = outs + (batch_gt_instances, batch_img_metas,
                              batch_gt_instances_ignore)
        
        losses = self.loss_by_feat(*loss_inputs) if len(outs) == 2 else self.loss_by_feat_new(*loss_inputs)

        return losses


    def _get_targets_single_new(self,
                                flat_anchors: Union[Tensor, BaseBoxes],
                                valid_flags: Tensor,
                                gt_instances: InstanceData,
                                img_meta: dict,
                                gt_instances_ignore: Optional[InstanceData] = None,
                                unmap_outputs: bool = True) -> tuple:
        inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                           img_meta['img_shape'][:2],
                                           self.train_cfg['allowed_border'])
        if not inside_flags.any():
            raise ValueError(
                'There is no valid anchor inside the image boundary. Please '
                'check the image size and anchor sizes, or set '
                '``allowed_border`` to -1 to skip the condition.')
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags]

        pred_instances = InstanceData(priors=anchors)
        assign_result = self.assigner.assign(pred_instances, gt_instances, gt_instances_ignore)

        size_gt_instances = copy.deepcopy(gt_instances)
        size_gt_instances.labels = gt_instances.size_labels
        size_gt_instances_ignore = copy.deepcopy(gt_instances_ignore)
        size_gt_instances_ignore.labels = gt_instances_ignore.size_labels
        size_assign_result = self.assigner.assign(pred_instances, size_gt_instances, size_gt_instances_ignore)
        
        # No sampling is required except for RPN and
        # Guided Anchoring algorithms
        sampling_result = self.sampler.sample(assign_result, pred_instances, gt_instances)
        size_sampling_result = self.sampler.sample(size_assign_result, pred_instances, size_gt_instances)

        num_valid_anchors = anchors.shape[0]
        target_dim = gt_instances.bboxes.size(-1) if self.reg_decoded_bbox \
            else self.bbox_coder.encode_size
        bbox_targets = anchors.new_zeros(num_valid_anchors, target_dim)
        bbox_weights = anchors.new_zeros(num_valid_anchors, target_dim)

        # TODO: Considering saving memory, is it necessary to be long?
        labels = anchors.new_full((num_valid_anchors, ),
                                  self.num_classes,
                                  dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)
        size_labels = anchors.new_full((num_valid_anchors, ),
                                        self.num_size_classes,
                                        dtype=torch.long)
        size_label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        # `bbox_coder.encode` accepts tensor or box type inputs and generates
        # tensor targets. If regressing decoded boxes, the code will convert
        # box type `pos_bbox_targets` to tensor.
        if len(pos_inds) > 0:
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(
                    sampling_result.pos_priors, sampling_result.pos_gt_bboxes)
            else:
                pos_bbox_targets = sampling_result.pos_gt_bboxes
                pos_bbox_targets = get_box_tensor(pos_bbox_targets)
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0

            labels[pos_inds] = sampling_result.pos_gt_labels
            size_labels[pos_inds] = size_sampling_result.pos_gt_labels
            if self.train_cfg['pos_weight'] <= 0:
                label_weights[pos_inds] = 1.0
                size_label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg['pos_weight']
                size_label_weights[pos_inds] = self.train_cfg['pos_weight']
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0
            size_label_weights[neg_inds] = 1.0

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            labels = unmap(labels, num_total_anchors, inside_flags, fill=self.num_classes)  # fill bg label
            label_weights = unmap(label_weights, num_total_anchors, inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)
            size_labels = unmap(size_labels, num_total_anchors, inside_flags, fill=self.num_size_classes)  # fill bg label
            size_label_weights = unmap(size_label_weights, num_total_anchors, inside_flags)

        return (labels, label_weights, bbox_targets, bbox_weights, size_labels, size_label_weights, pos_inds,
                neg_inds, sampling_result)

    def get_targets_new(self,
                    anchor_list: List[List[Tensor]],
                    valid_flag_list: List[List[Tensor]],
                    batch_gt_instances: InstanceList,
                    batch_img_metas: List[dict],
                    batch_gt_instances_ignore: OptInstanceList = None,
                    unmap_outputs: bool = True,
                    return_sampling_results: bool = False) -> tuple:
        num_imgs = len(batch_img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs

        if batch_gt_instances_ignore is None:
            batch_gt_instances_ignore = [None] * num_imgs

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors to a single tensor
        concat_anchor_list = []
        concat_valid_flag_list = []
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            concat_anchor_list.append(cat_boxes(anchor_list[i]))
            concat_valid_flag_list.append(torch.cat(valid_flag_list[i]))

        # compute targets for each image
        results = multi_apply(
            self._get_targets_single_new,
            concat_anchor_list,
            concat_valid_flag_list,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore,
            unmap_outputs=unmap_outputs)
        (all_labels, all_label_weights, all_bbox_targets, all_bbox_weights, all_size_labels, all_size_label_weights, 
         pos_inds_list, neg_inds_list, sampling_results_list) = results[:9]
        rest_results = list(results[9:])  # user-added return values
        # Get `avg_factor` of all images, which calculate in `SamplingResult`.
        # When using sampling method, avg_factor is usually the sum of
        # positive and negative priors. When using `PseudoSampler`,
        # `avg_factor` is usually equal to the number of positive priors.
        avg_factor = sum(
            [results.avg_factor for results in sampling_results_list])
        # update `_raw_positive_infos`, which will be used when calling
        # `get_positive_infos`.
        self._raw_positive_infos.update(sampling_results=sampling_results_list)
        # split targets to a list w.r.t. multiple levels
        labels_list = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights, num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets, num_level_anchors)
        bbox_weights_list = images_to_levels(all_bbox_weights, num_level_anchors)
        size_labels_list = images_to_levels(all_size_labels, num_level_anchors)
        size_label_weights_list = images_to_levels(all_size_label_weights, num_level_anchors)
        res = (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, 
               size_labels_list, size_label_weights_list, avg_factor)
        if return_sampling_results:
            res = res + (sampling_results_list, )
        for i, r in enumerate(rest_results):  # user-added return values
            rest_results[i] = images_to_levels(r, num_level_anchors)

        return res + tuple(rest_results)
    
    def loss_by_feat_single_new(self, cls_score: Tensor, bbox_pred: Tensor, size_score: Tensor,
                                anchors: Tensor, 
                                labels: Tensor, label_weights: Tensor, 
                                bbox_targets: Tensor, bbox_weights: Tensor, 
                                size_labels: Tensor, size_label_weights: Tensor, 
                                avg_factor: int) -> tuple:
        # size loss
        size_labels = size_labels.reshape(-1)
        size_label_weights = size_label_weights.reshape(-1)
        size_score = size_score.permute(0, 2, 3, 1).reshape(-1, self.size_out_channels)
        loss_size = self.loss_size(size_score, size_labels, size_label_weights, avg_factor=avg_factor)        
        
        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
        loss_cls = self.loss_cls(cls_score, labels, label_weights, avg_factor=avg_factor)
        # regression loss
        target_dim = bbox_targets.size(-1)
        bbox_targets = bbox_targets.reshape(-1, target_dim)
        bbox_weights = bbox_weights.reshape(-1, target_dim)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, self.bbox_coder.encode_size)
        if self.reg_decoded_bbox:
            # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
            # is applied directly on the decoded bounding boxes, it
            # decodes the already encoded coordinates to absolute format.
            anchors = anchors.reshape(-1, anchors.size(-1))
            bbox_pred = self.bbox_coder.decode(anchors, bbox_pred)
            bbox_pred = get_box_tensor(bbox_pred)
        loss_bbox = self.loss_bbox(
            bbox_pred, bbox_targets, bbox_weights, avg_factor=avg_factor)
        return loss_cls, loss_bbox, loss_size

    def loss_by_feat_new(
            self,
            cls_scores: List[Tensor],
            bbox_preds: List[Tensor],
            size_scores: List[Tensor],
            batch_gt_instances: InstanceList,
            batch_img_metas: List[dict],
            batch_gt_instances_ignore: OptInstanceList = None) -> dict:
        """Calculate the loss based on the features extracted by the detection
        head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                has shape (N, num_anchors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict: A dictionary of loss components.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels

        device = cls_scores[0].device

        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, batch_img_metas, device=device)
        cls_reg_targets = self.get_targets_new(
            anchor_list,
            valid_flag_list,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, size_labels_list, size_label_weights_list,
         avg_factor) = cls_reg_targets

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors and flags to a single tensor
        concat_anchor_list = []
        for i in range(len(anchor_list)):
            concat_anchor_list.append(cat_boxes(anchor_list[i]))
        all_anchor_list = images_to_levels(concat_anchor_list,
                                           num_level_anchors)

        losses_cls, losses_bbox, losses_size = multi_apply(
            self.loss_by_feat_single_new,
            cls_scores,
            bbox_preds,
            size_scores,
            all_anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            size_labels_list,
            size_label_weights_list,
            avg_factor=avg_factor)
        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox, loss_size=losses_size)
