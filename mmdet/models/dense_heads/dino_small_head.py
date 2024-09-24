from typing import Dict, Tuple

import torch
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures.bbox import bbox_xyxy_to_cxcywh

from .dino_head import DINOHead

@MODELS.register_module()
class DINOSmallHead(DINOHead):

    def filter_gt_bboxes_labels(self, gt_bboxes: Tensor, gt_labels: Tensor) -> Tuple:
        bboxes_wh = bbox_xyxy_to_cxcywh(gt_bboxes)[:, 2:]
        bboxes_area = torch.prod(bboxes_wh, dim=1)
        chosen_small_size_indice = torch.nonzero(bboxes_area < 1024).view(-1)       # 32^2
        if len(chosen_small_size_indice) == 0:
            chosen_small_size_indice = torch.nonzero(bboxes_area < 4096).view(-1)   # 64^2
        filter_bboxes = gt_bboxes[chosen_small_size_indice]
        filter_labels = gt_labels[chosen_small_size_indice]

        if len(chosen_small_size_indice) == 0:  # There is not any small and middle size object
            filter_bboxes = gt_bboxes
            filter_labels = gt_labels

        # threshold = 32
        # chosen_small_size_indice = []
        # while(len(chosen_small_size_indice) == 0):
        #     chosen_small_size_indice = torch.nonzero(bboxes_area < threshold*threshold).view(-1)  # Note `* 0.5`
        #     threshold += 16
        return filter_bboxes, filter_labels

    def _get_dn_targets_single(self, gt_instances: InstanceData,
                               img_meta: dict, dn_meta: Dict[str, int]) -> tuple:
        gt_bboxes = gt_instances.bboxes
        gt_labels = gt_instances.labels
        gt_bboxes, gt_labels = self.filter_gt_bboxes_labels(gt_bboxes, gt_labels)

        num_groups = dn_meta['num_denoising_groups']
        num_denoising_queries = dn_meta['num_denoising_queries']
        num_queries_each_group = int(num_denoising_queries / num_groups)
        device = gt_bboxes.device

        if len(gt_labels) > 0:
            t = torch.arange(len(gt_labels), dtype=torch.long, device=device)
            t = t.unsqueeze(0).repeat(num_groups, 1)
            pos_assigned_gt_inds = t.flatten()
            pos_inds = torch.arange(num_groups, dtype=torch.long, device=device)
            pos_inds = pos_inds.unsqueeze(1) * num_queries_each_group + t
            pos_inds = pos_inds.flatten()
        else:
            pos_inds = pos_assigned_gt_inds = gt_bboxes.new_tensor([], dtype=torch.long)

        neg_inds = pos_inds + num_queries_each_group // 2

        # label targets
        labels = gt_bboxes.new_full((num_denoising_queries, ), self.num_classes, dtype=torch.long)
        labels[pos_inds] = gt_labels[pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_denoising_queries)

        # bbox targets
        bbox_targets = torch.zeros(num_denoising_queries, 4, device=device)
        bbox_weights = torch.zeros(num_denoising_queries, 4, device=device)
        bbox_weights[pos_inds] = 1.0
        img_h, img_w = img_meta['img_shape']

        # DETR regress the relative position of boxes (cxcywh) in the image.
        # Thus the learning target should be normalized by the image size, also
        # the box format should be converted from defaultly x1y1x2y2 to cxcywh.
        factor = gt_bboxes.new_tensor([img_w, img_h, img_w, img_h]).unsqueeze(0)
        gt_bboxes_normalized = gt_bboxes / factor
        gt_bboxes_targets = bbox_xyxy_to_cxcywh(gt_bboxes_normalized)
        bbox_targets[pos_inds] = gt_bboxes_targets.repeat([num_groups, 1])

        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds, neg_inds)