from typing import Tuple

import torch
from torch import Tensor

from mmdet.structures import SampleList
from mmdet.structures.bbox import bbox_xyxy_to_cxcywh

from .utils import inverse_sigmoid
from .dino_layers import CdnQueryGenerator

class CdnSmallQueryGenerator(CdnQueryGenerator):

    def __call__(self, batch_data_samples: SampleList) -> tuple:
        """Generate contrastive denoising (cdn) queries with ground truth."""
        # normalize bbox and collate ground truth (gt)
        gt_labels_list = []
        gt_bboxes_list = []
        for sample in batch_data_samples:
            img_h, img_w = sample.img_shape
            bboxes = sample.gt_instances.bboxes
            factor = bboxes.new_tensor([img_w, img_h, img_w, img_h]).unsqueeze(0)
            bboxes_normalized = bboxes / factor
            labels = sample.gt_instances.labels
            gt_bboxes, labels = \
                self.filter_gt_bboxes_labels(absolute_gt_bboxes=bboxes,
                                             relative_gt_bboxes=bboxes_normalized, gt_labels=labels)
            # gt_bboxes_list.append(bboxes_normalized)
            gt_bboxes_list.append(gt_bboxes)
            gt_labels_list.append(labels)
        gt_labels = torch.cat(gt_labels_list)  # (num_target_total, 4)
        gt_bboxes = torch.cat(gt_bboxes_list)


        num_target_list = [len(bboxes) for bboxes in gt_bboxes_list]
        max_num_target = max(num_target_list)
        num_groups = self.get_num_groups(max_num_target)

        dn_label_query = self.generate_dn_label_query(gt_labels, num_groups)
        dn_bbox_query = self.generate_dn_bbox_query(gt_bboxes, num_groups)

        # The `batch_idx` saves the batch index of the corresponding sample
        # for each target, has shape (num_target_total).
        batch_idx = torch.cat([
            torch.full_like(t.long(), i) for i, t in enumerate(gt_labels_list)
        ])
        dn_label_query, dn_bbox_query = self.collate_dn_queries(
            dn_label_query, dn_bbox_query, batch_idx, len(batch_data_samples),
            num_groups)

        attn_mask = self.generate_dn_mask(
            max_num_target, num_groups, device=dn_label_query.device)

        dn_meta = dict(
            num_denoising_queries=int(max_num_target * 2 * num_groups),
            num_denoising_groups=num_groups)

        return dn_label_query, dn_bbox_query, attn_mask, dn_meta

    def filter_gt_bboxes_labels(self, absolute_gt_bboxes: Tensor, relative_gt_bboxes: Tensor, gt_labels: Tensor) -> Tuple:
        bboxes_wh = bbox_xyxy_to_cxcywh(absolute_gt_bboxes)[:, 2:]
        bboxes_area = torch.prod(bboxes_wh, dim=1)
        
        chosen_small_size_indice = torch.nonzero(bboxes_area < 1024).view(-1)       # 32^2
        # if len(chosen_small_size_indice) == 0:
        #     chosen_small_size_indice = torch.nonzero(bboxes_area < 2304).view(-1)   # 48^2      
        if len(chosen_small_size_indice) == 0:
            chosen_small_size_indice = torch.nonzero(bboxes_area < 4096).view(-1)   # 64^2  
        filter_bboxes = relative_gt_bboxes[chosen_small_size_indice]
        filter_labels = gt_labels[chosen_small_size_indice]

        if len(chosen_small_size_indice) == 0:  # There is not any small and middle size object
            filter_bboxes = relative_gt_bboxes
            filter_labels = gt_labels
            
        return filter_bboxes, filter_labels
    
    def generate_dn_bbox_query_new(self, gt_bboxes: Tensor, num_groups: int) -> Tensor:
        assert self.box_noise_scale > 0
        device = gt_bboxes.device

        # expand gt_bboxes as groups
        gt_bboxes_expand = gt_bboxes.repeat(2 * num_groups, 1)  # xyxy

        # obtain index of negative queries in gt_bboxes_expand
        positive_idx = torch.arange(
            len(gt_bboxes), dtype=torch.long, device=device)
        positive_idx = positive_idx.unsqueeze(0).repeat(num_groups, 1)
        positive_idx += 2 * len(gt_bboxes) * torch.arange(
            num_groups, dtype=torch.long, device=device)[:, None]
        positive_idx = positive_idx.flatten()
        negative_idx = positive_idx + len(gt_bboxes)

        # determine the sign of each element in the random part of the added
        # noise to be positive or negative randomly.
        rand_sign = torch.randint_like(
            gt_bboxes_expand, low=0, high=2,
            dtype=torch.float32) * 2.0 - 1.0  # [low, high), 1 or -1, randomly

        # calculate the random part of the added noise
        rand_part = torch.rand_like(gt_bboxes_expand)  # [0, 1)
        rand_part[negative_idx] += 1.0  # pos: [0, 1); neg: [1, 2)
        rand_part *= rand_sign  # pos: (-1, 1); neg: (-2, -1] U [1, 2)

        # unify the coordinate deviation of a point
        rand_part[positive_idx, 1] = rand_part[positive_idx, 0]
        rand_part[positive_idx, 3] = rand_part[positive_idx, 2]

        # add noise to the bboxes
        bboxes_whwh = bbox_xyxy_to_cxcywh(gt_bboxes_expand)[:, 2:].repeat(1, 2)
        noisy_bboxes_expand = gt_bboxes_expand + torch.mul(
            rand_part, bboxes_whwh) * self.box_noise_scale / 2  # xyxy
        noisy_bboxes_expand = noisy_bboxes_expand.clamp(min=0.0, max=1.0)
        noisy_bboxes_expand = bbox_xyxy_to_cxcywh(noisy_bboxes_expand)

        dn_bbox_query = inverse_sigmoid(noisy_bboxes_expand, eps=1e-3)
        return dn_bbox_query