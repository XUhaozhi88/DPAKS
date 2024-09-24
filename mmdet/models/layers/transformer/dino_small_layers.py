from typing import Tuple

import torch
from torch import Tensor

from mmdet.structures import SampleList
from mmdet.structures.bbox import bbox_xyxy_to_cxcywh

from .dino_layers import CdnQueryGenerator

class CdnSmallQueryGenerator(CdnQueryGenerator):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.count = {
            '0~32': 0, '32~48': 0, '48~all': 0
        }

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
        if len(chosen_small_size_indice) == 0:
            # chosen_small_size_indice = torch.nonzero(bboxes_area < 2304).view(-1)   # 48^2
            chosen_small_size_indice = torch.nonzero(bboxes_area < 4096).view(-1)  # 64^2
        #     if len(chosen_small_size_indice) == 0:  # There is not any small and middle size object
        #         # filter_bboxes = relative_gt_bboxes
        #         # filter_labels = gt_labels
        #         self.count['48~all'] += 1
        #     else:
        #         self.count['32~48'] += 1
        # else:
        #     self.count['0~32'] += 1
        # if len(chosen_small_size_indice) == 0:
        #     chosen_small_size_indice = torch.nonzero(bboxes_area < 4096).view(-1)   # 64^2
        filter_bboxes = relative_gt_bboxes[chosen_small_size_indice]
        filter_labels = gt_labels[chosen_small_size_indice]

        # if len(chosen_small_size_indice) == 0:  # There is not any small and middle size object
        #     filter_bboxes = relative_gt_bboxes
        #     filter_labels = gt_labels
            # self.count['48~all'] += 1
        # threshold = 32
        # chosen_small_size_indice = []
        # while(len(chosen_small_size_indice) == 0):
        #     chosen_small_size_indice = torch.nonzero(bboxes_area < threshold*threshold).view(-1)  # Note `* 0.5`
        #     threshold += 16
        return filter_bboxes, filter_labels