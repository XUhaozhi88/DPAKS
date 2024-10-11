from typing import Dict, Tuple

import torch
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures.bbox import bbox_xyxy_to_cxcywh

from .dino_head import DINOHead

from mmengine.structures import InstanceData
from mmengine.model import constant_init, bias_init_with_prob
from mmdet.structures.bbox import (bbox_cxcywh_to_xyxy, bbox_overlaps,
                                   bbox_xyxy_to_cxcywh)
from mmdet.utils import (ConfigType, reduce_mean)
from ..losses import QualityFocalLoss
from ..utils import multi_apply

import copy
from typing import Dict, List, Tuple
from scipy.stats import norm

import torch
import torch.nn as nn
from mmcv.cnn import Linear
from torch import Tensor
from torch.nn import functional as F

from mmdet.registry import MODELS
from mmdet.utils import InstanceList, OptInstanceList
from ..layers import inverse_sigmoid

from .dino_head import DINOHead

@MODELS.register_module()
class DPAKSHead(DINOHead):    
    def __init__(
            self,
            *args,
            num_aux_classes=3,
            loss_aux: ConfigType = dict(),
            **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.num_aux_classes = num_aux_classes
        self.loss_aux = MODELS.build(loss_aux)

        if self.loss_aux.use_sigmoid:
            self.aux_out_channels = num_aux_classes
        else:
            self.aux_out_channels = num_aux_classes + 1

        self._init_aux_layers()

    def _init_aux_layers(self) -> None:
        # cls old -> size new
        aux_branch = nn.Sequential(
            nn.ReLU(),
            Linear(self.cls_out_channels, self.aux_out_channels))
        
        if self.share_pred_layer:
            self.aux_branches = nn.ModuleList(
                [aux_branch for _ in range(self.num_pred_layer)])
        else:
            self.aux_branches = nn.ModuleList([
                copy.deepcopy(aux_branch) for _ in range(self.num_pred_layer)])

    def init_weights(self) -> None:
        super(DPAKSHead, self).init_weights()
        # old -> size new
        if self.loss_aux.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.aux_branches:
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, bias_init)    

    def forward(self, hidden_states: Tensor, references: List[Tensor]) -> Tuple[Tensor, Tensor]:
        all_layers_outputs_classes = []
        all_layers_outputs_coords = []
        if self.training: all_layers_outputs_aux = []

        for layer_id in range(hidden_states.shape[0]):
            reference = inverse_sigmoid(references[layer_id])
            # NOTE The last reference will not be used.
            hidden_state = hidden_states[layer_id]
            outputs_class = self.cls_branches[layer_id](hidden_state)
            tmp_reg_preds = self.reg_branches[layer_id](hidden_state)
            if self.training: 
                # cls old -> size new
                outputs_aux = self.aux_branches[layer_id](outputs_class)

            if reference.shape[-1] == 4:
                # When `layer` is 0 and `as_two_stage` of the detector
                # is `True`, or when `layer` is greater than 0 and
                # `with_box_refine` of the detector is `True`.
                tmp_reg_preds += reference
            else:
                # When `layer` is 0 and `as_two_stage` of the detector
                # is `False`, or when `layer` is greater than 0 and
                # `with_box_refine` of the detector is `False`.
                assert reference.shape[-1] == 2
                tmp_reg_preds[..., :2] += reference
            outputs_coord = tmp_reg_preds.sigmoid()
            all_layers_outputs_classes.append(outputs_class)
            all_layers_outputs_coords.append(outputs_coord)
            if self.training: all_layers_outputs_aux.append(outputs_aux)

        all_layers_outputs_classes = torch.stack(all_layers_outputs_classes)
        all_layers_outputs_coords = torch.stack(all_layers_outputs_coords)
        if self.training: all_layers_outputs_aux = torch.stack(all_layers_outputs_aux)

        if self.training:
            return all_layers_outputs_classes, all_layers_outputs_coords, all_layers_outputs_aux
        return all_layers_outputs_classes, all_layers_outputs_coords

    def loss_by_feat(
        self,
        all_layers_cls_scores: Tensor,
        all_layers_bbox_preds: Tensor,
        all_layers_aux_scores: Tensor,
        enc_cls_scores: Tensor,
        enc_bbox_preds: Tensor,
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        dn_meta: Dict[str, int],
        batch_gt_instances_ignore: OptInstanceList = None
    ) -> Dict[str, Tensor]:
        # extract denoising and matching part of outputs
        (all_layers_matching_cls_scores, all_layers_matching_bbox_preds, all_layers_matching_aux_scores,
         all_layers_denoising_cls_scores, all_layers_denoising_bbox_preds, all_layers_denoising_aux_scores) = \
            self.split_outputs(all_layers_cls_scores, all_layers_bbox_preds, all_layers_aux_scores, dn_meta)

        loss_dict = self.loss_by_feat_matching(
            all_layers_matching_cls_scores, all_layers_matching_bbox_preds, all_layers_matching_aux_scores,
            batch_gt_instances, batch_img_metas, batch_gt_instances_ignore)

        # loss of proposal generated from encode feature map.
        if enc_cls_scores is not None:
            enc_loss_cls, enc_losses_bbox, enc_losses_iou = \
                super(DINOHead, self).loss_by_feat_single(
                    enc_cls_scores, enc_bbox_preds,
                    batch_gt_instances=batch_gt_instances,
                    batch_img_metas=batch_img_metas)
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_bbox'] = enc_losses_bbox
            loss_dict['enc_loss_iou'] = enc_losses_iou

        if all_layers_denoising_cls_scores is not None:
            # calculate denoising loss from all decoder layers
            dn_losses_cls, dn_losses_bbox, dn_losses_iou, dn_losses_aux = self.loss_dn(
                all_layers_denoising_cls_scores,
                all_layers_denoising_bbox_preds,
                all_layers_denoising_aux_scores,
                batch_gt_instances=batch_gt_instances,
                batch_img_metas=batch_img_metas,
                dn_meta=dn_meta)
            # collate denoising loss
            loss_dict['dn_loss_cls'] = dn_losses_cls[-1]
            loss_dict['dn_loss_bbox'] = dn_losses_bbox[-1]
            loss_dict['dn_loss_iou'] = dn_losses_iou[-1]
            loss_dict['dn_loss_aux'] = dn_losses_aux[-1]
            for num_dec_layer, (loss_cls_i, loss_bbox_i, loss_iou_i, loss_aux_i) in \
                    enumerate(zip(dn_losses_cls[:-1], dn_losses_bbox[:-1], dn_losses_iou[:-1], dn_losses_aux[:-1])):
                loss_dict[f'd{num_dec_layer}.dn_loss_cls'] = loss_cls_i
                loss_dict[f'd{num_dec_layer}.dn_loss_bbox'] = loss_bbox_i
                loss_dict[f'd{num_dec_layer}.dn_loss_iou'] = loss_iou_i
                loss_dict[f'd{num_dec_layer}.dn_loss_aux'] = loss_aux_i

        return loss_dict

    def loss_by_feat_matching(
        self,
        all_layers_cls_scores: Tensor,
        all_layers_bbox_preds: Tensor,
        all_layers_aux_scores: Tensor,
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        batch_gt_instances_ignore: OptInstanceList = None
    ) -> Dict[str, Tensor]:
        assert batch_gt_instances_ignore is None, \
            f'{self.__class__.__name__} only supports for batch_gt_instances_ignore setting to None.'

        losses_cls, losses_bbox, losses_iou, losses_aux = multi_apply(
            self.loss_by_feat_matching_single,
            all_layers_cls_scores,
            all_layers_bbox_preds,
            all_layers_aux_scores,
            batch_gt_instances=batch_gt_instances,
            batch_img_metas=batch_img_metas)

        loss_dict = dict()
        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]
        loss_dict['loss_iou'] = losses_iou[-1]
        loss_dict['loss_aux'] = losses_aux[-1]
        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i, loss_iou_i, loss_aux_i in \
                zip(losses_cls[:-1], losses_bbox[:-1], losses_iou[:-1], losses_aux[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            loss_dict[f'd{num_dec_layer}.loss_iou'] = loss_iou_i
            loss_dict[f'd{num_dec_layer}.loss_aux'] = loss_aux_i
            num_dec_layer += 1
        return loss_dict

    def loss_by_feat_matching_single(self, cls_scores: Tensor, bbox_preds: Tensor, aux_scores: Tensor,
                            batch_gt_instances: InstanceList,
                            batch_img_metas: List[dict]) -> Tuple[Tensor]:
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_targets_matching(cls_scores_list, bbox_preds_list,
                                                    batch_gt_instances, batch_img_metas)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, aux_labels_list, aux_label_weights_list,
         num_total_pos, num_total_neg, pos_inds_list) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)
        aux_labels = torch.cat(aux_labels_list, 0)
        aux_label_weights = torch.cat(aux_label_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        # auxiliary loss
        aux_scores = aux_scores.reshape(-1, self.aux_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        aux_avg_factor = num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            aux_avg_factor = reduce_mean(aux_scores.new_tensor([aux_avg_factor]))
        aux_avg_factor = max(aux_avg_factor, 1)

        if isinstance(self.loss_cls, QualityFocalLoss):
            bg_class_ind = self.num_classes
            pos_inds = ((labels >= 0) & (labels < bg_class_ind)).nonzero().squeeze(1)
            scores = label_weights.new_zeros(labels.shape)
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_decode_bbox_targets = bbox_cxcywh_to_xyxy(pos_bbox_targets)
            pos_bbox_pred = bbox_preds.reshape(-1, 4)[pos_inds]
            pos_decode_bbox_pred = bbox_cxcywh_to_xyxy(pos_bbox_pred)
            scores[pos_inds] = bbox_overlaps(
                pos_decode_bbox_pred.detach(),
                pos_decode_bbox_targets,
                is_aligned=True)
            loss_cls = self.loss_cls(
                cls_scores, (labels, scores), label_weights, avg_factor=cls_avg_factor)

            # auxiliary
            bg_class_ind = self.num_aux_classes
            pos_inds = ((aux_labels >= 0) & (aux_labels < bg_class_ind)).nonzero().squeeze(1)
            scores = aux_label_weights.new_zeros(aux_labels.shape)
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_decode_bbox_targets = bbox_cxcywh_to_xyxy(pos_bbox_targets)
            pos_bbox_pred = bbox_preds.reshape(-1, 4)[pos_inds]
            pos_decode_bbox_pred = bbox_cxcywh_to_xyxy(pos_bbox_pred)
            scores[pos_inds] = bbox_overlaps(
                pos_decode_bbox_pred.detach(),
                pos_decode_bbox_targets,
                is_aligned=True)
            # loss_aux = self.loss_aux(
            #     aux_scores, (aux_labels, scores), aux_label_weights, avg_factor=aux_avg_factor)
            pos_mask = torch.cat(pos_inds_list, dim=0)
            loss_aux = self.loss_aux(aux_scores[pos_mask], aux_labels, aux_label_weights) / num_total_pos
        else:
            loss_cls = self.loss_cls(
                cls_scores, labels, label_weights, avg_factor=cls_avg_factor)
            # loss_aux = self.loss_aux(
            #     aux_scores, aux_labels, aux_label_weights, avg_factor=aux_avg_factor)
            pos_mask = torch.cat(pos_inds_list, dim=0)
            loss_aux = self.loss_aux(aux_scores[pos_mask], aux_labels, aux_label_weights) / num_total_pos

        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # construct factors used for rescale bboxes
        factors = []
        for img_meta, bbox_pred in zip(batch_img_metas, bbox_preds):
            img_h, img_w, = img_meta['img_shape']
            factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                           img_h]).unsqueeze(0).repeat(
                bbox_pred.size(0), 1)
            factors.append(factor)
        factors = torch.cat(factors, 0)

        # DETR regress the relative position of boxes (cxcywh) in the image,
        # thus the learning target is normalized by the image size. So here
        # we need to re-scale them for calculating IoU loss
        bbox_preds = bbox_preds.reshape(-1, 4)
        bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors
        bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors

        # regression IoU loss, defaultly GIoU loss
        loss_iou = self.loss_iou(
            bboxes, bboxes_gt, bbox_weights, avg_factor=num_total_pos)

        # regression L1 loss
        loss_bbox = self.loss_bbox(
            bbox_preds, bbox_targets, bbox_weights, avg_factor=num_total_pos)
        return loss_cls, loss_bbox, loss_iou, loss_aux

    def get_targets_matching(self, cls_scores_list: List[Tensor],
                             bbox_preds_list: List[Tensor],
                             batch_gt_instances: InstanceList, batch_img_metas: List[dict]) -> tuple:
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, aux_labels_list, aux_label_weights_list,
         pos_inds_list, neg_inds_list) = multi_apply(self._get_targets_matching_single,
                                                     cls_scores_list, bbox_preds_list,
                                                     batch_gt_instances, batch_img_metas)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
                aux_labels_list, aux_label_weights_list,
                num_total_pos, num_total_neg, pos_inds_list)

    def _get_targets_matching_single(self, cls_score: Tensor, bbox_pred: Tensor,
                                     gt_instances: InstanceData, img_meta: dict) -> tuple:
        img_h, img_w = img_meta['img_shape']
        factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                       img_h]).unsqueeze(0)
        num_bboxes = bbox_pred.size(0)
        # convert bbox_pred from xywh, normalized to xyxy, unnormalized
        bbox_pred = bbox_cxcywh_to_xyxy(bbox_pred)
        bbox_pred = bbox_pred * factor

        pred_instances = InstanceData(scores=cls_score, bboxes=bbox_pred)
        # assigner and sampler
        assign_result = self.assigner.assign(
            pred_instances=pred_instances,
            gt_instances=gt_instances,
            img_meta=img_meta)

        gt_bboxes = gt_instances.bboxes
        gt_labels = gt_instances.labels
        gt_size_labels = gt_instances.size_labels
        pos_inds = torch.nonzero(
            assign_result.gt_inds > 0, as_tuple=False).squeeze(-1).unique()
        neg_inds = torch.nonzero(
            assign_result.gt_inds == 0, as_tuple=False).squeeze(-1).unique()
        pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1
        pos_gt_bboxes = gt_bboxes[pos_assigned_gt_inds.long(), :]

        # label targets
        labels = gt_bboxes.new_full((num_bboxes, ),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred, dtype=gt_bboxes.dtype)
        bbox_weights = torch.zeros_like(bbox_pred, dtype=gt_bboxes.dtype)
        bbox_weights[pos_inds] = 1.0

        # # aux label targets
        # aux_labels = gt_bboxes.new_full((num_bboxes,), self.num_aux_classes, dtype=torch.long)
        # aux_labels[pos_inds] = gt_size_labels[pos_assigned_gt_inds]
        # aux_label_weights = gt_bboxes.new_ones(num_bboxes)

        # DETR regress the relative position of boxes (cxcywh) in the image.
        # Thus the learning target should be normalized by the image size, also
        # the box format should be converted from defaultly x1y1x2y2 to cxcywh.
        pos_gt_bboxes_normalized = pos_gt_bboxes / factor
        pos_gt_bboxes_targets = bbox_xyxy_to_cxcywh(pos_gt_bboxes_normalized)
        bbox_targets[pos_inds] = pos_gt_bboxes_targets

        # aux label targets
        # aux_labels = gt_bboxes.new_full((num_bboxes,), self.num_aux_classes, dtype=torch.long)
        pos_gt_bboxes = bbox_xyxy_to_cxcywh(pos_gt_bboxes)
        pos_gt_bboxes_area = pos_gt_bboxes[:, [-1]] * pos_gt_bboxes[:, [-2]]
        pos_gt_bboxes_area = torch.clamp(pos_gt_bboxes_area, 1e-6)
        pos_gt_bboxes_area = torch.sqrt(pos_gt_bboxes_area)
        aux_labels = self.size_distributed(pos_gt_bboxes_area)
        aux_label_weights = torch.ones_like(pos_gt_bboxes_area)
        # size_distribution = self.size_distribution(pos_gt_bboxes_area, a0=0.0, a1=32.0, a2=96.0, var=16.0)
        # aux_labels[pos_inds] = size_distribution[pos_assigned_gt_inds]
        return (labels, label_weights, bbox_targets, bbox_weights, aux_labels, aux_label_weights, pos_inds, neg_inds)
        
    def size_distributed(self, input, a0=0.0, a1=32.0, a2=96.0, var=16.0):
        output = torch.zeros_like(input)
        output = torch.repeat_interleave(output, 3, dim=-1)
        input = input.cpu().numpy()
        sA0 = norm.cdf((input - a0) / var)
        sA1 = norm.cdf((input - a1) / var)
        sA2 = norm.cdf((input - a2) / var)
        output[:, [0]] = torch.tensor((sA1 - sA0), dtype=output.dtype, device=output.device)
        output[:, [1]] = torch.tensor((sA2 - sA1), dtype=output.dtype, device=output.device)
        output[:, [2]] = torch.ones_like(output[:, [2]]) - torch.tensor(sA2, dtype=output.dtype, device=output.device)
        return F.softmax(output, dim=-1)
    
    def loss_dn(self, all_layers_denoising_cls_scores: Tensor,
                all_layers_denoising_bbox_preds: Tensor,
                all_layers_denoising_aux_scores: Tensor,
                batch_gt_instances: InstanceList, batch_img_metas: List[dict],
                dn_meta: Dict[str, int]) -> Tuple[List[Tensor]]:
        return multi_apply(
            self._loss_dn_single,
            all_layers_denoising_cls_scores,
            all_layers_denoising_bbox_preds,
            all_layers_denoising_aux_scores,
            batch_gt_instances=batch_gt_instances,
            batch_img_metas=batch_img_metas,
            dn_meta=dn_meta)

    def _loss_dn_single(self, dn_cls_scores: Tensor, dn_bbox_preds: Tensor, dn_aux_scores: Tensor,
                        batch_gt_instances: InstanceList,
                        batch_img_metas: List[dict],
                        dn_meta: Dict[str, int]) -> Tuple[Tensor]:
        cls_reg_targets = self.get_dn_targets(batch_gt_instances, batch_img_metas, dn_meta)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, aux_labels_list, aux_label_weights_list,
         num_total_pos, num_total_neg, pos_inds_list) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)
        aux_labels = torch.cat(aux_labels_list, 0)
        aux_label_weights = torch.cat(aux_label_weights_list, 0)

        # classification loss
        cls_scores = dn_cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        # auxiliary loss
        aux_scores = dn_aux_scores.reshape(-1, self.aux_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        aux_avg_factor = num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            aux_avg_factor = reduce_mean(aux_scores.new_tensor([aux_avg_factor]))
        aux_avg_factor = max(aux_avg_factor, 1)

        if len(cls_scores) > 0:
            if isinstance(self.loss_cls, QualityFocalLoss):
                bg_class_ind = self.num_classes
                pos_inds = ((labels >= 0) & (labels < bg_class_ind)).nonzero().squeeze(1)
                scores = label_weights.new_zeros(labels.shape)
                pos_bbox_targets = bbox_targets[pos_inds]
                pos_decode_bbox_targets = bbox_cxcywh_to_xyxy(pos_bbox_targets)
                pos_bbox_pred = dn_bbox_preds.reshape(-1, 4)[pos_inds]
                pos_decode_bbox_pred = bbox_cxcywh_to_xyxy(pos_bbox_pred)
                scores[pos_inds] = bbox_overlaps(
                    pos_decode_bbox_pred.detach(),
                    pos_decode_bbox_targets,
                    is_aligned=True)
                loss_cls = self.loss_cls(
                    cls_scores, (labels, scores), weight=label_weights, avg_factor=cls_avg_factor)

                # auxiliary
                bg_class_ind = self.num_aux_classes
                pos_inds = ((aux_labels >= 0) & (aux_labels < bg_class_ind)).nonzero().squeeze(1)
                scores = aux_label_weights.new_zeros(aux_labels.shape)
                pos_bbox_targets = bbox_targets[pos_inds]
                pos_decode_bbox_targets = bbox_cxcywh_to_xyxy(pos_bbox_targets)
                pos_bbox_pred = dn_bbox_preds.reshape(-1, 4)[pos_inds]
                pos_decode_bbox_pred = bbox_cxcywh_to_xyxy(pos_bbox_pred)
                scores[pos_inds] = bbox_overlaps(
                    pos_decode_bbox_pred.detach(),
                    pos_decode_bbox_targets,
                    is_aligned=True)
                # loss_aux = self.loss_aux(
                #     aux_scores, (aux_labels, scores), aux_label_weights, avg_factor=aux_avg_factor)
                pos_mask = torch.cat(pos_inds_list, dim=0)
                loss_aux = self.loss_aux(aux_scores[pos_mask], aux_labels, aux_label_weights) / num_total_pos
            else:
                loss_cls = self.loss_cls(cls_scores, labels, label_weights, avg_factor=cls_avg_factor)
                # loss_aux = self.loss_aux(
                #     aux_scores, aux_labels, aux_label_weights, avg_factor=aux_avg_factor)
                pos_mask = torch.cat(pos_inds_list, dim=0)
                loss_aux = self.loss_aux(aux_scores[pos_mask], aux_labels, aux_label_weights) / num_total_pos
        else:
            loss_cls = torch.zeros(
                1, dtype=cls_scores.dtype, device=cls_scores.device)
            loss_aux = torch.zeros(
                1, dtype=aux_scores.dtype, device=aux_scores.device)

        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # construct factors used for rescale bboxes
        factors = []
        for img_meta, bbox_pred in zip(batch_img_metas, dn_bbox_preds):
            img_h, img_w = img_meta['img_shape']
            factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                           img_h]).unsqueeze(0).repeat(
                                               bbox_pred.size(0), 1)
            factors.append(factor)
        factors = torch.cat(factors)

        # DETR regress the relative position of boxes (cxcywh) in the image,
        # thus the learning target is normalized by the image size. So here
        # we need to re-scale them for calculating IoU loss
        bbox_preds = dn_bbox_preds.reshape(-1, 4)
        bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors
        bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors

        # regression IoU loss, defaultly GIoU loss
        loss_iou = self.loss_iou(
            bboxes, bboxes_gt, bbox_weights, avg_factor=num_total_pos)

        # regression L1 loss
        loss_bbox = self.loss_bbox(
            bbox_preds, bbox_targets, bbox_weights, avg_factor=num_total_pos)
        return loss_cls, loss_bbox, loss_iou, loss_aux

    def get_dn_targets(self, batch_gt_instances: InstanceList,
                       batch_img_metas: dict, dn_meta: Dict[str, int]) -> tuple:
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, aux_labels_list, aux_label_weights_list,
         pos_inds_list, neg_inds_list) = multi_apply(
             self._get_dn_targets_single,
             batch_gt_instances,
             batch_img_metas,
             dn_meta=dn_meta)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
                aux_labels_list, aux_label_weights_list, num_total_pos, num_total_neg, pos_inds_list)

    def _get_dn_targets_single(self, gt_instances: InstanceData,
                               img_meta: dict, dn_meta: Dict[str, int]) -> tuple:
        gt_bboxes = gt_instances.bboxes
        gt_labels = gt_instances.labels
        gt_size_labels = gt_instances.size_labels
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

        # aux label targets
        aux_labels = gt_bboxes.new_full((num_denoising_queries, ), self.num_aux_classes, dtype=torch.long)
        aux_labels[pos_inds] = gt_size_labels[pos_assigned_gt_inds]
        aux_label_weights = gt_bboxes.new_ones(num_denoising_queries)

        # aux label targets
        # aux_labels = gt_bboxes.new_full((num_bboxes,), self.num_aux_classes, dtype=torch.long)
        pos_gt_bboxes = gt_bboxes[pos_assigned_gt_inds.long(), :]
        pos_gt_bboxes = bbox_xyxy_to_cxcywh(pos_gt_bboxes)
        pos_gt_bboxes_area = pos_gt_bboxes[:, [-1]] * pos_gt_bboxes[:, [-2]]
        pos_gt_bboxes_area = torch.clamp(pos_gt_bboxes_area, 1e-6)
        pos_gt_bboxes_area = torch.sqrt(pos_gt_bboxes_area)
        aux_labels = self.size_distributed(pos_gt_bboxes_area)
        aux_label_weights = torch.ones_like(pos_gt_bboxes_area)

        return (labels, label_weights, bbox_targets, bbox_weights, aux_labels, aux_label_weights, pos_inds, neg_inds)

    def filter_gt_bboxes_labels(self, gt_bboxes: Tensor, gt_labels: Tensor) -> Tuple:
        bboxes_wh = bbox_xyxy_to_cxcywh(gt_bboxes)[:, 2:]
        bboxes_area = torch.prod(bboxes_wh, dim=1)
        
        chosen_small_size_indice = torch.nonzero(bboxes_area < 1024).view(-1)       # 32^2
        # if len(chosen_small_size_indice) == 0:
        #     chosen_small_size_indice = torch.nonzero(bboxes_area < 2304).view(-1)   # 48^2        
        if len(chosen_small_size_indice) == 0:
            chosen_small_size_indice = torch.nonzero(bboxes_area < 4096).view(-1)   # 64^2           
        filter_bboxes = gt_bboxes[chosen_small_size_indice]
        filter_labels = gt_labels[chosen_small_size_indice]
        
        if len(chosen_small_size_indice) == 0:  # There is not any small and middle size object
            filter_bboxes = gt_bboxes
            filter_labels = gt_labels

        return filter_bboxes, filter_labels    
    
    @staticmethod
    def split_outputs(all_layers_cls_scores: Tensor,
                          all_layers_bbox_preds: Tensor,
                          all_layers_aux_scores: Tensor,
                          dn_meta: Dict[str, int]) -> Tuple[Tensor]:
        num_denoising_queries = dn_meta['num_denoising_queries']
        if dn_meta is not None:
            all_layers_denoising_cls_scores = \
                all_layers_cls_scores[:, :, : num_denoising_queries, :]
            all_layers_denoising_bbox_preds = \
                all_layers_bbox_preds[:, :, : num_denoising_queries, :]
            all_layers_denoising_aux_scores = \
                all_layers_aux_scores[:, :, : num_denoising_queries, :]
            all_layers_matching_cls_scores = \
                all_layers_cls_scores[:, :, num_denoising_queries:, :]
            all_layers_matching_bbox_preds = \
                all_layers_bbox_preds[:, :, num_denoising_queries:, :]
            all_layers_matching_aux_scores = \
                all_layers_aux_scores[:, :, num_denoising_queries:, :]
        else:
            all_layers_denoising_cls_scores = None
            all_layers_denoising_bbox_preds = None
            all_layers_denoising_aux_scores = None
            all_layers_matching_cls_scores = all_layers_cls_scores
            all_layers_matching_bbox_preds = all_layers_bbox_preds
            all_layers_matching_aux_scores = all_layers_aux_scores
        return (all_layers_matching_cls_scores, all_layers_matching_bbox_preds, all_layers_matching_aux_scores,
                all_layers_denoising_cls_scores, all_layers_denoising_bbox_preds, all_layers_denoising_aux_scores)