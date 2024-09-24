from abc import ABCMeta
from mmdet.registry import MODELS
from mmdet.structures import OptSampleList
from mmdet.models.layers import CdnQueryGenerator

from mmdet.models.layers.transformer.utils import inverse_sigmoid

import argparse
import os.path as osp

from mmengine.config import Config
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

from mmdet.utils import setup_cache_size_limit_of_dynamo

from typing import Tuple

import torch
from torch import Tensor

from mmdet.structures import SampleList
from mmdet.structures.bbox import bbox_xyxy_to_cxcywh


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
            bboxes = bboxes.tensor
            # bboxes = torch.Tensor(bboxes)
            bboxes_normalized = bboxes / factor
            labels = sample.gt_instances.labels
            gt_bboxes, labels = self.filter_gt_bboxes_labels(absolute_gt_bboxes=bboxes,
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

        return gt_bboxes, dn_bbox_query

        # The `batch_idx` saves the batch index of the corresponding sample
        # for each target, has shape (num_target_total).
        # batch_idx = torch.cat([
        #     torch.full_like(t.long(), i) for i, t in enumerate(gt_labels_list)
        # ])
        # dn_label_query, dn_bbox_query = self.collate_dn_queries(
        #     dn_label_query, dn_bbox_query, batch_idx, len(batch_data_samples),
        #     num_groups)
        #
        # attn_mask = self.generate_dn_mask(
        #     max_num_target, num_groups, device=dn_label_query.device)
        #
        # dn_meta = dict(
        #     num_denoising_queries=int(max_num_target * 2 * num_groups),
        #     num_denoising_groups=num_groups)
        #
        # return dn_label_query, dn_bbox_query, attn_mask, dn_meta

    def generate_dn_bbox_query(self, gt_bboxes: Tensor,
                               num_groups: int) -> Tensor:
        """Generate noisy bboxes and their query embeddings.

        The strategy for generating noisy bboxes is as follow:

        .. code:: text

            +--------------------+
            |      negative      |
            |    +----------+    |
            |    | positive |    |
            |    |    +-----|----+------------+
            |    |    |     |    |            |
            |    +----+-----+    |            |
            |         |          |            |
            +---------+----------+            |
                      |                       |
                      |        gt bbox        |
                      |                       |
                      |             +---------+----------+
                      |             |         |          |
                      |             |    +----+-----+    |
                      |             |    |    |     |    |
                      +-------------|--- +----+     |    |
                                    |    | positive |    |
                                    |    +----------+    |
                                    |      negative      |
                                    +--------------------+

         The random noise is added to the top-left and down-right point
         positions, hence, normalized (x, y, x, y) format of bboxes are
         required. The noisy bboxes of positive queries have the points
         both within the inner square, while those of negative queries
         have the points both between the inner and outer squares.

        Besides, the length of outer square is twice as long as that of
        the inner square, i.e., self.box_noise_scale * w_or_h / 2.
        NOTE The noise is added to all the bboxes. Moreover, there is still
        unconsidered case when one point is within the positive square and
        the others is between the inner and outer squares.

        Args:
            gt_bboxes (Tensor): The concatenated gt bboxes of all samples
                in the batch, has shape (num_target_total, 4) with the last
                dimension arranged as (cx, cy, w, h) where
                `num_target_total = sum(num_target_list)`.
            num_groups (int): The number of denoising query groups.

        Returns:
            Tensor: The output noisy bboxes, which are embedded by normalized
            (cx, cy, w, h) format bboxes going through inverse_sigmoid, has
            shape (num_noisy_targets, 4) with the last dimension arranged as
            (cx, cy, w, h), where
            `num_noisy_targets = num_target_total * num_groups * 2`.
        """
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

        # add noise to the bboxes
        bboxes_whwh = bbox_xyxy_to_cxcywh(gt_bboxes_expand)[:, 2:].repeat(1, 2)
        noisy_bboxes_expand = gt_bboxes_expand + torch.mul(
            rand_part, bboxes_whwh) * self.box_noise_scale / 2  # xyxy
        noisy_bboxes_expand = noisy_bboxes_expand.clamp(min=0.0, max=1.0)
        noisy_bboxes_expand = bbox_xyxy_to_cxcywh(noisy_bboxes_expand)

        dn_bbox_query = inverse_sigmoid(noisy_bboxes_expand, eps=1e-3)
        return dn_bbox_query

    def filter_gt_bboxes_labels(self, absolute_gt_bboxes: Tensor, relative_gt_bboxes: Tensor, gt_labels: Tensor) -> Tuple:
        bboxes_wh = bbox_xyxy_to_cxcywh(absolute_gt_bboxes)[:, 2:]
        bboxes_area = torch.prod(bboxes_wh, dim=1)
        chosen_small_size_indice = torch.nonzero(bboxes_area < 1024).view(-1)       # 32^2
        if len(chosen_small_size_indice) == 0:
            chosen_small_size_indice = torch.nonzero(bboxes_area < 4096).view(-1)   # 64^2
        filter_bboxes = relative_gt_bboxes[chosen_small_size_indice]
        filter_labels = gt_labels[chosen_small_size_indice]

        if len(chosen_small_size_indice) == 0:  # There is not any small and middle size object
            filter_bboxes = relative_gt_bboxes
            filter_labels = gt_labels
        # threshold = 32
        # chosen_small_size_indice = []
        # while(len(chosen_small_size_indice) == 0):
        #     chosen_small_size_indice = torch.nonzero(bboxes_area < threshold*threshold).view(-1)  # Note `* 0.5`
        #     threshold += 16
        return filter_bboxes, filter_labels


@MODELS.register_module()
class GetNoise(object):

    def __init__(self,
                 num_classes,
                 embed_dims,
                 num_queries,
                 dn_cfg) -> None:

        if dn_cfg is not None:
            assert 'num_classes' not in dn_cfg and \
                   'num_queries' not in dn_cfg and \
                   'hidden_dim' not in dn_cfg, \
                'The three keyword args `num_classes`, `embed_dims`, and ' \
                '`num_matching_queries` are set in `detector.__init__()`, ' \
                'users should not set them in `dn_cfg` config.'
            dn_cfg['num_classes'] = num_classes
            dn_cfg['embed_dims'] = embed_dims
            dn_cfg['num_matching_queries'] = num_queries
        self.dn_query_generator = CdnSmallQueryGenerator(**dn_cfg)

    def get_noise(self, batch_data_samples: OptSampleList = None):
        dn_label_query, dn_bbox_query, dn_mask, dn_meta = self.dn_query_generator(batch_data_samples)

if __name__ == '__main__':

    def parse_args():
        parser = argparse.ArgumentParser(description='Train a detector')
        parser.add_argument('--config',
                            default='D:/Files/Code/Python/DINO-Small/mmdetection/configs/dino_small/dino_noise-4scale_r50_8xb2-12e_visdrone.py',
                            help='train config file path')
        parser.add_argument('--work-dir',
                            default='D:/Files/Code/Python/DINO-Small/mmdetection/results/VisDroneDET/dino_noise/data/',
                            help='the dir to save logs and models')
        args = parser.parse_args()
        return args

    args = parse_args()

    setup_cache_size_limit_of_dynamo()

        # load config
    cfg = Config.fromfile(args.config)

        # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
            # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
            # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])

        # build the runner from config
    if 'runner_type' not in cfg:
        runner = Runner.from_cfg(cfg)
    else:
        runner = RUNNERS.build(cfg)

    # get model
    num_classes = runner.model.bbox_head.num_classes
    embed_dims = runner.model.encoder.embed_dims
    num_queries = runner.model.num_queries
    dn_cfg = dict(label_noise_scale=0.5, box_noise_scale=1.0,
                  group_cfg=dict(dynamic=True, num_groups=None, num_dn_queries=100))
    Net = GetNoise(num_classes, embed_dims, num_queries, dn_cfg)

    train_dataloader = runner.train_dataloader
    # batch_sampler = train_dataloader.batch_sampler

    for idx, data_batch in enumerate(train_dataloader):
        gt_bboxes, dn_bbox_query = Net.get_noise(data_batch['data_samples'])
        pass
