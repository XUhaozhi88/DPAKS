import numpy as np

from .loading import LoadAnnotations
from mmdet.registry import TRANSFORMS

@TRANSFORMS.register_module()
class AUXLoadAnnotations(LoadAnnotations):

    def _load_labels(self, results: dict) -> None:
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:``mmengine.BaseDataset``.

        Returns:
            dict: The dict contains loaded label annotations.
        """
        gt_bboxes_labels = []
        gt_bboxes_size_labels = []
        for instance in results.get('instances', []):
            gt_bboxes_labels.append(instance['bbox_label'])
            gt_bboxes_size_labels.append(instance['size_label'])
        # TODO: Inconsistent with mmcv, consider how to deal with it later.
        results['gt_bboxes_labels'] = np.array(gt_bboxes_labels, dtype=np.int64)
        results['gt_bboxes_size_labels'] = np.array(gt_bboxes_size_labels, dtype=np.int64)