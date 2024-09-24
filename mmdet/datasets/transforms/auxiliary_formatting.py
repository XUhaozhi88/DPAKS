from mmdet.registry import TRANSFORMS

from .formatting import PackDetInputs


@TRANSFORMS.register_module()
class AUXPackDetInputs(PackDetInputs):
    mapping_table = {
        'gt_bboxes': 'bboxes',
        'gt_bboxes_labels': 'labels',
        'gt_bboxes_size_labels': 'size_labels',
        'gt_masks': 'masks'
    }