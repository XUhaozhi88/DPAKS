from mmengine.model import is_model_wrapper
from mmengine.hooks import Hook
from mmdet.registry import HOOKS


@HOOKS.register_module()
class FreezeBackboneEpochBasedHook(Hook):
    """Unfreeze backbone network Hook.

    Args:
        freeze_epoch (int): The epoch freezing the backbone network.
    """

    def __init__(self, freeze_epoch=0):
        self.freeze_epoch = freeze_epoch

    def before_train_epoch(self, runner):

        if runner.epoch == self.freeze_epoch:
            model = runner.model
            if is_model_wrapper(model):
                model = model.module
            backbone = model.backbone
            backbone.eval()
            for _, param in backbone.named_parameters():
                if param.requires_grad is True:
                    param.requires_grad = False


@HOOKS.register_module()
class FreezeAuxiliaryBranchEpochBasedHook(Hook):
    """Freeze auxiliary branch network Hook.

    Args:
        freeze_epoch (int): The epoch freezing the backbone network.
    """

    def __init__(self, freeze_epoch=0):
        self.freeze_epoch = freeze_epoch

    def before_train_epoch(self, runner):

        if runner.epoch == self.freeze_epoch:
            model = runner.model
            if is_model_wrapper(model):
                model = model.module
            
            # Backbone
            backbone = model.aux_backbone
            backbone.eval()
            for _, param in backbone.named_parameters():
                if param.requires_grad is True:
                    param.requires_grad = False

            # Neck
            neck = model.aux_neck
            neck.eval()
            for _, param in neck.named_parameters():
                if param.requires_grad is True:
                    param.requires_grad = False

            # Head
            bbox_head = model.aux_bbox_head
            bbox_head.eval()
            for _, param in bbox_head.named_parameters():
                if param.requires_grad is True:
                    param.requires_grad = False


@HOOKS.register_module()
class FreezeModelEpochBasedHook(Hook):
    """Unfreeze backbone network Hook.

    Args:
        unfreeze_epoch (int): The epoch unfreezing the backbone network.
    """

    def __init__(self, unfreeze_epoch=1):
        self.unfreeze_epoch = unfreeze_epoch

    def before_train_epoch(self, runner):
        # Unfreeze the backbone network.
        # Only valid for resnet.
        if runner.epoch == self.unfreeze_epoch:
            model = runner.model
            if is_model_wrapper(model):
                model = model.module
            model.train()
            for _, param in model.named_parameters():
                param.requires_grad = True



            # backbone = model.backbone
            # if backbone.frozen_stages >= 0:
            #     if backbone.deep_stem:
            #         backbone.stem.train()
            #         for param in backbone.stem.parameters():
            #             param.requires_grad = True
            #     else:
            #         backbone.norm1.train()
            #         for m in [backbone.conv1, backbone.norm1]:
            #             for param in m.parameters():
            #                 param.requires_grad = True
            #
            # for i in range(1, backbone.frozen_stages + 1):
            #     m = getattr(backbone, f'layer{i}')
            #     m.train()
            #     for param in m.parameters():
            #         param.requires_grad = True