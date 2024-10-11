from mmengine.model import is_model_wrapper
from mmengine.hooks import Hook
from mmdet.registry import HOOKS


@HOOKS.register_module()
class UnfreezeBackboneEpochBasedHook(Hook):
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
            backbone = model.backbone
            if backbone.frozen_stages >= 0:
                if backbone.deep_stem:
                    backbone.stem.train()
                    for param in backbone.stem.parameters():
                        param.requires_grad = True
                else:
                    backbone.norm1.train()
                    for m in [backbone.conv1, backbone.norm1]:
                        for param in m.parameters():
                            param.requires_grad = True

            for i in range(1, backbone.frozen_stages + 1):
                m = getattr(backbone, f'layer{i}')
                m.train()
                for param in m.parameters():
                    param.requires_grad = True


@HOOKS.register_module()
class UnfreezeAuxiliaryBranchBasedHook(Hook):
    """Unfreeze backbone network Hook.

    Args:
        unfreeze_epoch (int): The epoch unfreezing the backbone network.
    """

    def __init__(self, unfreeze_epoch=1, is_backbone_special=True):
        self.unfreeze_epoch = unfreeze_epoch
        self.is_backbone_special = is_backbone_special

    def before_train_epoch(self, runner):
        # Unfreeze the backbone network.
        # Only valid for resnet.
        if runner.epoch == self.unfreeze_epoch:
            model = runner.model
            if is_model_wrapper(model):
                model = model.module

            model.train()
            for param in model.parameters():
                if param.requires_grad is False:
                    param.requires_grad = True

            # Backbone
            backbone = model.aux_backbone
            backbone.train()
            for param in backbone.parameters():
                if param.requires_grad is False:
                    param.requires_grad = True
            # resnet freeze layer 1
            if self.is_backbone_special:
                self.backbone_special(backbone)

            # Neck
            neck = model.aux_neck
            neck.train()
            for _, param in neck.named_parameters():
                if param.requires_grad is False:
                    param.requires_grad = True

            # Head
            bbox_head = model.aux_bbox_head
            bbox_head.train()
            for _, param in bbox_head.named_parameters():
                if param.requires_grad is False:
                    param.requires_grad = True

    def backbone_special(self, backbone):
        if backbone.frozen_stages >= 0:
            if backbone.deep_stem:
                backbone.stem.eval()
                for param in backbone.stem.parameters():
                    param.requires_grad = False
            else:
                backbone.norm1.eval()
                for m in [backbone.conv1, backbone.norm1]:
                    for param in m.parameters():
                        param.requires_grad = False

        for i in range(1, backbone.frozen_stages + 1):
            m = getattr(backbone, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False


@HOOKS.register_module()
class UnfreezeModelEpochBasedHook(Hook):
    """Unfreeze backbone network Hook.

    Args:
        unfreeze_epoch (int): The epoch unfreezing the backbone network.
    """

    def __init__(self, unfreeze_epoch=1,
                 # is_backbone_special=True
                 ):
        self.unfreeze_epoch = unfreeze_epoch
        # self.is_backbone_special = is_backbone_special

    def before_train_epoch(self, runner):
        # Unfreeze the backbone network.
        # Only valid for resnet.
        if runner.epoch == self.unfreeze_epoch:
            model = runner.model
            if is_model_wrapper(model):
                model = model.module
            model.train()
            for _, param in model.named_parameters():
                if param.requires_grad is False:
                    param.requires_grad = True

            # resnet freeze layer 1
            # if self.is_backbone_special:
            #     self.backbone_special(model.backbone)
            backbone = model.bakcbone
            if backbone.frozen_stages >= 0:
                if backbone.deep_stem:
                    backbone.stem.eval()
                    for param in backbone.stem.parameters():
                        param.requires_grad = False
                else:
                    backbone.norm1.eval()
                    for m in [backbone.conv1, backbone.norm1]:
                        for param in m.parameters():
                            param.requires_grad = False

            for i in range(1, backbone.frozen_stages + 1):
                m = getattr(backbone, f'layer{i}')
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False


    def backbone_special(self, bakcbone):
        if self.frozen_stages >= 0:
            if self.deep_stem:
                self.stem.eval()
                for param in self.stem.parameters():
                    param.requires_grad = False
            else:
                self.norm1.eval()
                for m in [self.conv1, self.norm1]:
                    for param in m.parameters():
                        param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False