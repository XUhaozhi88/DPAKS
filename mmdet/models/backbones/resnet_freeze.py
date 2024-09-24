from mmdet.registry import MODELS

from .resnet import ResNet

@MODELS.register_module()
class ResNetFreeze(ResNet):
    def __init__(self, freeeze_all=False, *args, **kwargs):
        super(ResNetFreeze, self).__init__(*args, **kwargs)
        if freeeze_all:
            self._freeze_all()

    def _freeze_all(self):
        self.eval()
        for _, param in self.named_parameters():
            param.requires_grad = False