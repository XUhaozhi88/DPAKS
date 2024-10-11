import torch
from torch import nn
import torch.nn.functional as F

from mmdet.registry import MODELS

from .resnet import ResNet

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class CBLinear(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):  # ch_in, ch_outs, kernel, stride, padding, groups
        super(CBLinear, self).__init__()
        self.c2 = c2
        if (isinstance(c1, int) is False) and (isinstance(c2, int) is False):
            raise AssertionError
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=True)

    def forward(self, x):
        out = self.conv(x)
        return out

class CBFuse(nn.Module):
    def __init__(self):
        super(CBFuse, self).__init__()

    def forward(self, xs):
        target_size = xs[-1].shape[2:]
        res = [F.interpolate(x, size=target_size, mode='nearest') for i, x in enumerate(xs[:-1])]
        out = torch.sum(torch.stack(res + xs[-1:]), dim=0)
        return out

@MODELS.register_module()
class AUXResNet(ResNet):
    """Auxiliary ResNet backbone.
    在这里我们没有使用和yolov9一样的多层级融合,仅是单层融合
    
    """

    def __init__(self, cbl_arch_settings=([512, 256], [1024, 512], [2048, 1024]),
                 *args, **kwargs):
        super().__init__(*args, **kwargs)        
        
        self.cbl_layers = []
        self.cbfuse_layers = []
        for i, cbl_arch in enumerate(cbl_arch_settings):
            cbl_layer = self.make_CBLinear(c1=cbl_arch[0], c2=cbl_arch[1])
            cbfuse_layer = self.make_CBFuse()
            cbl_layer_name = f'cbl_layer{i + 1}'
            cbfuse_layer_name = f'cbfuse_layer{i + 1}'
            self.add_module(cbl_layer_name, cbl_layer)
            self.add_module(cbfuse_layer_name, cbfuse_layer)
            self.cbl_layers.append(cbl_layer_name)
            self.cbfuse_layers.append(cbfuse_layer_name)

    def make_CBLinear(self, **kwargs):
        """Pack all blocks in a stage into a ``ResLayer``."""
        return CBLinear(**kwargs)
    
    def make_CBFuse(self, **kwargs):
        """Pack all blocks in a stage into a ``ResLayer``."""
        return CBFuse(**kwargs)

    def forward(self, x, resnet_xs):
        """Forward function."""

        if len(resnet_xs) == 4: resnet_xs = resnet_xs[1:]
        
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        x = self.maxpool(x)

        # resnet output
        resnet_outs = []
        for i, layer_name in enumerate(self.cbl_layers):
            cbl_layer = getattr(self, layer_name)
            resnet_x = cbl_layer(resnet_xs[i])
            resnet_outs.append(resnet_x)    

        # auxiliary resent output
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i < len(self.cbfuse_layers):
                cbfuse_layer = getattr(self, self.cbfuse_layers[i])
                x = cbfuse_layer([resnet_outs[i], x])
            if i in self.out_indices:
                outs.append(x)

        return tuple(outs)