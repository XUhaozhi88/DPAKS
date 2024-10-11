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
    def __init__(self, c1, c2s, k=1, s=1, p=None, g=1):  # ch_in, ch_outs, kernel, stride, padding, groups
        super(CBLinear, self).__init__()
        self.c2s = c2s
        if isinstance(c1, list):
            assert len(c1) == 1
            c1 = c1[0]
        elif isinstance(c1, int):
            pass
        else:
            raise AssertionError
        self.conv = nn.Conv2d(c1, sum(c2s), k, s, autopad(k, p), groups=g, bias=True)

    def forward(self, x):
        outs = self.conv(x).split(self.c2s, dim=1)
        return outs

class CBFuse(nn.Module):
    def __init__(self):
        super(CBFuse, self).__init__()

    def forward(self, xs):
        target_size = xs[0].shape[2:]
        res = [F.interpolate(x, size=target_size, mode='nearest') for i, x in enumerate(xs[1:])]
        out = torch.sum(torch.stack([xs[0]] + res), dim=0)
        return out

@MODELS.register_module()
class AUXResNet(ResNet):
    """Auxiliary ResNet backbone."""

    def __init__(self,
                 depth, in_channels=3, stem_channels=None, base_channels=64, num_stages=4,
                 strides=(1, 2, 2, 2), dilations=(1, 1, 1, 1), out_indices=(0, 1, 2, 3),
                 style='pytorch', deep_stem=False, avg_down=False, frozen_stages=-1,
                 conv_cfg=None, norm_cfg=dict(type='BN', requires_grad=True), norm_eval=True,
                 dcn=None, stage_with_dcn=(False, False, False, False), plugins=None,
                 with_cp=False, zero_init_residual=True, pretrained=None, init_cfg=None):
        super().__init__(depth, in_channels, stem_channels, base_channels, num_stages, 
                         strides, dilations, out_indices, style, deep_stem, avg_down, 
                         frozen_stages, conv_cfg, norm_cfg, norm_eval, dcn, stage_with_dcn, 
                         plugins, with_cp, zero_init_residual, pretrained, init_cfg)
        
        # P1、P2、P3
        cbl_arch_settings = (
            [[512], [256]],
            [[1024], [256, 512]],
            [[2048], [256, 512, 1024]])
        self.cbl_layers = []
        self.cbfuse_layers = []
        # for i in range(len(self.out_indices)):
        #     cbl_layer = self.make_CBLinear(
        #             c1=cbl_arch_settings[i][0],
        #             c2s=cbl_arch_settings[i][1])
        for i, cbl_arch in enumerate(cbl_arch_settings):
            cbl_layer = self.make_CBLinear(c1=cbl_arch[0], c2s=cbl_arch[1])
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
    
    def  make_CBFuse(self, **kwargs):
        """Pack all blocks in a stage into a ``ResLayer``."""
        return CBFuse(**kwargs)
    
    def forward0(self, x, resnet_xs):
        """Forward function."""
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

        # cbfuse inputs
        cbfuse_ins = []
        for i in range(len(self.cbl_layers)):
            cbfuse_ins.append([resnet_out[i] for resnet_out in resnet_outs])
            _ = resnet_outs.pop(0)        

        # auxiliary resent output
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i < len(self.cbfuse_layers):
                cbfuse_layer = getattr(self, self.cbfuse_layers[i])
                x = cbfuse_layer([*cbfuse_ins[i], x])
            if i in self.out_indices:
                outs.append(x)                
        return tuple(outs)

    def forward1(self, x, resnet_xs):
        """Forward function."""
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
            if i < len(self.res_layers) - 1:
                cbfuse_layer = getattr(self, self.cbfuse_layers[i])
                x = [x, *[resnet_x[i] for resnet_x in resnet_outs[i:]]]
                x = cbfuse_layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)

    def forward(self, x, resnet_xs):
        """Forward function."""
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
                x = [x, *[resnet_x[i] for resnet_x in resnet_outs[i:]]]
                x = cbfuse_layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)