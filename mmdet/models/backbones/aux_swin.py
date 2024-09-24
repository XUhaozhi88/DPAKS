import torch
from torch import nn
import torch.nn.functional as F

from mmdet.registry import MODELS

from .swin import SwinTransformer

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
class AUXSwinTransformer(SwinTransformer):
    """Auxiliary SwinTransformer backbone.
    在这里我们没有使用和yolov9一样的多层级融合,仅是单层融合
    
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        cbl_arch_settings = (
            [384, 192],
            [768, 384],
            [1536, 768])
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

    def forward(self, x, swin_xs):
        """Forward function."""
        if len(swin_xs) == 4: swin_xs = swin_xs[1:]

        x, hw_shape = self.patch_embed(x)

        if self.use_abs_pos_embed:
            x = x + self.absolute_pos_embed
        x = self.drop_after_pos(x)

        # swin output
        swin_outs = []
        for i, layer_name in enumerate(self.cbl_layers):
            cbl_layer = getattr(self, layer_name)
            swin_x = cbl_layer(swin_xs[i])
            swin_outs.append(swin_x)    

        # auxiliary swin output
        outs = []
        for i, stage in enumerate(self.stages):
            x, hw_shape, out, out_hw_shape = stage(x, hw_shape)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                out = norm_layer(out)                
                out = out.view(-1, *out_hw_shape, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                if i < len(self.cbfuse_layers):
                    cbfuse_layer = getattr(self, self.cbfuse_layers[i])
                    out = cbfuse_layer([swin_outs[i], out])
                outs.append(out)

        return tuple(outs)