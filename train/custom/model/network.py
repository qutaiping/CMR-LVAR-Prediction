import numpy as np
import torch
import torch.nn as nn
from .registry import NETWORKS, HEADS, BACKBONES
from custom.model.utils import build_backbone
from einops import rearrange, repeat

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution."""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.dropout = nn.Dropout3d(0.2)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding."""
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )
def make_res_layer(inplanes, planes, blocks, stride=1):
    downsample = nn.Sequential(conv1x1(inplanes, planes, stride), nn.BatchNorm3d(planes),)

    layers = []
    layers.append(BasicBlock(inplanes, planes, stride, downsample))
    for _ in range(1, blocks):
        layers.append(BasicBlock(planes, planes))

    return nn.Sequential(*layers)

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, kernel_size=3):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=int(kernel_size / 2)),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, dilation=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, input):
        return self.conv(input)
    
@BACKBONES.register_module
class ResUnet_enc(nn.Module):
    def __init__(self, in_ch, channels=16, blocks=3):
        super(ResUnet_enc, self).__init__()

        self.in_conv = DoubleConv(in_ch, channels, stride=2, kernel_size=3)
        self.layer1 = make_res_layer(channels, channels * 2, blocks, stride=2)
        self.layer2 = make_res_layer(channels * 2, channels * 4, blocks, stride=2)
        self.layer3 = make_res_layer(channels * 4, channels * 8, blocks, stride=2)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def forward(self, input):
        c1 = self.in_conv(input)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.avgpool(c4)
        x = torch.flatten(c5, 1)
        return x
    
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

@BACKBONES.register_module
class CNNTrans(nn.Module):
    def __init__(self, in_ch=1, channels=32, blocks=3):
        super(CNNTrans, self).__init__()
        self.CNN = ResUnet_enc(in_ch=in_ch, channels=channels, blocks=blocks)
        self.CNN.fc = nn.Sequential(nn.Linear(channels * 8, 128))

        self.transformer = Transformer(dim=128, depth=3, heads=4, dim_head=32, mlp_dim=256, dropout=0.1)
        self.mlp_head_cls = nn.Sequential(
            nn.LayerNorm(128),
            nn.Linear(128, 1)
        )
        self.deep_sup_cls = nn.Linear(128, 1)


    def forward(self, img):
        img_cur = img.detach()
        x = self.CNN(img_cur)
        x = self.CNN.fc(x)
        deep_sup_cls = self.deep_sup_cls(x)
        x = self.transformer(x.unsqueeze(1))
        x_cls = self.mlp_head_cls(x)
        return x_cls, deep_sup_cls

# @HEADS.register_module
# class Classification_Head(nn.Module):
#     def __init__(self,):
#         super(Classification_Head, self).__init__()
#         self.loss_bce_func = torch.nn.BCEWithLogitsLoss(reduction='none')


#     def forward(self, inputs):
#         pass


#     def loss(self, inputs, targets):

#         with torch.no_grad():
#             targets = targets.view(-1, 1)
        
#         outs_cls, deep_sup_cls = inputs[0], inputs[1]
#         loss_cls = self.loss_bce_func(outs_cls.float(), targets.float())
#         loss_cls = loss_cls.mean()
#         loss_cls_deep_sup = self.loss_bce_func(deep_sup_cls.float(), targets.float()) * 0.5
#         loss_cls_deep_sup = loss_cls_deep_sup.mean()


#         return {
#                 "loss_ce": loss_cls,
#                 "loss_cls_deep_sup": loss_cls_deep_sup,
#                 }
    
#     def _smooth_tgt(self, tgt, iter=3):
#         tgt = tgt[None, None, :, :, :]
#         for _ in range(iter):
#             tgt = self.avg_conv(tgt)
#         scale_ratio = 1 / tgt.max()
#         tgt = torch.clamp(tgt, 0, 1)[0, 0] * scale_ratio
#         return tgt

@NETWORKS.register_module
class Classification_Network(nn.Module):
    def __init__(self, backbone, head, apply_sync_batchnorm=False, pipeline=[], train_cfg=None, test_cfg=None):
        super(Classification_Network, self).__init__()
        self.backbone = build_backbone(backbone)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self._show_count = 0

        if apply_sync_batchnorm:
            self._apply_sync_batchnorm()

    @torch.jit.export
    def forward_test(self, img):
        outs_cls, _ = self.backbone(img)
        outs_cls = outs_cls.view(outs_cls.shape[0] * outs_cls.shape[1], -1)
        
        pred = torch.sigmoid(outs_cls)
        return pred

    def single_test(self, img, gt):
        pred_cls = self.forward_test(img)
        return pred_cls, gt

    def _apply_sync_batchnorm(self):
        self.backbone = nn.SyncBatchNorm.convert_sync_batchnorm(self.backbone)
        # self.head = nn.SyncBatchNorm.convert_sync_batchnorm(self.head)