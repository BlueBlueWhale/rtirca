import torch
import torch.nn as nn


class IRCA(nn.Module):
    def __init__(self, in_channels, scale=16, k_size=None):
        super().__init__()
        if k_size is None:
            k_size = [3, 5, 7]
        self.k_size = k_size
        self.in_channels = in_channels
        self.out_channels = self.in_channels // scale

        self.Conv_key = nn.Conv2d(self.in_channels, 1, 1)
        self.SoftMax = nn.Softmax(dim=1)

        self.Conv_value = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, 1),
            nn.LayerNorm([self.out_channels, 1, 1]),
            nn.ReLU(),
            nn.Conv2d(self.out_channels, self.in_channels, 1),
        )

        self.conv = nn.Conv2d(self.in_channels, self.in_channels + len(k_size) + 1, 1, 1, 0)
        self.focal_layers = nn.ModuleList()
        for k in range(len(k_size)):
            self.focal_layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        in_channels,
                        kernel_size=k_size[k],
                        stride=1,
                        groups=in_channels,
                        padding=k_size[k] // 2,
                        bias=False,
                    ),
                    nn.GELU(),
                )
            )
        self.pooling = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        b, c, h, w = x.size()
        x = self._contextAggregation(x)
        # key -> [b, 1, H, W] -> [b, 1, H*W] ->  [b, H*W, 1]
        key = self.SoftMax(self.Conv_key(x).view(b, 1, -1).permute(0, 2, 1).view(b, -1, 1).contiguous())
        query = x.view(b, c, h * w)
        # [b, c, h*w] * [b, H*W, 1]
        concate_QK = torch.matmul(query, key)
        concate_QK = concate_QK.view(b, c, 1, 1).contiguous()
        value = self.Conv_value(concate_QK)
        out = x + value
        return out

    def _contextAggregation(self, x):
        focal_level = len(self.k_size)
        C = x.shape[1]
        x = self.conv(x)
        ctx, gates = torch.split(x, (C, focal_level + 1), 1)
        ctx_all = 0
        for l in range(focal_level):
            ctx = self.focal_layers[l](ctx)
            ctx_all = ctx_all + ctx * gates[:, l : l + 1]
        ctx_global = self.pooling(ctx)
        ctx_all = ctx_all + ctx_global * gates[:, focal_level:]
        return ctx_all
