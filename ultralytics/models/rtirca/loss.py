from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.models.yolo.model import YOLO
from ultralytics.utils.loss import v8DetectionLoss
from .activation_hooks import ActivationHooks


class MLKDLoss(v8DetectionLoss):
    """Multi-Level Knowledge Distillation (MLKD)"""

    def __init__(
        self,
        model: torch.nn.Module,
    ):
        """Initialize the MLKDLoss module."""
        super().__init__(model)

        self.alpha = model.alpha
        self.beta = model.beta
        self.gamma = model.gamma
        self.temperature = model.temperature
        self.has_rev = model.has_rev
        self.has_irca = model.has_irca
        self.has_mut = model.has_mut

        self.activation_layers = model.activation_layers
        self.student_channels = model.student_channels
        self.teacher_channels = model.teacher_channels

        self.device = next(model.parameters()).device

        self.student_model = model
        self.student_activation_hooks = ActivationHooks()
        self.student_activations = self.student_activation_hooks.activations
        self.student_activation_hooks.register_hooks(self.student_model.model, self.activation_layers)

        self.teacher_model = YOLO(model.teacher_ckpt)
        self.teacher_model.eval()
        self.teacher_activation_hooks = ActivationHooks()
        self.teacher_activations = self.teacher_activation_hooks.activations
        self.teacher_activation_hooks.register_hooks(self.teacher_model.model, self.activation_layers)

        # Get the model's dtype to ensure consistency
        model_dtype = next(model.parameters()).dtype

        # relevance loss modules
        if self.has_rev:
            self.l_rev_modules = [
                Attention_Loss2(self.student_channels[idx], self.teacher_channels[idx], self.alpha, self.temperature)
                .to(self.device)
                .to(dtype=model_dtype)
                .train()
                for idx in self.activation_layers
            ]

        # global feature distillation loss modules
        if self.has_irca:
            self.l_irca_modules = [
                GC_FocalModulationLoss(self.student_channels[idx], self.teacher_channels[idx], self.beta)
                .to(self.device)
                .to(dtype=model_dtype)
                .train()
                for idx in self.activation_layers
            ]

        # mutual information loss modules
        if self.has_mut:
            self.l_mut_modules = [
                MINE9(self.student_channels[idx], self.teacher_channels[idx], self.gamma)
                .to(self.device)
                .to(dtype=model_dtype)
                .train()
                for idx in self.activation_layers
            ]

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the total loss for the student model.

        Args:
            preds (Any): Predictions from the student model.
            batch (dict[str, torch.Tensor]): Input batch data.

        Returns:
            (tuple[torch.Tensor, torch.Tensor]): Total loss and detached loss.
        """
        # detection loss
        l_det, l_det_detached = super().__call__(preds, batch)

        # During validation after each training batch, the framework performs a forward pass, initializes the criterion, and then computes the loss.
        # This may result in empty student_activations, leading to incorrect loss calculations.
        # Thus, if student_activations is empty, a forward pass is performed to populate it.
        # This doubles the forward passes and slows down inference in validation.
        # TODO: Initialize the criterion before the forward pass during validation.
        if not self.student_activations:
            _ = self.student_model(batch["img"])

        # teacher forward pass without gradient
        with torch.no_grad():
            _ = self.teacher_model(batch["img"])

        # intermediate layer distillation loss, including:
        # relevance loss, global feature distillation loss, and mutual information loss
        l_rev_irca_mut = torch.zeros(3, device=self.device)

        # Loop over each activation layer index
        for i in range(len(self.activation_layers)):
            student_activation = self.student_activations[f"{self.activation_layers[i]}"]
            teacher_activation = self.teacher_activations[f"{self.activation_layers[i]}"].clone().detach()

            # If relevance loss is enabled, compute the relevance loss
            if self.has_rev:
                l_rev_irca_mut[0] += self.l_rev_modules[i](student_activation, teacher_activation)
            # If global feature distillation loss is enabled, compute the global feature distillation loss
            if self.has_irca:
                l_rev_irca_mut[1] += self.l_irca_modules[i](student_activation, teacher_activation)
            # If mutual information loss is enabled, compute the mutual information loss
            if self.has_mut:
                l_rev_irca_mut[2] += self.l_mut_modules[i](student_activation, teacher_activation)

        batch_size = batch["img"].shape[0]
        l_total = torch.cat([l_det, l_rev_irca_mut * batch_size], dim=0)
        l_total_detached = torch.cat([l_det_detached, l_rev_irca_mut.detach()], dim=0)

        # self.student_activation_hooks.remove_hooks()
        # self.teacher_activation_hooks.remove_hooks()

        return l_total, l_total_detached


class Attention_Loss2(nn.Module):
    def __init__(self, in_channel, out_channel, weight=1e-3, temp=0.5):
        super().__init__()
        self.temp = temp
        self.conv = nn.Conv2d(in_channel, out_channel, 1, 1, 0)
        self.pool_spatial = nn.AvgPool2d(4, 4, 0)
        self.pool_channel = nn.AvgPool1d(2, 2, 0)
        self.weight = weight

    def forward(self, s, t):
        s = self.conv(s)

        channel_map = self._get_channel_attention(s, t, self.temp).unsqueeze(2).unsqueeze(2)
        spatial_map = self._get_spatial_attention(s, t, self.temp).unsqueeze(1)

        loss = torch.sum(channel_map * spatial_map * ((s - t) ** 2))
        return loss * self.weight

    def _get_channel_attention(self, s, t, temp=0.5):
        pool_s = self.pool_spatial(s).view(s.shape[0], s.shape[1], -1).unsqueeze(2)
        pool_t = self.pool_spatial(t).view(t.shape[0], t.shape[1], -1).unsqueeze(3)

        channel_map = F.softmax(torch.matmul(pool_s, pool_t).squeeze() / temp, dim=-1)
        return channel_map

    def _get_spatial_attention(self, s, t, temp=0.5):
        ns, c, h, w = s.shape
        pool_s = self.pool_channel(s.permute(0, 2, 3, 1).view(ns, h * w, -1)).view(ns, h, w, -1).unsqueeze(3)
        pool_t = self.pool_channel(t.permute(0, 2, 3, 1).view(ns, h * w, -1)).view(ns, h, w, -1).unsqueeze(4)

        spatial_map = torch.matmul(pool_s, pool_t).squeeze() / temp
        spatial_map = F.softmax(spatial_map.view(ns, h * w), dim=-1).view(ns, h, w)
        return spatial_map


class GC_FocalModulation(nn.Module):
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


class GC_FocalModulationLoss(nn.Module):
    def __init__(self, in_channel, out_channel, weight=1e-8):
        super().__init__()
        self.model_s = GC_FocalModulation(out_channel)
        self.model_t = GC_FocalModulation(out_channel)
        self.conv = nn.Conv2d(in_channel, out_channel, 1, 1, 0)
        self.weight = weight

    def forward(self, s, t):
        s = self.conv(s)

        s = self.model_s(s)
        t = self.model_t(t)

        return F.mse_loss(s, t, reduction="sum") * self.weight


class MINE9(nn.Module):
    """Mutual Information Neural Estimation"""

    def __init__(self, in_channel, out_channel, weight=1e-6, query_dim=512):
        super().__init__()
        if out_channel == 256:
            resolution = 80
        elif out_channel == 512:
            resolution = 40
        elif out_channel == 1024:
            resolution = 20
        else:
            # Ensure resolution is at least 10
            resolution = max(10, int(20480 / out_channel))
            # Optional: Add warning
            # import warnings
            # warnings.warn(f"out_channel={out_channel} not in predefined values, using calculated resolution: {resolution}")
        self.weight = weight
        self.conv = nn.Conv2d(in_channel, out_channel, 1, 1, 0)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channel * 2, out_channel, 1, 1, 0),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
        )
        self.query = QueryExtractor2(resolution, query_dim)
        self.mlp = MLP2(query_dim)

    def forward(self, x, y):
        x = self.conv(x)

        # joint probabilities
        joint = torch.cat((x, y), dim=1)

        # marginal probabilities
        marginal_x = torch.cat((x, torch.zeros_like(y)), dim=1)
        marginal_y = torch.cat((torch.zeros_like(x), y), dim=1)

        # Forward through the network
        t = self.mlp(self.query(self.conv2(joint))).squeeze(1)
        et_x = self.mlp(self.query(self.conv2(marginal_x))).squeeze(1)
        et_y = self.mlp(self.query(self.conv2(marginal_y))).squeeze(1)

        # Compute the loss
        mi_loss = 1 / (
            F.kl_div(torch.log_softmax(t, dim=-1), torch.softmax(et_x * et_y, dim=-1), reduction="batchmean") + 1e-8
        )

        return mi_loss * self.weight


class QueryExtractor2(nn.Module):
    def __init__(self, resolution, query_dim=256):
        super(QueryExtractor2, self).__init__()
        # Declare but don't define linear layers
        self.query_extraction = None
        self.key_extraction = None  # Add key matrix
        self.softmax = nn.Softmax(dim=-1)
        self.scale = query_dim**-0.5  # Scaling factor
        self.query_dim = query_dim
        # self.resolution = resolution
        self._initialized = False

    def forward(self, x):
        b, c, h, w = x.shape
        
        # Check if already initialized
        if not self._initialized:
            # Initialize linear layers based on actual input dimensions
            input_dim = h * w
            self.query_extraction = nn.Linear(input_dim, self.query_dim).to(device=x.device, dtype=x.dtype)
            self.key_extraction = nn.Linear(input_dim, self.query_dim).to(device=x.device, dtype=x.dtype)
            
            # Manually register modules to the current module
            self.add_module('query_extraction', self.query_extraction)
            self.add_module('key_extraction', self.key_extraction)
            
            self._initialized = True

        x = x.view(b, c, h * w)
        queries = self.query_extraction(x)
        keys = self.key_extraction(x)  # Generate keys

        # Scaled dot-product attention mechanism
        attention_map = torch.matmul(queries, keys.transpose(1, 2)) * self.scale

        attention_weight = self.softmax(torch.sum(attention_map, dim=-1)).unsqueeze(-1)
        total_query = torch.sum(attention_weight * queries, dim=1)
        return total_query


class MLP2(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        dim2 = input_dim // 2
        dim3 = input_dim // 4
        dim4 = input_dim // 8
        dim5 = input_dim // 16
        self.layers = nn.Sequential(
            AttentionMLP(input_dim, dim2, (input_dim + dim2) // 2),
            nn.LayerNorm(dim2),
            nn.ReLU(),
            AttentionMLP(dim2, dim2, dim2),
            nn.LayerNorm(dim2),
            nn.ReLU(),
            AttentionMLP(dim2, dim3, (dim3 + dim2) // 2),
            nn.LayerNorm(dim3),
            nn.ReLU(),
            AttentionMLP(dim3, dim3, dim3),
            nn.LayerNorm(dim3),
            nn.ReLU(),
            AttentionMLP(dim3, dim4, (dim3 + dim4) // 2),
            nn.LayerNorm(dim4),
            nn.ReLU(),
            AttentionMLP(dim4, dim4, dim4),
            nn.LayerNorm(dim4),
            nn.ReLU(),
            AttentionMLP(dim4, dim5, (dim3 + dim2) // 2),
            nn.LayerNorm(dim5),
            nn.ReLU(),
            AttentionMLP(dim5, dim5, dim5),
            nn.LayerNorm(dim5),
            nn.ReLU(),
            nn.Linear(dim5, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.layers(x)


class AttentionMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, dropout_rate=0.1):
        super().__init__()
        self.attention_fc = nn.Linear(input_dim, input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Attention weights
        attention_weights = F.softmax(self.attention_fc(x), dim=-1)
        # Weighted features
        weighted_x = x * attention_weights
        # MLP processing
        x = F.relu(self.ln(self.fc1(weighted_x)))
        x = self.dropout(x)  # Apply dropout
        x = self.fc2(x)
        return x
