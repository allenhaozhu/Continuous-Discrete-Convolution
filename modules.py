import numpy as np
import math
import os.path as osp
from functools import partial
from typing import Any, Callable, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from torch_scatter import scatter_max, scatter_min, scatter_mean, scatter_sum

from torch_sparse import SparseTensor, set_diag
from torch_geometric.data import Data
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptTensor, PairOptTensor, PairTensor
from torch_geometric.utils import add_self_loops, remove_self_loops
import torch_geometric.transforms as T
from torch_geometric.nn import MLP, fps, global_max_pool, global_mean_pool, radius
from torch_geometric.nn.pool import avg_pool, max_pool

def kaiming_uniform(tensor, size):
    fan = 1
    for i in range(1, len(size)):
        fan *= size[i]
    gain = math.sqrt(2.0 / (1 + math.sqrt(5) ** 2))
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)

class WeightNet(nn.Module):
    def __init__(self, l: int, kernel_channels: list[int], use_attention: bool = False):
        super(WeightNet, self).__init__()

        self.l = l
        self.kernel_channels = kernel_channels
        self.use_attention = use_attention

        self.Ws = nn.ParameterList()
        self.bs = nn.ParameterList()

        for i, channels in enumerate(kernel_channels):
            if i == 0:
                self.Ws.append(torch.nn.Parameter(torch.empty(l, 3 + 3 + 1, channels)))
                self.bs.append(torch.nn.Parameter(torch.empty(l, channels)))
            else:
                self.Ws.append(torch.nn.Parameter(torch.empty(l, kernel_channels[i-1], channels)))
                self.bs.append(torch.nn.Parameter(torch.empty(l, channels)))

        self.relu = nn.LeakyReLU(0.2)

        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(kernel_channels[-1], 1),
                nn.Sigmoid()
            )

    def reset_parameters(self):
        for i, channels in enumerate(self.kernel_channels):
            if i == 0:
                kaiming_uniform(self.Ws[0].data, size=[self.l, 3 + 3 + 1, channels])
            else:
                kaiming_uniform(self.Ws[i].data, size=[self.l, self.kernel_channels[i-1], channels])
            self.bs[i].data.fill_(0.0)

    def forward(self, input, idx):
        for i in range(len(self.kernel_channels)):
            W = torch.index_select(self.Ws[i], 0, idx)
            b = torch.index_select(self.bs[i], 0, idx)
            if i == 0:
                weight = self.relu(torch.bmm(input.unsqueeze(1), W).squeeze(1) + b)
            else:
                weight = self.relu(torch.bmm(weight.unsqueeze(1), W).squeeze(1) + b)

        if self.use_attention:
            # Add attention weights
            att_weights = self.attention(weight)
            weight = weight * att_weights
            
        return weight

class EnhancedWeightNet(nn.Module):
    def __init__(self, l: int, hidden_dim: int = 32, use_gating: bool = True):
        super().__init__()
        self.l = l
        
        # Position-wise encoding
        self.pos_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU()
        )
        
        # Orientation encoding
        self.ori_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU()
        )
        
        # Distance encoding
        self.dist_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU()
        )
        
        # Sequence-aware gating
        self.seq_gate = nn.Parameter(torch.ones(l, hidden_dim))
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Optional gating mechanism
        self.use_gating = use_gating
        if use_gating:
            self.gate = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Sigmoid()
            )

    def forward(self, input, seq_idx):
        # Split input into components
        pos, ori, dist = torch.split(input, [3, 3, 1], dim=1)
        
        # Encode each component separately
        pos_feat = self.pos_encoder(pos)
        ori_feat = self.ori_encoder(ori)
        dist_feat = self.dist_encoder(dist)
        
        # Combine features
        combined = torch.cat([pos_feat, ori_feat, dist_feat], dim=1)
        weight = self.fusion(combined)
        
        # Apply sequence-aware gating
        seq_weights = torch.index_select(self.seq_gate, 0, seq_idx)
        weight = weight * seq_weights
        
        # Optional gating mechanism
        if self.use_gating:
            gate = self.gate(weight)
            weight = weight * gate
            
        return weight

    def reset_parameters(self):
        # Initialize linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Initialize sequence gate
        nn.init.ones_(self.seq_gate)

class CDConv(MessagePassing):
    def __init__(self, r: float, l: float, kernel_channels: list[int], in_channels: int, out_channels: int, add_self_loops: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'sum')
        super().__init__(**kwargs)
        self.r = r
        self.l = l
        self.kernel_channels = kernel_channels
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.WeightNet = WeightNet(l, kernel_channels)
        #self.WeightNet = EnhancedWeightNet(l, kernel_channels[0])
        self.W = torch.nn.Parameter(torch.empty(kernel_channels[-1] * in_channels, out_channels))

        self.add_self_loops = add_self_loops

        self.reset_parameters()

    def reset_parameters(self):
        self.WeightNet.reset_parameters()
        kaiming_uniform(self.W.data, size=[self.kernel_channels * self.in_channels, self.out_channels])

    def forward(self, x: OptTensor, pos: Tensor, seq: Tensor, ori: Tensor, batch: Tensor) -> Tensor:
        row, col = radius(pos, pos, self.r, batch, batch, max_num_neighbors=9999)
        edge_index = torch.stack([col, row], dim=0)

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index, num_nodes=min(pos.size(0), pos.size(0)))

            elif isinstance(edge_index, SparseTensor):
                edge_index = set_diag(edge_index)

        out = self.propagate(edge_index, x=(x, None), pos=(pos, pos), seq=(seq, seq), ori=(ori.reshape((-1, 9)), ori.reshape((-1, 9))), size=None)
        out = torch.matmul(out, self.W)

        return out

    def message(self, x_j: Optional[Tensor], pos_i: Tensor, pos_j: Tensor, 
                seq_i: Tensor, seq_j: Tensor, ori_i: Tensor, ori_j: Tensor) -> Tensor:
        # Compute spatial displacement (reuse tensors)
        pos_diff = pos_j - pos_i
        distance = torch.norm(pos_diff, p=2, dim=-1, keepdim=True)
        pos_diff.div_(distance + 1e-9)  # in-place normalization
        
        # Transform position (minimize intermediate tensors)
        ori_i_mat = ori_i.reshape((-1, 3, 3))
        pos_transformed = torch.matmul(ori_i_mat, pos_diff.unsqueeze(2)).squeeze(2)
        
        # Compute orientation (reuse memory)
        ori = torch.sum(ori_i_mat * ori_j.reshape((-1, 3, 3)), dim=2)
        del ori_i_mat  # explicitly free memory
        
        # Prepare base values
        seq_diff = seq_j - seq_i
        s = self.l//2
        base_window = s//4
        
        # Initialize accumulator for final message
        final_msg = None
        
        # Process scales sequentially to save memory
        scales = [1.0, 2.0, 4.0]
        for scale_idx, scale in enumerate(scales):
            # Compute temporal factors
            curr_window = base_window * scale
            normalized_time = seq_diff / curr_window
            normalized_time.clamp_(-1.0, 1.0)  # in-place clamp
            
            # Compute delta features (reuse memory)
            delta = torch.cat([
                pos_transformed,
                ori,
                distance/self.r
            ], dim=1)
            
            # Generate weights
            scale_indices = torch.clamp((seq_diff + s) / scale, min=0, max=self.l-1).to(torch.int64)
            kernel_weight = self.WeightNet(delta, scale_indices.squeeze(1))
            
            # Compute smoothing (minimize operations)
            smooth = torch.exp(-distance/self.r - torch.abs(normalized_time))
            
            # Apply weights and compute message
            msg = torch.matmul((kernel_weight * smooth).unsqueeze(2), x_j.unsqueeze(1))
            
            # Accumulate message (avoid storing all scales)
            if final_msg is None:
                final_msg = msg
            else:
                final_msg.add_(msg)  # in-place addition
                
            # Clean up intermediate tensors
            del kernel_weight, smooth, msg
            
        # Average accumulated messages
        final_msg.div_(len(scales))  # in-place division
        
        # Reshape final output
        return final_msg.reshape((-1, final_msg.size(1) * final_msg.size(2)))

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(r={self.r}, '
                f'l={self.l},'
                f'kernel_channels={self.kernel_channels},'
                f'in_channels={self.in_channels},'
                f'out_channels={self.out_channels})')

class MaxPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, pos, seq, ori, batch):
        idx = torch.div(seq.squeeze(1), 2, rounding_mode='floor')
        idx = torch.cat([idx, idx[-1].view((1,))])

        idx = (idx[0:-1] != idx[1:]).to(torch.float32)
        idx = torch.cumsum(idx, dim=0) - idx
        idx = idx.to(torch.int64)
        x = scatter_max(src=x, index=idx, dim=0)[0]
        pos = scatter_mean(src=pos, index=idx, dim=0)
        seq = scatter_max(src=torch.div(seq, 2, rounding_mode='floor'), index=idx, dim=0)[0]
        ori = scatter_mean(src=ori, index=idx, dim=0)
        ori = torch.nn.functional.normalize(ori, 2, -1)
        batch = scatter_max(src=batch, index=idx, dim=0)[0]

        return x, pos, seq, ori, batch

class AvgPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, pos, seq, ori, batch):
        idx = torch.div(seq.squeeze(1), 2, rounding_mode='floor')
        idx = torch.cat([idx, idx[-1].view((1,))])

        idx = (idx[0:-1] != idx[1:]).to(torch.float32)
        idx = torch.cumsum(idx, dim=0) - idx
        idx = idx.to(torch.int64)
        x = scatter_mean(src=x, index=idx, dim=0)
        pos = scatter_mean(src=pos, index=idx, dim=0)
        seq = scatter_max(src=torch.div(seq, 2, rounding_mode='floor'), index=idx, dim=0)[0]
        ori = scatter_mean(src=ori, index=idx, dim=0)
        ori = torch.nn.functional.normalize(ori, 2, -1)
        batch = scatter_max(src=batch, index=idx, dim=0)[0]

        return x, pos, seq, ori, batch

#Time Series Multi-scale Convolution
class TSMConv(MessagePassing):
    def __init__(self, r: float, l: float, kernel_channels: list[int], in_channels: int, out_channels: int, 
                 scales: list[float] = [1.0, 2.0, 4.0], base_window: float = 1.0, base_radius: float = 1.0,
                 add_self_loops: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'sum')
        super().__init__(**kwargs)
        self.r = r
        self.l = l
        self.kernel_channels = kernel_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Pre-compute scale factors for efficiency
        self.register_buffer('scale_factors', torch.tensor(scales))
        self.register_buffer('scale_windows', torch.tensor(scales) * base_window)
        self.register_buffer('scale_radii', torch.tensor(scales) * base_radius)
        
        # Simplified network for pattern transformation
        self.pattern_transform = nn.Sequential(
            nn.Linear(4, kernel_channels[0]),
            nn.ReLU(),
            nn.Linear(kernel_channels[0], kernel_channels[-1])
        )

        self.W = torch.nn.Parameter(torch.empty(kernel_channels[-1] * in_channels, out_channels))
        self.add_self_loops = add_self_loops
        self.reset_parameters()

    def reset_parameters(self):
        kaiming_uniform(self.W.data, size=[self.kernel_channels[-1] * self.in_channels, self.out_channels])
        # Initialize scale transforms
        for layer in self.pattern_transform:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight)
                if layer.bias is not None:
                    layer.bias.data.zero_()
                        
    def forward(self, x: OptTensor, pos: Tensor, seq: Tensor, ori: Tensor, batch: Tensor) -> Tensor:
        row, col = radius(pos, pos, self.r, batch, batch, max_num_neighbors=9999)
        edge_index = torch.stack([col, row], dim=0)

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index, num_nodes=min(pos.size(0), pos.size(0)))

            elif isinstance(edge_index, SparseTensor):
                edge_index = set_diag(edge_index)

        out = self.propagate(edge_index, x=(x, None), pos=(pos, pos), seq=(seq, seq), ori=(ori.reshape((-1, 9)), ori.reshape((-1, 9))), size=None)
        out = torch.matmul(out, self.W)

        return out

    def message(self, x_j: Optional[Tensor], pos_i: Tensor, pos_j: Tensor, seq_i: Tensor, seq_j: Tensor) -> Tensor:
        # Pre-compute shared values
        seq_dist = seq_j - seq_i  # [N, 1]
        spatial_dist = pos_j - pos_i  # [N, 3]
        spatial_norm = torch.norm(spatial_dist, p=2, dim=-1, keepdim=True)
        normalized_directions = spatial_dist / (spatial_norm + 1e-6)
        
        # Initialize accumulated messages
        accumulated_messages = torch.zeros(x_j.size(0), self.kernel_channels[-1] * self.in_channels, 
                                        device=x_j.device)
        
        # Process each scale separately to save memory
        for scale_idx in range(len(self.scale_factors)):
            # Compute weights for current scale
            norm_time = seq_dist / self.scale_windows[scale_idx]
            time_weight = torch.exp(-norm_time.pow(2))
            
            spatial_weight = torch.exp(-spatial_norm / self.scale_radii[scale_idx])
            
            # Combine weights
            weight = time_weight * spatial_weight  # [N, 1]
            
            # Prepare pattern features for current scale
            pattern_features = torch.cat([
                normalized_directions,  # [N, 3]
                spatial_norm,  # [N, 1]
            ], dim=-1)
            
            # Transform pattern
            scale_features = self.pattern_transform(pattern_features)  # [N, kernel_channels[-1]]
            
            # Weight features
            weighted_features = scale_features * weight
            
            # Compute message for current scale
            msg = torch.matmul(weighted_features.unsqueeze(2), x_j.unsqueeze(1))
            msg = msg.reshape(msg.size(0), -1)  # Flatten to [N, kernel_channels[-1] * in_channels]
            
            # Accumulate messages
            accumulated_messages += msg
            
        return accumulated_messages
