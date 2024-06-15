# Copyright (c) 2022, Zikang Zhou. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch_geometric.data import Data


# 定义一个用于时间序列图数据的类
class TemporalData(Data):

    # 类的初始化方法
    def __init__(self,
                 x: Optional[torch.Tensor] = None,  # 节点特征矩阵
                 positions: Optional[torch.Tensor] = None,  # 节点的位置信息
                 edge_index: Optional[torch.Tensor] = None,  # 边索引，指明图中各边的连接关系
                 edge_attrs: Optional[List[torch.Tensor]] = None,  # 边的属性列表，可用于存储每个时间步的边属性
                 y: Optional[torch.Tensor] = None,  # 目标值，用于监督学习
                 num_nodes: Optional[int] = None,  # 图中节点的数量
                 padding_mask: Optional[torch.Tensor] = None,  # 填充掩码，用于标记无效的节点或时间步
                 bos_mask: Optional[torch.Tensor] = None,  # 开始标记掩码，可能用于标记序列开始的位置
                 rotate_angles: Optional[torch.Tensor] = None,  # 节点的旋转角度，可能用于调整节点位置
                 lane_vectors: Optional[torch.Tensor] = None,  # 车道向量，描述车道的方向和长度
                 is_intersections: Optional[torch.Tensor] = None,  # 标记是否在交叉口的布尔张量
                 turn_directions: Optional[torch.Tensor] = None,  # 车道的转向方向
                 traffic_controls: Optional[torch.Tensor] = None,  # 交通控制信息，如红绿灯或停车标志
                 lane_actor_index: Optional[torch.Tensor] = None,  # 描述车道和节点之间关系的索引
                 lane_actor_vectors: Optional[torch.Tensor] = None,  # 描述车道和节点之间相对位置的向量
                 seq_id: Optional[int] = None,  # 序列ID，用于标识不同的数据序列
                 **kwargs) -> None:  # 其他关键字参数
        if x is None:
            super(TemporalData, self).__init__()
            return
        # 调用父类的初始化方法，并传入相应的参数
        super(TemporalData, self).__init__(x=x, positions=positions, edge_index=edge_index, y=y, num_nodes=num_nodes,
                                           padding_mask=padding_mask, bos_mask=bos_mask, rotate_angles=rotate_angles,
                                           lane_vectors=lane_vectors, is_intersections=is_intersections,
                                           turn_directions=turn_directions, traffic_controls=traffic_controls,
                                           lane_actor_index=lane_actor_index, lane_actor_vectors=lane_actor_vectors,
                                           seq_id=seq_id, **kwargs)
        if edge_attrs is not None:
            # 如果提供了边的属性，则为每个时间步创建一个特定的属性
            for t in range(self.x.size(1)):
                self[f'edge_attr_{t}'] = edge_attrs[t]

    # 定义如何增加特定键的值
    def __inc__(self, key, value):
        if key == 'lane_actor_index':
            # 特殊处理'lane_actor_index'的增加，返回一个张量，指明如何增加
            return torch.tensor([[self['lane_vectors'].size(0)], [self.num_nodes]])
        else:
            # 对于其他键，调用父类的方法
            return super().__inc__(key, value)
    



# 定义一个根据边的距离来过滤边的类
class DistanceDropEdge(object):

    # 类的初始化方法
    def __init__(self, max_distance: Optional[float] = None) -> None:
        # 最大距离阈值，超过这个距离的边将被移除。如果为None，则不移除任何边。
        self.max_distance = max_distance

    # 当实例被调用作为函数时执行的操作
    def __call__(self,
                 edge_index: torch.Tensor,  # 边的索引，通常是一个2xN的张量，N是边的数量
                 edge_attr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:  # 边的属性，可以是边的长度或其他特征
        # 如果没有设置最大距离，则不过滤任何边，直接返回原始的边索引和属性
        if self.max_distance is None:
            return edge_index, edge_attr
        
        # 分解边的索引为起点和终点
        row, col = edge_index
        # 计算每条边的属性的L2范数，并与最大距离进行比较，生成一个布尔掩码
        # 这个掩码表示哪些边的属性小于最大距离，即应该被保留的边
        mask = torch.norm(edge_attr, p=2, dim=-1) < self.max_distance
        # 使用掩码过滤边的索引，只保留满足条件的边
        edge_index = torch.stack([row[mask], col[mask]], dim=0)
        # 同样使用掩码过滤边的属性
        edge_attr = edge_attr[mask]
        # 返回过滤后的边索引和属性
        return edge_index, edge_attr



# 定义权重初始化函数，适用于多种类型的神经网络层
def init_weights(m: nn.Module) -> None:
    # 如果是全连接层（Linear）
    if isinstance(m, nn.Linear):
        # 使用Xavier均匀初始化权重
        nn.init.xavier_uniform_(m.weight)
        # 如果有偏置项，则将偏置初始化为0
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    # 如果是卷积层（1D、2D或3D）
    elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        # 计算fan_in和fan_out，用于初始化范围的计算
        fan_in = m.in_channels / m.groups
        fan_out = m.out_channels / m.groups
        bound = (6.0 / (fan_in + fan_out)) ** 0.5
        # 使用均匀分布初始化权重
        nn.init.uniform_(m.weight, -bound, bound)
        # 如果有偏置项，则将偏置初始化为0
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    # 如果是嵌入层（Embedding）
    elif isinstance(m, nn.Embedding):
        # 使用正态分布初始化权重
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
    # 如果是批量归一化层（1D、2D或3D）
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        # 将权重初始化为1，偏置初始化为0
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    # 如果是层归一化（LayerNorm）
    elif isinstance(m, nn.LayerNorm):
        # 将权重初始化为1，偏置初始化为0
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    # 如果是多头注意力机制（MultiheadAttention）
    elif isinstance(m, nn.MultiheadAttention):
        # 初始化投影权重和偏置
        if m.in_proj_weight is not None:
            # 使用均匀分布初始化in_proj_weight
            fan_in = m.embed_dim
            fan_out = m.embed_dim
            bound = (6.0 / (fan_in + fan_out)) ** 0.5
            nn.init.uniform_(m.in_proj_weight, -bound, bound)
        else:
            # 使用Xavier均匀初始化q、k、v投影权重
            nn.init.xavier_uniform_(m.q_proj_weight)
            nn.init.xavier_uniform_(m.k_proj_weight)
            nn.init.xavier_uniform_(m.v_proj_weight)
        # 初始化偏置项
        if m.in_proj_bias is not None:
            nn.init.zeros_(m.in_proj_bias)
        nn.init.xavier_uniform_(m.out_proj.weight)
        if m.out_proj.bias is not None:
            nn.init.zeros_(m.out_proj.bias)
        # 特殊偏置项的初始化
        if m.bias_k is not None:
            nn.init.normal_(m.bias_k, mean=0.0, std=0.02)
        if m.bias_v is not None:
            nn.init.normal_(m.bias_v, mean=0.0, std=0.02)
    # 如果是LSTM层
    elif isinstance(m, nn.LSTM):
        # 初始化LSTM层的权重和偏置
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                # 对输入到隐藏层的权重进行Xavier均匀初始化
                for ih in param.chunk(4, 0):
                    nn.init.xavier_uniform_(ih)
            elif 'weight_hh' in name:
                # 对隐藏层到隐藏层的权重进行正交初始化
                for hh in param.chunk(4, 0):
                    nn.init.orthogonal_(hh)
            elif 'bias_ih' in name:
                # 初始化偏置项为0
                nn.init.zeros_(param)
            elif 'bias_hh' in name:
                # 初始化隐藏层偏置项，其中忘记门偏置初始化为1
                nn.init.zeros_(param)
                nn.init.ones_(param.chunk(4, 0)[1])
    # 如果是GRU层
    elif isinstance(m, nn.GRU):
        # 初始化GRU层的权重和偏置
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                # 对输入到隐藏层的权重进行Xavier均匀初始化
                for ih in param.chunk(3, 0):
                    nn.init.xavier_uniform_(ih)
            elif 'weight_hh' in name:
                # 对隐藏层到隐藏层的权重进行正交初始化
                for hh in param.chunk(3, 0):
                    nn.init.orthogonal_(hh)
            elif 'bias_ih' in name:
                # 初始化偏置项为0
                nn.init.zeros_(param)
            elif 'bias_hh' in name:
                # 初始化隐藏层偏置项为0
                nn.init.zeros_(param)

