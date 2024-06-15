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
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.data import Data
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj
from torch_geometric.typing import OptTensor
from torch_geometric.typing import Size
from torch_geometric.utils import softmax
from torch_geometric.utils import subgraph

from models import MultipleInputEmbedding
from models import SingleInputEmbedding
from utils import DistanceDropEdge
from utils import TemporalData
from utils import init_weights


# 定义本地编码器类，用于编码时空图的局部特征
class LocalEncoder(nn.Module):

    # 类的初始化方法
    def __init__(self,
                 historical_steps: int,  # 历史时间步数，表示要考虑的历史信息长度
                 node_dim: int,  # 节点特征的维度
                 edge_dim: int,  # 边特征的维度
                 embed_dim: int,  # 嵌入特征的目标维度
                 num_heads: int = 8,  # 多头注意力机制中的头数，默认为8
                 dropout: float = 0.1,  # Dropout比率，默认为0.1，用于防止过拟合
                 num_temporal_layers: int = 4,  # 时间编码层的数量，默认为4
                 local_radius: float = 50,  # 本地感知范围的半径，默认为50，用于决定哪些节点是相邻的
                 parallel: bool = False) -> None:  # 是否并行处理，默认为False
        super(LocalEncoder, self).__init__()  # 调用父类的构造函数进行初始化
        self.historical_steps = historical_steps  # 保存历史时间步数
        self.parallel = parallel  # 保存是否并行处理的设置

        # 初始化距离阈值边丢弃模块，用于根据本地感知范围过滤边
        self.drop_edge = DistanceDropEdge(local_radius)
        # 初始化自注意力编码器，用于编码节点和边的特征
        self.aa_encoder = AAEncoder(historical_steps=historical_steps,
                                    node_dim=node_dim,
                                    edge_dim=edge_dim,
                                    embed_dim=embed_dim,
                                    num_heads=num_heads,
                                    dropout=dropout,
                                    parallel=parallel)
        # 初始化时间编码器，用于编码时间序列信息
        self.temporal_encoder = TemporalEncoder(historical_steps=historical_steps,
                                                embed_dim=embed_dim,
                                                num_heads=num_heads,
                                                dropout=dropout,
                                                num_layers=num_temporal_layers)
        # 初始化自适应局部编码器，用于进一步提取特征
        self.al_encoder = ALEncoder(node_dim=node_dim,
                                    edge_dim=edge_dim,
                                    embed_dim=embed_dim,
                                    num_heads=num_heads,
                                    dropout=dropout)


    # 定义前向传播过程
    def forward(self, data: TemporalData) -> torch.Tensor:
        # 对每个历史时间步进行遍历
        for t in range(self.historical_steps):
            # 使用subgraph函数基于填充掩码来过滤边索引，保留有效的边
            data[f'edge_index_{t}'], _ = subgraph(subset=~data['padding_mask'][:, t], edge_index=data.edge_index)
            # 计算边属性为相应节点位置的差值
            data[f'edge_attr_{t}'] = \
                data['positions'][data[f'edge_index_{t}'][0], t] - data['positions'][data[f'edge_index_{t}'][1], t]

        # 如果设置为并行处理
        if self.parallel:
            snapshots = [None] * self.historical_steps
            for t in range(self.historical_steps):
                # 使用drop_edge方法基于最大距离过滤边
                edge_index, edge_attr = self.drop_edge(data[f'edge_index_{t}'], data[f'edge_attr_{t}'])
                # 创建每个时间步的Data对象
                snapshots[t] = Data(x=data.x[:, t], edge_index=edge_index, edge_attr=edge_attr,
                                    num_nodes=data.num_nodes)
            # 将多个Data对象合并为一个Batch对象
            batch = Batch.from_data_list(snapshots)
            # 使用自注意力编码器对Batch对象进行编码
            out = self.aa_encoder(x=batch.x, t=None, edge_index=batch.edge_index, edge_attr=batch.edge_attr,
                                bos_mask=data['bos_mask'], rotate_mat=data['rotate_mat'])
            # 重塑输出以匹配时间步、批次大小和特征维度
            out = out.view(self.historical_steps, out.shape[0] // self.historical_steps, -1)
        else:
            # 串行处理的情况
            out = [None] * self.historical_steps
            for t in range(self.historical_steps):
                # 为每个时间步过滤边并编码
                edge_index, edge_attr = self.drop_edge(data[f'edge_index_{t}'], data[f'edge_attr_{t}'])
                out[t] = self.aa_encoder(x=data.x[:, t], t=t, edge_index=edge_index, edge_attr=edge_attr,
                                        bos_mask=data['bos_mask'][:, t], rotate_mat=data['rotate_mat'])
            # 将列表中的编码结果堆叠成张量
            out = torch.stack(out)  # [T, N, D]

        # 使用时间编码器进一步处理编码结果
        out = self.temporal_encoder(x=out, padding_mask=data['padding_mask'][:, : self.historical_steps])
        # 过滤车道和动态实体之间的边
        edge_index, edge_attr = self.drop_edge(data['lane_actor_index'], data['lane_actor_vectors'])
        # 使用自适应局部编码器对最终结果进行编码
        out = self.al_encoder(x=(data['lane_vectors'], out), edge_index=edge_index, edge_attr=edge_attr,
                            is_intersections=data['is_intersections'], turn_directions=data['turn_directions'],
                            traffic_controls=data['traffic_controls'], rotate_mat=data['rotate_mat'])
        # 返回编码后的结果
        return out



# 定义AAEncoder类，继承自MessagePassing
class AAEncoder(MessagePassing):

    # 类的初始化方法
    def __init__(self,
                 historical_steps: int,  # 历史步数，表示考虑的时间范围
                 node_dim: int,  # 节点特征维度
                 edge_dim: int,  # 边特征维度
                 embed_dim: int,  # 嵌入向量维度
                 num_heads: int = 8,  # 多头注意力机制的头数
                 dropout: float = 0.1,  # Dropout比率
                 parallel: bool = False,  # 是否并行处理
                 **kwargs) -> None:  # 其他参数
        super(AAEncoder, self).__init__(aggr='add', node_dim=0, **kwargs)  # 调用父类的构造函数
        self.historical_steps = historical_steps
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.parallel = parallel

        # 初始化用于节点和邻居节点嵌入的线性层
        self.center_embed = SingleInputEmbedding(in_channel=node_dim, out_channel=embed_dim)
        self.nbr_embed = MultipleInputEmbedding(in_channels=[node_dim, edge_dim], out_channel=embed_dim)
        # 初始化用于计算查询向量（Q）、键向量（K）和值向量（V）的线性层
        self.lin_q = nn.Linear(embed_dim, embed_dim)
        self.lin_k = nn.Linear(embed_dim, embed_dim)
        self.lin_v = nn.Linear(embed_dim, embed_dim)
        # 初始化自注意力后的线性层和Dropout
        self.lin_self = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        # 初始化用于残差连接的线性层和Dropout
        self.lin_ih = nn.Linear(embed_dim, embed_dim)
        self.lin_hh = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)
        # 初始化层归一化（LayerNorm）
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        # 初始化多层感知机（MLP）及其Dropout
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout))
        # 初始化序列开始（BOS）标记的参数，并随机初始化
        self.bos_token = nn.Parameter(torch.Tensor(historical_steps, embed_dim))
        nn.init.normal_(self.bos_token, mean=0., std=.02)
        # 应用权重初始化函数
        self.apply(init_weights)


    # 定义前向传播方法
    def forward(self,
                x: torch.Tensor,  # 节点特征矩阵
                t: Optional[int],  # 当前时间步（仅在非并行模式下使用）
                edge_index: Adj,  # 边索引，表示图的连接关系
                edge_attr: torch.Tensor,  # 边的特征或属性
                bos_mask: torch.Tensor,  # 序列开始标记的掩码
                rotate_mat: Optional[torch.Tensor] = None,  # 旋转矩阵，用于坐标旋转
                size: Size = None) -> torch.Tensor:  # 图的大小，可选参数
        # 如果设置为并行处理
        if self.parallel:
            # 如果没有提供旋转矩阵，直接对x进行节点嵌入
            if rotate_mat is None:
                center_embed = self.center_embed(x.view(self.historical_steps, x.shape[0] // self.historical_steps, -1))
            else:
                # 如果提供了旋转矩阵，先将x旋转再进行节点嵌入
                center_embed = self.center_embed(
                    torch.matmul(x.view(self.historical_steps, x.shape[0] // self.historical_steps, -1).unsqueeze(-2),
                                rotate_mat.expand(self.historical_steps, *rotate_mat.shape)).squeeze(-2))
            # 应用BOS标记，如果bos_mask为True，则使用bos_token替换center_embed中的对应元素
            center_embed = torch.where(bos_mask.t().unsqueeze(-1),
                                    self.bos_token.unsqueeze(-2),
                                    center_embed).view(x.shape[0], -1)
        else:
            # 非并行模式下的处理逻辑
            if rotate_mat is None:
                center_embed = self.center_embed(x)
            else:
                center_embed = self.center_embed(torch.bmm(x.unsqueeze(-2), rotate_mat).squeeze(-2))
            # 应用BOS标记
            center_embed = torch.where(bos_mask.unsqueeze(-1), self.bos_token[t], center_embed)
        # 通过自注意力机制更新center_embed
        center_embed = center_embed + self._mha_block(self.norm1(center_embed), x, edge_index, edge_attr, rotate_mat, size)
        # 通过前馈网络进一步处理center_embed
        center_embed = center_embed + self._ff_block(self.norm2(center_embed))
        # 返回经过编码的节点特征
        return center_embed



    # 定义消息传递函数
    def message(self,
                edge_index: Adj,  # 边索引，表示图中的连接关系
                center_embed_i: torch.Tensor,  # 中心节点嵌入
                x_j: torch.Tensor,  # 邻居节点特征
                edge_attr: torch.Tensor,  # 边属性
                rotate_mat: Optional[torch.Tensor],  # 旋转矩阵，用于旋转特征
                index: torch.Tensor,  # 用于聚合操作的索引
                ptr: OptTensor,  # 用于分段聚合的指针
                size_i: Optional[int]) -> torch.Tensor:  # 输入大小
        # 如果没有提供旋转矩阵，直接对邻居节点和边属性进行嵌入
        if rotate_mat is None:
            nbr_embed = self.nbr_embed([x_j, edge_attr])
        else:
            # 如果提供了旋转矩阵并且设置为并行处理，重复旋转矩阵以匹配历史步数
            if self.parallel:
                center_rotate_mat = rotate_mat.repeat(self.historical_steps, 1, 1)[edge_index[1]]
            else:
                # 如果不是并行处理，直接使用旋转矩阵
                center_rotate_mat = rotate_mat[edge_index[1]]
            # 对邻居节点和边属性应用旋转矩阵后进行嵌入
            nbr_embed = self.nbr_embed([torch.bmm(x_j.unsqueeze(-2), center_rotate_mat).squeeze(-2),
                                        torch.bmm(edge_attr.unsqueeze(-2), center_rotate_mat).squeeze(-2)])
        # 计算查询、键、值向量
        query = self.lin_q(center_embed_i).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        key = self.lin_k(nbr_embed).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        value = self.lin_v(nbr_embed).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        # 计算缩放因子，用于缩放点积注意力的结果
        scale = (self.embed_dim // self.num_heads) ** 0.5
        # 计算注意力权重
        alpha = (query * key).sum(dim=-1) / scale
        # 应用softmax进行归一化，并使用dropout
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = self.attn_drop(alpha)
        # 返回加权的值向量
        return value * alpha.unsqueeze(-1)


    # 定义更新函数
    def update(self,
            inputs: torch.Tensor,  # 输入特征
            center_embed: torch.Tensor) -> torch.Tensor:  # 中心节点嵌入
        # 将输入重塑为二维张量
        inputs = inputs.view(-1, self.embed_dim)
        # 计算门控信号
        gate = torch.sigmoid(self.lin_ih(inputs) + self.lin_hh(center_embed))
        # 计算并返回更新后的特征
        return inputs + gate * (self.lin_self(center_embed) - inputs)


    # 定义多头自注意力模块
    def _mha_block(self,
                center_embed: torch.Tensor,  # 中心节点嵌入
                x: torch.Tensor,  # 节点特征
                edge_index: Adj,  # 边索引
                edge_attr: torch.Tensor,  # 边属性
                rotate_mat: Optional[torch.Tensor],  # 旋转矩阵
                size: Size) -> torch.Tensor:  # 图的大小
        # 使用propagate方法传递消息，并应用输出投影
        center_embed = self.out_proj(self.propagate(edge_index=edge_index, x=x, center_embed=center_embed,
                                                    edge_attr=edge_attr, rotate_mat=rotate_mat, size=size))
        # 应用dropout并返回结果
        return self.proj_drop(center_embed)


    # 定义前馈网络模块
    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        # 通过多层感知机处理输入x并返回结果
        return self.mlp(x)



# 定义TemporalEncoder类，用于编码时间序列数据
class TemporalEncoder(nn.Module):

    # 类的初始化方法
    def __init__(self,
                 historical_steps: int,  # 历史时间步数
                 embed_dim: int,  # 嵌入维度
                 num_heads: int = 8,  # Transformer中多头注意力机制的头数
                 num_layers: int = 4,  # Transformer编码器层的数量
                 dropout: float = 0.1) -> None:  # Dropout比率
        super(TemporalEncoder, self).__init__()  # 调用父类的构造函数进行初始化
        # 定义单个Transformer编码器层
        encoder_layer = TemporalEncoderLayer(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        # 利用定义的编码器层构建Transformer编码器
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers,
                                                         norm=nn.LayerNorm(embed_dim))
        # 初始化填充（padding）和分类（cls）标记的参数，以及位置嵌入
        self.padding_token = nn.Parameter(torch.Tensor(historical_steps, 1, embed_dim))
        self.cls_token = nn.Parameter(torch.Tensor(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.Tensor(historical_steps + 1, 1, embed_dim))
        # 生成注意力掩码，用于Transformer编码器
        attn_mask = self.generate_square_subsequent_mask(historical_steps + 1)
        self.register_buffer('attn_mask', attn_mask)
        # 初始化参数
        nn.init.normal_(self.padding_token, mean=0., std=.02)
        nn.init.normal_(self.cls_token, mean=0., std=.02)
        nn.init.normal_(self.pos_embed, mean=0., std=.02)
        # 应用权重初始化函数
        self.apply(init_weights)

    # 定义前向传播方法
    def forward(self,
                x: torch.Tensor,  # 输入的时间序列特征
                padding_mask: torch.Tensor) -> torch.Tensor:  # 填充掩码
        # 使用填充标记替换被掩码的位置
        x = torch.where(padding_mask.t().unsqueeze(-1), self.padding_token, x)
        # 扩展并添加分类标记到序列的末尾
        expand_cls_token = self.cls_token.expand(-1, x.shape[1], -1)
        x = torch.cat((x, expand_cls_token), dim=0)
        # 加入位置嵌入
        x = x + self.pos_embed
        # 通过Transformer编码器进行编码
        out = self.transformer_encoder(src=x, mask=self.attn_mask, src_key_padding_mask=None)
        # 返回最后的输出，对应于分类标记的编码结果
        return out[-1]  # [N, D]

    # 静态方法：生成注意力掩码
    @staticmethod
    def generate_square_subsequent_mask(seq_len: int) -> torch.Tensor:
        # 生成一个上三角矩阵，用于屏蔽未来的时间步
        mask = (torch.triu(torch.ones(seq_len, seq_len)) == 1).transpose(0, 1)
        # 将掩码值设置为-inf和0，分别表示屏蔽和不屏蔽
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask



# 定义TemporalEncoderLayer类，表示Transformer编码器中的单个层
class TemporalEncoderLayer(nn.Module):

    # 类的初始化方法
    def __init__(self,
                 embed_dim: int,  # 嵌入向量的维度
                 num_heads: int = 8,  # 多头注意力机制中头的数量
                 dropout: float = 0.1) -> None:  # Dropout比率
        super(TemporalEncoderLayer, self).__init__()  # 调用父类的构造函数进行初始化
        # 初始化多头自注意力机制
        self.self_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        # 初始化前馈网络的第一个线性层，扩大维度
        self.linear1 = nn.Linear(embed_dim, embed_dim * 4)
        # 初始化Dropout层
        self.dropout = nn.Dropout(dropout)
        # 初始化前馈网络的第二个线性层，恢复维度
        self.linear2 = nn.Linear(embed_dim * 4, embed_dim)
        # 初始化两个层归一化层
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        # 初始化额外的Dropout层，用于自注意力和前馈网络
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    # 定义前向传播方法
    def forward(self,
                src: torch.Tensor,  # 输入的源数据
                src_mask: Optional[torch.Tensor] = None,  # 可选的源数据掩码，用于自注意力计算
                src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:  # 可选的键填充掩码
        x = src
        # 应用自注意力块，并添加到原始输入上，实现残差连接
        x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
        # 应用前馈网络块，并添加到自注意力的结果上，实现残差连接
        x = x + self._ff_block(self.norm2(x))
        # 返回处理后的数据
        return x

    # 定义自注意力块的私有方法
    def _sa_block(self,
                  x: torch.Tensor,
                  attn_mask: Optional[torch.Tensor],
                  key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        # 通过多头自注意力层处理数据，不返回注意力权重
        x = self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)[0]
        # 应用Dropout，并返回结果
        return self.dropout1(x)

    # 定义前馈网络块的私有方法
    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        # 通过两层线性变换和ReLU激活函数，间隔一个Dropout层
        x = self.linear2(self.dropout(F.relu_(self.linear1(x))))
        # 应用Dropout，并返回结果
        return self.dropout2(x)



# 定义ALEncoder类，用于自适应局部编码
class ALEncoder(MessagePassing):

    # 类的初始化方法
    def __init__(self,
                 node_dim: int,  # 节点特征维度
                 edge_dim: int,  # 边特征维度
                 embed_dim: int,  # 嵌入向量维度
                 num_heads: int = 8,  # 多头注意力机制的头数
                 dropout: float = 0.1,  # Dropout比率
                 **kwargs) -> None:
        super(ALEncoder, self).__init__(aggr='add', node_dim=0, **kwargs)  # 调用父类的构造函数
        self.embed_dim = embed_dim  # 保存嵌入向量维度
        self.num_heads = num_heads  # 保存头数

        # 初始化车道特征的嵌入层
        self.lane_embed = MultipleInputEmbedding(in_channels=[node_dim, edge_dim], out_channel=embed_dim)
        # 初始化用于计算查询（Q）、键（K）和值（V）的线性变换层
        self.lin_q = nn.Linear(embed_dim, embed_dim)
        self.lin_k = nn.Linear(embed_dim, embed_dim)
        self.lin_v = nn.Linear(embed_dim, embed_dim)
        # 初始化自注意力后的线性变换层和Dropout
        self.lin_self = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        # 初始化额外的线性变换层和Dropout
        self.lin_ih = nn.Linear(embed_dim, embed_dim)
        self.lin_hh = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)
        # 初始化层归一化（LayerNorm）
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        # 初始化多层感知机（MLP）及其Dropout
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout))
        # 初始化交叉口、转向方向和交通控制特征的嵌入向量
        self.is_intersection_embed = nn.Parameter(torch.Tensor(2, embed_dim))
        self.turn_direction_embed = nn.Parameter(torch.Tensor(3, embed_dim))
        self.traffic_control_embed = nn.Parameter(torch.Tensor(2, embed_dim))
        # 对嵌入向量进行正态分布初始化
        nn.init.normal_(self.is_intersection_embed, mean=0., std=.02)
        nn.init.normal_(self.turn_direction_embed, mean=0., std=.02)
        nn.init.normal_(self.traffic_control_embed, mean=0., std=.02)
        # 应用权重初始化函数
        self.apply(init_weights)

    # 定义前向传播方法
    def forward(self,
                x: Tuple[torch.Tensor, torch.Tensor],  # 输入的车道和动态实体特征
                edge_index: Adj,  # 边索引
                edge_attr: torch.Tensor,  # 边特征
                is_intersections: torch.Tensor,  # 是否为交叉口
                turn_directions: torch.Tensor,  # 转向方向
                traffic_controls: torch.Tensor,  # 交通控制信息
                rotate_mat: Optional[torch.Tensor] = None,  # 可选的旋转矩阵
                size: Size = None) -> torch.Tensor:  # 图的大小
        x_lane, x_actor = x  # 解包输入特征
        # 将分类特征转换为长整型
        is_intersections = is_intersections.long()
        turn_directions = turn_directions.long()
        traffic_controls = traffic_controls.long()
        # 对动态实体特征应用多头自注意力块和前馈网络块
        x_actor = x_actor + self._mha_block(self.norm1(x_actor), x_lane, edge_index, edge_attr, is_intersections,
                                            turn_directions, traffic_controls, rotate_mat, size)
        x_actor = x_actor + self._ff_block(self.norm2(x_actor))
        # 返回处理后的动态实体特征
        return x_actor

    def message(self,
                edge_index: Adj,  # 边索引
                x_i: torch.Tensor,  # 源节点特征
                x_j: torch.Tensor,  # 目标节点特征
                edge_attr: torch.Tensor,  # 边特征
                is_intersections_j,  # 目标节点是否为交叉口
                turn_directions_j,  # 目标节点的转向方向
                traffic_controls_j,  # 目标节点的交通控制信息
                rotate_mat: Optional[torch.Tensor],  # 旋转矩阵
                index: torch.Tensor,  # 用于聚合的索引
                ptr: OptTensor,  # 可选的聚合指针
                size_i: Optional[int]) -> torch.Tensor:  # 输入大小
        # 根据是否提供旋转矩阵，选择是否对节点和边特征进行旋转变换
        # 检查是否提供了旋转矩阵
        if rotate_mat is None:
            # 如果没有提供旋转矩阵，直接使用目标节点特征、边特征
            # 和交通相关特征的嵌入向量进行嵌入操作
            x_j = self.lane_embed([x_j, edge_attr],
                                [self.is_intersection_embed[is_intersections_j],  # 是否交叉口的嵌入向量
                                self.turn_direction_embed[turn_directions_j],  # 转向方向的嵌入向量
                                self.traffic_control_embed[traffic_controls_j]])  # 交通控制的嵌入向量
        else:
            # 如果提供了旋转矩阵，先根据边索引获取对应的旋转矩阵
            rotate_mat = rotate_mat[edge_index[1]]
            # 使用批量矩阵乘法（batch matrix multiplication, bmm）对目标节点特征和边特征进行旋转变换，
            # 然后将变换后的特征与交通相关特征的嵌入向量一起进行嵌入操作
            x_j = self.lane_embed([torch.bmm(x_j.unsqueeze(-2), rotate_mat).squeeze(-2),  # 对目标节点特征进行旋转变换
                                torch.bmm(edge_attr.unsqueeze(-2), rotate_mat).squeeze(-2)],  # 对边特征进行旋转变换
                                [self.is_intersection_embed[is_intersections_j],  # 是否交叉口的嵌入向量
                                self.turn_direction_embed[turn_directions_j],  # 转向方向的嵌入向量
                                self.traffic_control_embed[traffic_controls_j]])  # 交通控制的嵌入向量

        # 计算查询、键、值向量，并通过点积注意力计算注意力权重
        # 使用线性层将源节点特征（x_i）转换为查询（Query）向量
        query = self.lin_q(x_i).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        # 使用线性层将目标节点特征（x_j）转换为键（Key）向量
        key = self.lin_k(x_j).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        # 使用线性层将目标节点特征（x_j）转换为值（Value）向量
        value = self.lin_v(x_j).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        # 计算缩放因子，用于缩放点积得到的注意力分数，防止梯度过小
        scale = (self.embed_dim // self.num_heads) ** 0.5
        # 计算查询向量和键向量的点积，然后除以缩放因子，得到注意力分数
        alpha = (query * key).sum(dim=-1) / scale
        # 使用softmax函数对注意力分数进行归一化，使得每个节点对其邻居的关注度总和为1
        alpha = softmax(alpha, index, ptr, size_i)
        # 对归一化后的注意力权重应用Dropout，这是一种常用的正则化技术
        alpha = self.attn_drop(alpha)
        # 将归一化的注意力权重应用于值（Value）向量，得到加权的值向量，注意力权重决定了每个值向量的重要程度
        return value * alpha.unsqueeze(-1)



    # 定义update函数，用于更新节点的特征
    def update(self,
            inputs: torch.Tensor,  # 从message函数中传递过来的加权的值向量
            x: torch.Tensor) -> torch.Tensor:  # 原始节点特征向量，其中包含了动态实体（actor）的特征
        x_actor = x[1]  # 获取动态实体节点特征
        inputs = inputs.view(-1, self.embed_dim)  # 将inputs重塑为二维张量，以匹配嵌入维度
        # 计算门控信号，这一步使用了两个线性变换层（lin_ih和lin_hh），
        # 并对它们的输出求和后应用sigmoid函数，以得到范围在[0, 1]之间的门控值
        gate = torch.sigmoid(self.lin_ih(inputs) + self.lin_hh(x_actor))
        # 使用计算得到的门控信号来加权更新节点特征，其中lin_self是另一个线性变换层，
        # 它对动态实体特征进行处理。这里inputs代表通过自注意力机制得到的新信息，
        # 而lin_self(x_actor) - inputs计算了原始动态实体特征与新信息之间的差异。
        # 通过门控信号调节这一差异对最终输出的影响程度，实现动态的特征融合。
        return inputs + gate * (self.lin_self(x_actor) - inputs)


    # 定义多头自注意力（MHA）块
    def _mha_block(self,
                x_actor: torch.Tensor,  # 动态实体特征向量
                x_lane: torch.Tensor,  # 车道特征向量
                edge_index: Adj,  # 边索引，表示图中的连接关系
                edge_attr: torch.Tensor,  # 边特征
                is_intersections: torch.Tensor,  # 是否交叉口的特征向量
                turn_directions: torch.Tensor,  # 转向方向的特征向量
                traffic_controls: torch.Tensor,  # 交通控制的特征向量
                rotate_mat: Optional[torch.Tensor],  # 旋转矩阵，用于坐标变换
                size: Size) -> torch.Tensor:  # 图的大小
        # 使用`propagate`方法执行消息传递。这一过程中，将动态实体和车道的特征向量、
        # 边的特征、交通特征（是否交叉口、转向方向、交通控制）以及可选的旋转矩阵作为输入。
        # `propagate`方法将基于这些信息进行自注意力机制的计算，得到更新后的动态实体特征向量。
        x_actor = self.out_proj(self.propagate(edge_index=edge_index, x=(x_lane, x_actor), edge_attr=edge_attr,
                                            is_intersections=is_intersections, turn_directions=turn_directions,
                                            traffic_controls=traffic_controls, rotate_mat=rotate_mat, size=size))
        # 对更新后的动态实体特征向量应用dropout，这是一种常用的正则化技术，有助于模型泛化能力的提升。
        return self.proj_drop(x_actor)


    # 定义前馈网络（FFN）块
    def _ff_block(self, x_actor: torch.Tensor) -> torch.Tensor:
        # 直接将动态实体节点特征通过一个多层感知机（MLP）进行处理
        # MLP在类的初始化时已经定义，这里不做结构上的改变，只是传递数据
        return self.mlp(x_actor)

