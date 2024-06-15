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
from typing import Optional

import torch
import torch.nn as nn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj
from torch_geometric.typing import OptTensor
from torch_geometric.typing import Size
from torch_geometric.utils import softmax
from torch_geometric.utils import subgraph

from models import MultipleInputEmbedding
from models import SingleInputEmbedding
from utils import TemporalData
from utils import init_weights


# 定义全局交互器类
class GlobalInteractor(nn.Module):

    # 初始化函数
    def __init__(self,
                 historical_steps: int,  # 历史步数
                 embed_dim: int,  # 嵌入维度
                 edge_dim: int,  # 边特征维度
                 num_modes: int = 6,  # 模式数量
                 num_heads: int = 8,  # 多头注意力机制中的头数
                 num_layers: int = 3,  # 交互层的层数
                 dropout: float = 0.1,  # Dropout比率
                 rotate: bool = True) -> None:  # 是否进行坐标旋转
        super(GlobalInteractor, self).__init__()  # 调用父类构造函数进行初始化
        self.historical_steps = historical_steps
        self.embed_dim = embed_dim
        self.num_modes = num_modes

        # 根据是否需要进行坐标旋转（rotate参数），选择使用不同类型的关系嵌入（rel_embed）
        if rotate:
            # 如果需要进行坐标旋转，则使用MultipleInputEmbedding类来创建关系嵌入
            # 这允许输入多个通道的信息（在这里是两个edge_dim维度的输入），输出为embed_dim维度的嵌入
            self.rel_embed = MultipleInputEmbedding(in_channels=[edge_dim, edge_dim], out_channel=embed_dim)
        else:
            # 如果不需要进行坐标旋转，则使用SingleInputEmbedding类来创建关系嵌入
            # 这处理单一通道的信息（edge_dim维度的输入），输出为embed_dim维度的嵌入
            self.rel_embed = SingleInputEmbedding(in_channel=edge_dim, out_channel=embed_dim)

        # 初始化全局交互层，这里使用nn.ModuleList来存储一系列的GlobalInteractorLayer层
        # 每个GlobalInteractorLayer层都以相同的嵌入维度（embed_dim）、头数（num_heads）和dropout率进行初始化
        # 通过循环num_layers次数，生成指定数量的全局交互层
        self.global_interactor_layers = nn.ModuleList(
            [GlobalInteractorLayer(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)])

        # 初始化层归一化
        self.norm = nn.LayerNorm(embed_dim)
        # 初始化多头预测的线性映射层
        self.multihead_proj = nn.Linear(embed_dim, num_modes * embed_dim)
        # 应用权重初始化
        self.apply(init_weights)

    # 前向传播函数
    def forward(self,
                data: TemporalData,  # 输入的时间序列数据
                local_embed: torch.Tensor) -> torch.Tensor:  # 本地嵌入表示
        # 使用子图函数根据填充掩码筛选出有效的边索引，排除那些因为时间步骤过早而被填充掩码掉的边
        edge_index, _ = subgraph(subset=~data['padding_mask'][:, self.historical_steps - 1], edge_index=data.edge_index)

        # 计算相对位置特征，即计算每条边的源节点和目标节点在最后一个时间步的位置差
        rel_pos = data['positions'][edge_index[0], self.historical_steps - 1] - data['positions'][
            edge_index[1], self.historical_steps - 1]

        # 判断是否提供了旋转矩阵，以确定如何计算相对位置的嵌入
        if data['rotate_mat'] is None:
            # 如果没有旋转矩阵，直接对相对位置特征进行嵌入
            rel_embed = self.rel_embed(rel_pos)
        else:
            # 如果有旋转矩阵，首先使用批矩阵乘法（batch matrix multiplication）将相对位置特征进行旋转变换
            rel_pos = torch.bmm(rel_pos.unsqueeze(-2), data['rotate_mat'][edge_index[1]]).squeeze(-2)
            # 计算源节点和目标节点旋转角度的差异
            rel_theta = data['rotate_angles'][edge_index[0]] - data['rotate_angles'][edge_index[1]]
            # 将角度差异转换为余弦和正弦值
            rel_theta_cos = torch.cos(rel_theta).unsqueeze(-1)
            rel_theta_sin = torch.sin(rel_theta).unsqueeze(-1)
            # 将转换后的相对位置特征和角度差异的余弦、正弦值进行嵌入
            rel_embed = self.rel_embed([rel_pos, torch.cat((rel_theta_cos, rel_theta_sin), dim=-1)])

        # 设置初始的节点特征为本地嵌入表示
        x = local_embed

        # 依次通过所有定义的全局交互层，每一层都将更新节点特征
        for layer in self.global_interactor_layers:
            x = layer(x, edge_index, rel_embed)

        # 对最终的节点特征进行层归一化处理
        x = self.norm(x)  # 应用层归一化

        # 使用全连接层将节点特征投影到多个模式的特征空间，每个模式对应一组特征
        x = self.multihead_proj(x).view(-1, self.num_modes, self.embed_dim)  # 应用多头预测映射

        # 将结果的维度进行调整，以适配模式数量的要求
        x = x.transpose(0, 1)  # 调整维度以适配模式数量

        # 返回处理后的全局特征
        return x




# 定义全局交互层类，继承自MessagePassing
class GlobalInteractorLayer(MessagePassing):

    # 初始化函数
    def __init__(self,
                 embed_dim: int,  # 嵌入向量的维度
                 num_heads: int = 8,  # 多头注意力机制中的头数
                 dropout: float = 0.1,  # Dropout比率
                 **kwargs) -> None:
        # 调用父类的初始化方法，设置聚合函数为加法
        super(GlobalInteractorLayer, self).__init__(aggr='add', node_dim=0, **kwargs)
        self.embed_dim = embed_dim  # 嵌入向量的维度
        self.num_heads = num_heads  # 注意力头数

        # 定义节点和边的线性变换层，用于计算查询(Q)、键(K)、值(V)
        self.lin_q_node = nn.Linear(embed_dim, embed_dim)  # 节点的查询层
        self.lin_k_node = nn.Linear(embed_dim, embed_dim)  # 节点的键层
        self.lin_k_edge = nn.Linear(embed_dim, embed_dim)  # 边的键层
        self.lin_v_node = nn.Linear(embed_dim, embed_dim)  # 节点的值层
        self.lin_v_edge = nn.Linear(embed_dim, embed_dim)  # 边的值层
        # 自注意力之后的线性变换层
        self.lin_self = nn.Linear(embed_dim, embed_dim)
        # Dropout层
        self.attn_drop = nn.Dropout(dropout)  # 注意力机制的Dropout
        # 门控信号的线性变换层
        self.lin_ih = nn.Linear(embed_dim, embed_dim)
        self.lin_hh = nn.Linear(embed_dim, embed_dim)
        # 输出投影层
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)  # 输出的Dropout
        # 层归一化
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        # 定义多层感知机(MLP)，进一步处理特征
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),  # 扩展维度
            nn.ReLU(inplace=True),  # 激活函数
            nn.Dropout(dropout),  # Dropout层
            nn.Linear(embed_dim * 4, embed_dim),  # 恢复维度
            nn.Dropout(dropout))  # Dropout层


    # 定义前向传播函数
    def forward(self,
                x: torch.Tensor,  # 输入的节点特征向量
                edge_index: Adj,  # 边索引，定义了图中每条边的起点和终点
                edge_attr: torch.Tensor,  # 边特征向量
                size: Size = None) -> torch.Tensor:  # 可选的图的大小参数，用于某些类型的图处理
        # 首先对输入的节点特征进行层归一化，然后传递给多头自注意力（MHA）块进行处理，
        # MHA块会根据边索引和边特征来更新节点特征。这里使用残差连接，即将MHA块的输出与原始特征相加
        x = x + self._mha_block(self.norm1(x), edge_index, edge_attr, size)
        # 接着，再次对节点特征进行层归一化，然后传递给前馈网络（FFN）块进行处理，
        # FFN块是一系列全连接层的组合，用于进一步提取特征。同样使用残差连接，即将FFN块的输出与上一步的结果相加
        x = x + self._ff_block(self.norm2(x))
        # 返回更新后的节点特征
        return x


    # 定义消息传递函数
    def message(self,
                x_i: torch.Tensor,  # 源节点的特征
                x_j: torch.Tensor,  # 目标节点的特征
                edge_attr: torch.Tensor,  # 边特征
                index: torch.Tensor,  # 指向目标节点的索引，用于聚合
                ptr: OptTensor,  # 用于稀疏张量的可选指针
                size_i: Optional[int]) -> torch.Tensor:  # 输入的大小
        # 计算查询向量，使用源节点特征
        query = self.lin_q_node(x_i).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        # 计算键向量，分别使用目标节点特征和边特征
        key_node = self.lin_k_node(x_j).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        key_edge = self.lin_k_edge(edge_attr).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        # 计算值向量，同样分别使用目标节点特征和边特征
        value_node = self.lin_v_node(x_j).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        value_edge = self.lin_v_edge(edge_attr).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        # 计算缩放因子，用于调整注意力分数的尺度
        scale = (self.embed_dim // self.num_heads) ** 0.5
        # 计算注意力分数，使用查询向量与键向量（节点和边的键向量之和）的点积，然后除以缩放因子
        alpha = (query * (key_node + key_edge)).sum(dim=-1) / scale
        # 使用softmax函数对注意力分数进行归一化处理，以便每个节点对其邻居的关注度加总为1
        alpha = softmax(alpha, index, ptr, size_i)
        # 应用Dropout到注意力分数上，增加模型的泛化能力
        alpha = self.attn_drop(alpha)
        # 返回加权的值向量，即将注意力分数应用于值向量（节点和边的值向量之和）
        return (value_node + value_edge) * alpha.unsqueeze(-1)


    def update(self,
            inputs: torch.Tensor,  # 消息传递方法计算得到的结果
            x: torch.Tensor) -> torch.Tensor:  # 节点的原始特征
        # 将输入重塑以匹配嵌入维度
        inputs = inputs.view(-1, self.embed_dim)
        # 计算门控信号，这是一种机制，用于决定保留多少原始特征与新计算特征
        gate = torch.sigmoid(self.lin_ih(inputs) + self.lin_hh(x))
        # 根据门控信号更新特征，结合原始输入和通过自注意力机制得到的新信息
        return inputs + gate * (self.lin_self(x) - inputs)

    # 定义多头自注意力块
    def _mha_block(self,
                x: torch.Tensor,  # 节点特征
                edge_index: Adj,  # 边索引
                edge_attr: torch.Tensor,  # 边特征
                size: Size) -> torch.Tensor:  # 图的大小
        # 使用propagate方法进行消息传递，然后通过一个输出投影层
        x = self.out_proj(self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr, size=size))
        # 应用Dropout进行正则化
        return self.proj_drop(x)

    # 定义前馈网络块
    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:  # 节点特征
        # 直接通过一个多层感知机（MLP）处理节点特征
        return self.mlp(x)
