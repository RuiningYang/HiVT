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
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import init_weights


# 定义GRU解码器类
class GRUDecoder(nn.Module):

    # 初始化函数
    def __init__(self,
                 local_channels: int,  # 本地特征维度
                 global_channels: int,  # 全局特征维度
                 future_steps: int,  # 需要预测的未来步数
                 num_modes: int,  # 预测的模态数
                 uncertain: bool = True,  # 是否预测不确定性
                 min_scale: float = 1e-3) -> None:  # 最小尺度值，防止尺度预测为0
        super(GRUDecoder, self).__init__()  # 调用父类的初始化方法
        # 初始化各项参数
        self.input_size = global_channels  # GRU的输入尺寸等于全局特征维度
        self.hidden_size = local_channels  # GRU的隐藏层尺寸等于本地特征维度
        self.future_steps = future_steps  # 需要预测的未来步数
        self.num_modes = num_modes  # 预测的模态数
        self.uncertain = uncertain  # 是否预测不确定性
        self.min_scale = min_scale  # 最小尺度值

        # 定义GRU单元
        self.gru = nn.GRU(input_size=self.input_size,
                          hidden_size=self.hidden_size,
                          num_layers=1,
                          bias=True,
                          batch_first=False,
                          dropout=0,
                          bidirectional=False)
        # 定义位置预测的线性层
        self.loc = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, 2))  # 输出二维位置信息
        # 如果需要预测不确定性，则定义尺度（scale）预测的线性层
        if uncertain:
            self.scale = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.LayerNorm(self.hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_size, 2))  # 输出二维尺度信息
        # 定义模式选择（pi）的线性层，用于多模态预测中的模式选择
        self.pi = nn.Sequential(
            nn.Linear(self.hidden_size + self.input_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, 1))  # 输出每个模态的权重
        self.apply(init_weights)  # 应用权重初始化

    # 定义前向传播函数
    def forward(self,
                local_embed: torch.Tensor,  # 本地嵌入向量
                global_embed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:  # 全局嵌入向量
        # 计算多模态输出的权重(pi)，通过将本地嵌入向量扩展到多模态数量并与全局嵌入向量拼接，然后通过pi网络处理
        pi = self.pi(torch.cat((local_embed.expand(self.num_modes, *local_embed.shape),
                                global_embed), dim=-1)).squeeze(-1).t()
        # 将全局嵌入向量调整形状，准备进行时间步展开
        global_embed = global_embed.reshape(-1, self.input_size)  # 调整全局嵌入向量的形状, # [F x N, D]
        # 将全局嵌入向量在时间维度上扩展，以匹配未来步数
        global_embed = global_embed.expand(self.future_steps, *global_embed.shape) # [H, F x N, D]
        # 将本地嵌入向量在模态维度上重复，并增加一个维度，以匹配GRU的输入要求
        local_embed = local_embed.repeat(self.num_modes, 1).unsqueeze(0) # [1, F x N, D]
        # 通过GRU处理扩展后的全局嵌入向量和本地嵌入向量，得到输出序列
        out, _ = self.gru(global_embed, local_embed)
        # 调整输出序列的维度顺序以适应后续处理
        out = out.transpose(0, 1) # [F x N, H, D]
        # 通过位置预测网络(loc)处理GRU的输出，得到位置预测
        loc = self.loc(out) # [F x N, H, 2]
        # 如果需要预测不确定性，则通过尺度预测网络(scale)处理GRU的输出，得到尺度预测
        if self.uncertain:
            # 使用ELU激活函数和最小尺度值处理尺度预测，以确保尺度值的有效性
            scale = F.elu_(self.scale(out), alpha=1.0) + 1.0 + self.min_scale # [F x N, H, 2]
            # 将位置和尺度预测合并，并调整形状以匹配输出格式
            return torch.cat((loc, scale), dim=-1).view(self.num_modes, -1, self.future_steps, 4), pi # [F, N, H, 4], [N, F]
        else:
            # 如果不预测不确定性，则直接返回位置预测和模态权重
            return loc.view(self.num_modes, -1, self.future_steps, 2), pi # [F, N, H, 2], [N, F]



# 定义多层感知机解码器类
class MLPDecoder(nn.Module):

    # 初始化函数
    def __init__(self,
                 local_channels: int,  # 本地特征的维度
                 global_channels: int,  # 全局特征的维度
                 future_steps: int,  # 需要预测的未来步数
                 num_modes: int,  # 预测的模态数量
                 uncertain: bool = True,  # 是否预测不确定性
                 min_scale: float = 1e-3) -> None:  # 最小尺度值，用于避免尺度预测为零
        super(MLPDecoder, self).__init__()  # 调用父类的初始化方法
        # 初始化模型参数
        self.input_size = global_channels  # 输入尺寸等于全局特征维度
        self.hidden_size = local_channels  # 隐藏层尺寸等于本地特征维度
        self.future_steps = future_steps  # 设置未来步数
        self.num_modes = num_modes  # 设置模态数量
        self.uncertain = uncertain  # 设置是否预测不确定性
        self.min_scale = min_scale  # 设置最小尺度值

        # 定义聚合嵌入层，用于将本地和全局特征融合
        self.aggr_embed = nn.Sequential(
            nn.Linear(self.input_size + self.hidden_size, self.hidden_size),  # 线性变换
            nn.LayerNorm(self.hidden_size),  # 层归一化
            nn.ReLU(inplace=True))  # ReLU激活函数
        # 定义位置预测层
        self.loc = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),  # 线性变换
            nn.LayerNorm(self.hidden_size),  # 层归一化
            nn.ReLU(inplace=True),  # ReLU激活函数
            nn.Linear(self.hidden_size, self.future_steps * 2))  # 输出每个未来步骤的2维位置信息
        # 如果需要预测不确定性，定义尺度预测层
        if uncertain:
            self.scale = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),  # 线性变换
                nn.LayerNorm(self.hidden_size),  # 层归一化
                nn.ReLU(inplace=True),  # ReLU激活函数
                nn.Linear(self.hidden_size, self.future_steps * 2))  # 输出每个未来步骤的尺度信息
        # 定义模态选择层，用于预测每个模态的权重
        self.pi = nn.Sequential(
            nn.Linear(self.hidden_size + self.input_size, self.hidden_size),  # 线性变换
            nn.LayerNorm(self.hidden_size),  # 层归一化
            nn.ReLU(inplace=True),  # ReLU激活函数
            nn.Linear(self.hidden_size, self.hidden_size),  # 另一层线性变换
            nn.LayerNorm(self.hidden_size),  # 层归一化
            nn.ReLU(inplace=True),  # ReLU激活函数
            nn.Linear(self.hidden_size, 1))  # 输出每个模态的权重
        self.apply(init_weights)  # 应用权重初始化


    # 定义前向传播函数
    def forward(self,
                local_embed: torch.Tensor,  # 本地嵌入向量，假设维度为 [N, D_local]
                global_embed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:  # 全局嵌入向量，假设维度为 [N, D_global]
        # 计算模态权重pi，将本地嵌入向量扩展到多模态数量并与全局嵌入向量拼接，然后通过pi网络处理
        # pi的维度为 [N, F] 转置后为 [F, N]，其中F为模态数，N为样本数
        pi = self.pi(torch.cat((local_embed.expand(self.num_modes, *local_embed.shape),
                                global_embed), dim=-1)).squeeze(-1).t()
        # 将全局嵌入和扩展的本地嵌入拼接后，通过聚合嵌入网络处理
        out = self.aggr_embed(torch.cat((global_embed, local_embed.expand(self.num_modes, *local_embed.shape)), dim=-1))
        # 通过位置预测网络，得到未来步骤的二维位置预测，维度变为 [F, N, H, 2]，其中H为未来步数
        loc = self.loc(out).view(self.num_modes, -1, self.future_steps, 2)
        if self.uncertain:
            # 如果预测不确定性，通过尺度预测网络处理，使用ELU激活函数加1确保尺度值非负，再加上最小尺度值，维度为 [F, N, H, 2]
            scale = F.elu_(self.scale(out), alpha=1.0).view(self.num_modes, -1, self.future_steps, 2) + 1.0
            scale = scale + self.min_scale
            # 返回位置和尺度预测以及模态权重，合并位置和尺度的维度为 [F, N, H, 4]
            return torch.cat((loc, scale), dim=-1), pi
        else:
            # 如果不预测不确定性，直接返回位置预测和模态权重，维度为 [F, N, H, 2] 和 [N, F]
            return loc, pi

