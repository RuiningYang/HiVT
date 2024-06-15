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
from typing import List, Optional

import torch
import torch.nn as nn

from utils import init_weights


# 定义一个处理单个输入的嵌入层类
class SingleInputEmbedding(nn.Module):

    # 初始化函数
    def __init__(self,
                 in_channel: int,  # 输入特征的维度
                 out_channel: int) -> None:  # 输出特征的维度
        super(SingleInputEmbedding, self).__init__()  # 调用父类的初始化方法
        # 定义一个嵌入网络，包含三个线性层和两个ReLU激活函数层，以及每个线性层之后的层归一化
        self.embed = nn.Sequential(
            nn.Linear(in_channel, out_channel),  # 第一个线性变换层
            nn.LayerNorm(out_channel),  # 第一个层归一化
            nn.ReLU(inplace=True),  # 第一个ReLU激活函数
            nn.Linear(out_channel, out_channel),  # 第二个线性变换层
            nn.LayerNorm(out_channel),  # 第二个层归一化
            nn.ReLU(inplace=True),  # 第二个ReLU激活函数
            nn.Linear(out_channel, out_channel),  # 第三个线性变换层
            nn.LayerNorm(out_channel))  # 第三个层归一化
        self.apply(init_weights)  # 应用权重初始化函数

    # 前向传播函数
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 将输入特征x通过嵌入网络进行处理，返回处理后的特征
        return self.embed(x)



# 定义处理多个输入的嵌入层类
class MultipleInputEmbedding(nn.Module):

    # 初始化函数
    def __init__(self,
                 in_channels: List[int],  # 输入通道列表，每个通道对应的特征维度
                 out_channel: int) -> None:  # 统一的输出特征维度
        super(MultipleInputEmbedding, self).__init__()  # 调用父类的初始化方法
        # 对每个输入通道创建一个模块，包含线性变换、层归一化和ReLU激活函数
        # 使用ModuleList来存储这些模块，以便它们可以被正确注册
        self.module_list = nn.ModuleList(
            [nn.Sequential(nn.Linear(in_channel, out_channel),  # 线性变换
                           nn.LayerNorm(out_channel),  # 层归一化
                           nn.ReLU(inplace=True),  # ReLU激活函数
                           nn.Linear(out_channel, out_channel))  # 另一个线性变换
             for in_channel in in_channels])
        # 定义一个聚合嵌入层，用于合并多个输入的处理结果
        self.aggr_embed = nn.Sequential(
            nn.LayerNorm(out_channel),  # 层归一化
            nn.ReLU(inplace=True),  # ReLU激活函数
            nn.Linear(out_channel, out_channel),  # 线性变换
            nn.LayerNorm(out_channel))  # 另一个层归一化
        self.apply(init_weights)  # 应用权重初始化函数

    # 前向传播函数
    def forward(self,
                continuous_inputs: List[torch.Tensor],  # 连续特征输入列表
                categorical_inputs: Optional[List[torch.Tensor]] = None) -> torch.Tensor:  # 可选的类别特征输入列表
        # 对每个连续特征输入使用对应的模块进行处理
        for i in range(len(self.module_list)):
            continuous_inputs[i] = self.module_list[i](continuous_inputs[i])
        # 将处理后的连续特征堆叠起来，并通过求和进行聚合
        output = torch.stack(continuous_inputs).sum(dim=0)
        # 如果提供了类别特征输入，则同样进行堆叠和求和，然后与连续特征的聚合结果相加
        if categorical_inputs is not None:
            output += torch.stack(categorical_inputs).sum(dim=0)
        # 将聚合后的输出通过聚合嵌入层进行进一步处理
        return self.aggr_embed(output)
