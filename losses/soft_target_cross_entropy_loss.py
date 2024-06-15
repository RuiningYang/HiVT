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
import torch
import torch.nn as nn
import torch.nn.functional as F


# 定义针对软目标的交叉熵损失类
class SoftTargetCrossEntropyLoss(nn.Module):

    # 初始化函数
    def __init__(self, reduction: str = 'mean') -> None:
        super(SoftTargetCrossEntropyLoss, self).__init__()
        self.reduction = reduction  # 损失降维方式，可选'mean'、'sum'或'none'

    # 前向传播函数
    def forward(self,
                pred: torch.Tensor,  # 预测值，一般为模型的输出，假设最后一个维度为类别维度
                target: torch.Tensor) -> torch.Tensor:  # 目标值，为软目标，形状与预测值相同
        # 计算交叉熵损失，首先对预测值进行log_softmax操作，然后与目标值的负点积求和
        cross_entropy = torch.sum(-target * F.log_softmax(pred, dim=-1), dim=-1)
        # 根据reduction参数决定损失的降维方式
        if self.reduction == 'mean':
            return cross_entropy.mean()  # 返回平均损失
        elif self.reduction == 'sum':
            return cross_entropy.sum()  # 返回总损失
        elif self.reduction == 'none':
            return cross_entropy  # 不降维，返回每个样本的损失
        else:
            raise ValueError('{} is not a valid value for reduction'.format(self.reduction))

