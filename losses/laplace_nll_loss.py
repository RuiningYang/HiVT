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


# 定义拉普拉斯分布的负对数似然损失类
class LaplaceNLLLoss(nn.Module):

    # 初始化函数
    def __init__(self,
                 eps: float = 1e-6,  # 用于确保尺度参数不为零的小正数
                 reduction: str = 'mean') -> None:  # 损失降维方式，可选'mean'、'sum'或'none'
        super(LaplaceNLLLoss, self).__init__()
        self.eps = eps  # 保存eps参数
        self.reduction = reduction  # 保存降维方式

    # 前向传播函数
    def forward(self,
                pred: torch.Tensor,  # 预测值，假设最后一个维度包含位置（loc）和尺度（scale）参数
                target: torch.Tensor) -> torch.Tensor:  # 目标值
        # 将预测值分为位置（loc）和尺度（scale）参数
        loc, scale = pred.chunk(2, dim=-1)
        scale = scale.clone()  # 克隆尺度参数以避免修改原始预测值
        with torch.no_grad():
            # 限制尺度参数的最小值为eps，避免除以零
            scale.clamp_(min=self.eps)
        # 计算负对数似然损失
        nll = torch.log(2 * scale) + torch.abs(target - loc) / scale
        # 根据reduction参数决定损失的降维方式
        if self.reduction == 'mean':
            return nll.mean()  # 返回平均损失
        elif self.reduction == 'sum':
            return nll.sum()  # 返回总损失
        elif self.reduction == 'none':
            return nll  # 不降维，返回每个样本的损失
        else:
            raise ValueError('{} is not a valid value for reduction'.format(self.reduction))

