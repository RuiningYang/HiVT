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
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from losses import LaplaceNLLLoss
from losses import SoftTargetCrossEntropyLoss
from metrics import ADE
from metrics import FDE
from metrics import MR
from models import GlobalInteractor
from models import LocalEncoder
from models import MLPDecoder
from utils import TemporalData


# 定义HiVT类，继承自PyTorch Lightning的LightningModule
class HiVT(pl.LightningModule):

    # 类的初始化方法
    def __init__(self,
                 historical_steps: int,  # 历史步数，即使用多少历史数据点
                 future_steps: int,  # 未来步数，即预测多少未来数据点
                 num_modes: int,  # 模式数量
                 rotate: bool,  # 是否进行坐标旋转以处理方向
                 node_dim: int,  # 节点特征维度
                 edge_dim: int,  # 边特征维度
                 embed_dim: int,  # 嵌入向量的维度
                 num_heads: int,  # 注意力head的数量
                 dropout: float,  # Dropout比率
                 num_temporal_layers: int,  # 时间编码层的数量
                 num_global_layers: int,  # 全局交互层的数量
                 local_radius: float,  # 本地感知范围半径
                 parallel: bool,  # 是否并行处理
                 lr: float,  # 学习率
                 weight_decay: float,  # 权重衰减，用于正则化
                 T_max: int,  # 用于学习率调度的最大迭代次数
                 **kwargs) -> None:  # 其他关键字参数
        super(HiVT, self).__init__()  # 调用父类的初始化方法
        self.save_hyperparameters('historical_steps', 'future_steps', 'num_modes', 'rotate', 
                                  'node_dim', 'edge_dim', 'embed_dim', 'num_heads', 
                                  'dropout', 'num_temporal_layers', 'num_global_layers', 
                                  'local_radius', 'lr', 'weight_decay', 'T_max')
        # self.save_hyperparameters()  # 保存超参数，以便之后可以通过self.hparams访问
        # 初始化模型参数
        self.historical_steps = historical_steps
        self.future_steps = future_steps
        self.num_modes = num_modes
        self.rotate = rotate
        self.parallel = parallel
        self.lr = lr
        self.weight_decay = weight_decay
        self.T_max = T_max

        # 初始化模型的各个组件
        self.local_encoder = LocalEncoder(historical_steps=historical_steps,
                                          node_dim=node_dim,
                                          edge_dim=edge_dim,
                                          embed_dim=embed_dim,
                                          num_heads=num_heads,
                                          dropout=dropout,
                                          num_temporal_layers=num_temporal_layers,
                                          local_radius=local_radius,
                                          parallel=parallel)
        self.global_interactor = GlobalInteractor(historical_steps=historical_steps,
                                                  embed_dim=embed_dim,
                                                  edge_dim=edge_dim,
                                                  num_modes=num_modes,
                                                  num_heads=num_heads,
                                                  num_layers=num_global_layers,
                                                  dropout=dropout,
                                                  rotate=rotate)
        self.decoder = MLPDecoder(local_channels=embed_dim,
                                  global_channels=embed_dim,
                                  future_steps=future_steps,
                                  num_modes=num_modes,
                                  uncertain=True)
        # 初始化损失函数
        self.reg_loss = LaplaceNLLLoss(reduction='mean')  # 回归损失
        self.cls_loss = SoftTargetCrossEntropyLoss(reduction='mean')  # 分类损失

        # 初始化评估指标
        self.minADE = ADE()  # 平均位移误差
        self.minFDE = FDE()  # 最终位移误差
        self.minMR = MR()  # 误差率


    # 定义模型的前向传播过程
    def forward(self, data: TemporalData):
        # 如果模型设置为旋转坐标
        if self.rotate:
            # 初始化旋转矩阵，用于将数据旋转到特定的方向
            rotate_mat = torch.empty(data.num_nodes, 2, 2, device=self.device)
            # 计算旋转角度的正弦和余弦值
            sin_vals = torch.sin(data['rotate_angles'])
            cos_vals = torch.cos(data['rotate_angles'])
            # 填充旋转矩阵
            rotate_mat[:, 0, 0] = cos_vals
            rotate_mat[:, 0, 1] = -sin_vals
            rotate_mat[:, 1, 0] = sin_vals
            rotate_mat[:, 1, 1] = cos_vals
            # 如果数据中包含目标值y，则对其应用旋转变换
            if data.y is not None:
                data.y = torch.bmm(data.y, rotate_mat)
            # 将旋转矩阵保存在数据对象中，以便后续使用
            data['rotate_mat'] = rotate_mat
        else:
            # 如果不进行旋转，则设置旋转矩阵为None
            data['rotate_mat'] = None

        # 使用本地编码器对数据进行编码，得到本地嵌入表示
        local_embed = self.local_encoder(data=data)
        # 使用全局交互器对本地嵌入进行进一步处理，得到全局嵌入表示
        global_embed = self.global_interactor(data=data, local_embed=local_embed)
        # 使用解码器基于本地和全局嵌入表示生成最终的预测结果
        y_hat, pi = self.decoder(local_embed=local_embed, global_embed=global_embed)
        # 返回预测结果和模式概率
        return y_hat, pi


    # 定义训练时的操作
    def training_step(self, data, batch_idx):
        # 通过模型前向传播得到预测结果和模式概率
        y_hat, pi = self(data)
        # 计算有效的回归掩码，用于过滤掉填充的数据点
        reg_mask = ~data['padding_mask'][:, self.historical_steps:]
        # 计算每个样本中有效的时间步数量
        valid_steps = reg_mask.sum(dim=-1)
        # 计算分类任务的掩码，只有当存在有效时间步时，样本才参与分类损失的计算
        cls_mask = valid_steps > 0
        # 计算预测结果和真实值之间的L2范数，只考虑有效的数据点
        l2_norm = (torch.norm(y_hat[:, :, :, :2] - data.y, p=2, dim=-1) * reg_mask).sum(dim=-1)  # [F, N]
        # 对每个节点，找到具有最小L2范数的模式
        best_mode = l2_norm.argmin(dim=0)
        # 根据最佳模式选择对应的预测结果
        y_hat_best = y_hat[best_mode, torch.arange(data.num_nodes)]
        # 计算回归损失
        reg_loss = self.reg_loss(y_hat_best[reg_mask], data.y[reg_mask])
        # 为分类任务创建软目标，基于最佳模式的L2范数，使用softmax进行归一化
        soft_target = F.softmax(-l2_norm[:, cls_mask] / valid_steps[cls_mask], dim=0).t().detach()
        # 计算分类损失
        cls_loss = self.cls_loss(pi[cls_mask], soft_target)
        # 总损失为回归损失和分类损失之和
        loss = reg_loss + cls_loss
        # 在训练过程中记录回归损失
        self.log('train_reg_loss', reg_loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
        # 返回总损失
        return loss


    # 定义验证时的操作
    def validation_step(self, data, batch_idx):
        # 通过模型前向传播得到预测结果和模式概率
        y_hat, pi = self(data)
        # 计算有效的回归掩码，用于过滤掉填充的数据点
        reg_mask = ~data['padding_mask'][:, self.historical_steps:]
        # 计算预测结果和真实值之间的L2范数，只考虑有效的数据点
        l2_norm = (torch.norm(y_hat[:, :, :, :2] - data.y, p=2, dim=-1) * reg_mask).sum(dim=-1)  # [F, N]
        # 对每个节点，找到具有最小L2范数的模式
        best_mode = l2_norm.argmin(dim=0)
        # 根据最佳模式选择对应的预测结果
        y_hat_best = y_hat[best_mode, torch.arange(data.num_nodes)]
        # 计算回归损失
        reg_loss = self.reg_loss(y_hat_best[reg_mask], data.y[reg_mask])
        # 在日志中记录验证集上的回归损失
        self.log('val_reg_loss', reg_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=1)

        # 针对代理计算预测性能指标
        y_hat_agent = y_hat[:, data['agent_index'], :, :2]
        y_agent = data.y[data['agent_index']]
        # 计算代理的最终位移误差
        fde_agent = torch.norm(y_hat_agent[:, :, -1] - y_agent[:, -1], p=2, dim=-1)
        # 找到具有最小最终位移误差的模式
        best_mode_agent = fde_agent.argmin(dim=0)
        # 根据最佳模式选择对应的预测结果
        y_hat_best_agent = y_hat_agent[best_mode_agent, torch.arange(data.num_graphs)]
        # 更新性能指标
        self.minADE.update(y_hat_best_agent, y_agent)
        self.minFDE.update(y_hat_best_agent, y_agent)
        self.minMR.update(y_hat_best_agent, y_agent)
        # 在日志中记录这些性能指标
        self.log('val_minADE', self.minADE, prog_bar=True, on_step=False, on_epoch=True, batch_size=y_agent.size(0))
        self.log('val_minFDE', self.minFDE, prog_bar=True, on_step=False, on_epoch=True, batch_size=y_agent.size(0))
        self.log('val_minMR', self.minMR, prog_bar=True, on_step=False, on_epoch=True, batch_size=y_agent.size(0))

    # 定义配置优化器的方法
    def configure_optimizers(self):
        decay = set()  # 将使用权重衰减的参数名称集合
        no_decay = set()  # 不使用权重衰减的参数名称集合
        # 定义应该使用权重衰减的模块类型白名单
        whitelist_weight_modules = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.MultiheadAttention, nn.LSTM, nn.GRU)
        # 定义不应使用权重衰减的模块类型黑名单
        blacklist_weight_modules = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.Embedding)
        # 遍历模型中的所有模块和参数
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                # 构造完整的参数名称
                full_param_name = '%s.%s' % (module_name, param_name) if module_name else param_name
                # 根据参数名称和模块类型决定是否使用权重衰减
                if 'bias' in param_name:
                    # 偏置参数不使用权重衰减
                    no_decay.add(full_param_name)
                elif 'weight' in param_name:
                    # 权重参数根据模块类型判断是否使用权重衰减
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                else:
                    # 非权重非偏置参数不使用权重衰减
                    no_decay.add(full_param_name)
        param_dict = {param_name: param for param_name, param in self.named_parameters()}
        # 验证参数集合的完整性和互斥性
        inter_params = decay & no_decay  # 应为空集，确保没有参数同时出现在两个集合中
        union_params = decay | no_decay  # 应包含所有参数，确保每个参数都被分类
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0

        # 构建优化器参数组，设置不同参数组的权重衰减策略
        optim_groups = [
            {"params": [param_dict[param_name] for param_name in sorted(list(decay))],
            "weight_decay": self.weight_decay},
            {"params": [param_dict[param_name] for param_name in sorted(list(no_decay))],
            "weight_decay": 0.0},
        ]

        # 初始化AdamW优化器
        optimizer = torch.optim.AdamW(optim_groups, lr=self.lr, weight_decay=self.weight_decay)
        # 初始化余弦退火学习率调度器
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.T_max, eta_min=0.0)
        # 返回优化器和调度器
        return [optimizer], [scheduler]


    @staticmethod
    def add_model_specific_args(parent_parser):
        # 创建一个参数组，专门用于HiVT模型的参数
        parser = parent_parser.add_argument_group('HiVT')
        # 向参数解析器中添加HiVT模型特定的命令行参数
        parser.add_argument('--historical_steps', type=int, default=20)  # 历史时间步数
        parser.add_argument('--future_steps', type=int, default=30)  # 未来时间步数
        parser.add_argument('--num_modes', type=int, default=6)  # 预测模式的数量
        parser.add_argument('--rotate', type=bool, default=True)  # 是否旋转坐标
        parser.add_argument('--node_dim', type=int, default=2)  # 节点特征的维度
        parser.add_argument('--edge_dim', type=int, default=2)  # 边特征的维度
        parser.add_argument('--embed_dim', type=int, required=True)  # 嵌入向量的维度，必须指定
        parser.add_argument('--num_heads', type=int, default=8)  # 多头注意力机制中的头数
        parser.add_argument('--dropout', type=float, default=0.1)  # Dropout比率
        parser.add_argument('--num_temporal_layers', type=int, default=4)  # 时间编码层的数量
        parser.add_argument('--num_global_layers', type=int, default=3)  # 全局交互层的数量
        parser.add_argument('--local_radius', type=float, default=50)  # 本地搜索半径
        parser.add_argument('--parallel', type=bool, default=False)  # 是否并行处理
        parser.add_argument('--lr', type=float, default=5e-4)  # 学习率
        parser.add_argument('--weight_decay', type=float, default=1e-4)  # 权重衰减系数
        parser.add_argument('--T_max', type=int, default=64)  # 余弦退火调度器的周期长度
        # 返回更新后的父解析器
        return parent_parser

