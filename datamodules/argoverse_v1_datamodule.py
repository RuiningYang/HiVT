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
from typing import Callable, Optional

from pytorch_lightning import LightningDataModule
from torch_geometric.data import DataLoader

from datasets import ArgoverseV1Dataset


# 定义一个数据模块类，用于处理Argoverse数据集
class ArgoverseV1DataModule(LightningDataModule):

    # 类的初始化方法
    def __init__(self,
                 root: str,  # 数据集的根目录
                 train_batch_size: int,  # 训练集批量大小
                 val_batch_size: int,  # 验证集批量大小
                 shuffle: bool = True,  # 是否在训练时打乱数据
                 num_workers: int = 8,  # 加载数据时使用的进程数
                 pin_memory: bool = True,  # 是否在加载数据时锁定内存，加快数据传输速度
                 persistent_workers: bool = True,  # 是否使用持久化工作进程来加载数据
                 train_transform: Optional[Callable] = None,  # 训练集数据变换函数
                 val_transform: Optional[Callable] = None,  # 验证集数据变换函数
                 local_radius: float = 50) -> None:  # 本地搜索半径，用于数据处理
        super(ArgoverseV1DataModule, self).__init__()  # 调用父类的初始化方法
        # 初始化类属性
        self.root = root
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.num_workers = num_workers
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.local_radius = local_radius

    # 准备数据的方法
    def prepare_data(self) -> None:
        print('3')
        # 实例化ArgoverseV1Dataset，准备训练和验证数据
        ArgoverseV1Dataset(self.root, 'train', self.train_transform, self.local_radius)
        ArgoverseV1Dataset(self.root, 'val', self.val_transform, self.local_radius)

    # 设置数据的方法，根据不同的阶段（训练或验证）来准备数据集
    def setup(self, stage: Optional[str] = None) -> None:
        print('1')
        # 创建训练和验证数据集的实例
        self.train_dataset = ArgoverseV1Dataset(self.root, 'train', self.train_transform, self.local_radius)
        self.val_dataset = ArgoverseV1Dataset(self.root, 'val', self.val_transform, self.local_radius)

    # 获取训练数据加载器的方法
    def train_dataloader(self):
        print('2')
        # 创建并返回训练数据加载器
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=self.shuffle,
                          num_workers=self.num_workers, pin_memory=self.pin_memory,
                          persistent_workers=self.persistent_workers)

    # 获取验证数据加载器的方法
    def val_dataloader(self):
        # 创建并返回验证数据加载器
        return DataLoader(self.val_dataset, batch_size=self.val_batch_size, shuffle=False, num_workers=self.num_workers,
                          pin_memory=self.pin_memory, persistent_workers=self.persistent_workers)

