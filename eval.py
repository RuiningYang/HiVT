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
from argparse import ArgumentParser

import pytorch_lightning as pl
from torch_geometric.data import DataLoader

from datasets import ArgoverseV1Dataset
from models.hivt import HiVT

if __name__ == '__main__':
    # 设置随机种子，确保实验结果具有可重复性
    pl.seed_everything(2022)

    # 创建一个命令行参数解析器
    parser = ArgumentParser()
    # 添加一个命令行参数，用来指定数据集的根目录
    parser.add_argument('--root', type=str, required=True)
    # 添加一个命令行参数，用来指定每个批次处理的数据量
    parser.add_argument('--batch_size', type=int, default=32)
    # 添加一个命令行参数，用来指定加载数据时使用的进程数
    parser.add_argument('--num_workers', type=int, default=8)
    # 添加一个命令行参数，用来指定是否在加载数据时将数据锁定在内存中，这有助于加快数据加载速度
    parser.add_argument('--pin_memory', type=bool, default=True)
    # 添加一个命令行参数，用来指定数据加载器是否应该使用持久化工作进程，这有助于提高加载效率
    parser.add_argument('--persistent_workers', type=bool, default=True)
    # 添加一个命令行参数，用来指定训练过程中使用的GPU数量
    parser.add_argument('--gpus', type=int, default=1)
    # 添加一个命令行参数，用来指定模型检查点（checkpoint）的路径
    parser.add_argument('--ckpt_path', type=str, required=True)
    # 解析命令行参数
    args = parser.parse_args()

    # 根据命令行参数创建一个训练器实例
    trainer = pl.Trainer.from_argparse_args(args)
    # 从指定的检查点路径加载模型，并设置是否并行处理
    model = HiVT.load_from_checkpoint(checkpoint_path=args.ckpt_path, parallel=True)
    # 创建验证集数据集实例，使用指定的根目录和本地半径参数
    val_dataset = ArgoverseV1Dataset(root=args.root, split='val', local_radius=model.hparams.local_radius)
    # 创建一个数据加载器，用于批量加载验证数据集，同时指定是否乱序、工作进程数量、是否锁存内存和是否使用持久化工作进程
    dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=args.pin_memory, persistent_workers=args.persistent_workers)
    # 使用训练器和数据加载器对模型进行验证
    trainer.validate(model, dataloader)
