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
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from datamodules import ArgoverseV1DataModule
from models.hivt import HiVT
import os

# 如果这个文件作为主程序运行
if __name__ == '__main__':
    # 设置全局随机种子以保证实验的可复现性
    pl.seed_everything(2022)

    # 创建命令行参数解析器
    parser = ArgumentParser()
    # 添加命令行参数
    parser.add_argument('--root', type=str, required=True)  # 数据集的根目录
    parser.add_argument('--train_batch_size', type=int, default=32)  # 训练批次大小
    parser.add_argument('--val_batch_size', type=int, default=32)  # 验证批次大小
    parser.add_argument('--shuffle', type=bool, default=True)  # 是否在训练时打乱数据
    parser.add_argument('--num_workers', type=int, default=8)  # 加载数据时使用的工作进程数
    parser.add_argument('--pin_memory', type=bool, default=True)  # 是否在加载数据时锁定内存
    parser.add_argument('--persistent_workers', type=bool, default=True)  # 是否使用持久工作进程加载数据
    parser.add_argument('--gpus', type=int, default=1)  # 使用的GPU数量
    parser.add_argument('--max_epochs', type=int, default=64)  # 训练的最大轮数
    parser.add_argument('--monitor', type=str, default='val_minFDE', choices=['val_minADE', 'val_minFDE', 'val_minMR'])  # 监控的验证指标
    parser.add_argument('--save_top_k', type=int, default=5)  # 保存性能最好的k个模型
    # 添加模型特定的参数
    parser = HiVT.add_model_specific_args(parser)
    # 解析命令行参数
    args = parser.parse_args()
    
    
     # 解析root参数以包括子目录（如10k_1/0.05）
    root_path_parts = args.root.strip('/').split('/')
    sub_dir = '/'.join(root_path_parts[-3:])  # 取最后3部分作为子目录
     
     # 日志文件路径
    log_file_path = os.path.join('/data/yangrn/HiVT', 'training_logs.txt')
    
     # 模型检查点保存目录
    checkpoint_dir = os.path.join('/data/yangrn/HiVT/lightning_logs', sub_dir)
    

    # 初始化模型检查点回调，用于保存性能最好的模型
    model_checkpoint = ModelCheckpoint(monitor=args.monitor, save_top_k=args.save_top_k, mode='min',dirpath=checkpoint_dir,filename='{epoch}-{step}-{val_minFDE:.2f}')
    
    # CSV Logger
    csv_logger = CSVLogger(save_dir='/data/yangrn/HiVT/logs', name=sub_dir)
    
    
    # 初始化训练器，并传入命令行参数和回调
    trainer = pl.Trainer.from_argparse_args(args, callbacks=[model_checkpoint],logger=[csv_logger])
    # 使用命令行参数初始化模型
    # 将args象中的所有属性转换为字典，然后将这个字典解包为关键字参数（key-value对）传递给HiVT的构造函数。
    # --embed_dim 128 --lr 0.001 -> args.embed_dim = 128, args.lr = 0.001 -> {'embed_dim': 128, 'lr': 0.001}
    # 返回HiVT类的一个实例
    print("para HiVT: ", vars(args))# para HiVT:  {'root': '/data/yangrn/try_100', 'train_batch_size': 32, 'val_batch_size': 32, 'shuffle': True, 'num_workers': 8, 'pin_memory': True, 'persistent_workers': True, 'gpus': 1, 'max_epochs': 64, 'monitor': 'val_minFDE', 'save_top_k': 5,
    
    
    # 'historical_steps': 20, 'future_steps': 30, 'num_modes': 6, 'rotate': True, 'node_dim': 2, 'edge_dim': 2, 'embed_dim': 64, 'num_heads': 8, 'dropout': 0.1, 'num_temporal_layers': 4, 'num_global_layers': 3, 'local_radius': 50, 'parallel': False, 'lr': 0.0005, 'weight_decay': 0.0001, 'T_max': 64}

    model = HiVT(**vars(args))
    # 使用命令行参数初始化数据模块
    datamodule = ArgoverseV1DataModule.from_argparse_args(args)
    # 开始训练模型
    trainer.fit(model, datamodule)
    
    # 记录模型保存信息和使用的数据集目录
    with open(log_file_path, 'a') as log_file:
        log_file.write(f"Training completed with root={args.root}, Model saved at {model_checkpoint.dirpath}\n")

    # train_loader = datamodule.train_dataloader()

    # dataloader_list = []
    #     # 遍历并打印前5个数据项
    # for i, data in enumerate(train_loader):
    #     dataloader_list.append(data)
    #     print(f"Data {i+1}:")
    #     print(data)


    # print("datamodule length: ", len(dataloader_list))
    # print("datamodule: ", dataloader_list)
    # print("Done")

