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
import os
from itertools import permutations
from itertools import product
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from argoverse.map_representation.map_api import ArgoverseMap
from torch_geometric.data import Data
from torch_geometric.data import Dataset
from tqdm import tqdm

from utils import TemporalData


# 定义一个处理Argoverse数据集的类，继承自Dataset基类
class ArgoverseV1Dataset(Dataset):

    # 类的初始化方法
    def __init__(self,
                 root: str,  # 数据集的根目录
                 split: str,  # 数据集的分割类型（如训练集、验证集等）
                 transform: Optional[Callable] = None,  # 可选的，用于数据变换的函数
                 local_radius: float = 50) -> None:  # 本地搜索半径，默认值为50
        self._split = split  # 存储数据集的分割类型
        self._local_radius = local_radius  # 存储本地搜索半径
        # 根据分割类型构建数据集的下载URL
        self._url = f'https://s3.amazonaws.com/argoai-argoverse/forecasting_{split}_v1.1.tar.gz'
        
        # 根据分割类型确定数据集的目录名
        if split == 'sample':
            self._directory = 'forecasting_sample'
        elif split == 'train':
            self._directory = 'train'
        elif split == 'val':
            self._directory = 'val'
        elif split == 'test':
            self._directory = 'test_obs'
        else:
            # 如果分割类型不是预定义的类型之一，则抛出异常
            raise ValueError(split + ' is not valid')
            
        self.root = root  # 存储数据集的根目录
        # 获取原始文件目录下的所有文件名
        self._raw_file_names = os.listdir(self.raw_dir)
        # 处理后的文件名，将原始文件名的扩展名替换为.pt, 18907.csv -> 18907.pt
        # 这种处理方式是为了标识这些文件已经被预处理过，并且保存为PyTorch的Tensor格式，便于后续直接加载使用。
        self._processed_file_names = [os.path.splitext(f)[0] + '.pt' for f in self.raw_file_names]
        # 构建处理后的文件路径列表: 拼接每个处理后的文件名(已经转为.pt格式)与处理后数据文件的目录路径。
        # 这样做的目的是为了方便后续直接通过文件路径来访问或加载这些处理过的数据
        # "/data/argoverse/train/processed/18907.pt",
        # "/data/argoverse/train/processed/28082.pt"
        self._processed_paths = [os.path.join(self.processed_dir, f) for f in self._processed_file_names]
        
        # 调用基类的初始化方法
        super(ArgoverseV1Dataset, self).__init__(root, transform=transform)

    @property
    def raw_dir(self) -> str:
        # 返回原始数据文件的目录路径
        return os.path.join(self.root, self._directory, 'data')

    @property
    def processed_dir(self) -> str:
        # 返回处理后数据文件的目录路径
        return os.path.join(self.root, self._directory, 'processed')

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        # 返回原始数据文件的文件名列表
        return self._raw_file_names

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        # 返回处理后数据文件的文件名列表
        return self._processed_file_names

    @property
    def processed_paths(self) -> List[str]:
        # 返回处理后数据文件的完整路径列表
        return self._processed_paths

    def process(self) -> None:
        # 处理原始数据文件，将它们转换成模型可以直接使用的格式
        # print('1111')
        am = ArgoverseMap()
        for raw_path in tqdm(self.raw_paths):
            # 使用process_argoverse函数处理每个原始数据文件
            kwargs = process_argoverse(self._split, raw_path, am, self._local_radius)
            # 将处理结果封装成TemporalData对象
            data = TemporalData(**kwargs)
            # print_file_path = "/data/yangrn/try/data_18907_info.txt"
            # # print("path: ", self.raw_paths)
            # with open(print_file_path, "w") as file:
            #     # 写入路径信息
            #     file.write("path: " + str(self.raw_paths) + "\n")
            
            #     # 遍历Data对象的所有属性并写入文件
            #     for key, value in data.__dict__.items():
            #         file.write(f"{key}: {value}\n")
            # 将处理后的数据保存为.pt文件
            torch.save(data, os.path.join(self.processed_dir, str(kwargs['seq_id']) + '.pt'))

    def len(self) -> int:
        # 返回数据集中的样本数
        return len(self._raw_file_names)

    def get(self, idx) -> Data:
        # 根据索引加载并返回一个样本
        return torch.load(self.processed_paths[idx])



# 定义一个函数用于处理Argoverse数据集的特定部分
def process_argoverse(split: str, # train, val, test? 数可能会以不同的方式处理数据，特别是在如何处理目标变量y方面，（例如，在测试集中可能不包含目标变量）。
                      raw_path: str, # Argoverse数据文件的路径。这个文件路径指向的是原始数据文件，通常是一个CSV文件
                      am: ArgoverseMap, # 一个ArgoverseMap对象，提供了一套API来访问和使用Argoverse数据集中的地图数据。这些地图数据包括道路网络、车道信息、交叉口、车道边界等
                      radius: float) -> Dict: # 定义了一个查询范围的半径，单位是米。它用于确定在演员周围多大范围内的车道信息是相关的并应被提取
    # 读取原始数据文件到DataFrame中
    df = pd.read_csv(raw_path)
    # print("2222")
    print('raw_path: ', raw_path)

    # 筛选出在历史时间步骤中出现的动态实体（例如，车辆或行人）
    timestamps = list(np.sort(df['TIMESTAMP'].unique()))  # 将所有独特的时间戳排序
    # print("timestamps: ", timestamps)
    historical_timestamps = timestamps[:20]  # 选取前20个时间戳作为历史时间步
    # print("historical_timestamps: ", historical_timestamps)
    historical_df = df[df['TIMESTAMP'].isin(historical_timestamps)]  # 筛选出历史时间步中的数据
    # print("hist_df: ", historical_df)

    # 计算每个track_id出现的次数
    # track_id_counts = historical_df['TRACK_ID'].value_counts()

    actor_ids = list(historical_df['TRACK_ID'].unique())  # 获取出现在历史时间步中的所有动态实体ID
    # print("hist_track_id: ", actor_ids)
    df = df[df['TRACK_ID'].isin(actor_ids)]  # 从原始DataFrame中筛选出这些动态实体的所有数据
    # print("actor_data: ", df)
    
    num_nodes = len(actor_ids)  # 计算动态实体的数量
    # print("num_nodes: ", num_nodes)

    # print('1')

    # x_list = []
    # seq_id_groups = {}

    
    # each_pair = [raw_path, num_nodes]
    # x_list.append(each_pair)

    
    # range_key = (num_nodes // 10 * 10, num_nodes // 10 * 10 + 10)

    # if range_key not in seq_id_groups:
    #     seq_id_groups[range_key] = {}
        
    # # seq_id_groups[range_key].append(df)
    # # 对每个范围的每个raw_path，存储track_id和它的计数
    # seq_id_groups[range_key][raw_path] = track_id_counts

    # print_file = "/data/yangrn/try/trackid_num.txt"
    # print("x_list: ", x_list)
    # # for range_key, seq_ids in seq_id_groups.items():
    # # 使用with语句打开文件，'w'模式表示写入模式，如果文件已存在则覆盖
    # with open(print_file, 'a') as file:
    #     # 在新内容前空一行
    #     file.write('\n')
    #     # 遍历seq_id_groups中的每个范围和对应的路径
    #     for range_key, paths in seq_id_groups.items():
    #         # 将信息写入文件
    #         file.write(f"Range {range_key}:\n")
    #         for raw_path, track_ids in paths.items():
    #             # 再次写入文件，对于每个路径和对应的track_id计数
    #             file.write(f"  {raw_path}: {track_ids}\n")
    


    # assert(0)

    # 选取自动驾驶车辆（AV）的数据
    av_df = df[df['OBJECT_TYPE'] == 'AV'].iloc  # 筛选出对象类型为AV的数据行
    av_index = actor_ids.index(av_df[0]['TRACK_ID'])  # 获取AV在动态实体ID列表中的索引
    # 选取代理车辆（AGENT）的数据
    agent_df = df[df['OBJECT_TYPE'] == 'AGENT'].iloc  # 筛选出对象类型为AGENT的数据行
    agent_index = actor_ids.index(agent_df[0]['TRACK_ID'])  # 获取AGENT在动态实体ID列表中的索引
    city = df['CITY_NAME'].values[0]  # 获取数据所在城市的名称

    # 以自动驾驶车辆（AV）为中心调整场景
    origin = torch.tensor([av_df[19]['X'], av_df[19]['Y']], dtype=torch.float)  # 获取AV在最后一个历史时间步的位置作为原点
    av_heading_vector = origin - torch.tensor([av_df[18]['X'], av_df[18]['Y']], dtype=torch.float)  # 计算AV的朝向向量，通过比较最后两个历史时间步的位置
    theta = torch.atan2(av_heading_vector[1], av_heading_vector[0])  # 计算AV的朝向角度
    rotate_mat = torch.tensor([[torch.cos(theta), -torch.sin(theta)],
                            [torch.sin(theta), torch.cos(theta)]])  # 构造一个旋转矩阵，用于调整其他动态实体的位置，使得AV朝向正北方向

    # 初始化
    x = torch.zeros(num_nodes, 50, 2, dtype=torch.float)  # 初始化一个张量用于存储所有动态实体的位置
    edge_index = torch.LongTensor(list(permutations(range(num_nodes), 2))).t().contiguous()  # 创建一个边索引，表示动态实体之间的所有可能关系
    padding_mask = torch.ones(num_nodes, 50, dtype=torch.bool)  # 初始化一个掩码，用于标记哪些时间步有有效的动态实体位置
    bos_mask = torch.zeros(num_nodes, 20, dtype=torch.bool)  # 初始化一个掩码，用于标记序列开始的位置
    rotate_angles = torch.zeros(num_nodes, dtype=torch.float)  # 初始化一个张量，用于存储每个动态实体的朝向角度

    # 遍历每个动态实体并处理其数据
    for actor_id, actor_df in df.groupby('TRACK_ID'):
        node_idx = actor_ids.index(actor_id)  # 获取当前动态实体的索引
        node_steps = [timestamps.index(timestamp) for timestamp in actor_df['TIMESTAMP']]  # 将动态实体出现的时间戳转换为索引
        padding_mask[node_idx, node_steps] = False  # 更新掩码，标记动态实体在这些时间步中出现
        if padding_mask[node_idx, 19]:  # 如果动态实体在当前时间步未出现，则在此之后不做预测
            padding_mask[node_idx, 20:] = True
        xy = torch.from_numpy(np.stack([actor_df['X'].values, actor_df['Y'].values], axis=-1)).float()  # 获取动态实体的位置信息
        x[node_idx, node_steps] = torch.matmul(xy - origin, rotate_mat)  # 调整动态实体的位置，使其相对于AV的位置和朝向
        node_historical_steps = list(filter(lambda node_step: node_step < 20, node_steps))  # 筛选出历史时间步
        if len(node_historical_steps) > 1:  # 如果动态实体在多个历史时间步中出现，计算其朝向
            heading_vector = x[node_idx, node_historical_steps[-1]] - x[node_idx, node_historical_steps[-2]]  # 通过位置变化计算朝向向量
            rotate_angles[node_idx] = torch.atan2(heading_vector[1], heading_vector[0])  # 计算朝向角度
        else:  # 如果有效时间步少于2，不对该动态实体进行预测
            padding_mask[node_idx, 20:] = True


    # 如果时间步t有效且时间步t-1无效，则bos_mask在该位置为True
    bos_mask[:, 0] = ~padding_mask[:, 0]  # 对于序列的第一个时间步，如果padding_mask为False（表示时间步有效），则bos_mask为True
    bos_mask[:, 1:20] = padding_mask[:, :19] & ~padding_mask[:, 1:20]  # 对于其他时间步，如果当前步有效且前一步无效，则标记为True

    positions = x.clone()  # 复制位置数据，用于保留原始位置信息
    # 对于时间步20及之后的位置，如果当前步或前一步被标记为无效，则位置设为0，否则计算相对于时间步19的位置变化
    x[:, 20:] = torch.where((padding_mask[:, 19].unsqueeze(-1) | padding_mask[:, 20:]).unsqueeze(-1),
                            torch.zeros(num_nodes, 30, 2),
                            x[:, 20:] - x[:, 19].unsqueeze(-2))
    # 对于时间步1到19，如果当前步或前一步被标记为无效，则位置设为0，否则计算相对于前一步的位置变化
    x[:, 1:20] = torch.where((padding_mask[:, :19] | padding_mask[:, 1:20]).unsqueeze(-1),
                            torch.zeros(num_nodes, 19, 2),
                            x[:, 1:20] - x[:, :19])
    x[:, 0] = torch.zeros(num_nodes, 2)  # 将序列的第一个位置设为0，因为没有前一步的位置进行比较

    # 在当前时间步获取车道特征
    df_19 = df[df['TIMESTAMP'] == timestamps[19]]  # 筛选出时间步19的数据
    node_inds_19 = [actor_ids.index(actor_id) for actor_id in df_19['TRACK_ID']]  # 获取时间步19出现的动态实体的索引
    node_positions_19 = torch.from_numpy(np.stack([df_19['X'].values, df_19['Y'].values], axis=-1)).float()  # 获取这些动态实体的位置
    # 调用get_lane_features函数，提取与这些位置相关的车道特征
    (lane_vectors, is_intersections, turn_directions, traffic_controls, lane_actor_index,
    lane_actor_vectors) = get_lane_features(am, node_inds_19, node_positions_19, origin, rotate_mat, city, radius)

    y = None if split == 'test' else x[:, 20:]  # 如果是测试集，不包含目标数据y；否则，y为时间步20之后的位置数据
    seq_id = os.path.splitext(os.path.basename(raw_path))[0]  # 从文件名提取序列ID


    return {
        'x': x[:, : 20],  # 时间步0到19的所有动态实体的位置，相对于自动驾驶车辆（AV）进行了调整。
                            # [N, 20, 2], N代表场景中动态实体的数量，20代表选择的时间步数量（这里是历史时间步）,2代表每个时间步动态实体的位置坐标（通常是x和y坐标）
        'positions': positions,  # 所有动态实体在所有50个时间步的原始位置数据。
                            # [N, 50, 2], N代表场景中动态实体的数量, 50代表总的时间步数量（包括历史和未来的位置）,2代表每个时间步动态实体的位置坐标 
        'edge_index': edge_index,  # 表示动态实体之间所有可能二元关系的边索引。
                            # [2, N x N - 1],这是一个用于图神经网络的边索引，表示节点间的连接关系。2表示每个边连接的两个节点的索引,N x N - 1代表在完全连接图中除了自环外所有可能的边的数量。 
        'y': y,  #  时间步20到49的位置数据，用作模型的预测目标。在测试集中，这个值可能为None。
                    # [N, 30, 2], N代表场景中动态实体的数量, 30代表未来时间步的数量（在给定的上下文中是时间步20到49）,2代表每个时间步动态实体的位置坐标 
        'num_nodes': num_nodes, # 场景中动态实体的总数  
        'padding_mask': padding_mask, #每个动态实体在每个时间步是否存在的掩码。如果动态实体在特定时间步未出现，则对应位置为True。
                    # [N, 50], N代表场景中动态实体的数量, 50代表总的时间步数量（包括历史和未来的位置）这个掩码张量用于标记每个动态实体在每个时间步是否有有效的位置数据。
        'bos_mask': bos_mask, # 标记序列开始的掩码，用于区分有效的开始位置。
                     # [N, 20], N代表场景中动态实体的数量，20代表选择的时间步数量（这里是历史时间步)这个掩码用于标记序列开始的位置，通常用于处理变长序列的任务。
        'rotate_angles': rotate_angles,  # 每个动态实体的朝向角度。
                    # [N],N代表场景中动态实体的数量
        'lane_vectors': lane_vectors, #  提取的车道向量，表示车道的方向和长度。
                     # [L, 2], L代表场景中车道线段的数量, 2代表车道线段的向量表示（通常是方向和长度）。
        'is_intersections': is_intersections, # 指示每条车道是否位于交叉口的布尔值
                     # [L]
        'turn_directions': turn_directions, # 每条车道的转向方向，如直行、左转或右转
                     # [L]
        'traffic_controls': traffic_controls, # 每条车道是否有交通控制措施的布尔值
                      # [L]
        'lane_actor_index': lane_actor_index, # 车道和动态实体之间关系的索引
                     # [2, E_{A-L}], 2代表每个关系对由两个元素组成，即一个车道节点的索引和一个动态实体节点的索引。E_{A-L}代表所有车道与动态实体之间关系对的数量
        'lane_actor_vectors': lane_actor_vectors, # 车道和动态实体之间的相对位置向量
                    # [E_{A-L}, 2], E_{A-L}代表所有车道与动态实体之间关系对的数量, 2代表车道与动态实体之间相对位置向量的x和y坐标
        'seq_id': int(seq_id), # 数据序列的ID, 18907
                    # 18907.csv
        'av_index': av_index, # 自动驾驶车辆（AV）在动态实体列表中的索引
                    # 0
        'agent_index': agent_index, # 代理车辆（AGENT）在动态实体列表中的索引
                    # 4
        'city': city, # 场景所在的城市
                    # PIT
        'origin': origin.unsqueeze(0), # 场景的原点位置，以自动驾驶车辆（AV）在最后一个历史时间步的位置为准
                    # [1, 2], 1代表只有一个原点坐标。这意味着我们不是在处理一系列的点，而是只关注一个具体的点，这个点用于整个数据集或场景的参考。
                    # 2表示每个坐标点由两个数值组成，即在二维空间中的x和y坐标
        'theta': theta, # 场景旋转的角度，以自动驾驶车辆的朝向为基准进行调整
    }


# 定义一个函数用于获取车道特征
def get_lane_features(am: ArgoverseMap,
                      node_inds: List[int], # 一个整数列表，代表需要提取车道特征的节点（或演员）索引
                      node_positions: torch.Tensor, # 这是一个张量，包含了每个节点（演员）的位置坐标。 [N, 2], N是演员数量，2代表二维空间中的x和y坐标
                      origin: torch.Tensor, # 表示数据处理或坐标转换的原点, 表示一个二维坐标
                      rotate_mat: torch.Tensor, # 这是一个旋转矩阵，用于将位置坐标旋转到一个新的坐标系中. [2, 2],代表一个二维旋转矩阵
                      city: str,
                      radius: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                                              torch.Tensor]:
    # 初始化几个列表，用于存储不同的车道特征
    lane_positions, lane_vectors, is_intersections, turn_directions, traffic_controls = [], [], [], [], []
    # 初始化一个集合，用于存储车道ID
    lane_ids = set()
    # 遍历每个节点位置
    for node_position in node_positions:
        # 更新车道ID集合，包括在给定半径内的所有车道ID
        lane_ids.update(am.get_lane_ids_in_xy_bbox(node_position[0], node_position[1], city, radius))
    # 对节点位置应用旋转和平移变换
    node_positions = torch.matmul(node_positions - origin, rotate_mat).float()
    # 遍历每个车道ID
    for lane_id in lane_ids:
        # 获取并转换车道中心线坐标
        lane_centerline = torch.from_numpy(am.get_lane_segment_centerline(lane_id, city)[:, :2]).float()
        lane_centerline = torch.matmul(lane_centerline - origin, rotate_mat)
        # 判断车道是否位于交叉口
        is_intersection = am.lane_is_in_intersection(lane_id, city)
        # 获取车道转向方向
        turn_direction = am.get_lane_turn_direction(lane_id, city)
        # 判断车道是否有交通控制措施
        traffic_control = am.lane_has_traffic_control_measure(lane_id, city)
        # 计算并存储车道位置和向量
        lane_positions.append(lane_centerline[:-1])
        lane_vectors.append(lane_centerline[1:] - lane_centerline[:-1])
        # 计算车道段的数量
        count = len(lane_centerline) - 1
        # 存储车道是否在交叉口和交通控制的信息
        is_intersections.append(is_intersection * torch.ones(count, dtype=torch.uint8))
        # 转换转向方向为数字编码
        if turn_direction == 'NONE':
            turn_direction = 0
        elif turn_direction == 'LEFT':
            turn_direction = 1
        elif turn_direction == 'RIGHT':
            turn_direction = 2
        else:
            # 如果转向方向无效，则抛出异常
            raise ValueError('turn direction is not valid')
        # 存储转向方向信息
        turn_directions.append(turn_direction * torch.ones(count, dtype=torch.uint8))
        # 存储交通控制信息
        traffic_controls.append(traffic_control * torch.ones(count, dtype=torch.uint8))
    # 将各个列表合并成张量
    lane_positions = torch.cat(lane_positions, dim=0)
    lane_vectors = torch.cat(lane_vectors, dim=0)
    is_intersections = torch.cat(is_intersections, dim=0)
    turn_directions = torch.cat(turn_directions, dim=0)
    traffic_controls = torch.cat(traffic_controls, dim=0)

    # 计算车道向量和节点索引的笛卡尔积，用于后续计算
    lane_actor_index = torch.LongTensor(list(product(torch.arange(lane_vectors.size(0)), node_inds))).t().contiguous()
    # 计算车道位置和节点位置之间的向量
    lane_actor_vectors = \
        lane_positions.repeat_interleave(len(node_inds), dim=0) - node_positions.repeat(lane_vectors.size(0), 1)
    # 创建一个掩码，用于筛选出在给定半径内的车道向量
    mask = torch.norm(lane_actor_vectors, p=2, dim=-1) < radius
    # 应用掩码
    lane_actor_index = lane_actor_index[:, mask]
    lane_actor_vectors = lane_actor_vectors[mask]

    # 返回车道向量、是否在交叉口、转向方向、交通控制信息、车道与节点的索引、车道与节点之间的向量
    return lane_vectors, is_intersections, turn_directions, traffic_controls, lane_actor_index, lane_actor_vectors

