import torch
from argoverse.map_representation.map_api import ArgoverseMap
from datasets.argoverse_v1_dataset import ArgoverseV1Dataset 

def main():
    # 设定数据集的根目录（请根据你的数据集位置进行修改）
    root_dir = '/data/yangrn/try_10000'
    # 初始化Argoverse地图API
    am = ArgoverseMap()
    # 假设我们处理训练集，并且设置本地搜索半径为50米
    dataset = ArgoverseV1Dataset(root=root_dir, split='train', local_radius=50)

    # 打印数据集的长度，即样本数量
    print(f'Dataset length: {len(dataset)}')

    # 获取并打印第一个样本的信息（仅作为示例，实际操作可能需要更复杂的处理）
    file_path = '/data/yangrn/try_10000/10000_pruning_temporalData_info.txt'
    with open(file_path, 'w', encoding='utf-8') as file:
        for each in dataset:
            # 将每个元素写入文件，每个元素占一行
            file.write(str(each) + '\n')

    # PIT: 49; MIA: 51
    # [8, 10, 46, 11, 19, 40, 11, 13, 28, 19, 9, 52, 10, 35, 10, 21, 21, 15, 9, 34, 16, 15, 11, 19, 16, 15, 11, 15, 25, 17, 11, 12, 26, 15, 26, 26, 23, 9, 12, 42, 17, 31, 15, 19, 29, 26, 19, 37, 27, 17, 25, 20, 19, 11, 21, 39, 20, 13, 25, 15, 8, 29, 21, 21, 13, 9, 46, 23, 19, 12, 35, 30, 72, 40, 39, 21, 32, 17, 21, 34, 21, 29, 16, 4, 17, 21, 22, 41, 28, 4, 12, 29, 11, 43, 16, 32, 47, 22, 15, 21]
    # x_list = []
    # seq_id_groups = {}

    # for each in dataset:
    #     each_pair = [each.seq_id, each.x.shape[0]]
    #     x_list.append(each_pair)

    #     num_actor = each.x.shape[0]
    #     range_key = (num_actor // 10 * 10, num_actor // 10 * 10 + 10)

    #     if range_key not in seq_id_groups:
    #         seq_id_groups[range_key] = []
        
    #     seq_id_groups[range_key].append(each.seq_id)

    # print("x_list: ", x_list)
    # for range_key, seq_ids in seq_id_groups.items():
    #     print(f"Range {range_key}: {seq_ids}")

    
    # sample = dataset[9]
    # print(sample)

if __name__ == '__main__':
    main()
