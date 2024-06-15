import os
import pandas as pd
import numpy as np

def process_csv_file(raw_path, delete_csv_path):
    print('-----------------------------------------------------------------')
    # 读取原始数据文件到DataFrame中
    df = pd.read_csv(raw_path)

    # 筛选出在历史时间步骤中出现的动态实体（例如，车辆或行人）
    timestamps = list(np.sort(df['TIMESTAMP'].unique()))  # 将所有独特的时间戳排序
    historical_timestamps = timestamps[:20]  # 选取前20个时间戳作为历史时间步
    historical_df = df[df['TIMESTAMP'].isin(historical_timestamps)]  # 筛选出历史时间步中的数据

    # 计算每个track_id出现的次数
    track_id_counts = historical_df['TRACK_ID'].value_counts()

    actor_ids = list(historical_df['TRACK_ID'].unique())  # 获取出现在历史时间步中的所有动态实体ID
    df = df[df['TRACK_ID'].isin(actor_ids)]  # 从原始DataFrame中筛选出这些动态实体的所有数据
    
    num_nodes = len(actor_ids)  # 计算动态实体的数量
    print('num_nodes: ', num_nodes)

    track_id_count_dict = track_id_counts.to_dict()
    # 打印track_id和它出现的次数的字典
    # print(f"Track ID count dictionary for {raw_path}:")
    # print(track_id_count_dict)



    # 选择需要删掉的track_id
    # range_key = num_nodes // 10 * 10
    # 根据range_key提取每个字典中超过range_key个元素的track_id和次数
    range_key = num_nodes // 10 * 10
    # print('range_key: ', range_key)

    remain = {}
    delete = {}

    for i, (track_id, count) in enumerate(track_id_count_dict.items()):
        if i < range_key:
            remain[track_id] = count
        else:
            delete[track_id] = count
    
    delete_track_id = []
    delete_track_id = list(delete.keys())

        
    #     # 打印结果
    # print(f"Remain dictionary for {raw_path} with range_key = {range_key}:")
    # print(remain)
    # print(f"Delete dictionary for {raw_path} with range_key = {range_key}:")
    # print(delete)
    # print(f"Delete Track IDs for {raw_path} with range_key = {range_key}:")
    # print(delete_track_id)

    # 删除delete_track_id列表中的track_id对应的所有数据
    df_remain = df[~df['TRACK_ID'].isin(delete_track_id)]

    # 检查df_remain是否为空
    if not df_remain.empty:
        # 检查是否同时存在'AV'和'AGENT'的数据
        has_av = any(df_remain['OBJECT_TYPE'] == 'AV')
        has_agent = any(df_remain['OBJECT_TYPE'] == 'AGENT')
        
        # 如果同时存在'AV'和'AGENT'，则保存到新的CSV文件中
        if has_av and has_agent:
            df_remain.to_csv(delete_csv_path, index=False)
            print(f"Saved filtered data to {delete_csv_path} as it contains both 'AV' and 'AGENT'.")
        else:
            # 如果不同时存在'AV'和'AGENT'，则不进行保存操作
            print(f"Data does not contain both 'AV' and 'AGENT'. {delete_csv_path} not created.")
    else:
        # 如果df_remain为空，则不进行保存操作
        print(f"No data remains after filtering. {delete_csv_path} not created.")


    # x_list = []
    # seq_id_groups = {}
    # each_pair = [raw_path, num_nodes]
    # x_list.append(each_pair)

    # range_key = (num_nodes // 10 * 10, num_nodes // 10 * 10 + 10)
    # if range_key not in seq_id_groups:
    #     seq_id_groups[range_key] = {}
        
    # # 对每个范围的每个raw_path，存储track_id和它的计数
    # seq_id_groups[range_key][raw_path] = track_id_counts

    # # 写入结果到文件
    # print_file = "/data/yangrn/try/trackid_num.txt"
    # with open(print_file, 'a') as file:
    #     file.write('\n')
    #     for range_key, paths in seq_id_groups.items():
    #         file.write(f"Range {range_key}:\n")
    #         for raw_path, track_ids in paths.items():
    #             file.write(f"  {raw_path}: {track_ids}\n")

def main():
    source_path = "/data/yangrn/argoverse_dataset/train/data"
    target_path = "/data/yangrn/argoverse_dataset/train_pruning/data"

     # 确保目标目录存在
    if not os.path.exists(target_path):
        os.makedirs(target_path)

    for filename in os.listdir(source_path):
        if filename.endswith(".csv"): 
            raw_path = os.path.join(source_path, filename)
            delete_path = os.path.join(target_path, filename)
            process_csv_file(raw_path, delete_path)
            print(f"Processed and saved {filename}")

if __name__ == "__main__":
    main()
