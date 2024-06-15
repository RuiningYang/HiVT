import numpy as np

# 假设已经有这些路径和索引
index_file = '/data/yangrn/select_bins/final/10k/0.1/select_indices_ArgoverseV1Dataset_0.1.npy'
selected_indices = np.load(index_file)
path_lst = train_dataset.processed_paths

# 使用索引从 path_lst 中提取对应的路径
selected_paths = [path_lst[idx] for idx in selected_indices]

# 指定要写入的文件路径
output_file_path = '/data/yangrn/Dataset_Quantization_hivt/results_try/hivt_try/epoch_16_50000_selected_0.5_data_paths.txt'

# 将选中的路径写入到文件中
with open(output_file_path, 'w') as file:
    for path in selected_paths:
        file.write(path + '\n')

print(f"路径已经写入到 {output_file_path}")
