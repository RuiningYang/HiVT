import os
import shutil

def create_folders():
    try:
        os.makedirs('/data/yangrn/coreset/HiVT/20k/0.02/train/data')
        # os.makedirs('/data/yangrn/try_10000_coreset/train/processed')
    except OSError as e:
        print(f"创建文件夹时出错：{e}")

def copy_files(source_file_paths, destination_folder):
    try:
        with open(source_file_paths, 'r') as file:
            file_paths = file.readlines()
            for file_path in file_paths:
                file_path = file_path.strip()
                if os.path.exists(file_path):
                    shutil.copy(file_path, destination_folder)
                else:
                    print(f"文件不存在：{file_path}")
    except Exception as e:
        print(f"复制文件时出错：{e}")

def main():
    try:
        create_folders()

        data_source_file_paths = '/data/yangrn/select_bins/final/20k/0.02/selected_data_paths.txt'
        # processed_source_file_paths = '/data/yangrn/Dataset_Quantization_hivt/results_try/hivt_try/9025_selected_0.2_processed_paths.txt'

        data_destination_folder = '/data/yangrn/coreset/HiVT/20k/0.02/train/data'
        # processed_destination_folder = '/data/yangrn/try_10000_coreset/train/processed'

        copy_files(data_source_file_paths, data_destination_folder)
        # copy_files(processed_source_file_paths, processed_destination_folder)
    except Exception as e:
        print(f"发生错误：{e}")

if __name__ == "__main__":
    main()
