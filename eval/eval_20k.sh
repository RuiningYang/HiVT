#!/bin/bash

# 指定使用的CUDA设备
export CUDA_VISIBLE_DEVICES=1
# 设置dataset根目录
dataset_root="/data/yangrn/random_select_20k/all"
# 结果日志存储根目录
result_log_dir="/data/yangrn/result_log/final_hivt"

# 遍历的根目录列表
base_dirs=(
    "/data/yangrn/HiVT/lightning_logs/balance_select_20k_64"
    "/data/yangrn/HiVT/lightning_logs/dy_select_20k_64"
    "/data/yangrn/HiVT/lightning_logs/random_select_20k_64"
)

# 遍历指定文件夹
for base_dir in "${base_dirs[@]}"; do
    echo "Searching in ${base_dir}"
    base_name=$(basename "$base_dir")
    for top_folder in "$base_dir"/2*; do
        if [ -d "$top_folder" ]; then
            top_folder_name=$(basename "$top_folder")
            for sub_folder in "$top_folder"/0*; do
                if [ -d "$sub_folder" ]; then
                    sub_folder_name=$(basename "$sub_folder")
                    # 获取最新的文件
                    last_file=$(ls -Art "$sub_folder" | tail -n 1)
                    if [[ "$last_file" != "" ]]; then
                        # 构建完整的checkpoint路径
                        ckpt_path="${sub_folder}/${last_file}"
                        echo "Found checkpoint: $ckpt_path"

                        # 构建日志文件目录
                        log_directory="${result_log_dir}/${base_name}/${top_folder_name}/${sub_folder_name}"
                        mkdir -p "$log_directory"

                        # 执行命令并记录日志
                        log_file="${log_directory}/eval_${top_folder_name}.log"
                        echo "Executing eval.py for ${top_folder_name}, logs will be saved to ${log_file}"
                        python /data/yangrn/HiVT/eval.py --root "$dataset_root" --batch_size 32 --ckpt_path "$ckpt_path" &> "$log_file"
                    fi
                fi
            done
        fi
    done
done
