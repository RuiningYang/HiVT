import os
import shutil
import random

# Define source and destination directories
src_dir = "/data/yangrn/dataset/20k/train/data"
dst_dir = "/data/yangrn/coreset/HiVT/random/0.5/train/data"

# Create destination directory if it does not exist
os.makedirs(dst_dir, exist_ok=True)

# Get a list of all files in the source directory
files = [file for file in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, file))]

# Randomly select 4510 files
selected_files = random.sample(files, 4000)

# Copy selected files to the destination directory
for file in selected_files:
    shutil.copy(os.path.join(src_dir, file), dst_dir)

"Files copied successfully."

