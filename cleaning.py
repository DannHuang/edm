import os
import shutil

def check_contains_txt_files(directory):
    # Check if the directory contains any txt files
    for file in os.listdir(directory):
        if file.startswith("sigma"):
            return True
    return False

root_dir='/root/autodl-tmp/imgnet'
for folder in os.listdir(root_dir):
    abs_folder=os.path.join(root_dir, folder)
    if not check_contains_txt_files(abs_folder):
        print(f"Removing folder: {abs_folder}")
        shutil.rmtree(abs_folder)