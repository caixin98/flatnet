#load .txt file and return the list of target_paths
import os
file_path = "/root/caixin/flatnet/data/text_files/train_source_imagenet_384_384_Feb_19.txt"
target_path = "flatnet/decoded_sim_captures"
with open(file_path, "r") as f:
    source_paths = f.readlines()
target_paths = [os.path.join(target_path, source_path.split("/")[-1]) for source_path in source_paths]
# save the target_paths to a new .txt file
with open("/root/caixin/flatnet/data/text_files/decoded_sim_captures_train.txt", "w") as f:
    for target_path in target_paths:
        f.write(target_path)
# for val
file_path = "/root/caixin/flatnet/data/text_files/val_source_imagenet_384_384_Feb_19.txt"
target_path = "flatnet_val/decoded_sim_captures"
with open(file_path, "r") as f:
    source_paths = f.readlines()
target_paths = [os.path.join(target_path, source_path.split("/")[-1]) for source_path in source_paths]
# save the target_paths to a new .txt file
with open("/root/caixin/flatnet/data/text_files/decoded_sim_captures_val.txt", "w") as f:
    for target_path in target_paths:
        f.write(target_path)