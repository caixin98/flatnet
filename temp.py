import os

# 设置文件夹A和B的路径
folder_a = '/root/RawSense/LenslessPiCam/outputs/2024-08-12/11-43-17/SimPhlatCam_raw_1518x2012'
folder_b = '/root/RawSense/flatnet/data/flatnet_val/decoded_sim_captures'

# 获取两个文件夹中的文件列表并排序
files_in_a = sorted(os.listdir(folder_a),key=lambda x: int(x.split('.')[0]))
files_in_b = sorted(os.listdir(folder_b))
# 检查文件数量是否相同
if len(files_in_a) != len(files_in_b):
    print("文件数量不匹配!")
else:
    # 对齐文件名
    for file_a, file_b in zip(files_in_a, files_in_b):
        src = os.path.join(folder_a, file_a)
        dst = os.path.join(folder_a, file_b)
        os.rename(src, dst)
        print(f"已将文件{file_a}重命名为{file_b}")
    print("文件名对齐完成。")