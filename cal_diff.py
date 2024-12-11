import cv2
import numpy as np

def calculate_absolute_difference(image1_path, image2_path):
    # 读取两张图片
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)
    img1 -= img1.min()
    img1 = img1 / img1.max() * 255
    img1 = img1.astype(np.uint8)
    img2 -= img2.min()
    img2 = img2 / img2.max() * 255
    img2 = img2.astype(np.uint8)
    
    # 确保两张图片大小相同
    if img1.shape != img2.shape:
        #resize
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    # 计算绝对差值
    diff = cv2.absdiff(img1, img2)

    return diff

# 使用示例
image1_path = '/root/RawSense/flatnet/output/flatnet_val/decoded_rgb_capture/sim_captures/n01818515_11302.png'
image2_path = '/root/RawSense/data/flatnet_output_384_val/gts/n01818515_11302.png'

# try:
result = calculate_absolute_difference(image1_path, image2_path)
    
    # # 显示结果
    # cv2.imshow('Absolute Difference', result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 保存结果（可选）
cv2.imwrite('absolute_difference.jpg', result)

# except Exception as e:
#     print(f"An error occurred: {str(e)}")