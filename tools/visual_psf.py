# load psf file and do the visualization

import numpy as np
import cv2
import torch
psf_path = '/root/caixin/flatnet/data/phase_psf/psf.npy'
psf = np.load(psf_path)
visual_path = "/root/caixin/flatnet/temp.png"
def visual_psf(matrix):
    print(matrix.shape)
    matrix =  (matrix - matrix.min()) / (matrix.max() - matrix.min())
    if torch.is_tensor(matrix):
        matrix = matrix.cpu().detach().numpy()
    matrix = (matrix * 255).astype(np.uint8)  
    cv2.imwrite(visual_path, matrix)
    print("visual_psf done")
visual_psf(psf)

    
