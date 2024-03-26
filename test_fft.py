#test the complexity of the fft in torch
import torch
import numpy as np
import time

img = torch.rand(4, 3, 1280, 1408)
img = img.to("cuda")
start = time.time()
fft = torch.fft.fft2(img)
end = time.time()
print("fft time: ", end - start)
#do the inverse fft
start = time.time()
ifft = torch.fft.ifft2(fft)
end = time.time()
print("ifft time: ", end - start)
# the difference between the original image and the ifft image
print("difference: ", torch.sum(torch.abs(img - ifft)))
print("difference rate: ", torch.sum(torch.abs(img - ifft)) / torch.sum(torch.abs(img)))