import cv2
import os
import numpy as np
from PIL import Image
from con_ssim import ssim
import glob
import torch
import torchvision.transforms as transforms

ref_path = '/home/l/my/dataset/TID_2013/reference_images/train/'
dis_path = '/home/l/my/result/yuan/'
s_path = '/home/l/my/result/s/'

transform = transforms.Compose([  # transforms.Resize((self.image_size, self.image_size), Image.BICUBIC),
    transforms.CenterCrop((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])])
data_list = sorted(glob.glob(os.path.join(dis_path + '/*.bmp')))
print(data_list)
for img in data_list:
    print(img)
    name = img.split('/')[-1]
    s = os.path.join(s_path, name)
    ref_name = "I" + name[1:3] + ".BMP"
    ref = os.path.join(ref_path, ref_name)
    # A = cv2.imread(img,0)
    # B = cv2.imread(ref,0)
    A = Image.open(img).convert("L")
    B = Image.open(ref).convert("L")
    A = transform(A)
    B = transform(B)
    tmpA = A.unsqueeze(0)
    tmpB = B.unsqueeze(0)
    map, ss = ssim(tmpA, tmpB, win_size=5, data_range=1)
    S = map.squeeze(0).squeeze(0)
    # SSS = Image.fromarray(np.array(S))
    # print(S.max(),S.min())
    # exit()
    cv2.imwrite(s, np.array((S + 1.0) / 2.0 * 250.0))
# SSS.save(s)
