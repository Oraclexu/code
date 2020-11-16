"""
将快衰落图像与参考图像拼接起来
"""

import os
import numpy as np
import cv2

img_fold_AB = "E:/Pycharm/code/code/dataset/combine_fr/train/"  # fr -> fastfading+refimgs
tpye_path = "E:/Pycharm/code/code/dataset/LIVE/fastfading/info.txt"
ref_paths = 'E:/Pycharm/code/code/dataset/LIVE/refimgs/'
dis_paths = 'E:/Pycharm/code/code/dataset/LIVE/fastfading/'
data_list = [line.strip().split(' ') for line in open(tpye_path, 'r')]
print(len(data_list))
N = len(data_list)
# print(data_list)


for index in range(N):
    if len(data_list[index]) == 3:
        ref, dis, mos = data_list[index]
        ref_path = os.path.join(ref_paths, ref)
        dis_path = os.path.join(dis_paths, dis)
        if os.path.isfile(ref_path) and os.path.isfile(dis_path):
            path_AB = os.path.join(img_fold_AB, str(index)) + '.bmp'
            im_A = cv2.imread(dis_path, cv2.IMREAD_COLOR)
            im_B = cv2.imread(ref_path, cv2.IMREAD_COLOR)
            im_AB = np.concatenate([im_A, im_B], 1)
            # print("a", im_A.shape)
            # print("b", im_B.shape)
            # print("ab", im_AB.shape)
            # exit()
            cv2.imwrite(path_AB, im_AB)
