import random
import json
import argparse
import os

parser = argparse.ArgumentParser(description='')
parser.add_argument('--data_path', type=str, default='E:/Pycharm/code/code/dataset/LIVE/all_4/test/')
parser.add_argument('--ref_path', type=str, default='E:/Pycharm/code/code/dataset/LIVE/all_4/test/')
parser.add_argument('--mos_path', type=str, default='E:/Pycharm/code/code/dataset/LIVE/all_4/N.txt')
parser.add_argument('--save_path', type=str, default='E:/Pycharm/code/code/dataset/LIVE/all_4/')
args = parser.parse_args()

SAVE_PATH = args.save_path
DATA_DIR = args.data_path
REF_DIR = args.ref_path
MOS_WITH_NAMES = args.mos_path

TRAIN_RATIO = 0.5
TEST_RATIO = 0.5

data_list = [line.strip().split(' ') for line in open(MOS_WITH_NAMES, 'r')]
print(len(data_list))
N = len(data_list)
idcs = list(range(0, N))
random.shuffle(idcs)  # 随机排序

train_idcs = idcs[:int(N * TRAIN_RATIO)]  # 0——0.5N
val_idcs = idcs[int(N * TRAIN_RATIO):-int(N * TEST_RATIO)]  # 0.5N——0.5N -> 0
test_idcs = idcs[-int(N * TEST_RATIO):]  # 0.5N —— N

train_images, train_labels, train_mos = [], [], []
test_images, test_labels, test_mos = [], [], []
print(data_list[1])
for index in range(N):
    # img, mos = data_list[index]
    if len(data_list[index]) == 3:
        ref, img, mos = data_list[index]
    print(index, mos, img)

    image = img.split(".")[0]
    ref = REF_DIR + image + '_B.png'
    img = DATA_DIR + image + '_A.png'
    # print(image)
    # ref = REF_DIR + ref
    # img = DATA_DIR + img

    if index in train_idcs:
        train_images.append(img)
        train_labels.append(ref)
        train_mos.append(float(mos))
    if index in val_idcs:
        train_images.append(img)
        train_labels.append(ref)
        train_mos.append(float(mos))
    if index in test_idcs:
        test_images.append(img)
        test_labels.append(ref)
        test_mos.append(float(mos))

print('len(train_images)', len(train_images))
print('len(test_images)', len(test_images))
# print(test_mos)


ns = vars()
for ph in ('train', 'test'):
    data_dict = dict(img=ns['{}_images'.format(ph)], ref=ns['{}_labels'.format(ph)], score=ns['{}_mos'.format(ph)])
    # print('/home/l/my/dataset/tid2013/04/fr_802/{}_data.json'.format(ph))
    print(os.path.join(SAVE_PATH, '{}_data.json'.format(ph)))

    # with open('/home/l/my/dataset/tid2013/04/fr_802/{}_data.json'.format(ph), 'w') as fp:
    with open(os.path.join(SAVE_PATH, '{}_data.json'.format(ph)), 'w') as fp:
        json.dump(data_dict, fp)
