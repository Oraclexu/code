'''
load train and test dataset
'''

import glob
import os
import random
import torch.utils.data as data
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from con_ssim import ssim
import json


class Loaders:
    '''
    Initialize dataloaders
    '''

    def __init__(self, config):
        self.dataset_path = config.dataset_path  # E:/Pycharm/code/code/dataset/LIVE/all_4/
        self.image_size = config.image_size  # 256
        self.batch_size = config.batch_size  # 1
        self.test_path = config.testdataset_path  # E:/# Pycharm/code/code/dataset/test/all_4/

        self.transforms = transforms.Compose([transforms.Resize((self.image_size, self.image_size), Image.BICUBIC),
                                              transforms.CenterCrop((self.image_size, self.image_size)),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

        train_set = ImageFolder(self.dataset_path, self.transforms)
        test_set = Imagetest(os.path.join(self.test_path, 'train/'), self.transforms)

        self.train_loader = data.DataLoader(dataset=train_set, batch_size=self.batch_size, shuffle=True, num_workers=4,
                                            drop_last=True)
        self.test_loader = data.DataLoader(dataset=test_set, batch_size=1, shuffle=False, num_workers=4,
                                           drop_last=False)


class ImageFolder(Dataset):
    """
    Load images given the path
    """

    def __init__(self, path, transform):
        self.fineSize = 256
        self.transform = transform
        # self._read_lists()
        # self.samples = sorted(glob.glob(os.path.join(path + '/*.bmp')))
        self.samples = self._read_lists(path)

    def _read_lists(self, path):
        img_path = os.path.join(path, 'train_data.json')
        print(img_path)
        assert os.path.exists(img_path)

        with open(img_path, 'r') as fp:
            data_dict = json.load(fp)

        self.samples = sorted(data_dict['img'])
        # return data_dict['img']  # _read_lists -> data_dict['img_abs']

        A = self.samples[1]  # index???

        # name =A.split("/")[-1].split("_")[0].replace("i","I")
        # B = os.path.join('/home/whao/data/quanliy/GAN/dataset/TID08train/TID2008/reference_images/train/'
        #                  ,name)+'.BMP'
        # B = B

        AB = Image.open(A)
        AB = self.transform(AB)
        w_total = AB.size(2)
        w = int(w_total / 2)
        h = AB.size(1)
        w_offset = random.randint(6, max(0, w - self.fineSize - 16))
        h_offset = random.randint(6, max(0, h - self.fineSize - 16))

        sample_source = AB[:, h_offset:h_offset + self.fineSize,
                        w_offset:w_offset + self.fineSize]
        sample_target = AB[:, h_offset:h_offset + self.fineSize,
                        w + w_offset:w + w_offset + self.fineSize]

        # As= A.unsqueeze(0)
        # Bs= B.unsqueeze(0)
        # map, ss = ssim(As, Bs, win_size=7, data_range=1)
        # S = map.squeeze(0)
        ################################################################
        tmp = sample_source[0, ...] * 0.299 + sample_source[1, ...] * 0.587 + sample_source[2, ...] * 0.114
        tmpA = tmp.unsqueeze(0).unsqueeze(0)
        tmp = sample_target[0, ...] * 0.299 + sample_target[1, ...] * 0.587 + sample_target[2, ...] * 0.114
        tmpB = tmp.unsqueeze(0).unsqueeze(0)
        map, ss = ssim(tmpA, tmpB, win_size=5, data_range=1)
        S = map.squeeze(0)

        # sample_source = Image.open(A)
        # sample_target = Image.open(B)
        # print(sample_source.shape)
        # exit()
        # w, h = sample.size
        # sample_target = sample.crop((w/2, 0, w, h))
        # sample_source = sample.crop((0, 0, w/2, h))

        # sample_source = self.transform(sample_source)
        # sample_target = self.transform(sample_target)
        # print(sample_source.shape)

        return sample_source, sample_target, S

    def __len__(self):
        return len(self.samples)


class Imagetest(Dataset):

    def __init__(self, path, transform):
        self.fineSize = 256
        self.transform = transforms.Compose([transforms.CenterCrop((self.fineSize, self.fineSize)),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

        self.samples = sorted(glob.glob(os.path.join(path + '/*.bmp')))
        # self.samples = self._read_lists(path)

    def _read_lists(self, path):
        img_path = os.path.join(path, 'train_data.json')
        print(img_path)
        assert os.path.exists(img_path)

        with open(img_path, 'r') as fp:
            data_dict = json.load(fp)

        # self.samples = sorted(data_dict['img'])
        return data_dict['img_abs']

    def __getitem__(self, index):
        A = self.samples[index]
        A_name = A.split('/')[-1].split('.')[0]
        A = Image.open(A)
        sample_source = self.transform(A)

        return sample_source, A_name

    def __len__(self):
        return len(self.samples)
