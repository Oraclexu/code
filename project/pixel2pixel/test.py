'''
Define Hyper-parameters
Init Dataset and Model
Run
'''

import argparse
import os

from solver import Solver
from loaders import Loaders


def test(self,config):
    # Save Images for debugging
    loaders = Loaders(config)
    self.loaders = loaders
    self.save_images_path = os.path.join(self.config.output_path, 'images/')
    self.save_models_path = os.path.join(self.config.output_path, 'models/')
    self.resume = self.save_models_path


    resume = os.path.join(self.resume, "G_")
    print("=> loading checkpoint '{}'".format(resume))
    checkpoint = torch.load(resume)
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (epoch {})"
          .format(resume, checkpoint['epoch']))
    with torch.no_grad():
        for i, (source_images) in enumerate(self.loaders.test_loader):
            source_images = source_images.to(self.device)
            self.generator = self.generator.eval()
            fake_images = self.generator(source_images)

            save_image(fake_images.cpu(),
                       os.path.join(self.save_images_path, 'images_{}_B.jpg'.format(i)))
            save_image(source_images.cpu(),
                       os.path.join(self.save_images_path, 'images_{}_A.jpg'.format(i)))


def main(config):

    # Environments Parameters
    os.environ['CUDA_VISIBLE_DEVICES'] = config.cuda
    if not os.path.exists(config.output_path,):
        os.makedirs(os.path.join(config.output_path, 'images/'))
        os.makedirs(os.path.join(config.output_path, 'models/'))

    # Initialize Dataset
    loaders = Loaders(config)

    # Initialize Pixel2Pixel and train
    solver = Solver(config, loaders)
    solver.test()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Environment Configuration
    parser.add_argument('--cuda', type=str, default='0', help='If -1, use cpu; if >0 use single GPU; if 2,3,4 for multi GPUS(2,3,4)')
    parser.add_argument('--output_path', type=str, default='/home/whao/data/quanliy/GAN/dataset/TID08train/TID2008/deblur/pix2pix/')

    # Dataset Configuration
    parser.add_argument('--dataset_path', type=str, default='/home/whao/data/quanliy/GAN/dataset/TID08train/TID2008/daabs/')
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=1)

    # Model Configuration
    parser.add_argument('--conv_dim', type=int, default=64)
    parser.add_argument('--layer_num', type=int, default=6)

    # Train Configuration
    parser.add_argument('--resume_epoch', type=int, default=-1, help='if -1, train from scratch; if >=0, resume and start to train')

    # Test Configuration
    parser.add_argument('--test_epoch', type=int, default=100)
    parser.add_argument('--test_image', type=str, default='', help='if is an image, only translate it; if a folder, translate all images in it')

    # main function
    config = parser.parse_args()
    main(config)
