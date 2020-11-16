import torch
import torch.optim as optim
from torchvision.utils import save_image
import os

from models_2d import Generator, Discriminator,DiscriminatorS, weights_init_normal
from iqa_loss import PerceptualLoss, contentFunc

class Solver:

    def __init__(self, config, loaders):

        # Parameters
        self.config = config
        self.loaders = loaders
        self.save_images_path = os.path.join(self.config.output_path, 'images/')
        self.save_models_path = os.path.join(self.config.output_path, 'models/')
        self.resume = self.save_models_path
        self.test_epoch = self.config.test_epoch

        self.test_images_path = self.config.test_images_path
        #Set Devices
        if self.config.cuda is not '-1':
            os.environ['CUDA_VISIBLE_DEVICES'] = config.cuda
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        # Initialize
        self._init_models()
        self._init_losses()
        self._init_optimizers()

        # Resume Model
        if self.config.resume_epoch == -1:
            self.start_epoch = 0
        elif self.config.resume_epoch >=0:
            self.start_epoch = self.config.resume_epoch
            self._restore_model(self.config.resume_epoch)


    def _init_models(self):

        # Init Model
        self.generator = Generator()
        self.discriminator = Discriminator(self.config.conv_dim, self.config.layer_num)
        # Init Weights
        self.generator.apply(weights_init_normal)
        self.discriminator.apply(weights_init_normal)
        # Move model to device (GPU or CPU)
        self.generator = torch.nn.DataParallel(self.generator).to(self.device)
        self.discriminator = torch.nn.DataParallel(self.discriminator).to(self.device)

        ##############################
        self.discriminatorS = DiscriminatorS(self.config.conv_dim, self.config.layer_num)
        self.discriminatorS.apply(weights_init_normal)
        self.discriminatorS = torch.nn.DataParallel(self.discriminatorS).to(self.device)
        ##############################


    def _init_losses(self):
        # Init GAN loss and Reconstruction Loss
        self.criterion_GAN = torch.nn.MSELoss()
        self.criterion_recon = torch.nn.L1Loss()


    def _init_optimizers(self):
        # Init Optimizer. Use Hyper-Parameters as DCGAN
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=0.0002, betas=[0.5, 0.999])
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=[0.5, 0.999])
        # Set learning-rate decay
        self.g_lr_decay = optim.lr_scheduler.StepLR(self.g_optimizer, step_size=100, gamma=0.1)
        self.d_lr_decay = optim.lr_scheduler.StepLR(self.d_optimizer, step_size=100, gamma=0.1)

        ###################3
        self.ds_optimizer = optim.Adam(self.discriminatorS.parameters(), lr=0.0002, betas=[0.5, 0.999])
        self.ds_lr_decay = optim.lr_scheduler.StepLR(self.ds_optimizer, step_size=100, gamma=0.1)
        ###################


    def _lr_decay_step(self, current_epoch):
        self.g_lr_decay.step(current_epoch)
        self.d_lr_decay.step(current_epoch)
        ###################
        self.ds_lr_decay.step(current_epoch)
        ################


    def _save_model(self, current_epoch):
        # Save generator and discriminator
        torch.save(self.generator.state_dict(), os.path.join(self.save_models_path, 'G_{}.pkl'.format(current_epoch)))
        torch.save(self.discriminator.state_dict(), os.path.join(self.save_models_path, 'D_{}.pkl'.format(current_epoch)))
        ###################
        torch.save(self.discriminatorS.state_dict(), os.path.join(self.save_models_path, 'DS_{}.pkl'.format(current_epoch)))
        #########################
        print( 'Note: Successfully save model as {}'.format(current_epoch))


    def _restore_model(self, resume_epoch):
        # Resume generator and discriminator
        ######################
        self.discriminatorS.load_state_dict(torch.load(os.path.join(self.save_models_path, 'DS_{}.pkl'.format(resume_epoch))))
        #######################
        self.discriminator.load_state_dict(torch.load(os.path.join(self.save_models_path, 'D_{}.pkl'.format(resume_epoch))))
        self.generator.load_state_dict(torch.load(os.path.join(self.save_models_path, 'G_{}.pkl'.format(resume_epoch))))
        print( 'Note: Successfully resume model from {}'.format(resume_epoch))


    def train(self):

        # Load 16 images as fixed image for displaying and debugging
        fixed_source_images, fixed_target_images,fixed_S = next(iter(self.loaders.train_loader))
        ones = torch.ones_like(self.discriminator(fixed_source_images, fixed_source_images))
        zeros = torch.zeros_like(self.discriminator(fixed_source_images, fixed_source_images))

        ###################
        onesS = torch.ones_like(self.discriminatorS(fixed_S, fixed_S))
        zerosS = torch.zeros_like(self.discriminatorS(fixed_S, fixed_S))
        ###################

        for ii in range(int(16/self.config.batch_size-1)):
            fixed_source_images_, fixed_target_images_ ,S_= next(iter(self.loaders.train_loader))
            fixed_source_images = torch.cat([fixed_source_images, fixed_source_images_], dim=0)
            fixed_target_images = torch.cat([fixed_target_images, fixed_target_images_], dim=0)
            fixed_S = torch.cat([fixed_S, S_], dim=0)
        fixed_S = fixed_S.to(self.device)
        fixed_source_images, fixed_target_images = fixed_source_images.to(self.device), fixed_target_images.to(self.device)

        # Train 200 epoches
        for epoch in range(self.start_epoch, 202):
            # Save Images for debugging
            with torch.no_grad():
                self.generator = self.generator.eval()
                fake_images,fake_S = self.generator(fixed_source_images)
                #print(fake_images.shape,fake_S.shape)
                all = torch.cat([fixed_source_images, fake_images, fixed_target_images], dim=0)
                all_S = torch.cat([fake_S, fixed_S], dim=0)
                save_image((all.cpu()+1.0)/2.0,
                           os.path.join(self.save_images_path, 'images_{}.jpg'.format(epoch)), 16)
                save_image((all_S.cpu() + 1.0) / 2.0,
                           os.path.join(self.save_images_path, 'images_S{}.jpg'.format(epoch)), 16)

            # Train
            self.generator = self.generator.train()
            self._lr_decay_step(epoch)
            for iteration, data in enumerate(self.loaders.train_loader):
                #########################################################################################################
                #                                            load a batch data                                          #
                #########################################################################################################
                source_images, target_images,target_S = data
                target_S = target_S.to(self.device)
                source_images, target_images = source_images.to(self.device), target_images.to(self.device)

                #########################################################################################################
                #                                                     Generator                                         #
                #########################################################################################################
                fake_images,fake_S = self.generator(source_images)

                gan_loss = self.criterion_GAN(self.discriminator(fake_images, source_images), ones)
                ####################
                ganS_loss = self.criterion_GAN(self.discriminatorS(fake_S, fake_S), onesS)
                #fff = torch.cat([fake_S,fake_S,fake_S,fake_S,fake_S],dim=1)
                #ganS_loss = self.criterion_GAN(self.discriminator(fff, fake_S), ones)
                iqaloss = contentFunc(fake_images,target_images)
                ##################
                recon_loss = self.criterion_recon(fake_images, target_images)
                reconS_loss = self.criterion_recon(fake_S, target_S)

                g_loss = gan_loss + 100 * recon_loss + 100 * reconS_loss + ganS_loss+iqaloss

                self.g_optimizer.zero_grad()
                g_loss.backward(retain_graph=True)
                self.g_optimizer.step()

                #########################################################################################################
                #                                                     Discriminator                                     #
                #########################################################################################################
                loss_real = self.criterion_GAN(self.discriminator(target_images, source_images), ones)
                loss_fake = self.criterion_GAN(self.discriminator(fake_images.detach(), source_images), zeros)
                d_loss = (loss_real + loss_fake) / 2.0

                lossS_real = self.criterion_GAN(self.discriminatorS(target_S, target_S), onesS)
                lossS_fake = self.criterion_GAN(self.discriminatorS(fake_S.detach(), fake_S.detach()), zerosS)
                dS_loss = (lossS_real + lossS_fake) / 2.0
                #fff_target=torch.cat([target_images, target_images, target_images, target_images, target_images], dim=1)
                #fff_fake=torch.cat([fake_images,fake_images,fake_images,fake_images,fake_images],dim=1)
                #loss_real = self.criterion_GAN(self.discriminator(target_images, fff_target), ones)
                #loss_fake = self.criterion_GAN(self.discriminator(fake_images.detach(), fff_fake.detach()), zeros)
                #dS_loss = (loss_real + loss_fake) / 2.0
                #d_loss = (d_loss + dS_loss)/2.0


                self.d_optimizer.zero_grad()
                d_loss.backward()
                self.d_optimizer.step()

                self.ds_optimizer.zero_grad()
                dS_loss.backward()
                self.ds_optimizer.step()
                if iteration%10 ==0:
                    print('[EPOCH:{}/{}]  [ITER:{}/{}]  [D_GAN:{}]  [G_GAN:{}]  [RECON:{} [RECONS:{}]'.
                      format(epoch, 200, iteration, len(self.loaders.train_loader), d_loss, gan_loss, recon_loss,reconS_loss))

            # Save model
            if epoch%20 == 0:
                self._save_model(epoch)


    def test(self):

        # Save Images for debugging
        self.generator = self.generator.eval()
        resume = os.path.join(self.resume,"G_{}.pkl".format(self.test_epoch))
        print("=> loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume)

        self.generator.load_state_dict(checkpoint)
        print(len(self.loaders.test_loader))

        #self.test_images_path = '/home/l/my/dataset/pix2pix//ssim_iqa_t/batch2/'

        with torch.no_grad():
            for i, (source_images,A_name) in enumerate(self.loaders.test_loader):

                source_images = source_images.to(self.device)
                fake_images, fake_S= self.generator(source_images)
                A_name = str(A_name).split("'")[1]
                #print(self.test_images_path)


                save_image((fake_images.cpu()+1.0)/2.0,
                           os.path.join(self.test_images_path , 'test/{}_B.png'.format(A_name)))
                save_image((source_images.cpu()+1.0)/2.0,
                           os.path.join(self.test_images_path , 'test/{}_A.png'.format(A_name)))
                save_image((fake_S.cpu() + 1.0) / 2.0,
                           os.path.join(self.test_images_path , 'test/{}_S.png'.format(A_name)))

