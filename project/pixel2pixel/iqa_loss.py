import torch
from torch import nn
from iqa_model import IQANet
#from dataset import TID2013Dataset, WaterlooDataset

def contentFunc(fakeIm, realIm):
	conv_3_3_layer = 14
	cnn = IQANet(weighted=False).cuda()
	resume = "/home/l/my/dataset/live/jpeg/FR/checkpoint_latest.pkl"
	#resume = "/home/l/my/dataset/pkl/checkpoint_851.pkl"
	checkpoint = torch.load(resume)
	cnn.load_state_dict(checkpoint['state_dict'])
	criterion = nn.MSELoss()
	f_fake = cnn.extract_feature(fakeIm)
	f_real = cnn.extract_feature(realIm)
	f_real_no_grad = f_real.detach()
	loss = criterion(f_fake, f_real_no_grad)
	return loss
import torch
from torch import nn
from iqa_model import IQANet
#from dataset import TID2013Dataset, WaterlooDataset

def contentFunc(fakeIm, realIm):
	conv_3_3_layer = 14
	cnn = IQANet(weighted=False).cuda()
	resume = "/home/l/my/dataset/live/jpeg/FR/checkpoint_latest.pkl"
	#resume = "/home/l/my/dataset/pkl/checkpoint_851.pkl"
	checkpoint = torch.load(resume)
	cnn.load_state_dict(checkpoint['state_dict'])
	criterion = nn.MSELoss()
	f_fake = cnn.extract_feature(fakeIm)
	f_real = cnn.extract_feature(realIm)
	f_real_no_grad = f_real.detach()
	loss = criterion(f_fake, f_real_no_grad)
	return loss


class PerceptualLoss():

	def contentFunc(self):
		conv_3_3_layer = 14
		cnn = IQANet(weighted=True).cuda()
		#resume = "/home/l/my/dataset/pkl/checkpoint_851.pkl"
		resume = "/home/l/my/dataset/live/FR_4/fr_802/checkpoint_latest.pkl"
		checkpoint = torch.load(resume)
		cnn.load_state_dict(checkpoint['state_dict'])



		# model = nn.Sequential()
		# model = model.cuda()
		# for i,layer in enumerate(list(cnn)):
		# 	model.add_module(str(i),layer)
		# 	if i == conv_3_3_layer:
		# 		break
		return model

	def __init__(self, loss):
		self.criterion = nn.MSELoss()
		self.contentFunc = self.contentFunc()

	def get_loss(self, fakeIm, realIm):
		f_fake = self.contentFunc.extract_feature(fakeIm)
		f_real = self.contentFunc.extract_feature(realIm)
		f_real_no_grad = f_real.detach()
		loss = self.criterion(f_fake, f_real_no_grad)
		return loss




class PerceptualLoss():

	def contentFunc(self):
		conv_3_3_layer = 14
		cnn = IQANet(weighted=True).cuda()
		#resume = "/home/l/my/dataset/pkl/checkpoint_851.pkl"
		resume = "/home/l/my/dataset/live/FR_4/fr_802/checkpoint_latest.pkl"
		checkpoint = torch.load(resume)
		cnn.load_state_dict(checkpoint['state_dict'])



		# model = nn.Sequential()
		# model = model.cuda()
		# for i,layer in enumerate(list(cnn)):
		# 	model.add_module(str(i),layer)
		# 	if i == conv_3_3_layer:
		# 		break
		return model

	def __init__(self, loss):
		self.criterion = nn.MSELoss()
		self.contentFunc = self.contentFunc()

	def get_loss(self, fakeIm, realIm):
		f_fake = self.contentFunc.extract_feature(fakeIm)
		f_real = self.contentFunc.extract_feature(realIm)
		f_real_no_grad = f_real.detach()
		loss = self.criterion(f_fake, f_real_no_grad)
		return loss


