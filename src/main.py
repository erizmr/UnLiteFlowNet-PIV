# -*- coding: utf-8 -*-
"""
UnLiteFlowNet-PIV

"""

import os, sys, errno
from google.colab import drive

drive.mount('/content/gdrive', force_remount=True)
data_path = "./gdrive/My Drive/PIV_data"
result_path = "./gdrive/My Drive/piv_result"
test_data_path = "./gdrive/My Drive/synthetic_particle_data"
piv_challenge_path = "./gdrive/My Drive/piv_challenge"

drive_path = "./gdrive/My Drive/Colab Notebooks"

# !pip install pycm livelossplot
# !pip3 install flowiz -U
# !pip3 install GPUtil


# Commented out IPython magic to ensure Python compatibility.
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import ShuffleSplit
import matplotlib as mpl
mpl.style.available
mpl.style.use('seaborn-paper') 
import matplotlib.patches as mpatches

import livelossplot
from livelossplot import PlotLosses
from pycm import *

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, drive_path)

from utils import *
from models import *
from model_FlowNetS import *
from loss_functions import *
from custom_dataset import *
from read_data import *
from train_functions import *
# %pylab inline

"""Loading traing dataset"""

img1_name_list, img2_name_list, gt_name_list = read_all(data_path)
flow_img1_name_list, flow_img2_name_list, flow_gt_name_list, flow_dir = read_by_type(data_path)

#print([f_dir for f_dir in flow_dir])
print([len(f_dir) for f_dir in flow_img1_name_list])
print([len(f_dir) for f_dir in flow_img2_name_list])
print([len(f_dir) for f_dir in flow_gt_name_list])

print(len(gt_name_list))
print(len(img1_name_list))
print(len(img2_name_list))

"""Construct dataset"""

amount = len(gt_name_list)
# Use how much of the data
total_data_index = np.arange(0, amount, 1)
total_label_index = np.arange(0, amount, 1)
total_data_index

# Divide train/validation and test data
shuffler = ShuffleSplit(n_splits=1, test_size=0.1, random_state=2).split(total_data_index, total_label_index)
indices = [(train_idx, test_idx) for train_idx, test_idx in shuffler][0]

# Divide train and validation data
shuffler_tv = ShuffleSplit(n_splits=1, test_size=0.1, random_state=2).split(indices[0], indices[0])
indices_tv = [(train_idx, validation_idx) for train_idx, validation_idx in shuffler_tv][0]

ratio = 1.0
train_data = indices_tv[0][:int(ratio*len(indices_tv[0]))]
validate_data = indices_tv[1][:int(ratio*len(indices_tv[1]))]
test_data = indices[1][:int(ratio*len(indices[1]))]
print("Check training data: ", len(train_data))
print("Check validate data: ", len(validate_data))
print("Check test data: ", len(test_data))

train_dataset = FlowDataset(train_data,[img1_name_list, img2_name_list], targets_index_list = train_data, targets = gt_name_list)
validate_dataset = FlowDataset(validate_data,[img1_name_list, img2_name_list], validate_data, gt_name_list)
test_dataset = FlowDataset(test_data, [img1_name_list, img2_name_list],test_data, gt_name_list)

flow_dataset = {}
for i, f_name in enumerate(flow_dir):
  total_index = np.arange(0, len(flow_img1_name_list[i]), 1)
  flow_dataset[f_name] = FlowDataset(total_index, [flow_img1_name_list[i], flow_img2_name_list[i]], targets_index_list = total_index, targets = flow_gt_name_list[i])

flow_dataset

"""Unsupervisored_LiteFlowNet"""

seed = 22
lr = 1e-4
momentum = 0.5
batch_size = 8
test_batch_size = 8
n_epochs = 100
new_train = True

#Load model for continue train
model = Network().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr,  weight_decay=1e-5, eps=1e-3, amsgrad=True)

if new_train:
  # New train
  model_trained = train_model(model,
                              train_dataset,
                              validate_dataset,
                              test_dataset,
                              batch_size,
                              test_batch_size,
                              lr,
                              n_epochs,
                              optimizer
                              )
else:
  model_save_name = 's2_c_UnsupervisoredLiteFlowNet_checkpoint_49_2020_03_22_08_28_38.pt'
  PATH = F"/content/gdrive/My Drive/models/{model_save_name}" 

  checkpoint = torch.load(PATH)
  model.load_state_dict(checkpoint['model_state_dict'])
  optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
  epoch = checkpoint['epoch']
  loss = checkpoint['loss']

  model_trained = train_model(model,
                              train_dataset,
                              validate_dataset,
                              test_dataset,
                              batch_size,
                              test_batch_size,
                              lr,
                              n_epochs,
                              optimizer,
                              epoch_trained = epoch+1)

"""Load Model"""

img1_name_list = json.load(open(piv_challenge_path+"/3th/3_Case_Time-resolved Jet Flow/img1_name_list.json",'r'))
img2_name_list = json.load(open(piv_challenge_path+"/3th/3_Case_Time-resolved Jet Flow/img2_name_list.json",'r'))


test_data = [x for x in range(100)]
test_dataset = FlowDataset(test_data, [img1_name_list, img2_name_list])

# LiteFlownet
model_save_name = 'LiteFlowNet_checkpoint_24_2020_02_15_05_31_34.pt'
#model_save_name = 'LiteFlowNet_checkpoint_29_2020_03_14_21_32_49.pt'
#model_save_name ='LiteFlowNet_checkpoint_19_2020_02_15_01_33_57.pt'
PATH = F"/content/gdrive/My Drive/models/{model_save_name}" 
liteflownet = Network()
liteflownet.load_state_dict(torch.load(PATH)['model_state_dict'])
#model_loaded.to(device)
liteflownet.eval()
liteflownet.to(device)
print('liteflownet load successfully.')

# UnsupervisoredLiteFlownet
model_save_name = 'UnsupervisoredLiteFlowNet_checkpoint_799_2020_03_16_21_57_51.pt'
PATH = F"/content/gdrive/My Drive/models/{model_save_name}" 
unliteflownet = Network()
unliteflownet.load_state_dict(torch.load(PATH)['model_state_dict'])
#model_loaded.to(device)
unliteflownet.eval()
unliteflownet.to(device)
print('unliteflownet load successfully.')


### Visualize 
# #Prepare the input
number_total = 100
# test_dataset.eval()
# input_data = test_dataset[number][0].view(-1, 2, 256, 256)
# label_data = test_dataset[number][1]

# # validate_dataset.eval()
# # input_data = validate_dataset[number][0].view(-1, 2, 256, 256)
# # label_data = validate_dataset[number][1]

test_dataset.eval()
for number in range(number_total):
  input_data = test_dataset[number][0][:,256:512,256:512].view(-1, 2, 256, 256)
  #input_data = F.interpolate(input_data.view(-1, 2, 512, 512), (256,256), mode='bilinear', align_corners=False)
  label_data = test_dataset[number][1]

  ##--------------------------------------
  h, w = input_data.shape[-2], input_data.shape[-1]
  x1 = input_data[:,0,...].view(-1, 1, h, w)
  x2 = input_data[:,1,...].view(-1, 1, h, w)

  print("Input size x1:", x1.shape)
  print("Input size x2:", x2.shape)
  # Do the estimation

  # Unliteflownet
  b, _, h, w = input_data.size()
  y_pre = estimate(x1.to(device), x2.to(device), unliteflownet,  train=False)
  y_pre = F.interpolate(y_pre, (h,w), mode='bilinear', align_corners=False)


  fig, axarr = plt.subplots(1, 3,figsize=(24, 8))
  resize_ratio_u = 1.0
  resize_ratio_v = 1.0
  u = y_pre[0][0].detach() * resize_ratio_u
  v = y_pre[0][1].detach() * resize_ratio_v

  color_data_pre = np.concatenate((u.view(h,w,1),v.view(h,w,1)), 2)
  u = u.numpy()
  v = v.numpy()

  mappable1 = axarr[1].imshow(fz.convert_from_flow(color_data_pre))
  X = np.arange(0, h, 4)
  Y = np.arange(0, w, 4)
  xx,yy=np.meshgrid(X, Y)
  U = u[xx.T, yy.T]
  V = v[xx.T, yy.T]

  #axarr[1].quiver(yy.T, xx.T, U, -V)
  axarr[1].axis('off')
  color_data_pre_unliteflownet = color_data_pre


  # liteflownet
  b, _, h, w = input_data.size()
  y_pre = estimate(x1.to(device), x2.to(device), liteflownet,  train=False)
  y_pre = F.interpolate(y_pre, (h,w), mode='bilinear', align_corners=False)


  u = y_pre[0][0].detach() * resize_ratio_u
  v = y_pre[0][1].detach() * resize_ratio_v

  color_data_pre = np.concatenate((u.view(h,w,1),v.view(h,w,1)), 2)
  u = u.numpy()
  v = v.numpy()

  mappable1 = axarr[2].imshow(fz.convert_from_flow(color_data_pre))
  X = np.arange(0, h, 4)
  Y = np.arange(0, w, 4)
  xx,yy=np.meshgrid(X, Y)
  U = u[xx.T, yy.T]
  V = v[xx.T, yy.T]

  #axarr[2].quiver(yy.T, xx.T, U, -V)
  axarr[2].axis('off')
  color_data_pre_liteflownet = color_data_pre

  # Label data
  # u = label_data[0].detach() 
  # v = label_data[1].detach()

  # color_data_label = np.concatenate((u.view(h,w,1),v.view(h,w,1)), 2)
  # u = u.numpy()
  # v = v.numpy()
  axarr[0].imshow(x1[0][0],cmap='gray')
  # mappable1 = axarr[0].imshow(fz.convert_from_flow(color_data_label))
  # X = np.arange(0, h, 8)
  # Y = np.arange(0, w, 8)
  # xx,yy=np.meshgrid(X, Y)
  # # 128 is the boundary of positive and negative velocity
  # # this is essential for arrows visualization
  # U = u[xx.T, yy.T]
  # V = v[xx.T, yy.T]

  # axarr[0].quiver(yy.T, xx.T, U, -V)
  axarr[0].axis('off')
  # color_data_pre_label = color_data_pre
  
  fig.savefig(result_path+'/jet_flow4/jet_flow_%d.png' % number, bbox_inches='tight')
  plt.close()

print('u mean', output_full_view[0].mean())
print('v mean', output_full_view.mean())

fig, ax = plt.subplots(1,2,figsize=(10, 5))
mappable = ax[0].imshow(fz.convert_from_flow(color_data_pre_flownet - color_data))
mappable = ax[1].imshow(fz.convert_from_flow(color_data_pre_liteflownet - color_data))
#fig.colorbar(mappable, ax=ax)
print('Img1 EPE error: %s, Img2 EPE error: %s' % (np.linalg.norm(color_data_pre_flownet - color_data,2,2).mean(), np.linalg.norm(color_data_pre_liteflownet - color_data,2,2).mean()))

fig, ax = plt.subplots(1,1,figsize=(5, 5))
mappable = ax.imshow(fz.convert_from_flow(color_data_pre_flownet - color_data_pre_liteflownet))
#fig.colorbar(mappable, ax=ax)
print('EPE error: ', np.linalg.norm(color_data_pre_flownet - color_data_pre_liteflownet,2,2).mean())

