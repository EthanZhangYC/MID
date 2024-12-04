import os
import argparse
import torch
import dill
import pdb
import numpy as np
import os.path as osp
import logging
import time
from torch import nn, optim, utils
import torch.nn as nn
from tensorboardX import SummaryWriter
from tqdm.auto import tqdm
import pickle

from dataset import EnvironmentDataset, collate, get_timesteps_data, restore
from models.autoencoder import AutoEncoder
from models.trajectron import Trajectron
from utils.model_registrar import ModelRegistrar
from utils.trajectron_hypers import get_traj_hypers
import evaluation

import torch
import torch.nn as nn
import numpy as np
import math
import datetime
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torchvision.models import resnet50#, ResNet50_Weights

import glob
from torchvision import transforms
from PIL import Image
from typing import Optional, List

from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange



class SamePadConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, groups=1):
        super().__init__()
        self.receptive_field = (kernel_size - 1) * dilation + 1
        padding = self.receptive_field // 2
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding,
            dilation=dilation,
            groups=groups
        )
        self.remove = 1 if self.receptive_field % 2 == 0 else 0
        
    def forward(self, x):
        out = self.conv(x)
        if self.remove > 0:
            out = out[:, :, : -self.remove]
        return out
    
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, final=False):
        super().__init__()
        self.conv1 = SamePadConv(in_channels, out_channels, kernel_size, dilation=dilation)
        self.conv2 = SamePadConv(out_channels, out_channels, kernel_size, dilation=dilation)
        self.projector = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels or final else None
    
    def forward(self, x):
        residual = x if self.projector is None else self.projector(x)
        x = F.gelu(x)
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        return x + residual

class DilatedConvEncoder(nn.Module):
    def __init__(self, in_channels, channels, kernel_size):
        super().__init__()
        self.net = nn.Sequential(*[
            ConvBlock(
                channels[i-1] if i > 0 else in_channels,
                channels[i],
                kernel_size=kernel_size,
                #dilation=min(2**i,1024),
                dilation=2**i,
                final=(i == len(channels)-1)
            )
            for i in range(len(channels))
        ])
        
    def forward(self, x):
        return self.net(x)
    
class TSEncoder_new(nn.Module):
    def __init__(self, input_dims=7, output_dims=64, hidden_dims=64, depth=10, mask_mode='binomial', n_class=4, reconstruct_dim=2):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.mask_mode = mask_mode
        
        self.input_fc = nn.Linear(input_dims, hidden_dims)
        self.feature_extractor = DilatedConvEncoder(
            hidden_dims,
            [hidden_dims] * depth + [output_dims],
            kernel_size=3
        )
        self.repr_dropout = nn.Dropout(p=0.1)
        self.fc = nn.Sequential(
            nn.Linear(output_dims, 64),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(64, n_class)
        )
        
    def forward(self, x, args=None, mask_early=False, mask_late=False):  
        nan_mask = ~x.isnan().any(axis=-1)
        ori_mask=None
        
        x = self.input_fc(x)  # B x T x Ch
        # conv encoder
        x = x.transpose(1, 2)  # B x Ch x T
        x = self.feature_extractor(x)  # B x Co x T
        x = x.transpose(1, 2)  # B x T x Co
        ori_feat = x.clone()
        
        feat = x = F.avg_pool1d(
            x.transpose(1, 2),
            kernel_size = x.size(1),
        ).transpose(1, 2).squeeze(1)
        
        logits=con_logits=va_logits=ori_mask=pred_each=None
        return logits, feat, ori_feat, con_logits, va_logits, ori_mask, pred_each
        
class Classifier_clf(nn.Module):
    def __init__(self, n_class=4, input_dim=64):
        super(Classifier_clf, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, n_class)
        self.dropout_p=0.1
        self.dropout = nn.Dropout(p=self.dropout_p)

    def forward(self, x, aux_feat=None): 
        x = self.dropout(F.relu(self.fc1(x)))
        feat = x
        x = self.fc2(x)
        return x

        
def get_label(single_dataset,idx,label_dict):
    label = single_dataset[idx][1].item()
    return label

class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
        callback_get_label func: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(self, dataset, indices=None, num_samples=None, callback_get_label=None):
                
        # if indices is not provided, 
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices

        # define custom callback
        self.callback_get_label = callback_get_label

        # if num_samples is not provided, 
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples
        
        
        label_dict={'0':0,'1':0,'2':1,'3':1,'4':1}
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx, label_dict)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1
        
        weights = [1.0 / label_to_count[self._get_label(dataset, idx, label_dict)] for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx, label_dict):
        if self.callback_get_label:
            return self.callback_get_label(dataset, idx, label_dict)
        elif isinstance(dataset, torchvision.datasets.MNIST):
            return dataset.train_labels[idx].item()
        elif isinstance(dataset, torchvision.datasets.ImageFolder):
            return dataset.imgs[idx][1]
        elif isinstance(dataset, torch.utils.data.Subset):
            return dataset.dataset.imgs[idx][1]
        else:
            raise NotImplementedError

    # sample class balance training batch 
    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples

class ForeverDataIterator:
    r"""A data iterator that will never stop producing data"""

    def __init__(self, data_loader: DataLoader, device=None):
        self.data_loader = data_loader
        self.iter = iter(self.data_loader)
        self.device = device

    def __next__(self):
        try:
            data = next(self.iter)
        except StopIteration:
            self.iter = iter(self.data_loader)
            data = next(self.iter)
        return data

    def __len__(self):
        return len(self.data_loader)

def filter_area_old(trajs, labels, pad_masks):
    new_list=[]
    new_list_y=[]
    lat_min,lat_max = (18.249901, 55.975593)
    lon_min,lon_max = (-122.3315333, 126.998528)
    len_traj = trajs.shape[0]
    # avg_lat_list=[]
    # avg_lon_list=[]
    for i in range(len_traj):
        traj = trajs[i]
        pad_mask = pad_masks[i]
        label = labels[i]
        
        new_traj = traj[~pad_mask][:,:2]
        new_traj[:,0] = new_traj[:,0] * (lat_max-lat_min) + lat_min
        new_traj[:,1] = new_traj[:,1] * (lon_max-lon_min) + lon_min
        avg_lat, avg_lon = np.mean(new_traj, axis=0)
        
        # avg_lat_list.append(avg_lat)
        # avg_lon_list.append(avg_lon)
        if avg_lat<41 and avg_lat>39:
            if avg_lon>115 and avg_lon<117:
                new_list.append(traj)
                new_list_y.append(label)
                
    return np.array(new_list), np.array(new_list_y)

def filter_area(trajs, labels, imgs, pad_masks):
    new_list=[]
    new_list_y=[]
    new_list_imgs=[]
    lat_min,lat_max = (18.249901, 55.975593)
    lon_min,lon_max = (-122.3315333, 126.998528)
    len_traj = trajs.shape[0]
    # avg_lat_list=[]
    # avg_lon_list=[]
    for i in range(len_traj):
        traj = trajs[i]
        pad_mask = pad_masks[i]
        label = labels[i]
        if imgs is not None:
            img = imgs[i]
        
        new_traj = traj[~pad_mask][:,:2]
        new_traj[:,0] = new_traj[:,0] * (lat_max-lat_min) + lat_min
        new_traj[:,1] = new_traj[:,1] * (lon_max-lon_min) + lon_min
        avg_lat, avg_lon = np.mean(new_traj, axis=0)
        
        # avg_lat_list.append(avg_lat)
        # avg_lon_list.append(avg_lon)
        if avg_lat<41 and avg_lat>39:
            if avg_lon>115 and avg_lon<117:
                new_list.append(traj)
                new_list_y.append(label)
                if imgs is not None:
                    new_list_imgs.append(img)
                
    if imgs is not None:
        return np.array(new_list), np.array(new_list_y), np.array(new_list_imgs)
    else:
        return np.array(new_list), np.array(new_list_y), None

def generate_posid(trajs, pad_masks, min_max=[(18.249901, 55.975593),(-122.3315333, 126.998528)]):
    lat_min,lat_max = min_max[0]
    lon_min,lon_max = min_max[1]
    
    new_list=[]
    new_list_y=[]
    len_traj = trajs.shape[0]
    
    max_list_lat=[]
    max_list_lon=[]
    min_list_lat=[]
    min_list_lon=[]
    for i in range(len_traj):
        traj = trajs[i]
        pad_mask = pad_masks[i]
        new_traj = traj[~pad_mask][:,:2]
        new_traj[:,0] = new_traj[:,0] * (lat_max-lat_min) + lat_min
        new_traj[:,1] = new_traj[:,1] * (lon_max-lon_min) + lon_min
        tmp_max_lat,tmp_max_lon = np.max(new_traj, axis=0)
        tmp_min_lat,tmp_min_lon = np.min(new_traj, axis=0)
        max_list_lat.append(tmp_max_lat)
        max_list_lon.append(tmp_max_lon)
        min_list_lat.append(tmp_min_lat)
        min_list_lon.append(tmp_min_lon)
    
    tmp_max_lat = np.max(np.array(max_list_lat))+1e-6
    tmp_max_lon = np.max(np.array(max_list_lon))+1e-6
    tmp_min_lat = np.min(np.array(min_list_lat))-1e-6
    tmp_min_lon = np.min(np.array(min_list_lon))-1e-6
    print(tmp_max_lat,tmp_max_lon,tmp_min_lat,tmp_min_lon)
        
    # tmp_max_lat,tmp_max_lon,tmp_min_lat,tmp_min_lon = 40.8855, 117.2707, 38.44578, 114.93135
    patchlen_lat = (tmp_max_lat-tmp_min_lat) / 16
    patchlen_lon = (tmp_max_lon-tmp_min_lon) / 16
    sid_list=[]
    eid_list=[]
    for i in range(len_traj):
        traj = trajs[i]
        pad_mask = pad_masks[i]
        # label = labels[i]
        
        new_traj = traj[~pad_mask][:,:2]
        new_traj[:,0] = new_traj[:,0] * (lat_max-lat_min) + lat_min
        new_traj[:,1] = new_traj[:,1] * (lon_max-lon_min) + lon_min
        # avg_lat, avg_lon = np.mean(new_traj, axis=0)
        
        sid = (new_traj[0,0]-tmp_min_lat)//patchlen_lat*16+(new_traj[0,1]-tmp_min_lon)//patchlen_lon
        eid = (new_traj[-1,0]-tmp_min_lat)//patchlen_lat*16+(new_traj[-1,1]-tmp_min_lon)//patchlen_lon
        sid_list.append(sid)
        eid_list.append(eid)
        # if sid>=256 or eid>=256:
        #     pdb.set_trace()

    return np.array(sid_list), np.array(eid_list)

class create_single_dataset(torch.utils.data.Dataset):
    def __init__(self, imgs, trajs, labels, seids, part, transform):
        super(create_single_dataset, self).__init__()
        self.imgs = imgs
        self.trajs = trajs
        self.labels = labels
        self.seids = seids
        self.default_idx = range(len(imgs))
        # self.label_dict={'0':0,'1':0,'2':1,'3':1,'4':1}
        self.transform = transform
        # self.dataset = dataset
        # if part=='train':                     
        #     random.shuffle(self.default_idx)
        
    def __getitem__(self, index):
        img_dir = self.imgs[self.default_idx[index]]

        img = Image.open(img_dir).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
            
        # traj,label = self.trajs[self.default_idx[index]]
        traj = np.array(self.trajs[self.default_idx[index]])#.astype(float)
        label = self.labels[self.default_idx[index]]
        if self.seids is not None:
            seid = self.seids[self.default_idx[index]]
        else:
            seid = None
        # if len(traj) < 600:
        #     extra = [[0 for j in range(len(traj[0]))] for i in range(600 - len(traj))]
        #     traj = traj + extra
        # traj = np.array(traj)
        # label = self.label_dict[img_dir.split('/')[-2]]
        return traj, img, seid, label 
        
    def __len__(self):
        return len(self.imgs)

class create_single_dataset_img(torch.utils.data.Dataset):
    def __init__(self, imgs, transform):
        super(create_single_dataset_img, self).__init__()
        self.imgs = imgs
        self.transform = transform
        
    def __getitem__(self, index):
        img_dir = self.imgs[index]
        img = Image.open(img_dir).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img
        
    def __len__(self):
        return len(self.imgs)
    
def load_data(config):
    batch_sizes = config.batch_size
    traj_length = config.traj_len
    # base_dir = "/home/xieyuan/Traj2Image-10.05/datas/"
    base_dir = "/home/yichen/data/xieyuan_datas/"
    assert config.traj_len<=600
    # assert not config.data.filter_area
    
    # "/home/xieyuan/Traj2Image-10.05/datas/cnn_data/traj2former_6class_segments_normalize_7x600dim.pickle"
    # "/home/xieyuan/Traj2Image-10.05/data_MTL/cnn_data/mtl_segment650_4class_650x7dim_normalize_latlon.pickle"
    traj_init_filename = base_dir + 'cnn_data/traj2image_6class_fixpixel_fixlat3_insert1s_train&test_cnn_0607_unnormalize.pickle'
    with open(traj_init_filename, "rb") as f:
        traj_dataset = pickle.load(f)
    train_init_traj, test_init_traj = traj_dataset
    train_x_ori, train_y_ori = map(list, zip(*train_init_traj)) 
    for i in range(len(train_x_ori)):
        tmp_trip_length = len(train_x_ori[i])
        if tmp_trip_length <= traj_length:
            train_x_ori[i] = np.pad(train_x_ori[i], ((0, traj_length - tmp_trip_length), (0, 0)), 'constant')
        else:
            train_x_ori[i] = train_x_ori[i][:traj_length]
    train_x_ori, train_y_ori = np.array(train_x_ori), np.array(train_y_ori)
    
    total_input_new = np.zeros((len(train_x_ori), 1, min(traj_length,600), 8))
    for i in range(len(train_x_ori)):
        total_input_new[i, 0, :, 0] = train_x_ori[i, :, 5]    #x
        total_input_new[i, 0, :, 1] = train_x_ori[i, :, 6]    #y
        total_input_new[i, 0, :, 2] = (train_x_ori[i, :, 0]!=0).astype(float)  #delta_time
        total_input_new[i, 0, 0, 2] = 1.
        total_input_new[i, 0, :, 3] = train_x_ori[i, :, 1]    #dist
        total_input_new[i, 0, :, 4] = train_x_ori[i, :, 1]    #speed
        total_input_new[i, 0, :, 5] = train_x_ori[i, :, 3]    #acc
        total_input_new[i, 0, :, 6] = train_x_ori[i, :, 2]    #bearing
        total_input_new[i, 0, :, 7] = train_x_ori[i, :, 4]    #bearing-rate
    train_x_ori = total_input_new.squeeze(1)
    
    # max_list = np.max(train_x_ori, axis=(0,1)) # [55.976195, 126.998528  , 1. , 80. , 79.62211779 , 179.98863883, 179.96395057]
    # min_list = np.min(train_x_ori, axis=(0,1)) # [18.249901, -122.3315333, 1. , 0.  , -72.77655734, 0.          , 0.          ]
    max_list = np.array([55.976195, 126.998528  , 1. , 80. , 80. , 79.62211779 , 179.98863883, 179.96395057])
    min_list = np.array([18.249901, -122.3315333, 0. , 0.  , 0.  , -72.77655734, 0.          , 0.          ])
    
    # normalize xy
    # min_max_list = [(18.249901, 55.975593),(-122.3315333, 126.998528)]
    for i in range(8):
        if i==2:
            continue
        train_x_ori[:,:,i] = (train_x_ori[:,:,i]-min_list[i])/(max_list[i]-min_list[i])
       
    pad_mask_source_train_ori = train_x_ori[:,:,2]==0
    train_x_ori[pad_mask_source_train_ori] = 0.
        
        
    if config.use_img:
        imgs_train = []
        img_dir = base_dir + "OpenStreetMap/global_map_tiles_satellite_zoom18_size50_train_size250/*.png"
        for file_name in glob.glob(img_dir, recursive=True):
            imgs_train.append(file_name)
        print('train:',len(imgs_train))#,'val:',len(tar_imgs_val))
    else:
        imgs_train = None
    
    # else:
    #     # batch_sizes = config.training.batch_size
    #     filename = '/home/yichen/TS2Vec/datafiles/Geolife/traindata_4class_xy_traintest_interpolatedNAN_5s_trip20_new_001meters_withdist_aligninterpolation_InsertAfterSeg_Both_11dim_doubletest_0218.pickle'
    #     with open(filename, 'rb') as f:
    #         kfold_dataset, X_unlabeled = pickle.load(f)
    #     dataset = kfold_dataset

    #     train_x = dataset[1].squeeze(1)
    #     train_y = dataset[3]
    #     train_x = train_x[:,:,4:]   
    #     pad_mask_source = train_x[:,:,0]==0
    #     train_x[pad_mask_source] = 0.
        
    #     # if config.data.interpolated:
    #     train_x_ori = dataset[1].squeeze(1)[:,:,2:]
    #     # else:
    #     # train_x_ori = dataset[0].squeeze(1)[:,:,2:]
    #     train_y_ori = dataset[3]
    #     imgs_train = None
    
    # pad_mask_source_train_ori = train_x_ori[:,:,2]==0
    # train_x_ori[pad_mask_source_train_ori] = 0.
        
        
    # class_id = 2
    # print('filtering class %d'%class_id)
    # mask_class = train_y_ori==class_id
    # train_x_ori = train_x_ori[mask_class] 
    # train_y_ori = train_y_ori[mask_class] 
    
    # if config.filter_area:
    print('filtering area')
    # train_x_ori,train_y_ori = filter_area(train_x_ori, train_y_ori, pad_mask_source_train_ori)
    train_x_ori,train_y_ori, imgs_train = filter_area(train_x_ori, train_y_ori, imgs_train, pad_mask_source_train_ori)
    pad_mask_source_train_ori = train_x_ori[:,:,2]==0


    if traj_length < train_x_ori.shape[1]:
        train_x_ori = train_x_ori[:,:traj_length,:]
        pad_mask_source_train_ori = pad_mask_source_train_ori[:,:traj_length]

    # if "seid" in config.model.mode:
    sid,eid = generate_posid(train_x_ori, pad_mask_source_train_ori)
    se_id = np.stack([sid, eid]).T
    
    if not config.not_filter_padding:
        print('filtering nopadding segments')
        pad_mask_source_incomplete = np.sum(pad_mask_source_train_ori,axis=1) == 0
        train_x_ori = train_x_ori[pad_mask_source_incomplete]
        train_y_ori = train_y_ori[pad_mask_source_incomplete]
        se_id = se_id[pad_mask_source_incomplete]
        if config.use_img:
            imgs_train = imgs_train[pad_mask_source_incomplete]
        
    class_dict={}
    for y in train_y_ori:
        if y not in class_dict:
            class_dict[y]=1
        else:
            class_dict[y]+=1
    print('Geolife:',dict(sorted(class_dict.items())))
    # print('Reading Data: (train: geolife + MTL, test: MTL)')
    print('GeoLife shape: '+str(train_x_ori.shape))
    # n_geolife = train_x.shape[0]
    # train_dataset_geolife = TensorDataset(
    #     torch.from_numpy(train_x).to(torch.float),
    #     torch.from_numpy(train_y),
    #     torch.from_numpy(np.array([0]*n_geolife)).float()
    # )
    
        
    if config.use_img:
        assert not config.use_traj
        input_size = 224
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
        transform_standard = {
            'train': transforms.Compose([
                transforms.Resize(input_size),
                transforms.RandomCrop(input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ]),
            'val': transforms.Compose([
                transforms.Resize(input_size),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                normalize
            ])
        }
        
        trainset_img = create_single_dataset_img(imgs_train, transform=transform_standard['train'])
        train_loader_img = DataLoader(trainset_img, batch_size=min(batch_sizes, len(trainset_img)), num_workers=0, shuffle=False, drop_last=False)
        
        img_encoder = resnet50(True).cuda()
        img_encoder.eval()
        with torch.no_grad():
            all_img_feat = []
            for idx,imgs in enumerate(train_loader_img):
                img_feat = img_encoder(imgs.cuda().float())
                all_img_feat.append(img_feat.cpu())
            all_img_feat = torch.cat(all_img_feat, dim=0)
            
        train_dataset_ori = TensorDataset(
            torch.from_numpy(train_x_ori).to(torch.float),
            torch.from_numpy(se_id).to(torch.float),
            all_img_feat,
            torch.from_numpy(train_y_ori)
        )
        train_loader_source_ori = DataLoader(train_dataset_ori, batch_size=min(batch_sizes, len(train_dataset_ori)), num_workers=0, shuffle=True, drop_last=True)
        
        # trainset_mix = create_single_dataset(
        #     imgs_train, 
        #     train_x_ori,
        #     train_y_ori,
        #     se_id,
        #     transform=transform_standard['train'],
        #     part='train'
        # )
        # train_loader_source_ori = DataLoader(trainset_mix, batch_size=min(batch_sizes, len(trainset_mix)), num_workers=0, shuffle=True, drop_last=False)
    
    elif config.use_traj:
        trainset_traj = TensorDataset(torch.from_numpy(train_x_ori).to(torch.float),torch.from_numpy(train_y_ori))
        train_loader_traj = DataLoader(trainset_traj, batch_size=min(batch_sizes, len(trainset_traj)), num_workers=0, shuffle=False, drop_last=False)
        
        G = TSEncoder_new(input_dims=6).cuda()
        F = Classifier_clf().cuda()
        
        ckpt_dir=config.resume
        print('resuming model', ckpt_dir)
        ckpt = torch.load(ckpt_dir, map_location='cuda:0')
        G.load_state_dict(ckpt['G'])#, strict=False)
        F.load_state_dict(ckpt['F2'])#, strict=False)
        
        G.eval()
        F.eval()
        with torch.no_grad():
            all_traj_feat = []
            top1 = AverageMeter('Acc', ':6.2f')
            for idx,(trajs,labels) in enumerate(train_loader_traj):
                _,traj_feat,_,_,_,_,_ = G(trajs[:,:,2:].cuda().float())
                all_traj_feat.append(traj_feat.cpu())

                y = F(traj_feat, None)
                acc1, = accuracy(y.cpu(), labels)
                top1.update(acc1.item(), y.size(0))
                
            print(traj_feat.shape)
            print(' * Acc {top1.avg:.3f} '.format(top1=top1))
            all_traj_feat = torch.cat(all_traj_feat, dim=0)
            
        train_dataset_ori = TensorDataset(
            torch.from_numpy(train_x_ori).to(torch.float),
            torch.from_numpy(se_id).to(torch.float),
            all_traj_feat,
            torch.from_numpy(train_y_ori)
        )
        train_loader_source_ori = DataLoader(train_dataset_ori, batch_size=min(batch_sizes, len(train_dataset_ori)), num_workers=0, shuffle=True, drop_last=True)
    
    else:
        
        batch_data_x = torch.from_numpy(train_x_ori).to(torch.float)
        label = torch.from_numpy(train_y_ori).unsqueeze(1)
        
        trip_len = torch.sum(batch_data_x[:,:,2]!=0, dim=1).unsqueeze(1)
        avg_feat = torch.sum(batch_data_x[:,:,3:5], dim=1) / (trip_len+1e-6)
        total_dist = torch.sum(batch_data_x[:,:,3], dim=1).unsqueeze(1)
        total_time = torch.sum(batch_data_x[:,:,2], dim=1).unsqueeze(1)
        avg_dist = avg_feat[:,0].unsqueeze(1)
        avg_speed = avg_feat[:,1].unsqueeze(1)
        # trip_len = trip_len / self.config.traj_len
        total_time = total_time / 3000.
        if config.encoder_dim==1:
            head = label.float()
        elif config.encoder_dim==8:
            head = torch.cat([label, total_dist, total_time, trip_len, avg_dist, avg_speed, sid, eid],dim=1)
        else:
            head = torch.cat([label, total_dist, total_time, trip_len, avg_dist, avg_speed],dim=1)
        
        batch_data_x[:,:,6] = batch_data_x[:,:,6] * 179.98863883
        speed = batch_data_x[:,:,4]
        x0 = torch.ones_like(batch_data_x[:,:,:2])
        x0[:,:,1] = speed * torch.sin(batch_data_x[:,:,6]/180*np.pi)
        x0[:,:,0] = speed * torch.cos(batch_data_x[:,:,6]/180*np.pi)
        
        
        train_dataset_ori = TensorDataset(
            torch.from_numpy(train_x_ori).to(torch.float),
            torch.from_numpy(se_id).to(torch.float),
            x0,head,
            torch.from_numpy(train_y_ori),
        )
        train_loader_source_ori = DataLoader(train_dataset_ori, batch_size=min(batch_sizes, len(train_dataset_ori)), num_workers=0, shuffle=True, drop_last=True)
        
    return train_loader_source_ori,None
    # train_tgt_iter = ForeverDataIterator(train_loader_source_ori)
    # return train_tgt_iter



def accuracy(output, target, topk=(1,)):
    r"""
    Computes the accuracy over the k top predictions for the specified values of k

    Args:
        output (tensor): Classification outputs, :math:`(N, C)` where `C = number of classes`
        target (tensor): :math:`(N)` where each value is :math:`0 \leq \text{targets}[i] \leq C-1`
        topk (sequence[int]): A list of top-N number.

    Returns:
        Top-N accuracies (N :math:`\in` topK).
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[None])

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            res.append(correct_k * (100.0 / batch_size))
        return res

class AverageMeter(object):
    r"""Computes and stores the average and current value.

    Examples::

        >>> # Initialize a meter to record loss
        >>> losses = AverageMeter()
        >>> # Update meter after every minibatch update
        >>> losses.update(loss_value, batch_size)
    """
    def __init__(self, name: str, fmt: Optional[str] = ':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    

class Proxy(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_ensembles=5):
        super().__init__()
        
        # self.models = nn.ModuleList([
        #     nn.Sequential(
        #         nn.Linear(input_dim, hidden_dim),
        #         nn.LeakyReLU(),
        #         nn.Linear(hidden_dim, hidden_dim),
        #         nn.LeakyReLU(),
        #         nn.Linear(hidden_dim, 1)
        #     )
        #  for _ in range(n_ensembles)])
        self.models = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim//4),
                nn.LeakyReLU(),
                nn.Linear(hidden_dim//4, hidden_dim//2),
                nn.LeakyReLU(),
                nn.Linear(hidden_dim//2, hidden_dim), # 256,200,1024
                # nn.AvgPool1d(200, stride=1),
                # nn.Linear(hidden_dim, 1)
            )
         for _ in range(n_ensembles)])
        self.fcs = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(n_ensembles)])
        print(self.models[0])
        
    def forward(self, x, confidence=False):
        xs = []
        for idx,model in enumerate(self.models):
            feat = model(x)
            feat = F.avg_pool1d(feat.transpose(1, 2), kernel_size = feat.size(1)).transpose(1, 2).squeeze(1)
            y_pred = self.fcs[idx](feat)
            xs.append(y_pred)
        xs = torch.stack(xs, dim=0)
        
        if confidence:
            return torch.mean(xs, dim=0), torch.std(xs, dim=0)
        else:
            return torch.mean(xs, dim=0)
        
    def get_loss(self, x, y):
        loss = 0.0
        for idx,model in enumerate(self.models):
            feat = model(x)
            feat = F.avg_pool1d(feat.transpose(1, 2), kernel_size = feat.size(1)).transpose(1, 2).squeeze(1)
            y_pred = self.fcs[idx](feat)
            loss += F.mse_loss(y, y_pred).mean()
        return loss
    
    
    def validate(self, epoch, val_loader, config):
        for model in self.models:
            model.eval()
            
        # top1_acc_list = [] 
        top1 = AverageMeter('Acc', ':6.2f')
        pbar = tqdm(val_loader, ncols=80)
        with torch.no_grad():
            for batch_data in pbar:
                bs = batch_data[-1].shape[0]
                x0 = batch_data[0][:,:,:2].float().cuda() # [256, 200, 9]
                label = batch_data[-1].unsqueeze(1).float()#.cuda()
                if "relative_xy" in config:
                    x0 = x0 - x0[:,0,:].unsqueeze(1)
                    lat_min,lat_max = (18.249901, 55.975593)
                    lon_min,lon_max = (-122.3315333, 126.998528)
                    x0[:,0] = x0[:,0] * (lat_max-lat_min) + lat_min
                    x0[:,1] = x0[:,1] * (lon_max-lon_min) + lon_min
                ys = self.forward(x0)
                
                pdb.set_trace()
                acc1, = accuracy(ys.cpu().numpy(), label)
                top1.update(acc1.item(), bs)
                pbar.set_description(f"Epoch {epoch}, Top 1: {acc1.item():.4f}")
        
        for model in self.models:
            model.train()
            
        self.log.info(f"Epoch {epoch}, Top 1: {top1.avg:.4f}")
        return top1.avg



class MID():
    def __init__(self, config):
        self.config = config
        torch.backends.cudnn.benchmark = True
        self._build()



    def train(self):
        for epoch in range(1, self.config.epochs + 1):
            # self.train_dataset.augment = self.config.augment
            # for node_type, data_loader in self.train_data_loader.items():
            node_type = ""
            data_loader = self.train_data_loader 
            pbar = tqdm(data_loader, ncols=80)
            epoch_losses = [] 
            for batch_data in pbar:
                self.optimizer.zero_grad()
                
                # batch_data_x = batch_data[0].float()#.cuda() # [256, 200, 9]
                # label = batch_data[-1].unsqueeze(1)#.cuda()
                
                # sid = batch_data[1][:,0].unsqueeze(1).float()#.cuda()
                # eid = batch_data[1][:,1].unsqueeze(1).float()#.cuda()
                
                if self.config.use_traj:
                    assert not self.config.use_img
                    # traj = batch_data_x[:,:,:2].cuda() 
                    # traj_feat = self.traj_encoder(traj.float())
                    traj_feat = batch_data[2].cuda()
                else:
                    traj_feat = None

                if self.config.use_img:
                    # img = batch_data[1].cuda() 
                    # img_feat = self.img_encoder(img.float())
                    img_feat = batch_data[2].cuda()
                else:
                    img_feat = None
                
                # trip_len = torch.sum(batch_data_x[:,:,2]!=0, dim=1).unsqueeze(1)
                # avg_feat = torch.sum(batch_data_x[:,:,3:5], dim=1) / (trip_len+1e-6)
                # total_dist = torch.sum(batch_data_x[:,:,3], dim=1).unsqueeze(1)
                # total_time = torch.sum(batch_data_x[:,:,2], dim=1).unsqueeze(1)
                # avg_dist = avg_feat[:,0].unsqueeze(1)
                # avg_speed = avg_feat[:,1].unsqueeze(1)
                # # trip_len = trip_len / self.config.traj_len
                # total_time = total_time / 3000.
                # if self.config.encoder_dim==1:
                #     head = label.float()
                # elif self.config.encoder_dim==8:
                #     head = torch.cat([label, total_dist, total_time, trip_len, avg_dist, avg_speed, sid, eid],dim=1)
                # else:
                #     head = torch.cat([label, total_dist, total_time, trip_len, avg_dist, avg_speed],dim=1)
                # head = head.cuda()
                
                # batch_data_x[:,:,6] = batch_data_x[:,:,6] * 179.98863883
                # speed = batch_data_x[:,:,4]
                # x0 = torch.ones_like(batch_data_x[:,:,:2])
                # x0[:,:,1] = speed * torch.sin(batch_data_x[:,:,6]/180*np.pi)
                # x0[:,:,0] = speed * torch.cos(batch_data_x[:,:,6]/180*np.pi)
                
                x0,head = batch_data[-3],batch_data[-2]
                x0,head = x0.cuda(),head.cuda()
                train_loss = self.model.get_loss(x0, latent=head, img_feat=img_feat, traj_feat=traj_feat)

                pbar.set_description(f"Epoch {epoch}, {node_type} MSE: {train_loss.item():.8f}")
                epoch_losses.append(train_loss.item())
                train_loss.backward()
                self.optimizer.step()
            self.log.info(f"Epoch {epoch}, {node_type} MSE: {np.array(epoch_losses).mean():.8f}")

            if epoch % self.config.save_every == 0:
                m_path = self.config.job_dir + f"/ckpt/unet_{epoch}.pt"
                torch.save(self.model.state_dict(), m_path)



    def eval(self, sampling, step):
        epoch = self.config.eval_at

        self.log.info(f"Sampling: {sampling} Stride: {step}")

        node_type = "PEDESTRIAN"
        eval_ade_batch_errors = []
        eval_fde_batch_errors = []
        ph = self.hyperparams['prediction_horizon']
        max_hl = self.hyperparams['maximum_history_length']


        for i, scene in enumerate(self.eval_scenes):
            print(f"----- Evaluating Scene {i + 1}/{len(self.eval_scenes)}")
            for t in tqdm(range(0, scene.timesteps, 10)):
                timesteps = np.arange(t,t+10)
                batch = get_timesteps_data(env=self.eval_env, scene=scene, t=timesteps, node_type=node_type, state=self.hyperparams['state'],
                               pred_state=self.hyperparams['pred_state'], edge_types=self.eval_env.get_edge_types(),
                               min_ht=7, max_ht=self.hyperparams['maximum_history_length'], min_ft=12,
                               max_ft=12, hyperparams=self.hyperparams)
                if batch is None:
                    continue
                test_batch = batch[0]
                nodes = batch[1]
                timesteps_o = batch[2]
                traj_pred = self.model.generate(test_batch, node_type, num_points=12, sample=20,bestof=True, sampling=sampling, step=step) # B * 20 * 12 * 2

                predictions = traj_pred
                predictions_dict = {}
                for i, ts in enumerate(timesteps_o):
                    if ts not in predictions_dict.keys():
                        predictions_dict[ts] = dict()
                    predictions_dict[ts][nodes[i]] = np.transpose(predictions[:, [i]], (1, 0, 2, 3))



                batch_error_dict = evaluation.compute_batch_statistics(predictions_dict,
                                                                       scene.dt,
                                                                       max_hl=max_hl,
                                                                       ph=ph,
                                                                       node_type_enum=self.eval_env.NodeType,
                                                                       kde=False,
                                                                       map=None,
                                                                       best_of=True,
                                                                       prune_ph_to_future=True)

                eval_ade_batch_errors = np.hstack((eval_ade_batch_errors, batch_error_dict[node_type]['ade']))
                eval_fde_batch_errors = np.hstack((eval_fde_batch_errors, batch_error_dict[node_type]['fde']))



        ade = np.mean(eval_ade_batch_errors)
        fde = np.mean(eval_fde_batch_errors)

        if self.config.dataset == "eth":
            ade = ade/0.6
            fde = fde/0.6
        elif self.config.dataset == "sdd":
            ade = ade * 50
            fde = fde * 50
        print(f"Sampling: {sampling} Stride: {step}")
        print(f"Epoch {epoch} Best Of 20: ADE: {ade} FDE: {fde}")
        #self.log.info(f"Best of 20: Epoch {epoch} ADE: {ade} FDE: {fde}")



    def train_proxy(self):
        raise NotImplementedError
        best_top1 = 0.
        for epoch in range(1, self.config.proxy_epochs + 1):
            node_type = ""
            train_loader = self.train_data_loader 
            pbar = tqdm(train_loader, ncols=80)
            epoch_losses = [] 
            for batch_data in pbar:
                self.optimizer_proxy.zero_grad()
                
                x0 = batch_data[0][:,:,:2].float().cuda() # [256, 200, 9]
                label = batch_data[-1].unsqueeze(1).float().cuda()
                if self.config.relative_xy:
                    x0 = x0 - x0[:,0,:].unsqueeze(1)
                    lat_min,lat_max = (18.249901, 55.975593)
                    lon_min,lon_max = (-122.3315333, 126.998528)
                    x0[:,0] = x0[:,0] * (lat_max-lat_min) + lat_min
                    x0[:,1] = x0[:,1] * (lon_max-lon_min) + lon_min
                train_loss = self.model_proxy.get_loss(x0, label)
                
                pbar.set_description(f"Epoch {epoch}, {node_type} MSE: {train_loss.item():.8f}")
                epoch_losses.append(train_loss.item())
                train_loss.backward()
                self.optimizer_proxy.step()
            # print(f"Epoch {epoch}, {node_type} MSE: {np.array(epoch_losses).mean():.8f}")
            self.log.info(f"Epoch {epoch}, {node_type} MSE: {np.array(epoch_losses).mean():.8f}")

            if epoch % self.config.proxy_save_every == 0:
                m_path = self.config.job_dir + f"/ckpt_proxy/unet_{epoch}.pt"
                torch.save(self.model_proxy.state_dict(), m_path)
                
            if epoch % self.config.proxy_eval_every == 0:
                top1 = self.model_proxy.validate(epoch, self.val_data_loader)
                if top1 > best_top1:
                    m_path = self.config.job_dir + f"/ckpt_proxy/unet_best.pt"
                    torch.save(self.model_proxy.state_dict(), m_path)
                    best_top1 = top1
    
    def eval_proxy(self, dataset_dir, ckpt_dir):
        raise NotImplementedError
        print(ckpt_dir)
        checkpoint = torch.load(ckpt_dir)
        self.model_proxy.load_state_dict(checkpoint)
        
        with open(dataset_dir, 'rb') as f:
            trajs, heads = pickle.load(f)
        test_dataset = TensorDataset(torch.from_numpy(trajs).to(torch.float), heads[:,0])
        
        test_loader = DataLoader(test_dataset, batch_size=min(self.config.batch_size, len(test_dataset)), num_workers=0, shuffle=False, drop_last=False)
        top1 = self.model_proxy.validate(0, test_loader, self.config)
        print("Top1: ",top1)



    def _build(self):
        self._build_dir()

        self._build_encoder_config()
        self._build_encoder()
        self._build_model()
        
        self._build_train_loader()
        self._build_eval_loader()

        self._build_optimizer()

        #self._build_offline_scene_graph()
        #pdb.set_trace()
        print("> Everything built. Have fun :)")

    def _build_dir(self):
        # self.model_dir = osp.join("./experiments",self.config.exp_name)
        self.model_dir = self.config.job_dir
        self.log_writer = SummaryWriter(log_dir=self.model_dir)
        os.makedirs(self.model_dir,exist_ok=True)
        log_name = '{}.log'.format(time.strftime('%Y-%m-%d-%H-%M'))
        log_name = f"{self.config.dataset}_{log_name}"

        log_dir = osp.join(self.model_dir, log_name)
        self.log = logging.getLogger()
        self.log.setLevel(logging.INFO)
        handler = logging.FileHandler(log_dir)
        handler.setLevel(logging.INFO)
        self.log.addHandler(handler)

        print(self.config)
        self.log.info("Config:")
        self.log.info(self.config)
        self.log.info("\n")
        self.log.info("Eval on:")
        self.log.info(self.config.dataset)
        self.log.info("\n")

        self.train_data_path = osp.join(self.config.data_dir,self.config.dataset + "_train.pkl")
        self.eval_data_path = osp.join(self.config.data_dir,self.config.dataset + "_test.pkl")
        print("> Directory built!")

    def _build_optimizer(self):
        self.optimizer = optim.Adam([{'params': self.registrar.get_all_but_name_match('map_encoder').parameters()},
                                     {'params': self.model.parameters()}
                                    ],
                                    lr=self.config.lr)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer,gamma=0.98)
        
        self.optimizer_proxy = optim.Adam(self.model_proxy.parameters(), lr=self.config.proxy_lr)
        self.scheduler_proxy = optim.lr_scheduler.ExponentialLR(self.optimizer_proxy,gamma=0.98)
        print("> Optimizer built!")

    def _build_encoder_config(self):

        self.hyperparams = get_traj_hypers()
        self.hyperparams['enc_rnn_dim_edge'] = self.config.encoder_dim//2
        self.hyperparams['enc_rnn_dim_edge_influence'] = self.config.encoder_dim//2
        self.hyperparams['enc_rnn_dim_history'] = self.config.encoder_dim//2
        self.hyperparams['enc_rnn_dim_future'] = self.config.encoder_dim//2
        # registar
        self.registrar = ModelRegistrar(self.model_dir, "cuda")

        # if self.config.eval_mode:
        #     epoch = self.config.eval_at
        #     checkpoint_dir = osp.join(self.model_dir, f"{self.config.dataset}_epoch{epoch}.pt")
        #     self.checkpoint = torch.load(osp.join(self.model_dir, f"{self.config.dataset}_epoch{epoch}.pt"), map_location = "cpu")

        #     self.registrar.load_models(self.checkpoint['encoder'])


        # with open(self.train_data_path, 'rb') as f:
        #     self.train_env = dill.load(f, encoding='latin1')
        # with open(self.eval_data_path, 'rb') as f:
        #     self.eval_env = dill.load(f, encoding='latin1')
        self.train_env = None

    def _build_encoder(self):
        self.encoder = None
        return
        self.encoder = Trajectron(self.registrar, self.hyperparams, "cuda")

        self.encoder.set_environment(self.train_env)
        self.encoder.set_annealing_params()

    def _build_model(self):
        """ Define Model """
        config = self.config
        model = AutoEncoder(config, encoder = self.encoder)

        self.model = model.cuda()
        if self.config.eval_mode:
            self.model.load_state_dict(self.checkpoint['ddpm'])
            
        model_proxy = Proxy(2, self.config.proxy_hidden_dim, 1, n_ensembles=self.config.proxy_n_ensemble)
        self.model_proxy = model_proxy.cuda()
        
        # if self.config.use_img:
        #     img_encoder = resnet50(True)
        #     for name,param in img_encoder.named_parameters():
        #         param.requires_grad = False 
        #     self.img_encoder = img_encoder.cuda()
        #     self.img_encoder.eval()

        print("> Model built!")

    def _build_train_loader(self):
        self.train_data_loader, self.val_data_loader = load_data(self.config)

    def _build_eval_loader(self):
        return
        config = self.config
        self.eval_scenes = []
        eval_scenes_sample_probs = None

        if config.eval_every is not None:
            with open(self.eval_data_path, 'rb') as f:
                self.eval_env = dill.load(f, encoding='latin1')

            for attention_radius_override in config.override_attention_radius:
                node_type1, node_type2, attention_radius = attention_radius_override.split(' ')
                self.eval_env.attention_radius[(node_type1, node_type2)] = float(attention_radius)

            if self.eval_env.robot_type is None and self.hyperparams['incl_robot_node']:
                self.eval_env.robot_type = self.eval_env.NodeType[0]  # TODO: Make more general, allow the user to specify?
                for scene in self.eval_env.scenes:
                    scene.add_robot_from_nodes(self.eval_env.robot_type)

            self.eval_scenes = self.eval_env.scenes
            eval_scenes_sample_probs = self.eval_env.scenes_freq_mult_prop if config.scene_freq_mult_eval else None
            self.eval_dataset = EnvironmentDataset(self.eval_env,
                                              self.hyperparams['state'],
                                              self.hyperparams['pred_state'],
                                              scene_freq_mult=self.hyperparams['scene_freq_mult_eval'],
                                              node_freq_mult=self.hyperparams['node_freq_mult_eval'],
                                              hyperparams=self.hyperparams,
                                              min_history_timesteps=self.hyperparams['minimum_history_length'],
                                              min_future_timesteps=self.hyperparams['prediction_horizon'],
                                              return_robot=not config.incl_robot_node)
            self.eval_data_loader = dict()
            for node_type_data_set in self.eval_dataset:
                node_type_dataloader = utils.data.DataLoader(node_type_data_set,
                                                             collate_fn=collate,
                                                             pin_memory=True,
                                                             batch_size=config.eval_batch_size,
                                                             shuffle=True,
                                                             num_workers=config.preprocess_workers)
                self.eval_data_loader[node_type_data_set.node_type] = node_type_dataloader

        print("> Dataset built!")

    def _build_offline_scene_graph(self):
        if self.hyperparams['offline_scene_graph'] == 'yes':
            print(f"Offline calculating scene graphs")
            for i, scene in enumerate(self.train_scenes):
                scene.calculate_scene_graph(self.train_env.attention_radius,
                                            self.hyperparams['edge_addition_filter'],
                                            self.hyperparams['edge_removal_filter'])
                print(f"Created Scene Graph for Training Scene {i}")

            for i, scene in enumerate(self.eval_scenes):
                scene.calculate_scene_graph(self.eval_env.attention_radius,
                                            self.hyperparams['edge_addition_filter'],
                                            self.hyperparams['edge_removal_filter'])
                print(f"Created Scene Graph for Evaluation Scene {i}")
