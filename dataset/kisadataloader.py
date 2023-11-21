from __future__ import division
from torch.utils.data import Dataset, DataLoader
import getpass
import os
import socket
import numpy as np
from .preprocess_data import *
from PIL import Image, ImageFilter
import pickle
import glob
#import dircache
import pdb

label_index = ['Abandonment',  'Falldown',  'FireDetection',  'Loitering',  'Intrusion', 'Violence', 'Normal']

def get_cctv(opt, frame_path, Total_frames):
    clip = []
    clip_image = []
    path_list = glob.glob(frame_path+'/*')
    path_list.sort()
    i = 0
    while(len(clip)!= 10):
        clip.append(path_list[i % len(path_list)])
    for i in clip:
        im = Image.open(i)
        clip_image.append(im.copy())
        im.close()

    return clip_image


class KISADataLoader(Dataset):
    def __init__(self, train, opt):
        if train == 1:
            self.clips = glob.glob(opt.root_path + 'train/*/*')
        else:
            self.clips = glob.glob(opt.root_path + 'val/*/*')
            self.clips.sort()
        self.opt = opt
        self.train_val_test = train
        self.train = train
    def __len__(self):
        return len(self.clips)
    def __getitem__(self, idx):
        path = self.clips[idx]
        label = path.split('/')[7]
        
        if self.train != 1:
            label_name = path.split('/')[8]
        else:
            label_name = 0
            
        label = label_index.index(label)
        Total_frames = len(glob.glob(path+'/*.jpg'))
        # print(path)
        clip = get_cctv(self.opt, path, Total_frames)
        return((scale_crop(clip, self.train_val_test, self.opt), label, label_name))
    


