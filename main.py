import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import torchvision
from torch import optim
from torch.optim import lr_scheduler

from tools.opts import *
from tools.utils import *
from dataset.kisadataloader import KISADataloader
from models.model import generate_model

import os

if __name__ == '__main__':
    opt = parse_opts()
    print(opt)
    
    opt.arch = '{}-{}'.format(opt.model, opt.model_depth)
    torch.manual_seed(opt.manual_seed)

    print("Preprocessing train data ...")
    
    train_data = KISADataloader(train=1, opt=opt)
    
    print("Length of train data = ", len(train_data))

    print("Preparing datatloaders ...")
    
    train_dataloader = DataLoader(train_data, batch_size = opt.batch_size, shuffle=True, num_workers = opt.n_workers, pin_memory = True)

    print("Length of train datatloader = ",len(train_dataloader))

    # define the model 
    print("Loading model... ", opt.model, opt.model_depth)
    model, parameters = generate_model(opt)
    
    criterion = nn.CrossEntropyLoss().cuda()

    criterion = nn.CrossEntropyLoss().cuda()

    if opt.resume_path1:
        print('loading checkpoint {}'.format(opt.resume_path1))
        checkpoint = torch.load(opt.resume_path1)
        
        assert opt.arch == checkpoint['arch']
        opt.begin_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
    
    log_path = os.path.join(opt.result_path, opt.dataset)
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    if opt.log == 1:
        if opt.pretrain_path:
            epoch_logger = Logger(os.path.join(log_path, 'PreKin_{}_{}_{}_train_batch{}_sample{}_clip{}_nest{}_damp{}_weight_decay{}_manualseed{}_model{}{}_ftbeginidx{}_varLR.log'
                            .format(opt.dataset, opt.split, opt.modality, opt.batch_size, opt.sample_size, opt.sample_duration, opt.nesterov, opt.dampening, opt.weight_decay, opt.manual_seed, opt.model, opt.model_depth, opt.ft_begin_index))
                            ,['epoch', 'loss', 'acc', 'lr'], opt.resume_path1, opt.begin_epoch-1)
            val_logger   = Logger(os.path.join(log_path, 'PreKin_{}_{}_{}_val_batch{}_sample{}_clip{}_nest{}_damp{}_weight_decay{}_manualseed{}_model{}{}_ftbeginidx{}_varLR.log'
                            .format(opt.dataset, opt.split, opt.modality, opt.batch_size, opt.sample_size, opt.sample_duration, opt.nesterov, opt.dampening, opt.weight_decay, opt.manual_seed, opt.model, opt.model_depth, opt.ft_begin_index))
                            ,['epoch', 'loss', 'acc'], opt.resume_path1, opt.begin_epoch-1)
        else:
            epoch_logger = Logger(os.path.join(log_path, '{}_{}_{}_train_batch{}_sample{}_clip{}_nest{}_damp{}_weight_decay{}_manualseed{}_model{}{}_ftbeginidx{}_varLR.log'
                            .format(opt.dataset, opt.split, opt.modality, opt.batch_size, opt.sample_size, opt.sample_duration, opt.nesterov, opt.dampening, opt.weight_decay, opt.manual_seed, opt.model, opt.model_depth, opt.ft_begin_index))
                            ,['epoch', 'loss', 'acc', 'lr'], opt.resume_path1, opt.begin_epoch-1)
            val_logger   = Logger(os.path.join(log_path, '{}_{}_{}_val_batch{}_sample{}_clip{}_nest{}_damp{}_weight_decay{}_manualseed{}_model{}{}_ftbeginidx{}_varLR.log'
                            .format(opt.dataset, opt.split, opt.modality, opt.batch_size, opt.sample_size, opt.sample_duration, opt.nesterov, opt.dampening, opt.weight_decay, opt.manual_seed, opt.model, opt.model_depth, opt.ft_begin_index))
                            ,['epoch', 'loss', 'acc'], opt.resume_path1, opt.begin_epoch-1)
           
    print("Initializing the optimizer ...")
    if opt.pretrain_path: 
        opt.weight_decay = 1e-5
        opt.learning_rate = 0.001

    if opt.nesterov: dampening = 0
    else: dampening = opt.dampening
        
    print("lr = {} \t momentum = {} \t dampening = {} \t weight_decay = {}, \t nesterov = {}"
                .format(opt.learning_rate, opt.momentum, dampening, opt. weight_decay, opt.nesterov))
    print("LR patience = ", opt.lr_patience)
    
    
    optimizer = optim.SGD(
        parameters,
        lr=opt.learning_rate,
        momentum=opt.momentum,
        dampening=dampening,
        weight_decay=opt.weight_decay,
        nesterov=opt.nesterov)

    if opt.resume_path1 != '':
        optimizer.load_state_dict(torch.load(opt.resume_path1)['optimizer'])

    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=opt.lr_patience)
    
    print(model)


