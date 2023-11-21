import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch.nn.functional as F

import torchvision
from torch import optim
from torch.optim import lr_scheduler

from tools.opts import *
from tools.utils import *
from dataset.kisadataloader import KISADataLoader
from models.model import generate_model

import os
import time

if __name__ == '__main__':
    opt = parse_opts()
    print(opt)
    
    opt.arch = '{}-{}'.format(opt.model, opt.model_depth)
    torch.manual_seed(opt.manual_seed)

    print("Preprocessing train data ...")
    
    train_data = KISADataLoader(train=1, opt=opt)
    
    print("Length of train data = ", len(train_data))

    print("Preparing datatloaders ...")
    
    train_dataloader = DataLoader(train_data, batch_size = opt.batch_size, shuffle=True, num_workers = opt.n_workers, pin_memory = True)

    print("Length of train datatloader = ",len(train_dataloader))

    print("Loading model... ", opt.model, opt.model_depth)
    model, parameters = generate_model(opt)
    
    criterion = nn.CrossEntropyLoss().cuda()

    if opt.resume_path1:
        print('loading checkpoint {}'.format(opt.resume_path1))
        checkpoint = torch.load(opt.resume_path1)
        
        assert opt.arch == checkpoint['arch']
        opt.begin_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
    
    log_path = os.path.join(opt.result_path + '/runs/', opt.dataset)
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
    
    print('run')
    for epoch in range(opt.begin_epoch, opt.n_epochs + 1):
        
        model.train()
        
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        accuracies = AverageMeter()

        end_time = time.time()
        #pdb.set_trace()
        for i, (inputs, targets,_) in enumerate(train_dataloader):
#             print(" inputs  : ",inputs.size())
            data_time.update(time.time() - end_time)
        
            targets = targets.cuda(non_blocking=True)
            inputs = Variable(inputs)
            targets = Variable(targets)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            acc = calculate_accuracy(outputs, targets)

            losses.update(loss.item(), inputs.size(0))
            accuracies.update(acc, inputs.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end_time)
            end_time = time.time()

            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch,
                      i + 1,
                      len(train_dataloader),
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses,
                      acc=accuracies))
                      
        if opt.log == 1:
            epoch_logger.log({
                'epoch': epoch,
                'loss': losses.avg,
                'acc': accuracies.avg,
                'lr': optimizer.param_groups[0]['lr']
            })

        
        if epoch % opt.checkpoint == 0:
            if opt.pretrain_path:
                save_file_path = os.path.join(log_path, 'PreKin_{}_{}_{}_train_batch{}_sample{}_clip{}_nest{}_damp{}_weight_decay{}_manualseed{}_model{}{}_ftbeginidx{}_varLR{}.pth'
                            .format(opt.dataset, opt.split, opt.modality, opt.batch_size, opt.sample_size, opt.sample_duration, opt.nesterov, opt.dampening, opt.weight_decay, opt.manual_seed, opt.model,
                                    opt.model_depth, opt.ft_begin_index, epoch))
            else:
                save_file_path = os.path.join(log_path, '{}_{}_{}_train_batch{}_sample{}_clip{}_nest{}_damp{}_weight_decay{}_manualseed{}_model{}{}_ftbeginidx{}_varLR{}.pth'
                            .format(opt.dataset, opt.split, opt.modality, opt.batch_size, opt.sample_size, opt.sample_duration, opt.nesterov, opt.dampening, opt.weight_decay, opt.manual_seed, opt.model,
                                    opt.model_depth, opt.ft_begin_index, epoch))
            states = {
                'epoch': epoch + 1,
                'arch': opt.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(states, save_file_path)
        
        model.eval()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        accuracies = AverageMeter()

        end_time = time.time()
        submit_file = './submit/submit'+str(epoch)+'.json'
        result = dict()
#         with torch.no_grad():
#             for i, (inputs, targets, label_name) in enumerate(val_dataloader):
# #                 print(" label_name  : ",label_name)
                
#                 # pdb.set_trace()
#                 data_time.update(time.time() - end_time)
#                 targets = targets.cuda(non_blocking=True)
#                 inputs = Variable(inputs)
#                 targets = Variable(targets)
#                 outputs = model(inputs)
                
#                 outputs2 = F.softmax(outputs, dim=1).cpu()
#                 _, label2 = torch.topk(outputs2, k=1)
#                 result[label_name[0]] = label2.squeeze().item()
                
                
#                 loss = criterion(outputs, targets)
#                 acc = calculate_accuracy(outputs, targets)
            
#                 losses.update(loss.item(), inputs.size(0))
#                 accuracies.update(acc, inputs.size(0))

#                 batch_time.update(time.time() - end_time)
#                 end_time = time.time()

#                 print('Val_Epoch: [{0}][{1}/{2}]\t'
#                     'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
#                     'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
#                     'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
#                     'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
#                         epoch,
#                         i + 1,
#                         len(val_dataloader),
#                         batch_time=batch_time,
#                         data_time=data_time,
#                         loss=losses,
#                         acc=accuracies))
                      
#         with open(submit_file, "w") as json_file:
#             json.dump(result, json_file)
#         if opt.log == 1:
#             val_logger.log({'epoch': epoch, 'loss': losses.avg, 'acc': accuracies.avg})
        scheduler.step(losses.avg)


