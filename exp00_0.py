#!/usr/bin/env python
# coding=utf-8

from __future__ import print_function
import argparse
import os
import shutil
import time
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets   as datasets
import models.cifar as models
from torch.autograd import Variable

from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig

model_names = sorted(name for name in models.__dict__
                    if name.islower() and not name.startswith("__")
                    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description = 'PyTorch CIFAR10 Training')
# Datasets
parser.add_argument('-d', '--dataset', default = 'cifar10', type = str)
parser.add_argument('-j', '--workers', default = 4, type = int, metavar = 'N', 
                   help = 'number of data loading workersers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default = 300, type = int, metavar = 'N', 
                   help = 'number of total epochs to run')
parser.add_argument('--start-epoch', default = 0, type = int, metavar = 'N', 
                   help = 'manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default = 128, type = int, metavar = 'N', 
                   help = 'train batchsize')
parser.add_argument('--test-batch', default = 100, type = int, metavar = 'N', 
                   help = 'test batchsize')
parser.add_argument('--lr', '--learing-rate', default = 0.1, type = float, metavar = 'N', help = 'initial learning rate')
parser.add_argument('--drop', '--dropout', default = 0.1, type = float, metavar = 'Dropout', help = 'Dropout ratio')
parser.add_argument('--schedule', type = int, nargs = "+", default = [150, 225], 
                   help = 'Decrease learning rate at these epochs')
parser.add_argument('--gamma', type = float, default = 0.1, help = 'LR is multiplied by gamma on schedule')
parser.add_argument('--momentum', default = 0.9, type = float, metavar = 'M', help = 'momentum')
parser.add_argument('--weight-decay', '--wd', default = 5e-4, type = float, metavar = 'W', help = 'weight decay (default:5e-4)')

# Checkpoints
parser.add_argument('-c', '--checkpoint', default = 'checkpoint', type = str, metavar = 'PATH', help = 'path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default = '', type = str, metavar = 'PATH', help = 'path to latest checkpoint (default: none)')

# Architecture
parser.add_argument('--arch', '-a', metavar = 'ARCH', default = 'resnet20', choices = model_names, help = 'model architecture: ' + ' | '.join(model_names) + ' (default: resnet 20)')
parser.add_argument('--depth', type = int, default = 29, help = 'Model depth')
parser.add_argument('--cardinality', type = int, default = 8, help = 'Model cardinality (group).')
parser.add_argument('--widen-factor', type = int, default = 4, help = 'Widen factor. 4 -> 64, 8 -> 128, ...')
parser.add_argument('--growthRate', type = int, default = 12, help = 'Growth rate for DenseNet')
parser.add_argument('--compressionRate', type = int, default = 2, help = 'Compression Rate (theta) for DenseNet')
parser.add_argument('--lam', default = 0.1, type = float, help = 'lambda to adjust the angle distance')

# Miscs
parser.add_argument('--manualSeed', type = int, help = 'manual seed')
parser.add_argument('-e', '--evaluate', dest = 'evaluate', action = 'store_true', 
                   help = 'evaluate model on validation set')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Validate dataset
assert args.dataset == 'cifar10' or args.dataset == 'cifar100', 'Dataset can only be cifar10 or cifar100'

# Use CUDA
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 19941229)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

# best test accuracy
best_acc = 0
global_writer = 0
global_num_classes = 0

def main():
    global best_acc
    global global_writer
    global global_num_classes
    start_epoch = args.start_epoch # start from 0 or last checkpoint

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    # Data
    print('==> Preparing dataset %s' % args.dataset)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding = 4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test   = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if args.dataset == 'cifar10':
        dataloader  = datasets.CIFAR10
        num_classes = 10
        global_num_classes = 10
    else:
        dataloader  = datasets.CIFAR100
        num_classes = 100
        global_num_classes = 100

    trainset        = dataloader(root = './data', train = True, download = True, transform = transform_train)
    trainloader     = data.DataLoader(trainset, batch_size = args.train_batch, shuffle = True, num_workers = args.workers)
    testset         = dataloader(root = './data', train = False, download = False, transform = transform_test)
    testloader      = data.DataLoader(testset, batch_size = args.test_batch, shuffle = False, num_workers = args.workers)

    # Model 
    print("==> Creating model '{}'".format(args.arch))
    if args.arch.startswith('resnext'):
        model = models.__dict__[args.arch](
            cardinality = args.cardinality,
            num_classes = num_classes,
            depth       = args.depth,
            widen_factor= args.widen_factor,
            dropRate    = args.drop,
        )
    elif args.arch.startswith('densenet'):
        model = models.__dict__[args.arch](
            num_classes = num_classes,
            depth       = args.depth,
            growthRate  = args.growthRate,
            compressionRate = args.compressionRate,
            dropRate    = args.drop,
        )
    elif args.arch.endswith('resnet'):
        model = models.__dict__[args.arch](
            num_classes = num_classes,
            depth       = args.depth,
        )
    else:
        model = models.__dict__[args.arch](num_classes = num_classes)

    
    # angleW = model.angle.weight

    model.cuda()
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True
    print('     Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/ 1000000.0))   

    # Define a-softmax learning
    nll    = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr = args.lr, momentum = args.momentum, weight_decay = args.weight_decay)

    title   = 'cifar-%d-' % (global_num_classes) + args.arch

    if args.resume:
        ###### TODO add !!
        pass
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title = title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])
        os.system('rm ' + os.path.join(args.checkpoint, 'info.txt'))
        global_writer = open(os.path.join(args.checkpoint, 'info.txt'), 'a', 0)


    if args.evaluate:
        print('\nEvaluation only')
        test_loss, test_acc = test(testloader, model, nll, start_epoch, use_cuda)
        print(' Test Loss: %.8f, Test Acc: %.2f' % (test_loss, test_acc))
        return
    
    # Train and evaluate
    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))
        train_loss, train_acc = train(trainloader, model, nll, optimizer, epoch, use_cuda, args.lam)
        test_loss, test_acc   = test(testloader,   model, nll, epoch, use_cuda)
        # append logger file
        logger.append([state['lr'], train_loss, test_loss, train_acc, test_acc])
        # save model
        is_best     = test_acc > best_acc
        best_acc    = max(test_acc, best_acc)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'acc':   test_acc,
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }, is_best, checkpoint = args.checkpoint)

    print('Best acc:')
    print(best_acc)

    logger.close()
    logger.plot()
    savefig(os.path.join(args.checkpoint, 'log.eps'))

def train(trainloader, model, nll, optimizer, epoch, use_cuda, lam):
    # switch to train mode
    global global_writer
    global global_num_classes
    model.train()
    print('lam = ', lam)

    batch_time = AverageMeter()
    data_time  = AverageMeter()
    losses     = AverageMeter()
    top1       = AverageMeter()
    top5       = AverageMeter()
    end        = time.time()

    bar        = Bar('Processing', max = len(trainloader))
    fixed   = torch.ones(global_num_classes, 1)
    if use_cuda:
        fixed = fixed.cuda()
    fixed   = Variable(fixed, requires_grad = False)
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda(async = True)
        inputs, targets = Variable(inputs), Variable(targets)

        # compute output
        outputs = model(inputs)
        angleW  = model.module.fc.weight
        angleWt = angleW.t() # 1024 x 10
        W       = angleWt.mm(fixed)
        Wt      = W.t()
        cos     = Wt.mm(W)
        cos     = cos * lam

        cos_loss_part= cos.data[0][0]

        loss    = nll(outputs, targets) + cos
        
        # measure acc and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk = (1, 5))
        losses.update(loss.data[0], inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))

        if batch_idx == len(trainloader) - 1:
        #if batch_idx > -1:
            x = angleW.data
            y = x.mm(x.t())
            n = x.norm(2, 1)
            m = n.mm(n.t())
            c = y / m
            c[c > 1.0] = 1.0
            import math
            PI = math.acos(-1.0)
            theta      = c.acos()
            theta      = theta / PI
            theta      = theta * 180.0
            print('batch_idx = ' + str(batch_idx), file = global_writer)
            print(theta, file = global_writer)
            print('min = ' + str(theta[theta > 10.0].min()), file = global_writer)
            print('max = ' + str(theta.max()), file = global_writer)
            print('mean = ' +str(theta[theta > 10.0].mean()), file = global_writer)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # TODO renormalize the angleW
        norm = angleW.data.norm(2, 1).expand_as(angleW.data)
        angleW.data.div_(norm)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        w = model.module.fc.weight.data
        w = w.norm(2, 1).squeeze()
        mw = w.mean()
        #s = ''
        #for x in range(w.size(0)):
        #    s += '%6.3f ' % w[x]
        #print(s)

        # plot progress
        bar.suffix = '({batch} / {size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f} | mean_W: {mw: .4f} | Old_Loss: {old_loss: .4f} | Cos_Loss: {cos_loss:.4f}'.format(
            batch = batch_idx + 1,
            size  = len(trainloader),
            data  = data_time.avg,
            bt    = batch_time.avg,
            total = bar.elapsed_td,
            eta   = bar.eta_td,
            loss  = losses.avg,
            top1  = top1.avg,
            top5  = top5.avg,
            mw    = mw,
            old_loss = loss.data[0] - cos_loss_part,
            cos_loss = cos_loss_part,
            )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)

def test(testloader, model, nll, epoch, use_cuda):
    global best_acc

    batch_time     = AverageMeter()
    data_time      = AverageMeter()
    losses         = AverageMeter()
    top1           = AverageMeter()
    top5           = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar('Processing', max = len(testloader))
    for batch_idx, (inputs, targets) in enumerate(testloader):
        data_time.update(time.time() - end)
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda(async = True)
        inputs, targets = Variable(inputs), Variable(targets)

        outputs = model(inputs)
        loss    = nll(outputs, targets)

        prec1, prec5 = accuracy(outputs.data, targets.data, topk = (1, 5))
        losses.update(loss.data[0], inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix = '({batch} / {size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
            batch = batch_idx + 1,
            size  = len(testloader),
            data  = data_time.avg,
            bt    = batch_time.avg,
            total = bar.elapsed_td,
            eta   = bar.eta_td,
            loss  = losses.avg,
            top1  = top1.avg,
            top5  = top5.avg,
            )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)

def save_checkpoint(state, is_best, checkpoint = 'checkpoint', filename = 'checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']

if __name__ == '__main__':
    main()






