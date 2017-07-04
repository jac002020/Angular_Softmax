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
parser = argparse.ArgumentParser(description = 'explore')
# Resume
parser.add_argument('--resume', default = '', type = str, metavar = 'PATH', help = 'path to latest checkpoint (default: none)')

# Architecture
parser.add_argument('--arch', '-a', metavar = 'ARCH', default = 'resnet20', choices = model_names, help = 'model architecture: ' + ' | '.join(model_names) + ' (default: resnet 20)')
parser.add_argument('--depth', type = int, default = 29, help = 'Model depth')
parser.add_argument('--cardinality', type = int, default = 8, help = 'Model cardinality (group).')
parser.add_argument('--widen-factor', type = int, default = 4, help = 'Widen factor. 4 -> 64, 8 -> 128, ...')
parser.add_argument('--growthRate', type = int, default = 12, help = 'Growth rate for DenseNet')
parser.add_argument('--compressionRate', type = int, default = 2, help = 'Compression Rate (theta) for DenseNet')
parser.add_argument('--drop', '--dropout', default = 0.1, type = float, metavar = 'Dropout', help = 'Dropout ratio')

# Parameters
num_classes = 10
dataloader = datasets.CIFAR10

args = parser.parse_args()
# load model
print('==> loading model {}'.format(args.arch))
model = 0
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

model = torch.nn.DataParallel(model).cuda()
cudnn.benchmark = True

if args.resume:
    print('==> Resuming from checkpoint...')
    assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
    args.checkpoint = os.path.dirname(args.resume)
    checkpoint      = torch.load(args.resume)
    model.load_state_dict(checkpoint['state_dict'])
    best_acc        = checkpoint['best_acc']
    print('best_acc = ', best_acc)
else:
    print('Please specify a checkpoint!')
    exit(0)

print(model.module.fc.bias.data)
exit()

x = model.module.fc.weight.data
#print(model.module.fc.weight.data)
y = x.mm(x.t())
n = x.norm(2, 1)
m = n.mm(n.t())
cos = y / m
cos[cos > 1.0] = 1.0
theta = cos.acos()
import math
PI = math.acos(-1.0)
theta = theta / PI
theta = theta * 180.0
print('min = ', theta[theta > 10.0].min())
print('max = ', theta.max())
print('mean = ', theta[theta > 10.0].mean())

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
testset = dataloader(root = './data', train = False, download = False, transform = transform_test)
testloader = data.DataLoader(testset, batch_size = 100, shuffle = False, num_workers = 4)

model.eval()

total_wrong = 0
confusion_matrix = torch.zeros(num_classes, num_classes)
for batch_idx, (inputs, targets) in enumerate(testloader):
    inputs, targets = inputs.cuda(), targets.cuda()
    inputs, targets = Variable(inputs, volatile=True), Variable(targets)
    outputs = model(inputs)

    _, predicted = torch.max(outputs.data, 1)

    for i in xrange(targets.data.size(0)):
        p = predicted[i][0]
        v = targets.data[i]
        if p != v:
            confusion_matrix[v][p] += 1
            total_wrong += 1
print(confusion_matrix)
print('total_wrong = ', total_wrong)

print('theta + confusion_matrix = ')
for i in range(theta.size(0)):
    for j in range(theta.size(1)):
        print('%6.2f|%2d  ' % (theta[i][j], confusion_matrix[i][j]), end = '')
    print('')
