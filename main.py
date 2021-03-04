#!/usr/bin/env python
# coding: utf-8

import torch
import torchvision
import numpy as np
import time 

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import argparse

parser = argparse.ArgumentParser(description='Test of memory/CPU/GPU')
parser.add_argument('--n_memory', default=1000, type=int, metavar='N',
                    help='number of iterations for memory access')
parser.add_argument('--n_batchs', default=1000, type=int, metavar='N',
                    help='Number of batchs for the only epoch of testing for cpu/gpu')
parser.add_argument('--n_test_gpus', default=20, type=int, metavar='N',
                    help='number of iterations of generating and training on one epoch on GPU')
parser.add_argument('--n_test_cpus', default=20, type=int, metavar='N',
                    help='number of iterations of generating and training on one epoch on CPU')


batch_size_train = 64
learning_rate = 0.01
momentum = 0.5
torch.manual_seed(1)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(train_loader, network, optimizer, usecuda, device):
  network.train()
  t0 = time.time()
  for batch_idx, (data, target) in enumerate(train_loader):
    if usecuda:
        data,  target = Variable(data.to(device)), Variable(target.to(device))
    optimizer.zero_grad()
    output = network(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()


def memory_test(args):
    train_loader = torch.utils.data.DataLoader(
                                torchvision.datasets.MNIST('../data/', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                 torchvision.transforms.ToTensor(),
                                 torchvision.transforms.Normalize(
                                   (0.1307,), (0.3081,))
                               ])),batch_size=batch_size_train, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
                                torchvision.datasets.MNIST('../data/', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                 torchvision.transforms.ToTensor(),
                                 torchvision.transforms.Normalize(
                                   (0.1307,), (0.3081,))
                               ])),batch_size=1000, shuffle=True)
    time_memory_access = []
    for _ in range(1000):
        t0 = time.time()
        train_loader = torch.utils.data.DataLoader(
                                torchvision.datasets.MNIST('../data/', train=True, download=False,
                               transform=torchvision.transforms.Compose([
                                 torchvision.transforms.ToTensor(),
                                 torchvision.transforms.Normalize(
                                   (0.1307,), (0.3081,))
                               ])),batch_size=batch_size_train, shuffle=True)

        test_loader = torch.utils.data.DataLoader(
                                torchvision.datasets.MNIST('../data/', train=False, download=False,
                               transform=torchvision.transforms.Compose([
                                 torchvision.transforms.ToTensor(),
                                 torchvision.transforms.Normalize(
                                   (0.1307,), (0.3081,))
                               ])),batch_size=1000, shuffle=True)
        time_memory_access.append(time.time() -t0)
    return(round(np.mean(time_memory_access), 4), train_loader) 


def cpu_test(args, train_loader):
    time_train = []
    time_loop = []
    device = torch.device('cpu')
    usecuda = False
    for _ in range(args.n_test_cpus):
        t1 = time.time()
        network = Net()
        optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                      momentum=momentum)
        for _ in range(args.n_batchs):
            t0 = time.time()
            train(train_loader, network, optimizer, usecuda, device)
            time_train.append(time.time()-t0)
        time_loop.append(time.time()-t1)
    return(round(np.mean(time_train), 4), round(np.mean(time_loop), 4))


def gpu_test(args, train_loader):
    time_train = []
    time_loop = []
    assert torch.cuda.is_available(), "Cuda not available, Can not test GPU"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    usecuda = True
    for _ in range(args.n_test_gpus):
        t1 = time.time()
        network = Net()
        optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                      momentum=momentum)
        network = network.cuda()
        for _ in range(args.n_batchs):
            t0 = time.time()
            train(train_loader, network, optimizer, usecuda, device)
            time_train.append(time.time()-t0)
        time_loop.append(time.time()-t1)
    return(round(np.mean(time_train), 4), round(np.mean(time_loop), 4))


def main():
    args = parser.parse_args()

    tm, train_loader = memory_test(args)
    print("Memory : " + str(tm).zfill(3) + "s.")

    tc, tl = cpu_test(args, train_loader)
    print("\nCPU:\nGen + Train: " + str(tl).zfill(3) + "s. \t Train batch: " + str(tc).zfill(3) + "s.")

    tc, tl = gpu_test(args, train_loader)
    print("\nGPU:\nGen + Train: " + str(tl).zfill(3) + "s. \t Train batch: " + str(tc).zfill(3) + "s.")



if __name__ == '__main__':
    main()


