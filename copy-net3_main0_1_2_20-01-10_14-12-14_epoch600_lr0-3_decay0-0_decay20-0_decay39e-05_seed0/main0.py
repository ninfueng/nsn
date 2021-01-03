#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The code implementation of SharedGradNet.
main0.py is for neural networks without hidden layer.
Some part from: https://jhui.github.io/2018/02/09/PyTorch-Variables-functionals-and-Autograd/
2019/06/17: Update with hyper-parameter tuning script.
2019/06/25: Committed main0.py.
"""
__author__ = 'Ninnart Fuengfusin'
__version__ = '0.0.1'
__email__ = 'ninnart.fuengfusin@yahoo.com'

import os
import time
import logging
import argparse
import torch
import torch.nn as nn
import model
from weight_decay import *
from dataset import load_dataset
from utils import *
from recorder import Recorder
from updater import UpdateMomentum
from namer import namer

parser = argparse.ArgumentParser(description='PyTorch implementation of SharedGradNet.')
parser.add_argument('--epoch', '-e', type=int, default=600, help='Number of training epoch.')
parser.add_argument('--learning_rate', '-lr', type=float, default=3e-1, help='A floating for initial learning rate.')
parser.add_argument('--train_batch', type=int, default=128, help='A integer for train batch amount.')
parser.add_argument('--test_batch', type=int, default=128, help='A integer for test batch amount')
parser.add_argument('--num_neuron', type=int, default=784,
                    help='Number of neurons in fully connected layer for produce codes')
parser.add_argument('--weight_decay', type=float, default=0, help='A floating for weight decay.')
parser.add_argument('--load', type=str2bool, default=False,
                    help='A boolean for loading weights from load_location or not.')
parser.add_argument('--load_location', type=str, default='model1-baseline',
                    help='A string of location for loading weights.')
parser.add_argument('--seed', '-s', type=int, default=0,
                    help='An integer for initialization randomness.')
args = parser.parse_args()

if __name__ == '__main__':
    save_loc = namer(
        f'epoch{args.epoch}', f'lr{args.learning_rate}',
        f'decay{args.weight_decay}', f'seed{args.seed}')

    set_logger(os.path.join(os.getcwd(), save_loc), 'info.log')
    logging.info(__doc__)
    logging.info(args)
    set_printoptions()
    seed_everywhere_torch(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    record = Recorder('test_acc', 'test_acc2', 'test_acc3', 'test_loss', 'test_loss2', 'test_loss3')
    train_loader, test_loader, img_size = load_dataset(
        num_train_batch=args.train_batch, num_test_batch=args.test_batch,
        num_extra_batch=0, num_worker=8, dataset='mnist')

    model1 = model.NetworkWithSub1()
    updaterW1_1 = UpdateMomentum()
    updaterB1_1 = UpdateMomentum()
    model1.to(device)

    BETA = 0.9
    t1 = time.time()
    for i in range(args.epoch):
        # Accumulating variables.
        total_train_loss = 0
        train_correct = 0
        train_total = 0
        total_test_loss = 0
        test_correct = 0
        test_total = 0
        model1.train()
        args.learning_rate = args.learning_rate/3 if i % 200 == 0 and i != 0 else args.learning_rate
        for train_data, train_label in train_loader:
            model1.zero_grad()
            train_data, train_label = train_data.to(device), train_label.to(device)
            train_output = model1.forward(train_data)
            train_loss = nn.CrossEntropyLoss()(
                train_output, train_label) #+ l2_weight_decay(args.weight_decay2, model2.w1)
            train_loss.backward()
            total_train_loss += train_loss.item()
            _, train_predicted = torch.max(train_output.data, 1)
            train_correct += (train_predicted == train_label).sum().item()
            train_total += train_label.data.size(0)
            model1.w1.data = updaterW1_1.update(
                model1.w1.data, BETA, args.learning_rate, model1.w1.grad.data)
            model1.b1.data = updaterB1_1.update(
                model1.b1.data, BETA, args.learning_rate, model1.b1.grad.data)

        logging.info(f'Epoch: {i + 1}')
        logging.info(f'Train Accuracy: {train_correct/train_total}, \nLoss: {total_train_loss/train_total}')

        with torch.no_grad():
            model1.eval()
            for test_data, test_label in test_loader:
                test_data, test_label = test_data.to(device), test_label.to(device)
                test_output = model1.forward(test_data)
                test_loss = nn.CrossEntropyLoss()(test_output, test_label)
                total_test_loss += test_loss.item()
                _, test_predicted = torch.max(test_output.data, 1)
                test_correct += (test_predicted == test_label).sum().item()
                test_total += test_label.data.size(0)

            if record.more_than_highest('test_acc', test_correct/test_total):
                save_model(model1, os.path.join(os.getcwd(), save_loc, 'checkpoint.pth'))
                logging.info(f'Save model')
            t2 = time.time() - t1
            logging.info(f'Test Accuracy: {test_correct/test_total}, \nLoss: {total_test_loss/test_total}')
            record.record('test_acc', test_correct/test_total)
            logging.info(f'Learning rate {args.learning_rate}')
            logging.info(f'Timer: {to_hhmmss(t2)}')
            logging.info(f'=====================================================================================')

    record.save_all(os.path.join(os.getcwd(), save_loc))
    logging.info(f'best test_acc: {record.highest("test_acc")}')
    logging.info(f'model1:w1 = {model1.w1.data}')

    record.plot(
        'test_acc', save=True,
        save_loc=os.path.join(os.getcwd(), save_loc, 'test_acc.png'))
    np.savetxt(
        os.path.join(os.getcwd(), save_loc, f'{record.highest("test_acc")}.txt'),
        record.highest("test_acc"), delimiter=',')
