#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The code implementation of SharedGradNet.
main1.py is for neural networks with a hidden layer.
Some part from: https://jhui.github.io/2018/02/09/PyTorch-Variables-functionals-and-Autograd/
2019/06/17: Update with hyper-parameter tuning scirpt.
2019/06/25: main1.py -> main1.py
"""
__author__ = 'Ninnart Fuengfusin'
__version__ = '0.0.1'
__email__ = 'ninnart.fuengfusin@yahoo.com'

import os
import time
import argparse
import torch
import torch.nn as nn
import model
from dataset import load_dataset
from utils import *
import logging
from recorder import Recorder
from updater import *
from namer import namer
from weight_decay import l2_weight_decay

parser = argparse.ArgumentParser(description='PyTorch implementation of SharedGradNet.')
parser.add_argument('--epoch', '-e', type=int, default=600, help='Number of training epoch.')
parser.add_argument('--learning_rate', '-lr', type=float, default=3e-1, help='A floating for initial learning rate.')
parser.add_argument('--train_batch', type=int, default=128, help='A integer for train batch amount.')
parser.add_argument('--test_batch', type=int, default=128, help='A integer for test batch amount')
parser.add_argument('--num_neuron', type=int, default=784,
                    help='Number of neurons in fully connected layer for produce codes')
parser.add_argument('--weight_decay3', type=float, default=3e-5, help='A floating for weight decay.')
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
        f'decay3{args.weight_decay3}', f'seed{args.seed}')

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

    model3 = model.NetworkWithSub3(args.num_neuron)
    logging.info(model3)
    model3.to(device)
    updaterW3_1 = UpdateMomentum()
    updaterB3_1 = UpdateMomentum()
    updaterW3_2 = UpdateMomentum()
    updaterB3_2 = UpdateMomentum()
    updaterW3_3 = UpdateMomentum()
    updaterB3_3 = UpdateMomentum()

    BETA = 0.9
    t1 = time.time()
    for i in range(args.epoch):
        # Accumulating variables.
        train_total = 0
        total_train_loss3 = 0
        train_correct3 = 0

        test_total = 0
        total_test_loss3 = 0
        test_correct3 = 0
        model3.train()
        args.learning_rate = args.learning_rate/3 if i % 100 == 0 and i != 0 else args.learning_rate
        for train_data, train_label in train_loader:
            model3.zero_grad()
            train_data, train_label = train_data.to(device), train_label.to(device)
            train_output3 = model3.forward(train_data)
            train_loss3 = nn.CrossEntropyLoss()(
                train_output3, train_label) + l2_weight_decay(args.weight_decay3, model=model3)
                #+ args.weight_decay2*(torch.norm(model2.w1) + torch.norm(model2.w2))
            train_loss3.backward()
            total_train_loss3 += train_loss3.item()
            _, train_predicted3 = torch.max(train_output3.data, 1)
            train_correct3 += (train_predicted3 == train_label).sum().item()
            train_total += train_label.data.size(0)
            model3.w1.data = updaterW3_1.update(
                model3.w1.data, BETA, args.learning_rate, model3.w1.grad.data)
            model3.b1.data = updaterB3_1.update(
                model3.b1.data, BETA, args.learning_rate, model3.b1.grad.data)
            model3.w2.data = updaterW3_2.update(
                model3.w2.data, BETA, args.learning_rate, model3.w2.grad.data)
            model3.b2.data = updaterB3_2.update(
                model3.b2.data, BETA, args.learning_rate, model3.b2.grad.data)
            model3.w2.data = updaterW3_2.update(
                model3.w2.data, BETA, args.learning_rate, model3.w2.grad.data)
            model3.b2.data = updaterB3_2.update(
                model3.b2.data, BETA, args.learning_rate, model3.b2.grad.data)

        logging.info(f'Epoch: {i + 1}')
        logging.info(f'Train Accuracy3: {train_correct3/train_total}, \nLoss: {total_train_loss3/train_total}')
        logging.info(f'=====================================================================================')

        with torch.no_grad():
            model3.eval()
            for test_data, test_label in test_loader:
                test_data, test_label = test_data.to(device), test_label.to(device)
                test_output3 = model3.forward(test_data)
                test_loss3 = nn.CrossEntropyLoss()(test_output3, test_label)
                total_test_loss3 += test_loss3.item()
                _, test_predicted3 = torch.max(test_output3.data, 1)
                test_correct3 += (test_predicted3 == test_label).sum().item()
                test_total += test_label.data.size(0)

            if record.more_than_highest('test_acc2', test_correct3/test_total):
                # Record and check the changing in the best test accuracy.
                save_model(model3, os.path.join(os.path.join(os.getcwd(), save_loc), 'checkpoint.pth'))
                logging.info(f'Save model')
            t2 = time.time() - t1
            logging.info(f'Test Accuracy: {test_correct3/test_total}, \nLoss: {total_test_loss3/test_total}')
            record.record('test_acc3', test_correct3/test_total)
            logging.info(f'Learning rate {args.learning_rate}')
            logging.info(f'Timer: {to_hhmmss(t2)}')
            logging.info(f'=====================================================================================')

    record.save_all(os.path.join(os.path.join(os.getcwd(), save_loc)))
    logging.info(f'best test_acc3: {record.highest("test_acc3")}')
    logging.info(f'model3:w1 = {model3.w1.data}')
    logging.info(f'model3:w2 = {model3.w2.data}')
    logging.info(f'model3:w2 = {model3.w3.data}')
    record.plot('test_acc3', save=True, save_loc=os.path.join(os.path.join(os.getcwd(), save_loc), 'test_acc3.png'))
    np.savetxt(
        os.path.join(os.getcwd(), save_loc, f'{record.highest("test_acc3")}.txt'),
        record.highest("test_acc3"), delimiter=',')
