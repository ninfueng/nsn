#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Network with Sub-Networks.
Some part from: https://jhui.github.io/2018/02/09/PyTorch-Variables-functionals-and-Autograd/
"""
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
from weight_decay import *

parser = argparse.ArgumentParser(description='Network with Sub-Networks.')
parser.add_argument('--epoch', '-e', type=int, default=600, help='Number of training epoch.')
parser.add_argument('--learning_rate', '-lr', type=float, default=3e-1, help='A floating for initial learning rate.')
parser.add_argument('--train_batch', type=int, default=128, help='A integer for train batch amount.')
parser.add_argument('--test_batch', type=int, default=128, help='A integer for test batch amount')
parser.add_argument('--num_neuron', type=int, default=784,
                    help='Number of neurons in fully connected layer for produce codes')
parser.add_argument('--epoch_unlearn', type=int, default=400,
                    help='Number of epoch for changing to general learning')
parser.add_argument('--weight_decay', type=float, default=0e-0, help='A floating for weight decay0.')
parser.add_argument('--weight_decay2', type=float, default=0e-0, help='A floating for weight decay2.')
parser.add_argument('--weight_decay3', type=float, default=(9e-5), help='A floating for weight decay3.')
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
        f'decay{args.weight_decay}', f'decay2{args.weight_decay2}', 
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

    model1 = model.NetworkWithSub1()
    model2 = model.NetworkWithSub2(args.num_neuron)
    model3 = model.NetworkWithSub3(args.num_neuron)
    logging.info(model1)
    logging.info(model2)
    logging.info(model3)
    model1.to(device)
    model2.to(device)
    model3.to(device)

    updaterW1_1 = UpdateMomentum()
    updaterB1_1 = UpdateMomentum()
    updaterW2_1 = UpdateMomentum()
    updaterB2_1 = UpdateMomentum()
    updaterW2_2 = UpdateMomentum()
    updaterB2_2 = UpdateMomentum()
    updaterW3_1 = UpdateMomentum()
    updaterB3_1 = UpdateMomentum()
    updaterW3_2 = UpdateMomentum()
    updaterB3_2 = UpdateMomentum()
    updaterW3_3 = UpdateMomentum()
    updaterB3_3 = UpdateMomentum()

    BETA = 0.9
    # ETA, for how much smaller model's gradient is applied with bigger model.
    t1 = time.time()
    for i in range(args.epoch):
        # Accumulating variables.
        total_train_loss = 0
        train_correct = 0
        train_total = 0
        total_train_loss2 = 0
        train_correct2 = 0
        train_total2 = 0
        total_train_loss3 = 0
        train_correct3 = 0
        train_total3 = 0

        total_test_loss = 0
        test_correct = 0
        test_total = 0
        total_test_loss2 = 0
        test_correct2 = 0
        test_total2 = 0
        total_test_loss3 = 0
        test_correct3 = 0
        test_total3 = 0
        model1.train()
        model2.train()
        model3.train()
        args.learning_rate = args.learning_rate/3 if i % 200 == 0 and i != 0 else args.learning_rate
        for train_data, train_label in train_loader:
            # if i >= args.epoch_unlearn:
            #     ETA = 1.0
            #     model1.w1.data = model3.w3.data.clone()
            #     model1.b1.data = model3.b3.data.clone()
            #     model2.w1.data = model3.w2.data.clone()
            #     model2.b1.data = model3.b2.data.clone()
            #     model2.w2.data = model1.w1.data.clone()
            #     model2.b2.data = model1.b1.data.clone()
            # elif i < args.epoch_unlearn:
            #     ETA = 1.0
            #     model2.w2.data = model1.w1.data.clone()
            #     model2.b2.data = model1.b1.data.clone()
            #     model3.w2.data = model2.w1.data.clone()
            #     model3.b2.data = model2.b1.data.clone()
            #     model3.w3.data = model1.w1.data.clone()
            #     model3.b3.data = model1.b1.data.clone()
            # else:
            #     raise NotImplementedError(f'args.epoch_unlearn is weird?: {args.epoch_unlearn}')

            ETA = 1.0
            model1.zero_grad()
            model2.zero_grad()
            model3.zero_grad()
            model2.w2.data = model1.w1.data.clone()
            model2.b2.data = model1.b1.data.clone()
            model3.w2.data = model2.w1.data.clone()
            model3.b2.data = model2.b1.data.clone()
            model3.w3.data = model1.w1.data.clone()
            model3.b3.data = model1.b1.data.clone()

            train_data, train_label = train_data.to(device), train_label.to(device)
            train_output = model1.forward(train_data)
            train_loss = nn.CrossEntropyLoss()(train_output, train_label)
            train_loss.backward()
            total_train_loss += train_loss.item()
            _, train_predicted = torch.max(train_output.data, 1)
            train_correct += (train_predicted == train_label).sum().item()
            train_total += train_label.data.size(0)

            train_output2 = model2.forward(train_data)
            train_loss2 = nn.CrossEntropyLoss()(
                train_output2, train_label) #+ l2_weight_decay(args.weight_decay2, model=model2)
            train_loss2.backward()
            total_train_loss2 += train_loss2.item()
            _, train_predicted2 = torch.max(train_output2.data, 1)
            train_correct2 += (train_predicted2 == train_label).sum().item()

            train_output3 = model3.forward(train_data)
            train_loss3 = nn.CrossEntropyLoss()(
                train_output3, train_label) + l2_weight_decay(args.weight_decay3, model=model3)
            train_loss3.backward()
            total_train_loss3 += train_loss3.item()
            _, train_predicted3 = torch.max(train_output3.data, 1)
            train_correct3 += (train_predicted3 == train_label).sum().item()

            model1.w1.data = updaterW1_1.update(
                model1.w1.data, BETA, args.learning_rate, ETA*model1.w1.grad.data, model2.w2.grad.data)
            model1.b1.data = updaterB1_1.update(
                model1.b1.data, BETA, args.learning_rate, ETA*model1.b1.grad.data, model2.b2.grad.data)
            model2.w1.data = updaterW2_1.update(
                model2.w1.data, BETA, args.learning_rate, ETA*model2.w1.grad.data, model3.w2.grad.data)
            model2.b1.data = updaterB2_1.update(
                model2.b1.data, BETA, args.learning_rate, ETA*model2.b1.grad.data, model3.b2.grad.data)

            # if i >= args.epoch_unlearn:
            #     model2.w2.data = updaterW1_1.update(
            #         model2.w2.data, BETA, args.learning_rate, ETA*model1.w1.grad.data, model2.w2.grad.data)
            #     model2.b2.data = updaterB1_1.update(
            #         model2.b2.data, BETA, args.learning_rate, ETA*model1.b1.grad.data, model2.b2.grad.data)
            #     model3.w2.data = updaterW2_1.update(
            #         model3.w2.data, BETA, args.learning_rate, ETA*model2.w1.grad.data, model3.w2.grad.data)
            #     model3.b2.data = updaterB2_1.update(
            #         model3.b2.data, BETA, args.learning_rate, ETA*model2.b1.grad.data, model3.b2.grad.data)
            # elif i < args.epoch_unlearn:
            #     model1.w1.data = updaterW1_1.update(
            #         model1.w1.data, BETA, args.learning_rate, ETA*model1.w1.grad.data, model2.w2.grad.data)
            #     model1.b1.data = updaterB1_1.update(
            #         model1.b1.data, BETA, args.learning_rate, ETA*model1.b1.grad.data, model2.b2.grad.data)
            #     model2.w1.data = updaterW2_1.update(
            #         model2.w1.data, BETA, args.learning_rate, ETA*model2.w1.grad.data, model3.w2.grad.data)
            #     model2.b1.data = updaterB2_1.update(
            #         model2.b1.data, BETA, args.learning_rate, ETA*model2.b1.grad.data, model3.b2.grad.data)
            # else:
            #     raise NotImplementedError(f'args.epoch_unlearn is weird?: {args.epoch_unlearn}')

            model3.w1.data = updaterW3_1.update(
                model3.w1.data, BETA, args.learning_rate, model3.w1.grad.data)
            model3.b1.data = updaterB3_1.update(
                model3.b1.data, BETA, args.learning_rate, model3.b1.grad.data)

        logging.info(f'Epoch: {i + 1}')
        logging.info(f'Train Accuracy: {train_correct/train_total}, \nLoss: {total_train_loss/train_total}')
        logging.info(f'Train Accuracy2: {train_correct2/train_total}, \nLoss: {total_train_loss2/train_total}')
        logging.info(f'Train Accuracy3: {train_correct3/train_total}, \nLoss: {total_train_loss3/train_total}')
        logging.info(f'=====================================================================================')

        with torch.no_grad():
            model1.eval()
            model2.eval()
            model3.eval()
            for test_data, test_label in test_loader:
                test_data, test_label = test_data.to(device), test_label.to(device)
                test_output = model1.forward(test_data)
                test_loss = nn.CrossEntropyLoss()(test_output, test_label)

                total_test_loss += test_loss.item()
                _, test_predicted = torch.max(test_output.data, 1)
                test_correct += (test_predicted == test_label).sum().item()
                test_total += test_label.data.size(0)

                test_output2 = model2.forward(test_data)
                test_loss2 = nn.CrossEntropyLoss()(test_output2, test_label)
                total_test_loss2 += test_loss2.item()
                _, test_predicted2 = torch.max(test_output2.data, 1)
                test_correct2 += (test_predicted2 == test_label).sum().item()

                test_output3 = model3.forward(test_data)
                test_loss3 = nn.CrossEntropyLoss()(test_output3, test_label)
                total_test_loss3 += test_loss3.item()
                _, test_predicted3 = torch.max(test_output3.data, 1)
                test_correct3 += (test_predicted3 == test_label).sum().item()

            if record.more_than_highest('test_acc', test_correct/test_total):
                # Record and check the changing in the best test accuracy.
                save_model(model1, os.path.join(os.path.join(os.getcwd(), save_loc), 'checkpoint.pth'))
                logging.info(f'Save model')

            if record.more_than_highest('test_acc2', test_correct2/test_total):
                # Record and check the changing in the best test accuracy.
                save_model(model2, os.path.join(os.path.join(os.getcwd(), save_loc), 'checkpoint2.pth'))
                logging.info(f'Save model2')

            if record.more_than_highest('test_acc3', test_correct3/test_total):
                # Record and check the changing in the best test accuracy.
                save_model(model3, os.path.join(os.path.join(os.getcwd(), save_loc), 'checkpoint3.pth'))
                logging.info(f'Save model3')

            logging.info(f'Test Accuracy: {test_correct/test_total}, \nLoss: {total_test_loss/test_total}')
            record.record('test_acc', test_correct/test_total)
            logging.info(f'Test Accuracy2: {test_correct2/test_total}, \nLoss: {total_test_loss2/test_total}')
            record.record('test_acc2', test_correct2/test_total)
            logging.info(f'Test Accuracy3: {test_correct3/test_total}, \nLoss: {total_test_loss3/test_total}')
            record.record('test_acc3', test_correct3/test_total)
            logging.info(f'Learning rate {args.learning_rate}')
            t2 = time.time() - t1
            logging.info(f'Timer: {to_hhmmss(t2)}')
            logging.info(f'=====================================================================================')

    record.save_all(os.path.join(os.getcwd(), save_loc))
    logging.info(f'best test_acc: {record.highest("test_acc")}')
    logging.info(f'best test_acc2: {record.highest("test_acc2")}')
    logging.info(f'best test_acc3: {record.highest("test_acc3")}')

    logging.info(f'model1:w1 = {model1.w1.data}')
    logging.info(f'model2:w1 = {model2.w1.data}')
    logging.info(f'model2:w2 = {model2.w2.data}')
    logging.info(f'model3:w1 = {model3.w1.data}')
    logging.info(f'model3:w2 = {model3.w2.data}')
    logging.info(f'model3:w3 = {model3.w3.data}')

    record.plot('test_acc', save=True, save_loc=os.path.join(os.getcwd(), save_loc, 'test_acc.png'))
    record.plot('test_acc2', save=True, save_loc=os.path.join(os.getcwd(), save_loc, 'test_acc2.png'))
    record.plot('test_acc3', save=True, save_loc=os.path.join(os.getcwd(), save_loc, 'test_acc3.png'))
    np.savetxt(
        os.path.join(os.getcwd(), save_loc, f'{record.highest("test_acc3")}.txt'),
        record.highest("test_acc3"), delimiter=',')
