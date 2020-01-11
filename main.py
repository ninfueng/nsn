#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Network with Sub-Networks.
Network as densely connected neural network (NN) with 2 hidden layers.
Sub-Networks as a hidden NN and softmax regression.
"""
import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import model
from dataset import load_dataset
from recorder import Recorder
from utils import seed_everywhere_torch
from updater import UpdateMomentum
from weight_decay import l2_weight_decay
parser = argparse.ArgumentParser(
    description='Network with Sub-Networks.')
parser.add_argument(
    '--epoch', '-e', type=int, default=600+1)
parser.add_argument(
    '--learning_rate', '-lr', type=float, default=3e-1)
parser.add_argument(
    '--train_batch', type=int, default=128)
parser.add_argument(
    '--test_batch', type=int, default=128)
parser.add_argument(
    '--num_neuron', type=int, default=784)
parser.add_argument(
    '--weight_decay', type=float, default=0e-0)
parser.add_argument(
    '--weight_decay2', type=float, default=0e-0)
parser.add_argument(
    '--weight_decay3', type=float, default=9e-5/2)
parser.add_argument(
    '--save_locat', type=str, default='save')
parser.add_argument(
    '--seed', '-s', type=int, default=0)
args = parser.parse_args()

if __name__ == '__main__':
    save_loc = args.save_locat
    record = Recorder(
        'test_acc', 'test_acc2', 'test_acc3', 
        'test_loss', 'test_loss2', 'test_loss3')
    print(__doc__)
    print(args)
    seed_everywhere_torch(args.seed)
    #torch.manual_seed(args.seed)
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader, img_size = load_dataset(
        num_train_batch=args.train_batch, num_test_batch=args.test_batch,
        num_extra_batch=0, num_worker=8, dataset='mnist')

    model1 = model.NetworkWithSub1()
    model2 = model.NetworkWithSub2(args.num_neuron)
    model3 = model.NetworkWithSub3(args.num_neuron)
    print(model1)
    print(model2)
    print(model3)
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
    
    if not os.path.isdir(os.path.join(os.getcwd(), save_loc)):
        os.mkdir(os.path.join(os.getcwd(), save_loc))

    t1 = time.time()
    for i in range(args.epoch):
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
                train_output3, train_label) + l2_weight_decay(
                    args.weight_decay3, model=model3)
            train_loss3.backward()
            total_train_loss3 += train_loss3.item()
            _, train_predicted3 = torch.max(train_output3.data, 1)
            train_correct3 += (train_predicted3 == train_label).sum().item()

            BETA = 0.9
            model1.w1.data = updaterW1_1.update(
                model1.w1.data, BETA, args.learning_rate, 
                model1.w1.grad.data, model2.w2.grad.data)
            model1.b1.data = updaterB1_1.update(
                model1.b1.data, BETA, args.learning_rate, 
                model1.b1.grad.data, model2.b2.grad.data)
            model2.w1.data = updaterW2_1.update(
                model2.w1.data, BETA, args.learning_rate, 
                model2.w1.grad.data, model3.w2.grad.data)
            model2.b1.data = updaterB2_1.update(
                model2.b1.data, BETA, args.learning_rate, 
                model2.b1.grad.data, model3.b2.grad.data)
            model3.w1.data = updaterW3_1.update(
                model3.w1.data, BETA, args.learning_rate, 
                model3.w1.grad.data)
            model3.b1.data = updaterB3_1.update(
                model3.b1.data, BETA, args.learning_rate, 
                model3.b1.grad.data)

        print(f'Epoch: {i + 1}')
        print(
            f'Train Accuracy: {train_correct/train_total},'
            + f'\nLoss: {total_train_loss/train_total}')
        print(
            f'Train Accuracy2: {train_correct2/train_total},'
            + f' \nLoss: {total_train_loss2/train_total}')
        print(
            f'Train Accuracy3: {train_correct3/train_total},'
            + f'\nLoss: {total_train_loss3/train_total}')
        print(
            f'========================================'
            + f'==========================================')

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
                torch.save(model1.state_dict(), os.path.join(
                    os.path.join(os.getcwd(), save_loc), 'checkpoint.pth'))

            if record.more_than_highest('test_acc2', test_correct2/test_total):
                torch.save(model2.state_dict(), os.path.join(
                    os.path.join(os.getcwd(), save_loc), 'checkpoint2.pth'))

            if record.more_than_highest('test_acc3', test_correct3/test_total):
                torch.save(model3.state_dict(), os.path.join(
                    os.path.join(os.getcwd(), save_loc), 'checkpoint3.pth'))

            print(
                f'Test Accuracy: {test_correct/test_total},' 
                + f'\nLoss: {total_test_loss/test_total}')
            record.record(
                'test_acc', test_correct/test_total)
            print(
                f'Test Accuracy2: {test_correct2/test_total},'
                + f'\nLoss: {total_test_loss2/test_total}')
            record.record('test_acc2', test_correct2/test_total)
            print(
                f'Test Accuracy3: {test_correct3/test_total},'
                + f' \nLoss: {total_test_loss3/test_total}')
            record.record('test_acc3', test_correct3/test_total)
            print(f'Learning rate {args.learning_rate}')
            t2 = time.time() - t1
            print(f'Timer: {time.strftime("%H:%M:%S", time.gmtime(t2))}')
            print(
                f'=========================================='
                + f'===========================================')

    record.save_all(os.path.join(os.getcwd(), save_loc))
    print(f'best test_acc: {record.highest("test_acc")}')
    print(f'best test_acc2: {record.highest("test_acc2")}')
    print(f'best test_acc3: {record.highest("test_acc3")}')
    print(f'model1:w1 = {model1.w1.data}')
    print(f'model2:w1 = {model2.w1.data}')
    print(f'model2:w2 = {model2.w2.data}')
    print(f'model3:w1 = {model3.w1.data}')
    print(f'model3:w2 = {model3.w2.data}')
    print(f'model3:w3 = {model3.w3.data}')
    record.plot(
        'test_acc', save=True, 
        save_loc=os.path.join(
            os.getcwd(), save_loc, 'test_acc.png'))
    record.plot(
        'test_acc2', save=True, 
        save_loc=os.path.join(
            os.getcwd(), save_loc, 'test_acc2.png'))
    record.plot(
        'test_acc3', save=True, 
        save_loc=os.path.join(
            os.getcwd(), save_loc, 'test_acc3.png'))
    np.savetxt(
        os.path.join(
            os.getcwd(), save_loc,
            f'{record.highest("test_acc3")}.txt'), 
        record.highest("test_acc3"), delimiter=',')
