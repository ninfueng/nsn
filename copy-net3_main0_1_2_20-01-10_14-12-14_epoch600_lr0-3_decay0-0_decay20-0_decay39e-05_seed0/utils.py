#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Update Log:
2019/06/21: Put pickle_load, pickle_save.
2019/06/25: Adding get_filename
"""
__author__ = 'Ninnart Fuengfusin'
__version__ = '0.0.1'
__email__ = 'ninnart.fuengfusin@yahoo.com'

import argparse
import random
import os
import time
import logging
import re
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pickle

def str2bool(v):
    """From: https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    :param v: A string inputted from argparse.
    :return: None
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def to_hhmmss(ss):
    """transform time.time() into hour:minute:second format.
    :param ss: A floating second from time.time()
    :return: A string indicates hour:minute:second.
    """
    assert type(ss) == float
    return time.strftime('%H:%M:%S', time.gmtime(ss))


def timer_wrapper(func):
    """Wrapper of function to printing out the running of the wrapped function.
    Note: to_hhmmss is required.
    :param func: function that wanted to wrap with.
    :return wrapped_func: wrapped function.
    """
    def wrapped_func(*args, **kwargs):
        t1 = time.time()
        res = func(*args, **kwargs)
        t2 = time.time() - t1
        logging.info(
            'Wrapper of {}: Run with timer: {}'.format(func.__name__, to_hhmmss(t2)))
        logging.info(
            '=========================================================================')
        return res
    return wrapped_func


def seed_everywhere_torch(seed):
    """Initialize a random seed in every possible places.
    From: https://github.com/pytorch/pytorch/issues/11278
    :param seed: an integer for setting an initialization.
    :return: None
    """
    assert type(seed) == int
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.tensor(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    logging.info('Plant the random seed: {}.'.format(seed))

def xavier_init(m):
    """
    From: https://github.com/pytorch/examples/blob/master/dcgan/main.py#L90-L96
    :param m:
    :return: None
    """
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        # Detect linear layer, if found, apply xavier.

        #m.weight
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def rotate_list(alist):
    """Pop the last element of list out then put it into the first of list.
    :param alist: A list that want to rotated.
    :return alist: A rotated list.
    """
    assert type(alist) == list
    last = alist.pop()
    alist.insert(0, last)
    return alist


def dist_plot(array, bins):
    """Distribution
    :param array: A dimensional array
    :return: None
    """
    assert type(bins) == int
    plt.hist(array, bins=bins)
    plt.title('Distribution')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.show()


@timer_wrapper
def train(model, train_loader, optimizer, device, epoch):
    """Generally train the model for a epoch.
    :param model:
    :param train_loader:
    :param optimizer:
    :param device:
    :param epoch:
    :return:
    """
    assert type(epoch) == int

    total_train_loss = 0
    train_correct = 0
    train_total = 0
    model.train()
    for train_data, train_label in train_loader:
        train_data, train_label = train_data.to(device), train_label.to(device)
        optimizer.zero_grad()
        train_output = model.forward(train_data)
        train_loss = nn.CrossEntropyLoss()(train_output, train_label)
        train_loss.backward()
        optimizer.step()
        # Accumulate loss, train_loss.data is torch format.
        total_train_loss += train_loss.item()
        # Reduce one-hot of softmax into a label per batch.
        _, train_predicted = torch.max(train_output.data, 1)
        # Correct prediction accumulates.
        train_correct += (train_predicted == train_label).sum().item()
        # Find the total number of object.
        train_total += train_label.size(0)
    observed_lr = get_lr(optimizer)
    logging.info('Epoch: {}, Train Accuracy: {}, lr: {} \nLoss: {}'.format(
        epoch + 1, train_correct/train_total, observed_lr, total_train_loss/train_total))
    return train_correct/train_total, total_train_loss/train_total

@timer_wrapper
def train_freeze(model, train_loader, optimizer, device, epoch):
    """Generally train the model for a epoch.
    :param model:
    :param train_loader:
    :param optimizer:tr
    :param device:
    :param epoch:
    :return:
    """
    assert type(epoch) == int

    total_train_loss = 0
    train_correct = 0
    train_total = 0
    model.train()
    for i in range(model.num_layer + 1):
        for train_data, train_label in train_loader:
            train_data, train_label = train_data.to(device), train_label.to(device)

            # TODO
            optimizer.zero_grad()
            train_output = model.forward(train_data)
            train_loss = nn.CrossEntropyLoss()(train_output, train_label)
            train_loss.backward()
            optimizer.step()
            # Accumulate loss, train_loss.data is torch format.
            total_train_loss += train_loss.item()
            # Reduce one-hot of softmax into a label per batch.
            _, train_predicted = torch.max(train_output.data, 1)
            # Correct prediction accumulates.
            train_correct += (train_predicted == train_label).sum().item()
            # Find the total number of object.
            train_total += train_label.data.size(0)
        observed_lr = get_lr(optimizer)
        logging.info('Epoch: {}, Train Accuracy: {}, lr: {} \nLoss: {}'.format(
            epoch + 1, train_correct/train_total, observed_lr, total_train_loss/train_total))
    return train_correct/train_total, total_train_loss/train_total

@timer_wrapper
def test(model, test_loader, device):
    """ Testing all possible sub-model.
    :param test_shuffle:
    :param model:
    :param test_loader:
    :param device: torch.device('cuda' or 'cpu').
    :return test_correct/test_total: A floating test accuracy.
    """
    model.eval()
    with torch.no_grad():
        # num_drop in range of [0, num_layer] hence, for 0 to num_layer + 1.
        total_test_loss = 0
        test_correct = 0
        test_total = 0
        for test_data, test_label in test_loader:
            test_data, test_label = test_data.to(device), test_label.to(device)
            #test_output = model.forward_randomly_drop(test_data, num_drop=i, shuffle=test_shuffle)
            test_output = model.forward(test_data)
            test_loss = nn.CrossEntropyLoss()(test_output, test_label)
            # Accumulate loss.
            total_test_loss += test_loss.item()
            # Reduce one-hot of softmax into a label per batch.
            _, test_predicted = torch.max(test_output.data, 1)
            # Correct prediction accumulates.
            test_correct += (test_predicted == test_label).sum().item()
            # Find the total number of object.
            test_total += test_label.data.size(0)
        logging.info('Test Accuracy: {},  Loss: {} '.format(
            test_correct/test_total, total_test_loss/test_total))
        # (num_hidden_layer+1) dimensions.
        return test_correct/test_total, total_test_loss/test_total


def set_logger(log_path, log_name):
    """From: https://github.com/cs230-stanford/cs230-code-examples/blob/master/pytorch/vision/utils.py
    Set the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    Note: using set_logger before using logging.info(string),
          set_logger is for indicate where logging text should go.
    """
    if not os.path.exists(log_path) and log_path is not '':
        # Create new dictionary if it was not existed and not the current folder.
        os.mkdir(log_path)
    log_path = os.path.join(log_path, log_name)
    if os.path.isfile(log_path):
        # Remove the file if old file was existed.
        os.remove(log_path)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # If Handler was not added before then go to this if.
        # Logging to a file.
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(message)s'))
        logger.addHandler(file_handler)
        # Logging to console.
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)

# class RecordBestAcc:
#     """Using class to keep a object, best_acc, inside.
#     Ex:
#         record_acc = RecordBestAcc()
#         for i in [0.1, 0.5, 0.3]:
#             best_acc = record_acc(i)
#             print(best_acc)
#             # 0.1
#             # 0.5
#             # 0.5
#     """
#     def __init__(self):
#         self.best_acc = None
#         self.change = False
#
#     def __call__(self, acc):
#         """Comparing best_acc with an input acc.
#         Assign best_acc = acc when best_acc < acc.
#         :param acc: A floating acc of the testing dataset.
#         :return best_acc: A floating acc.
#         :return change: A boolean indicates
#         """
#         if self.best_acc is None:
#             # Detect this one first.
#             self.best_acc = acc
#         elif self.best_acc < acc:
#             # If the upper one is not detected, using this one.
#             self.best_acc = acc
#             self.change = True
#         else:
#             self.change = False
#         return self.best_acc, self.change


def save_model(model, path):
    """From: https://pytorch.org/tutorials/beginner/saving_loading_models.html
    Save the model including all of parameters.
    :param model: torch.nn.Module model for saving.
    :param path: A string for saving location and name of file.
    :return: None
    """
    assert type(path) == str
    torch.save(model.state_dict(), path)


def load_model_or_initial_weights(model, path):
    """From: https://pytorch.org/tutorials/beginner/saving_loading_models.html
    Load the model or if the file is not exist in the path then initial the weight.
    :param path: A string for loading location and name of file.
    :return model: The loaded model
    """
    if os.path.isfile('checkpoint.pth'):
        model.load_state_dict(torch.load(path))
        logging.info('Load the model from this location: @{}.'.format(path))
    else:
        model.apply(xavier_init)


def switch_to_SGD(epoch_count, change_epoch, lr, model, optimizer, scheduler, **kwargs):
    """Changing from one optimizer into other optimizer with given epoch.
    :param epoch_count: A integer counter of epoch.
    :param change_epoch: A integer. If the epoch_count equal to change_epoch then,
           return SGD optimizer.
    :param model: torch.Module for using model.parameters.
    :param lr: A floating learning rate to pass thought.
    :param kwargs: EX: {'momentum': 0, 'weight_decay': 0, 'nesterov': False} for common SGD.
    :param optimizer: For pass though, if the change_epoch is not passed.
    :return optimizer:
    """
    assert type(epoch_count) == int
    assert type(change_epoch) == int
    assert type(lr) == float

    if epoch_count > change_epoch and epoch_count != 0:
        optimizer = optim.SGD(
            model.parameters(), lr=lr, momentum=kwargs['momentum'],
            nesterov=kwargs['nesterov'], weight_decay=kwargs['weight_decay'])
    elif epoch_count == change_epoch and epoch_count != 0:
        logging.info('Switch to SGD with momentum @epoch {}'.format(epoch_count))
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', factor=kwargs['factor'], patience=kwargs['patience'])
    return optimizer, scheduler


# TODO
# class ChangeToSGD():
#     def __init__(self, change_epoch, lr, model, optimizer, scheduler, **kwargs):
#
#
#
#         self.change_epoch = change_epoch
#
#
#     def __call__(self, ):
#         if self.
#         return




def save_np(path, name, vari):
    """Save a variable in the format of numpy.
    :param path: A string location for saving.
    :param name: A string of saved name of file.
    :param vari: A variable for saving.
    :return: None
    """
    assert type(path) == str
    assert type(name) == str
    if not os.path.isdir(path):
        os.mkdir(path)
    np.save(os.path.join(path, name), vari)


def plot_hist(
        vari, name_x='', name_y='', title='', label='',
        loc='upper left', bins=10, conti=False):
    """Plotting histogram of a variable.
    Ex: Plot for a histogram but using two type of graph.
    plot_hist(vari=[0, 1, 2, 3], name_x='x', name_y='y', title='title',
              label='one', loc='upper left', bins=10, conti=True)
    plot_hist(vari=[4, 5, 6, 7], name_x='x', name_y='y', title='title',
              label='two', loc='upper left', bins=10, conti=False)

    :param vari: A variable for plotting
    :param name_x: A string for x axis name.
    :param name_y: A string for y axis name.
    :param title: A string for the title of histogram.
    :param label: A label for using with the legend.
    :param loc: A string for locate the position of legend.
    :param bins: A integer for resolution?
    :param conti: A boolean for using plt.show() or not.
    :return: None
    """
    assert type(name_x) == str
    assert type(name_y) == str
    assert type(title) == str
    assert type(bins) == int
    assert type(label) == str
    plt.hist(vari, bins=bins, label=label)
    if not conti:
        plt.xlabel(name_x)
        plt.ylabel(name_y)
        plt.title(title)
        plt.legend(loc=loc)
        plt.grid(True)
        plt.show()
    elif conti:
        pass
    else:
        raise NotImplementedError(
            'conti is a boolean, but your input is {}.'.format(conti))


def plot_graph(
        x, y, name_x='', name_y='', title='', label='',
        loc='upper left', conti=False):
    """Plotting histogram of a variable.
    Ex: Plot for a graph but using two type of graph.
    plot_graph([1, 2, 3, 4], [1, 4, 9, 16], label='one', conti=True)
    plot_graph([5, 6, 7, 8], [1, 16, 81, 189],
     name_x='x', name_y='y', title='title', label='two', conti=False)
    :param vari: A variable for plotting
    :param name_x: A string for x axis name.
    :param name_y: A string for y axis name.
    :param title: A string for the title of histogram.
    :param label: A label for using with the legend.
    :param loc: A string for locate the position of legend.
    :param conti: A boolean for using plt.show() or not.
    :return: None
    """
    assert type(name_x) == str
    assert type(name_y) == str
    assert type(title) == str
    assert type(label) == str
    plt.plot(x, y, label=label)
    if not conti:
        plt.xlabel(name_x)
        plt.ylabel(name_y)
        plt.title(title)
        plt.legend(loc=loc)
        plt.grid(True)
        plt.show()
    elif conti:
        pass
    else:
        raise NotImplementedError(
            'conti is a boolean, but your input is {}.'.format(conti))


def get_lr(optimizer):
    """From: https://stackoverflow.com/questions/52660985/pytorch-how-to-get-learning-rate-during-training
    :param optimizer: torch.optim.
    :return param_group['lr']: A floating point of learning rate in optimizer.
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']


def eliminate_loaded_param(loaded_state, eliminate):
    """Remove all parameters that have a the name as part of string in eliminate list.
    :param loaded_state: An OrderDict after loaded parameters with torch.load()
    :param eliminate: A list of string that used to detect parameters for removing.
    :return loaded_state: An OrderDict after freeze and eliminated.
    """
    assert type(eliminate) == list
    eliminate_list = []
    kept_list = []
    for key, value in loaded_state.items():
        if any(i in key for i in eliminate):
            # Cannot remove key during this for loop.
            eliminate_list.append(key)
        else:
            kept_list.append(key)
            logging.info('{} was kept.'.format(key))
    for i in eliminate_list:
        del loaded_state[i]
        logging.info('{} was removed.'.format(i))
    logging.info('########################################################')
    return loaded_state, kept_list


def eliminate_except_param(loaded_param, not_eliminate):
    """
    Ex:
        loaded2 = torch.load(os.path.join('model2-baseline', 'checkpoint.pth'))
        loaded2 = eliminate_except_param(loaded2, ['net.1.weight', 'net.1.bias'])
        loaded.update(loaded2)
    :param loaded_param: An OrderedDict.
    :param not_eliminate: A list of key.
    :return adict: OrderedDict with keys in not_eliminate and their values.

    """
    assert type(not_eliminate) == list
    keep_key = []
    keep_value = []
    for key, value in loaded_param.items():
        if key in not_eliminate:
            keep_key.append(key)
            keep_value.append(value)

    adict = OrderedDict()
    for i, j in zip(keep_key, keep_value):
        adict[i] = j
    return adict



def extract_txt(string, expression=r'^[a-zA-Z0-9_]+'):
    """
    :param string:
    :param expression:
    :return:
    """
    pattern = re.compile(expression)
    match = pattern.findall(string)
    return match


def extract_name(alist, expression=r'^[a-zA-Z0-9_]+'):
    """
    :param alist:
    :param expression:
    :return:
    """
    pattern = re.compile(expression)
    match_list = []
    for i in alist:
        match_list.append(pattern.findall(i).pop())
    return match_list


def remove_duplicate_in_list(alist):
    """Remove replacements on the list.
    Ex: [1, 1, 2, 2, 3, 3] => [1, 2, 3]
    :param alist: A list that contains the replacements.
    :return alist: A list without replacements.
    """
    return list(set(alist))


def freeze_param_given_name(model, freeze_list):
    """
    :param freeze_list:
    :param model:
    :return:
    """
    for name, param in model.named_parameters():
        if name in freeze_list:
            param.requires_grad = False
            logging.info('{} was freeze.'.format(name))
        else:
            logging.info('{} was not freeze'.format(name))
    logging.info('########################################################')


def swap_param(adict, akey, anew_key):
    """Swap akey in adict into anew_key with the anew_key contains value of akey.
    :param adict: A dictionary.
    :param akey: A string the key in adict that wanted to change.
    :param anew_key: A string that is used for changing akey.
    :return adict: A key changed dictionary.
    """
    keep_value = adict[akey]
    del adict[akey]
    adict[anew_key] = keep_value
    return adict


def pickle_load(namefile: str):
    """Load Python variable, given name of file.
    :param namefile: A string of file to load.
    :return output: A loaded variable.
    """
    with open(namefile, 'rb') as load_file:
        output = pickle.load(load_file)
    return output


def pickle_save(var, namefile: str):
    """Save python variable, into namefile.pkl format.
    :param var: A Python variable to save.
    :param namefile: A string of saving file.
    :return None:
    """
    with open(namefile, 'wb') as save_file:
        pickle.dump(var, save_file)
    print(f'{var} was loaded to {namefile}.')


def set_printoptions():
    """Default for Ninnart's print setting.
    """
    torch.set_printoptions(precision=6, threshold=2_000)
    np.set_printoptions(precision=6, threshold=2_000)
