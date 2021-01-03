#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Recorder for saving and tracking variables during training.
2019/06/13: Update record_check_update_highest for using with save NNs model.
          : Adding typing hint that works only for python3.5.
2019/06/16: Add option to save the plotting.
2019/06/17: Checking the saving plotting and extend to hist.
2019/06/19: Updating record_check_update_highest -> more_than_highest
"""

__author__ = 'Ninnart Fuengfusin'
__version__ = '0.0.1'
__email__ = 'ninnart.fuengfusin@yahoo.com'

import os
from shutil import copy2
import matplotlib.pyplot as plt
import numpy as np


class Recorder:
    """Requirement: numpy, matplotlib, os
    Work as dictionary with keys are string and value as list.
    Example:
        recorder = Recorder('test1', 'test2')
        recorder.record('test1', 1)
        # Test with not existed key.
        # recorder.record('test3', 1)
        recorder.record('test1', 2)
        print(recorder('test1'))
        recorder.plot('test1')
        recorder.hist('test1', bins=10)
        print('highest: {}'.format(recorder.highest('test1')))
        print('lowest: {}'.format(recorder.lowest('test1')))
        recorder.save_all('save')
        recorder.record_check_update_highest('test1', 3) and print('update {}'.format(True))
        # Not printing from and condition.
        recorder.record_check_update_highest('test1', 0) and print('update {}'.format(False))
    """
    def __init__(self, *args: str) -> None:
        """Need to input all string that will use as keys.
        :param args: A string for key of dictionary.
        """
        self.dict_ = {}
        for arg in args:
            self.dict_[str(arg)] = []

    def __call__(self, key: str) -> list:
        """Return a list of record variables.
        :param key: A string of key.
        :return: A list of input key.
        """
        assert type(key) == str
        return self.dict_[key]

    def record(self, key: str, value: float) -> None:
        """Put a value into dictionary
        :param key: A string of key.
        :param value: A value that wants to record in to the dictionary.
        :return: None
        """
        assert type(key) == str
        if key not in self.dict_.keys():
            raise KeyError(f'Recorder did not have key: {key}.')
        else:
            self.dict_[key].append(value)

    def plot(self, key: str, save=False, save_loc='') -> None:
        """Plot graph of a list of the input key.
        :param key: A string of key.
        :param save_loc: A string of save location.
        :param save: A boolean indicate to save figure or not.
        """
        assert type(key) == str
        assert type(save) == bool
        assert type(save_loc) == str
        x = range(len(self.dict_[key]))
        y = self.dict_[key]
        plt.plot(x, y, 'x-')
        plt.grid(True)
        save and plt.savefig(save_loc)
        save and plt.close()
        not save and plt.show()

    def hist(self, key: str, bins: int, save=False, save_loc='') -> None:
        """Plot histogram of a list of the input key.
        :param key: A string of key.
        :param save_loc: A string of save location.
        :param save: A boolean indicate to save figure or not.
        :param bins: A integer indicates resolution of plotting.
        """
        assert type(key) == str
        assert type(bins) == int
        assert type(save) == bool
        assert type(save_loc) == str
        plt.hist(self.dict_[key], bins=bins)
        plt.grid(True)
        plt.show()
        save and plt.savefig(save_loc)
        save and plt.close()
        not save and plt.show()

    def highest(self, key: str):
        """Return max value in list.
        :param key: A string of key for the dictionary.
        :return: A maximum value in a list in the input key dictionary.
        """
        assert type(key) == str
        return max(self.dict_[key]), self.dict_[key].index(max(self.dict_[key]))

    def lowest(self, key: str):
        """Return min value in list.
        :param key: A string of key for the dictionary.
        :return: A minimum value in a list in the input key dictionary.
        """
        assert type(key) == str
        return min(self.dict_[key]), self.dict_[key].index(min(self.dict_[key]))

    def save_np(self, save_loc: str, *keys: str) -> None:
        """Save a list from the key with numpy format.
        :param save_locat: A string of save location.
        :param keys: Strings of key.
        :return: None
        """
        assert type(save_loc) == str
        current_loc = os.getcwd()
        _save_loc = os.path.join(current_loc, save_loc)
        if not (os.path.exists(_save_loc)):
            os.mkdir(_save_loc)
        for key in keys:
            assert type(key) == str
            np.save(os.path.join(_save_loc, key), self.dict_[key])

    def save_all(self, save_locat: str) -> None:
        """Save all of lists in Recorder and also save all of Python file (.py) in the current folder.
        :return: None
        """
        assert type(save_locat) == str
        current_loc = os.getcwd()
        _save_locat = os.path.join(current_loc, save_locat)
        if not (os.path.exists(_save_locat)):
            os.mkdir(_save_locat)
        for key in self.dict_.keys():
            np.save(os.path.join(_save_locat, key), self.dict_[key])
        self.save_py(_save_locat)

    # def record_check_update_highest(self, key: str, value: float) -> bool:
    #     """Record and return a boolean for the highest value change or not
    #     :param key: A string of key.
    #     :param value: A floating variable that want to record.
    #     :return Boolean: True if value > old_best else False
    #     """
    #     try:
    #         old_best, _ = self.highest(key)
    #     except ValueError:
    #         # for detecting the first iteration error (max([])).
    #         old_best = -9999
    #     self.record(key, value)
    #     return True if value > old_best else False

    def more_than_highest(self, key: str, value: float) -> bool:
        """Record and return a boolean for the highest value change or not
        :param key: A string of key.
        :param value: A floating variable that want to record.
        :return Boolean: True if value > old_best else False
        """
        try:
            old_best, _ = self.highest(key)
        except ValueError:
            # for detecting the first iteration error (max([])).
            old_best = -9999
        return True if value > old_best else False

    def save_py(self, save_locat: str) -> None:
        """Copy all Python files in current folder into save_locat.
        :param save_locat: A string of save location.
        :return None:
        """
        if not os.path.exists(os.path.join(os.curdir, save_locat)):
            os.mkdir(os.path.join(os.curdir, save_locat))
        [copy2(i, os.path.join(os.curdir, save_locat, i)) for i in os.listdir(os.curdir) if i.endswith('.py')]

    def save_csv(self, save_locat: str, *keys: str) -> None:
        if not os.path.exists(os.path.join(os.curdir, save_locat)):
            os.mkdir(os.path.join(os.curdir, save_locat))
        for key in keys:
            assert type(key) == str
            np.savetxt(os.path.join(save_locat, key), self.dict_[key], delimiter=",")
    # TODO save and load csv! accuracy so on.


if __name__ == '__main__':
    recorder = Recorder('test1', 'test2')
    recorder.record('test1', 1)
    # Test with not existed key.
    # recorder.record('test3', 1)
    recorder.record('test1', 2)
    print(recorder('test1'))
    recorder.plot('test1')
    recorder.hist('test1', bins=10)
    print('highest: {}'.format(recorder.highest('test1')))
    print('lowest: {}'.format(recorder.lowest('test1')))
    recorder.save_all('save')
    # recorder.record_check_update_highest('test1', 3) and print('update {}'.format(True))
    # Not printing from and condition.
    # recorder.record_check_update_highest('test1', 0) and print('update {}'.format(False))
