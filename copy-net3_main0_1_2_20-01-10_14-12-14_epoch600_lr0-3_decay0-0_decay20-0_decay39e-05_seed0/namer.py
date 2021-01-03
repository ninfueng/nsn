#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate a string for using into the saving file in pattern: FolderName_Time_HyperParameters.
19/06/15: Initial committing.
19/06/25: Update get_filename.
"""

__author__ = 'Ninnart Fuengfusin'
__version__ = '0.0.1'
__email__ = 'ninnart.fuengfusin@yahoo.com'

import os
from datetime import datetime
# Getting name of main Python script.
from __main__ import __file__


def namer(*hyp_params):
    """ Return A string in format of {folder}_{file}_{date}_{hyp_param}.
    :param hyp_params: Strings that we want to include in the folder name.
    :return: A string for saving.
    """
    def get_foldername() -> str:
        """Return a string of folder name.
        :return folder_name: A string of folder name.
        """
        full_path = os.path.realpath(__file__)
        path, _ = os.path.split(full_path)
        _, folder_name = os.path.split(path)
        return folder_name

    def get_date() -> str:
        """Return a string of datetime.
        :return:
        """
        return str(datetime.now().strftime('%y-%m-%d_%H-%M-%S'))

    def dot_to_dash(string):
        """From the input string, find . change into _.
        :param string: A string that have .
        :return out_string: A string that have not . but _.
        """
        assert type(string) == str
        return string.replace('.', '-')

    def get_filename() -> str:
        """Return a string name of python scirpt.
        :return: a string name of python scirpt
        """

        return os.path.basename(__file__).replace('.py', '')

    folder = get_foldername()
    file = get_filename()
    date = get_date()
    name = f'{folder}_{file}_{date}'
    for hyp_param in hyp_params:
        hyp_param = dot_to_dash(hyp_param)
        name += '_' + hyp_param
    return name


if __name__ == '__main__':
    print(__doc__)
    print(namer('lr0.1', 'sgd'))
