from shutil import copy2
import os


def save_py(save_locat: str) -> None:
    if not os.path.exists(os.path.join(os.curdir, save_locat)):
        os.mkdir(os.path.join(os.curdir, save_locat))
    [copy2(i, os.path.join(os.curdir, save_locat, i)) for i in os.listdir(os.curdir) if i.endswith('.py')]


if __name__ == '__main__':
    save_py('save_py')