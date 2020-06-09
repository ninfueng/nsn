#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Update Log:
2019/06/26: Update UpdateMomentumShape, fixing w -> w_b. Checking back to UpdateMomentum for already correct.
2019/07/05: Update UpdateMomentum as self.mov_grad = beta*self.mov_grad + avg_grad following Tensorflow.
2019/07/06: Reverse to self.mov_grad = beta*self.mov_grad + (1.0 - beta)*avg_grad.
"""
__author__ = 'Ninnart Fuengfusin'
__version__ = '0.0.1'
__email__ = 'ninnart.fuengfusin@yahoo.com'


class UpdateMomentum:
    """Updating the aver_grad to mov_grad for avoiding the confession.
    """
    def __init__(self):
        self.mov_grad = 0

    def __repr__(self):
        return f'Moving average gradient: {self.mov_grad}'

    def update(self, w_b, beta, lr, *grads):
        """Return weight w, by using self.aver_grad and grads.
        From: https://www.tensorflow.org/api_docs/python/tf/train/MomentumOptimizer
        :param w: torch.tensor format.
        :param beta: a floating [0, 1]
        :param lr: a floating
        :param grads: w.grad.data and b.grad.data.
        :return: Tuned weights for an iteration.
        """
        assert type(lr) is float
        assert type(beta) is float
        sum_grad = None
        for grad in grads:
            if sum_grad is None:
                sum_grad = grad.data
            else:
                sum_grad = sum_grad + grad.data
        avg_grad = sum_grad/len(grads)
        # Using same as accumulation = momentum * accumulation + gradient (Tensorflow doc)
        self.mov_grad = beta*self.mov_grad + (1.0 - beta)*avg_grad
        # self.mov_grad = beta*self.mov_grad + avg_grad
        return w_b.data - lr*self.mov_grad.data

    def clear(self):
        self.mov_grad = 0

if __name__ == '__main__':
    import torch
    print(__doc__)
    w = torch.tensor(10)
    beta = 0.9
    lr = 1e-1
    grad = torch.tensor(1e-1)
    updater = UpdateMomentum()
    print(w)
    grad = torch.tensor(2e-1)
    w = updater.update(w, beta, lr, grad)
    print(w)
    grad = torch.tensor(3e-1)
    w = updater.update(w, beta, lr, grad)
    print(w)
