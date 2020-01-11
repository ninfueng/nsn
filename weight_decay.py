#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def l2_weight_decay(lamda, model):
    """Getting l2 weight decay loss by using the same concept as link below:
    https://www.tensorflow.org/api_docs/python/tf/nn/l2_loss
    :param lamda: A floating point, regularization parameter l2
    :param model: A neural networks model in torch.nn.Module format.
    :return regularization_loss:
    """
    assert isinstance(lamda, float)
    # To make sure that sum_weights accumulation is start with tensor.
    sum_weight = None
    for name, w in model.named_parameters():
        if name.find('w') != -1:
            if sum_weight is None:
                sum_weight = (w.pow(2)).sum()
            else:
                sum_weight = sum_weight + (w.pow(2)).sum()
    return lamda*sum_weight

