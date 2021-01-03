# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """The collection of weight decay for using for regularization.
# 2019/06/26: Committed weight_decay.py.
# 2019/07/04: Update l2_weight_decay, l2 = 3e-5, with the best test accuracy: 0.9882.
# """
# __author__ = 'Ninnart Fuengfusin'
# __version__ = '0.0.2'
# __email__ = 'ninnart.fuengfusin@yahoo.com'
#
#
# def l2_weight_decay(lamda, model):
#     """Getting l2 weight decay loss by using the same concept as link below:
#     https://www.tensorflow.org/api_docs/python/tf/nn/l2_loss
#     :param lamda: A floating point, regularization parameter l2
#     :param model: A neural networks model in torch.nn.Module format.
#     :return regularization_loss:
#     """
#     assert type(lamda) is float
#     sum_weight = None
#     for i in model.parameters():
#         if sum_weight is None:
#             sum_weight = (i.pow(2)).sum()
#         else:
#             sum_weight = sum_weight + (i.pow(2)).sum()
#     return lamda*sum_weight
#
#
# def l2_weight_decay_(lamda, model):
#     """Getting l2 weight decay loss by using the same concept as link below:
#     https://stackoverflow.com/questions/44641976/in-pytorch-how-to-add-l1-regularizer-to-activations
#     :param lamda: A floating point, regularization parameter l2
#     :param model: A neural networks model in torch.nn.Module format.
#     :return regularization_loss:
#     """
#     assert type(lamda) is float
#     sum_weight = None
#     for i in model.parameters():
#         if sum_weight is None:
#             sum_weight = i.norm(2).sum()
#         else:
#             sum_weight = sum_weight + i.norm(2).sum()
#     return lamda*sum_weight
#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""The collection of weight decay for using for regularization.
2019/06/26: Committed weight_decay.py.
2019/07/04: Update l2_weight_decay, l2 = 3e-5, with the best test accuracy: 0.9882.
"""
__author__ = 'Ninnart Fuengfusin'
__version__ = '0.0.2'
__email__ = 'ninnart.fuengfusin@yahoo.com'


def l2_weight_decay(lamda, model):
    """Getting l2 weight decay loss by using the same concept as link below:
    https://www.tensorflow.org/api_docs/python/tf/nn/l2_loss
    :param lamda: A floating point, regularization parameter l2
    :param model: A neural networks model in torch.nn.Module format.
    :return regularization_loss:
    """
    assert type(lamda) is float
    # To make sure that sum_weights accumulation is start with tensor.
    sum_weight = None
    for name, w in model.named_parameters():
        if name.find('w') != -1:
            if sum_weight is None:
                sum_weight = (w.pow(2)).sum()
            else:
                sum_weight = sum_weight + (w.pow(2)).sum()
    return lamda*sum_weight


def l2_weight_decay_(lamda, model):
    """Getting l2 weight decay loss by using the same concept as link below:
    https://stackoverflow.com/questions/44641976/in-pytorch-how-to-add-l1-regularizer-to-activations
    :param lamda: A floating point, regularization parameter l2
    :param model: A neural networks model in torch.nn.Module format.
    :return regularization_loss:
    """
    assert type(lamda) is float
    sum_weight = None
    for i in model.parameters():
        if sum_weight is None:
            sum_weight = i.norm(2).sum()
        else:
            sum_weight = sum_weight + i.norm(2).sum()
    return lamda*sum_weight
