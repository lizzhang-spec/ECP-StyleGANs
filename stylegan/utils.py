import random
import time
import datetime
import sys

from torch.autograd import Variable
import torch
from visdom import Visdom
import numpy as np

# # initiate a mask for convolution layer
# first_conv_out = 512
# # Define the scaling factors for each convolution layer
# scaling_factors = [1] * 7 + [0.5] * 2 + [0.25] * 1 + [0.125] * 1 + [0.0625] * 1 + [0.03125] * 1
# # Use a list comprehension to compute the channel numbers
# mask_chns = [int(first_conv_out * scale) for scale in scaling_factors]
# for i in range(9):
#     mask_chns.append(3)
# bit_len = 0
# for mask_chn in mask_chns:
#     bit_len += mask_chn


def compute_layer_mask_longMasked(mask, mask_chns):
    cfg_mask = []

    start_id = 0
    end_id = start_id + mask_chns[0]
    cfg_mask.append(mask[:end_id])
    start_id = end_id
    end_id = start_id + mask_chns[1]
    cfg_mask.append(mask[start_id:end_id])
    start_id = end_id
    end_id = start_id + mask_chns[2]
    cfg_mask.append(mask[start_id:end_id])
    start_id = end_id
    end_id = start_id + mask_chns[3]
    cfg_mask.append(mask[start_id:end_id])
    start_id = end_id
    end_id = start_id + mask_chns[4]
    cfg_mask.append(mask[start_id:end_id])
    start_id = end_id
    end_id = start_id + mask_chns[5]
    cfg_mask.append(mask[start_id:end_id])
    start_id = end_id
    end_id = start_id + mask_chns[6]
    cfg_mask.append(mask[start_id:end_id])
    start_id = end_id
    end_id = start_id + mask_chns[7]
    cfg_mask.append(mask[start_id:end_id])
    start_id = end_id
    end_id = start_id + mask_chns[8]
    cfg_mask.append(mask[start_id:end_id])
    # start_id = end_id
    # end_id = start_id + mask_chns[9]
    # cfg_mask.append(mask[start_id:end_id])
    # start_id = end_id
    # end_id = start_id + mask_chns[10]
    # cfg_mask.append(mask[start_id:end_id])
    # start_id = end_id
    # end_id = start_id + mask_chns[11]
    # cfg_mask.append(mask[start_id:end_id])
    # start_id = end_id
    # end_id = start_id + mask_chns[12]
    # cfg_mask.append(mask[start_id:end_id])
    # start_id = end_id
    # end_id = start_id + mask_chns[13]
    # cfg_mask.append(mask[start_id:end_id])
    # start_id = end_id
    # end_id = start_id + mask_chns[14]
    # cfg_mask.append(mask[start_id:end_id])
    # start_id = end_id
    # end_id = start_id + mask_chns[15]
    # cfg_mask.append(mask[start_id:end_id])
    # start_id = end_id
    # end_id = start_id + mask_chns[16]
    # cfg_mask.append(mask[start_id:end_id])
    # start_id = end_id
    # end_id = start_id + mask_chns[17]
    # cfg_mask.append(mask[start_id:end_id])
    # start_id = end_id
    # end_id = start_id + mask_chns[18]
    # cfg_mask.append(mask[start_id:end_id])
    # start_id = end_id
    # end_id = start_id + mask_chns[19]
    # cfg_mask.append(mask[start_id:end_id])
    # start_id = end_id
    # end_id = start_id + mask_chns[20]
    # cfg_mask.append(mask[start_id:end_id])
    cfg_mask.append(np.ones(1))

    return cfg_mask


def compute_layer_mask(mask, mask_chns):
    cfg_mask = []

    start_id = 0
    end_id = start_id + mask_chns[0]
    cfg_mask.append(mask[:end_id])
    start_id = end_id
    end_id = start_id + mask_chns[1]
    cfg_mask.append(mask[start_id:end_id])
    start_id = end_id
    end_id = start_id + mask_chns[2]
    cfg_mask.append(mask[start_id:end_id])
    start_id = end_id
    end_id = start_id + mask_chns[3]
    cfg_mask.append(mask[start_id:end_id])
    start_id = end_id
    end_id = start_id + mask_chns[4]
    cfg_mask.append(mask[start_id:end_id])
    start_id = end_id
    end_id = start_id + mask_chns[5]
    cfg_mask.append(mask[start_id:end_id])
    start_id = end_id
    end_id = start_id + mask_chns[6]
    cfg_mask.append(mask[start_id:end_id])
    start_id = end_id
    end_id = start_id + mask_chns[7]
    cfg_mask.append(mask[start_id:end_id])
    start_id = end_id
    end_id = start_id + mask_chns[8]
    cfg_mask.append(mask[start_id:end_id])
    # start_id = end_id
    # end_id = start_id + mask_chns[9]
    # cfg_mask.append(mask[start_id:end_id])
    # start_id = end_id
    # end_id = start_id + mask_chns[10]
    # cfg_mask.append(mask[start_id:end_id])
    # start_id = end_id
    # end_id = start_id + mask_chns[11]
    # cfg_mask.append(mask[start_id:end_id])
    # start_id = end_id
    # end_id = start_id + mask_chns[12]
    # cfg_mask.append(mask[start_id:end_id])
    # start_id = end_id
    # end_id = start_id + mask_chns[13]
    # cfg_mask.append(mask[start_id:end_id])
    # start_id = end_id
    # end_id = start_id + mask_chns[14]
    # cfg_mask.append(mask[start_id:end_id])
    # start_id = end_id
    # end_id = start_id + mask_chns[15]
    # cfg_mask.append(mask[start_id:end_id])
    # start_id = end_id
    # end_id = start_id + mask_chns[16]
    # cfg_mask.append(mask[start_id:end_id])
    # start_id = end_id
    # end_id = start_id + mask_chns[17]
    # cfg_mask.append(mask[start_id:end_id])
    # start_id = end_id
    # end_id = start_id + mask_chns[18]
    # cfg_mask.append(mask[start_id:end_id])
    # start_id = end_id
    # end_id = start_id + mask_chns[19]
    # cfg_mask.append(mask[start_id:end_id])
    # start_id = end_id
    # end_id = start_id + mask_chns[20]
    # cfg_mask.append(mask[start_id:end_id])
    cfg_mask.append(np.ones(1))

    return cfg_mask

# def compute_layer_mask_afhq(mask, mask_chns):
#     cfg_mask = []
#
#     start_id = 0
#     end_id = start_id + mask_chns[0]
#     cfg_mask.append(mask[:end_id])
#     start_id = end_id
#     end_id = start_id + mask_chns[1]
#     cfg_mask.append(mask[start_id:end_id])
#     start_id = end_id
#     end_id = start_id + mask_chns[2]
#     cfg_mask.append(mask[start_id:end_id])
#     start_id = end_id
#     end_id = start_id + mask_chns[3]
#     cfg_mask.append(mask[start_id:end_id])
#     start_id = end_id
#     end_id = start_id + mask_chns[4]
#     cfg_mask.append(mask[start_id:end_id])
#     start_id = end_id
#     end_id = start_id + mask_chns[5]
#     cfg_mask.append(mask[start_id:end_id])
#     start_id = end_id
#     end_id = start_id + mask_chns[6]
#     cfg_mask.append(mask[start_id:end_id])
#     # start_id = end_id
#     # end_id = start_id + mask_chns[7]
#     # cfg_mask.append(mask[start_id:end_id])
#     # start_id = end_id
#     # end_id = start_id + mask_chns[8]
#     # cfg_mask.append(mask[start_id:end_id])
#     cfg_mask.append(np.ones(1))
#
#     return cfg_mask

# def compute_layer_mask_weight(mask, mask_chns):
#     cfg_mask = []
#
#     start_id = 0
#     end_id = start_id + mask_chns[0]
#     cfg_mask.append(mask[:end_id])
#     start_id = end_id
#     end_id = start_id + mask_chns[1]
#     cfg_mask.append(mask[start_id:end_id])
#     start_id = end_id
#     end_id = start_id + mask_chns[2]
#     cfg_mask.append(mask[start_id:end_id])
#     start_id = end_id
#     end_id = start_id + mask_chns[3]
#     cfg_mask.append(mask[start_id:end_id])
#     start_id = end_id
#     end_id = start_id + mask_chns[4]
#     cfg_mask.append(mask[start_id:end_id])
#     start_id = end_id
#     end_id = start_id + mask_chns[5]
#     cfg_mask.append(mask[start_id:end_id])
#     start_id = end_id
#     end_id = start_id + mask_chns[6]
#     cfg_mask.append(mask[start_id:end_id])
#     start_id = end_id
#     end_id = start_id + mask_chns[7]
#     cfg_mask.append(mask[start_id:end_id])
#     start_id = end_id
#     end_id = start_id + mask_chns[8]
#     cfg_mask.append(mask[start_id:end_id])
#     start_id = end_id
#     end_id = start_id + mask_chns[9]
#     cfg_mask.append(mask[start_id:end_id])
#     start_id = end_id
#     end_id = start_id + mask_chns[10]
#     cfg_mask.append(mask[start_id:end_id])
#     start_id = end_id
#     end_id = start_id + mask_chns[11]
#     cfg_mask.append(mask[start_id:end_id])
#     start_id = end_id
#     end_id = start_id + mask_chns[12]
#     cfg_mask.append(mask[start_id:end_id])
#     start_id = end_id
#     end_id = start_id + mask_chns[13]
#     cfg_mask.append(mask[start_id:end_id])
#     start_id = end_id
#     end_id = start_id + mask_chns[14]
#     cfg_mask.append(mask[start_id:end_id])
#     start_id = end_id
#     end_id = start_id + mask_chns[15]
#     cfg_mask.append(mask[start_id:end_id])
#     start_id = end_id
#     end_id = start_id + mask_chns[16]
#     cfg_mask.append(mask[start_id:end_id])
#     cfg_mask.append(np.ones(1))
#
#     return cfg_mask

