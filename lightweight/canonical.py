#     ___   _     __               __
#    / _ | (_)___/ /  ___ ___ ____/ /
#   / __ |/ / __/ _ \/ -_) _ `/ _  /
#  /_/ |_/_/_/ /_//_/\__/\_,_/\_,_/
#

"""
Lightweight convolutional layers

CANONICAL POLYADIC FORM
"""

import tensornetwork as tn
import torch
import torch.nn as nn
from torch import tensor
from utils.helper import set_params


tn.set_default_backend('pytorch')


class CanonicalLayer(nn.Module):

    def __init__(self,
                 rank:int = 5):

        super().__init__()

        '''We'll go with a single image for now, as input. Question: to what extent should I take something like
        batch size into account? If we input more than one person's data, does that not add a new dimension to
        the input vector, making it a 5th order tensor rather than a 4th-order one?'''

        # Rank
        self.rank = rank

        # 5 cores
        c_prime = torch.rand(1, 4, dim, dim, dim, requires_grad=True)
        h_prime = torch.rand(1, 4, dim, dim, dim, requires_grad=True)
        w_prime = torch.rand(1, 4, dim, dim, dim, requires_grad=True)
        d_prime = torch.rand(1, 4, dim, dim, dim, requires_grad=True)
        c =  torch.rand(1, 4, dim, dim, dim, requires_grad=True)




if __name__ == '__main__':
    # Console parameters
    set_params(verbosity=3, timestamped=False)

    # Quick test (currently no cuda support on my end)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create appropriately dimensioned tensor with random values
    dim = 128
    x = torch.rand(1, 4, dim, dim, dim)
    x = x.to(device)

    # Sanity check: do dimensions make sense? Let's 'benchmark' a classic Conv3D layer
    in_channels = 4
    out_channels = 4
    kernel_dim = 3
    layer_classic = nn.Conv3d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_dim,padding=1)
    layer_classic.to(device)

    # Output
    y = layer_classic(x)

