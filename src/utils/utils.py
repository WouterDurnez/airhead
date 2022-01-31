#     ___   _     __               __
#    / _ | (_)___/ /  ___ ___ ____/ /
#   / __ |/ / __/ _ \/ -_) _ `/ _  /
#  /_/ |_/_/_/ /_//_/\__/\_,_/\_,_/
#

"""
Various utilities
"""
import inspect
import math
import os
import sys
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pytorch_lightning.callbacks import Callback
from sympy import Symbol
from sympy.solvers import solve
from torch.optim.lr_scheduler import LambdaLR

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import src.utils.helper as hlp
from src.utils.helper import KUL_PAL, TENSOR_NET_TYPES
from src.utils.helper import log


#################
# LR Schedulers #
#################

class WarmupCosineSchedule(LambdaLR):
    """
    Learning rate schedule with linear warmup and then cosine decay.
    Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
    Decreases learning rate from 1. to 0. over remaining steps.
    """

    def __init__(self, optimizer, warmup_steps, total_steps, last_epoch=-1):

        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        super().__init__(
            optimizer, self.lr_lambda, last_epoch=last_epoch
        )

    # Determine lambda from step
    def lr_lambda(self, step):

        # During warmup period, linear increase to LR = 1
        if step < self.warmup_steps:
            factor = step / self.warmup_steps

        # After warmup, cosine schedule
        #    See: He, T., Zhang, Z., Zhang, H., Zhang, Z., Xie, J., & Li, M. (2019).
        #    Bag of tricks for image classification with convolutional neural networks.
        #    Proceedings of the IEEE Computer Society Conference on Computer Vision and
        #    Pattern Recognition, 2019-June, 558–567. https://doi.org/10.1109/CVPR.2019.00065

        else:

            # Discount the warmup period
            progress = step - self.warmup_steps

            # Calculate frequency
            w = math.pi / (self.total_steps - self.warmup_steps)  # frequency
            factor = 0.5 * (1.0 + math.cos(w * progress))

        return factor


#############
# Callbacks #
#############

class TakeSnapshot(Callback):
    """
    Callback to store training progress/results
    """

    def __init__(self, epochs=None, save_dir=None):
        super(TakeSnapshot, self).__init__()
        self.epochs = () if epochs is None else epochs
        self.save_dir = save_dir

    # Call this every time the validation loop ends
    def on_validation_end(self, trainer, pl_module):

        # Store to specified dir, or to log dir of trainer if none provided
        if self.save_dir is None:
            self.save_dir = os.path.join(trainer.logger.log_dir, "checkpoints")

        # Get current epoch from trainer
        epoch = trainer.current_epoch

        # Execute this at specified epochs
        if epoch in self.epochs:
            # Save checkpoint
            filepath = os.path.join(self.save_dir, f"epoch={epoch}.ckpt")
            trainer.save_checkpoint(filepath)
            log(f"\r Snapshot taken, epoch = {epoch}", timestamped=True)

    # Get learning rate from trainer
    def get_lr(self, trainer):

        # (... apparently it's buried deep)
        optimizer = trainer.lr_schedulers[0]["scheduler"].optimizer
        for param_group in optimizer.param_groups:
            return param_group["lr"]


#############################
# Compression/rank tradeoff #
#############################

def get_tuning_par(compression: int, tensor_net_type: str, in_channels: int, out_channels: int,
                   kernel_size: int = 3) -> int:
    """
    Given a compression rate, return the appropriate tuning parameter,
    depending on the tensor network that is used for the low-rank convolution
    :param compression: compression rate
    :param tensor_net_type: tensor network type (cpd, tensor train or tucker)
    :param in_channels: input channels
    :param out_channels: output channels
    :param kernel_size: kernel size (default 3)
    :return:
    """

    # Make sure the right type of tensor network was passed]
    assert tensor_net_type in TENSOR_NET_TYPES, f"Choose a valid tensor network {TENSOR_NET_TYPES}"

    # Number of parameters for a full-rank convolution
    max_params = in_channels * out_channels * kernel_size ** 3

    #######
    # CPD #
    #######

    if tensor_net_type in ('cp', 'cpd', 'canonical'):
        ''''
        cpd_params:
        r * (in_channels + out_channels + 3 * kernel_size)
        
        compression = max_params / cpd_params
        '''
        rank = max_params / (compression * (in_channels + out_channels + 3 * kernel_size))
        return round(rank)

    ##########
    # TUCKER #
    ##########

    if tensor_net_type in ['tucker']:
        '''
        tucker params:
        (C_in**2/S)        + (C_out**/S)       + (C_in/S * C_out/S * k_h * k_w * k_d)
        = input node shape + output node shape + central node shape
        
        compression = max_params / tucker_params        
        '''

        S = Symbol('S', positive=True, real=True)
        solutions = solve((max_params / (
                ((in_channels ** 2) / S) +
                ((out_channels ** 2) / S) +
                ((in_channels / S) * (out_channels / S) * kernel_size ** 3)
        )) - compression, S)

        # Check for appropriate S values
        evaluated_solutions = []
        for s in solutions:
            evaluated_solutions.append(s.evalf())

        # Check if unique positive, real solution
        assert len(evaluated_solutions) == 1, 'Too many solutions!'

        return evaluated_solutions[0]

    ################
    # TENSOR TRAIN #
    ################

    if tensor_net_type in ['tt', 'tensor-train', 'train']:
        '''
        tensor train params:
        r * (in_channels + r*(3+3+3) + out_channels)
        compression = max_params / tt_params     
        '''
        r = Symbol('r', real=True, positive=True)
        solutions = solve(max_params /
                          (r * (in_channels + 3 * (3 * r) + out_channels)) - compression, r)

        # Check for appropriate S values
        evaluated_solutions = []
        for s in solutions:
            evaluated_solutions.append(s.evalf())

        # Check if unique positive, real solution
        assert len(evaluated_solutions) == 1, 'Too many solutions!'

        return evaluated_solutions[0]


def get_network_size(tuning_param: float, tensor_net_type: str, in_channels: int, out_channels: int,
                     kernel_size: int = 3) -> float:
    # Make sure the right type of tensor network was passed
    assert tensor_net_type in TENSOR_NET_TYPES, f"Choose a valid tensor network {TENSOR_NET_TYPES}"

    #######
    # CPD #
    #######

    if tensor_net_type in ['cp', 'cpd', 'canonical']:
        ''''
        cpd_params:
        r * (in_channels + out_channels + 3 * kernel_size)

        compression = max_params / cpd_params
        '''

        return int(tuning_param) * (in_channels + out_channels + 3 * kernel_size)

    ##########
    # TUCKER #
    ##########

    elif tensor_net_type in ['tucker']:
        '''
        tucker params:
        (C_in**2/S)        + (C_out**/S)       + (C_in/S * C_out/S * k_h * k_w * k_d)
        = input node shape + output node shape + central node shape
        '''
        in_param = round(in_channels / tuning_param) if round(in_channels / tuning_param) > 0 else 1
        out_param = round(out_channels / tuning_param) if round(out_channels / tuning_param) > 0 else 1
        in_param_squared = in_param * in_channels
        out_param_squared = out_param * out_channels

        return (
                in_param_squared + out_param_squared + (in_param * out_param * kernel_size ** 3)
        )

    ################
    # TENSOR TRAIN #
    ################

    elif tensor_net_type in ['tt', 'tensor-train', 'train']:
        '''
        tensor train params:
        r * (in_channels + r*(3+3+3) + out_channels)
        '''
        return tuning_param * (in_channels + tuning_param * 9 + out_channels)


if __name__ == '__main__':

    hlp.hi('Utils')
    vis_dir = join(hlp.DATA_DIR, 'visuals')

    # Convolutional layer parameters
    in_channels = 64
    out_channels = 32
    kernel_size = 3
    comp = 2

    # Number of parameters in full kernel
    max_param = in_channels * out_channels * kernel_size ** 3

    # We'll calculate tuning parameters and resulting actual compression rates
    # for a number of compression rate settings
    tuning_cpd, tuning_tt, tuning_tucker = [], [], []
    cpd_params, tt_params, tucker_params = [], [], []
    theoretical_params = []
    compression_rates = range(2, 100, 1)

    for comp in compression_rates:
        theoretical_params.append(max_param / comp)
        tuning_cpd.append(get_tuning_par(compression=comp, tensor_net_type='cpd', in_channels=in_channels,
                                         out_channels=out_channels, kernel_size=kernel_size))
        cpd_params.append(get_network_size(tuning_param=tuning_cpd[-1], tensor_net_type='cpd', in_channels=in_channels,
                                           out_channels=out_channels, kernel_size=kernel_size))

        tuning_tucker.append(get_tuning_par(compression=comp, tensor_net_type='tucker', in_channels=in_channels,
                                            out_channels=out_channels, kernel_size=kernel_size))
        tucker_params.append(
            get_network_size(tuning_param=tuning_tucker[-1], tensor_net_type='tucker', in_channels=in_channels,
                             out_channels=out_channels, kernel_size=kernel_size))
        tuning_tt.append(get_tuning_par(compression=comp, tensor_net_type='tt', in_channels=in_channels,
                                        out_channels=out_channels, kernel_size=kernel_size))
        tt_params.append(get_network_size(tuning_param=tuning_tt[-1], tensor_net_type='tt', in_channels=in_channels,
                                          out_channels=out_channels, kernel_size=kernel_size))

    # Create data frame
    df = pd.DataFrame({
        'Compression': compression_rates,
        'Theoretical': theoretical_params,
        'Canonical polyadic': cpd_params,
        'Tensor train': tt_params,
        'Tucker': tucker_params,

    })
    df = df.melt(id_vars='Compression', var_name='Format',
                 value_vars=['Theoretical', 'Canonical polyadic', 'Tensor train', 'Tucker'],
                 value_name='parameters')
    df.parameters = df.parameters.astype('int')
    df['actual_compression'] = max_param / df.parameters
    df['compression_error'] = np.absolute(df.actual_compression - df.Compression) / df.Compression
    df = df.loc[df['Format'] != 'Theoretical']

    # Plot
    colors = sns.color_palette(KUL_PAL[:3])  # + ["#DD8A2E"])[1:]
    fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=300)
    sns.set_theme(context='paper', font_scale=1.3, style='white', palette=colors)
    sns.lineplot(data=df, x='Compression', y='actual_compression', hue='Format', style='Format',
                 markers=False, markersize=8, dashes=True, palette=colors, ax=ax)
    ax.set_ylabel('Actual compression rate', fontweight='bold', size=15)
    ax.set_xlabel('Chosen compression rate', fontweight='bold', size=15)
    plt.setp(ax.get_legend().get_title(), fontweight='bold')
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.savefig(join(vis_dir, 'param_vs_compression.pdf'), bbox_inches='tight', pad_inches=0)
    plt.show()
