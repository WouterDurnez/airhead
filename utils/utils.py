#     ___   _     __               __
#    / _ | (_)___/ /  ___ ___ ____/ /
#   / __ |/ / __/ _ \/ -_) _ `/ _  /
#  /_/ |_/_/_/ /_//_/\__/\_,_/\_,_/
#

"""
Various utilities
"""

import os
import math
from sympy.solvers import solve
from sympy import Symbol
from torch.optim.lr_scheduler import LambdaLR
from pytorch_lightning.callbacks import Callback
from utils.helper import log
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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

# TODO: clean up, moved function to LowRank class

def get_tuning_var(compression: int, tensor_net_type:str, in_channels: int, out_channels:int, kernel_size:int = 3) -> int:
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

    # Make sure the right type of tensor network was passed
    types = ['cpd', 'canonical', 'tucker', 'train', 'tensor-train', 'tt']
    assert tensor_net_type in types, f"Choose a valid tensor network {types}"

    # Number of parameters for a full-rank convolution
    max_params = in_channels * out_channels * kernel_size**3

    #######
    # CPD #
    #######

    if tensor_net_type in ['cpd','canonical']:

        ''''
        cpd_params:
        r * (in_channels + out_channels + 3 * kernel_size)
        
        compression = max_params / cpd_params
        '''

        rank = max_params/(compression * (in_channels + out_channels + 3*kernel_size))
        return round(rank)

    ##########
    # TUCKER #
    ##########

    if tensor_net_type in ['tucker']:

        '''
        tucker params:
        ((C_in/S) * 3 * 3 * 3 * (C_out/S)) + 3 * (3*3)             + (input_channels**2/S) + (output_channels**2/S)
        = core tensor shape                + 3 * kernel node shape + input node shape      + output node shape 
        
        compression = max_params / tucker_params        
        '''
        S = Symbol('S',real=True)
        solutions = solve((max_params/((3**3*(in_channels/S)*(out_channels/S)) + 3**3 + ((in_channels**2)/S) + ((out_channels**2)/S))) - compression,S)

        # Check for appropriate S values
        evaluated_solutions = []
        for s in solutions:
            evaluated = s.evalf()
            if evaluated > 0:
                evaluated_solutions.append(evaluated)

        # Check if unique positive, real solution
        assert len(evaluated_solutions) == 1, 'Too many solutions!'

        return evaluated_solutions[0]

    ################
    # TENSOR TRAIN #
    ################

    if tensor_net_type in ['tt','tensor-train','train']:
        '''
        tensor train params:
        r * (in_channels + r*(3+3+3) + out_channels)
        
        compression = max_params / tt_params     
        '''
        r = Symbol('r', real=True)
        solutions = solve(max_params/
                          (r*(in_channels + 3*(3*r) + out_channels)) - compression,r)

        # Check for appropriate S values
        evaluated_solutions = []
        for s in solutions:
            evaluated = s.evalf()
            if evaluated > 0:
                evaluated_solutions.append(evaluated)

        # Check if unique positive, real solution
        assert len(evaluated_solutions) == 1, 'Too many solutions!'

        return evaluated_solutions[0]


def get_network_size(tuning_param: float, tensor_net_type:str, in_channels: int, out_channels:int, kernel_size:int = 3) -> float:
    # Make sure the right type of tensor network was passed
    types = ['cpd', 'canonical', 'tucker', 'train', 'tensor-train', 'tt']
    assert tensor_net_type in types, f"Choose a valid tensor network {types}"

    #######
    # CPD #
    #######

    if tensor_net_type in ['cpd', 'canonical']:
        ''''
        cpd_params:
        r * (in_channels + out_channels + 3 * kernel_size)

        compression = max_params / cpd_params
        '''

        return int(tuning_param) * (in_channels + out_channels + 3*kernel_size)

    ##########
    # TUCKER #
    ##########

    if tensor_net_type in ['tucker']:

        '''
        tucker params:
        ((C_in/S) * 3 * 3 * 3 * (C_out/S)) + 3 * (3*3)             + (input_channels**2/S) + (output_channels**2/S)
        = core tensor shape                + 3 * kernel node shape + input node shape      + output node shape 

        '''
        in_param = round(in_channels/tuning_param)
        out_param = round(out_channels/tuning_param)
        return (in_param * 3*3*3 * out_param) +\
               3**3 +\
               (in_channels*in_param) +\
               (out_channels*out_param)

    ################
    # TENSOR TRAIN #
    ################

    if tensor_net_type in ['tt', 'tensor-train', 'train']:
        '''
        tensor train params:
        r * (in_channels + r*(3+3+3) + out_channels)
        '''
        return tuning_param*(in_channels + tuning_param*9 + out_channels)


if __name__ == '__main__':

    in_channels = 4
    out_channels = 32
    kernel_size = 3
    comp = 10

    max_param = in_channels*out_channels*kernel_size**3

    #tune_cpd = get_tuning_var(compression=comp, tensor_net_type='cpd', in_channels=in_channels, out_channels=out_channels, kernel_size=3)

    '''for comp in (5, 10, 50, 100):
        for in_channels, out_channels in zip((4,32,32,64,64,128,128,256,256,320,320),
                                             (32,32,64,64,128,128,256,256,320,320,320)):
            tune_tucker = get_tuning_var(compression=comp, tensor_net_type='tucker', in_channels=in_channels, out_channels=out_channels, kernel_size=3)
            print(f'comp {comp} - in {in_channels} - out {out_channels} --> S={tune_tucker}')'''

    '''for comp in (2, 5, 10, 50, 100):
        for in_channels, out_channels in zip((4, 32, 32, 64, 64, 128, 128, 256, 256, 320, 320),
                                             (32, 32, 64, 64, 128, 128, 256, 256, 320, 320, 320)):
            tune_train = get_tuning_var(compression=comp, tensor_net_type='tt', in_channels=in_channels,
                                         out_channels=out_channels, kernel_size=3)
            print(f'comp {comp} - in {in_channels} - out {out_channels} --> R={tune_train}')'''


    tuning_cpd, tuning_tt, tuning_tucker = [], [], []
    cpd_params, tt_params, tucker_params = [], [], []
    theoretical_params = []
    compression_rates = range(5,100,5)
    #compression_rates = (1,5,10,50,100)
    for comp in compression_rates:
        theoretical_params.append(max_param/comp)
        tuning_cpd.append(get_tuning_var(compression=comp, tensor_net_type='cpd', in_channels=in_channels,
                                  out_channels=out_channels, kernel_size=3))
        cpd_params.append(get_network_size(tuning_param=tuning_cpd[-1], tensor_net_type='cpd', in_channels=in_channels,
                                  out_channels=out_channels, kernel_size=3))

        tuning_tucker.append(get_tuning_var(compression=comp, tensor_net_type='tucker', in_channels=in_channels,
                                     out_channels=out_channels, kernel_size=3))
        tucker_params.append(get_network_size(tuning_param=tuning_tucker[-1], tensor_net_type='tucker', in_channels=in_channels,
                                  out_channels=out_channels, kernel_size=3))

        tuning_tt.append(get_tuning_var(compression=comp, tensor_net_type='tt', in_channels=in_channels,
                                     out_channels=out_channels, kernel_size=3))
        tt_params.append(get_network_size(tuning_param=tuning_tt[-1], tensor_net_type='tt', in_channels=in_channels,
                                  out_channels=out_channels, kernel_size=3))

    # Create data frame
    df = pd.DataFrame({
        'Compression': compression_rates,
        'Theoretical': theoretical_params,
        'CPD': cpd_params,
        'Tucker':tucker_params,
        'Tensor train': tt_params
    })
    df=df.melt(id_vars='Compression',var_name='Tensor network',
               value_vars=['Theoretical','CPD','Tucker','Tensor train'],value_name='parameters')
    df.parameters = df.parameters.astype('int')

    # Plot
    sns.set_theme(context='paper', style='white', palette='deep',font_scale=1.1, font='serif')
    sns.lineplot(data=df,x='Compression',y='parameters', hue='Tensor network',)
    plt.ylabel('# Parameters')
    plt.xscale('log')
    plt.yscale('log')
    sns.despine(left=True,bottom=True)

    plt.show()
