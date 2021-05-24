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

def get_rank(compression: int, tensor_net_type:str, in_channels: int, out_channels:int, kernel_size:int) -> int:

    types = ['cpd', 'canonical', 'tucker', 'train', 'tensor-train', 'tt']
    assert tensor_net_type in types, f"Choose a valid tensor network {types}"

    max_params = in_channels * out_channels * kernel_size**3

    if tensor_net_type in ['cpd','canonical']:

        ''''
        cpd params:
        r * (in_channels + out_channels + 3 * kernel_size)
        
        compression = max_params / cpd_params
        '''

        rank = max_params/(compression * (in_channels + out_channels + 3*kernel_size))
        return int(rank)

    if tensor_net_type in ['tucker']:

        '''
        tucker params (assuming all r the same):
        (r**5) + (r * c_in) + (r * c_out) + (r * 3 * kernel_size)
        
        '''
        r = Symbol('r')
        return solve(max_params/(r**5 + r * (in_channels + out_channels + 3*kernel_size)) - compression,r)

if __name__ == '__main__':

    in_channels = 16
    out_channels = 8
    kernel_size = 3
    comp = 10

    rank_cpd = get_rank(compression=comp, tensor_net_type='cpd', in_channels=in_channels, out_channels=out_channels, kernel_size=3)

    rank_tucker = get_rank(compression=comp, tensor_net_type='tucker', in_channels=in_channels, out_channels=out_channels, kernel_size=3)