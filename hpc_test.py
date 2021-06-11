import utils.helper as h
import os
import torch
from utils.helper import log
from layers import air_conv

if __name__ == '__main__':

    h.hi('HPC test')
    log(torch.cuda.device_count(), color='red')
