"""
Helper functions
"""

import os
import time
from os.path import join
from pathlib import Path
from typing import Callable

from colorama import Fore, Style
from pytorch_lightning import seed_everything
from torch import Tensor
from torch.nn import Module

VERBOSITY = 3
TIMESTAMPED = True
DATA_DIR = join(
    Path(os.path.dirname(os.path.abspath(__file__))).parents[1], 'data'
)
LOG_DIR = join(
    Path(os.path.dirname(os.path.abspath(__file__))).parents[1], 'logs'
)
TENSOR_NET_TYPES = (
    'cp',
    'cpd',
    'canonical',
    'tucker',
    'train',
    'tensor-train',
    'tt',
)
KUL_PAL = ['#FF7251', '#C58B85', '#8CA5B8', '#52BEEC']
KUL_PAL2 = [
    '#FF7251',
    '#E67D67',
    '#CE887D',
    '#B59393',
    '#9C9DAA',
    '#83A8C0',
    '#6BB3D6',
    '#52BEEC',
]

# Set parameters
def set_params(
    verbosity: int = None,
    timestamped: bool = None,
    data_dir: str = None,
    log_dir: str = None,
):
    global VERBOSITY
    global TIMESTAMPED
    global DATA_DIR
    global LOG_DIR

    VERBOSITY = verbosity if verbosity else VERBOSITY
    TIMESTAMPED = timestamped if timestamped is not None else TIMESTAMPED
    DATA_DIR = data_dir if data_dir else DATA_DIR
    LOG_DIR = log_dir if log_dir else LOG_DIR

    set_dir(DATA_DIR, LOG_DIR)


def hi(title=None, **params):
    """
    Say hello. (It's stupid, I know.)
    If there's anything to initialize, do so here.
    """

    print('\n')
    print(Fore.BLUE, end='')
    print('     ___   _     __               __')
    print('    / _ | (_)___/ /  ___ ___ ____/ /')
    print('   / __ |/ / __/ _ \/ -_) _ `/ _  /')
    print('  /_/ |_/_/_/ /_//_/\__/\_,_/\_,_/', end='')
    print(Style.RESET_ALL)
    print()

    if title:
        log(title, title=True, color='blue')

    # Set params on request
    if params:
        set_params(**params)

    print()

    # Set directories
    if not os.path.exists(DATA_DIR) or not os.path.exists(LOG_DIR):
        set_dir(DATA_DIR, LOG_DIR)

    # Set seed
    seed_everything(616, workers=True)


# Expand on what happens to input when sent through layer
def whatsgoingon(layer: Callable, input: Tensor):
    """
    Processes input through layer, and prints the effect on the dimensionality
    """

    # Generate output
    output = layer(input)

    # Log the effect
    log(f'{layer.__name__}: {input.shape} --> {output.shape}')

    return output


# Fancy print
def log(
    *message, verbosity=3, sep='', timestamped=None, title=False, color=None
):
    """
    Print wrapper that adds timestamp, and can be used to toggle levels of logging info.

    :param message: message to print
    :param verbosity: importance of message: level 1 = top importance, level 3 = lowest importance
    :param timestamped: include timestamp at start of log
    :param sep: separator
    :param title: toggle whether this is a title or not
    :param color: text color
    :return: /
    """

    # Set colors
    color_dict = {
        'red': Fore.RED,
        'blue': Fore.BLUE,
        'green': Fore.GREEN,
        'yellow': Fore.YELLOW,
        'magenta': Fore.MAGENTA,
        'cyan': Fore.CYAN,
    }
    if color and color in color_dict:
        color = color_dict[color]

    # Title always get shown
    verbosity = 1 if title else verbosity

    # Print if log level is sufficient
    if verbosity <= VERBOSITY:

        # Print title
        if title:
            n = len(*message)
            if color:
                print(color, end='')
            print('\n' + (n + 4) * '#')
            print('# ', *message, ' #', sep='')
            print((n + 4) * '#' + '\n' + Style.RESET_ALL)

        # Print regular
        else:
            ts = timestamped if timestamped is not None else TIMESTAMPED
            t = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
            if color:
                print(color, end='')
            print(
                (str(t) + (' - ' if sep == '' else '-')) if ts else '',
                *message,
                Style.RESET_ALL,
                sep=sep,
            )

    return


def time_it(f: Callable):
    """
    Timer decorator: shows how long execution of function took.
    :param f: function to measure
    :return: /
    """

    def timed(*args, **kwargs):
        t1 = time.time()
        res = f(*args, **kwargs)
        t2 = time.time()

        log(
            "'",
            f.__name__,
            "' took ",
            round(t2 - t1, 3),
            ' seconds to complete.',
            sep='',
        )

        return res

    return timed


def set_dir(*dirs):
    """
    If folder doesn't exist, make it.

    :param dir: directory to check/create
    :return: path to dir
    """

    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir, exist_ok=True)
            log(
                'WARNING: Data directory <{dir}> did not exist yet, and was created.'.format(
                    dir=dir
                ),
                verbosity=1,
            )
        else:
            log("'{}' folder accounted for.".format(dir), verbosity=3)


def count_params(module: Module, format=None, precision=3):
    """Count total parameters in a module/model"""

    n = sum(p.numel() for p in module.parameters())

    if format == 'k':

        return f'{round(n/1000,precision)}k params'
    elif format == 'M':

        return f'{round(n/1000000,precision)}M params'
    elif format == 'G':

        return f'{round(n/1000000000,precision)}G params'
    elif format == 'base':

        return f'{round(n, precision)} params'

    else:
        return n


if __name__ == '__main__':
    hi('Test!')
