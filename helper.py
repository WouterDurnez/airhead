"""
Helper functions

-- Coded by Wouter Durnez
"""

from typing import Callable
import time
import os, sys
from torch import Tensor

VERBOSITY = 3
TIMESTAMPED = False

# Set script parameters
def set_params(verbosity:int = 3, timestamped:bool = False):

    global VERBOSITY
    global TIMESTAMPED

    VERBOSITY = verbosity
    TIMESTAMPED = timestamped

# Expand on what happens to input when sent through layer
def whatsgoingon(layer: Callable, input:Tensor):
    """
    Processes input through layer, and prints the effect on the dimensionality
    """

    # Generate output
    output = layer(input)

    # Log the effect
    log(f'{layer.__name__}: {input.shape} --> {output.shape}')

    return output


# Fancy print function
def log(*message, verbosity=3, timestamped=TIMESTAMPED, sep="", title=False):
    """
    Print wrapper that adds timestamp, and can be used to toggle levels of logging info.

    :param message: message to print
    :param verbosity: importance of message: level 1 = top importance, level 3 = lowest importance
    :param sep: separator
    :param title: toggle whether this is a title or not
    :return: /
    """

    # Title always get shown
    verbosity = 1 if title else verbosity

    # Print if log level is sufficient
    if verbosity <= VERBOSITY:

        # Print title
        if title:
            n = len(*message)
            print('\n' + (n + 4) * '#')
            print('# ', *message, ' #', sep='')
            print((n + 4) * '#' + '\n')

        # Print regular
        else:
            t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            print((str(t) +  (" - " if sep == "" else "-")) if timestamped else "", *message, sep=sep)

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

        log("\'", f.__name__, "\' took ", round(t2 - t1, 3), " seconds to complete.", sep="")

        return res

    return timed


def set_dir(*dirs):
    """
    If folders don't exist, make them.

    :param dirs: directories to check/create
    :return: None
    """

    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)
            log("WARNING: Data directory <{dir}> did not exist yet, and was created.".format(dir=dir), verbosity=1)
        else:
            log("\'{}\' folder accounted for.".format(dir), verbosity=3)