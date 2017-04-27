import random
import numpy as np


def shuffle_data(*args):
    for i in range(len(args)-1):
        assert len(args[i]) == len(args[i+1]), "Arguments length doesn't match"
    idxs = list(range(len(args[0])))
    random.shuffle(idxs)
    return [np.array([arg[i] for i in idxs]) for arg in args]

