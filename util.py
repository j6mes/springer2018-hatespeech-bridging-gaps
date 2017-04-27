import random
import numpy as np

r = random.Random(1)


def shuffle_data(*args):
    for i in range(len(args)-1):
        assert len(args[i]) == len(args[i+1]), "Arguments length doesn't match"
    idxs = list(range(len(args[0])))
    r.shuffle(idxs)
    return [np.array([arg[i] for i in idxs]) for arg in args]


def early_stopping(curve, patience=3, lower_is_better=True):
    best = min if lower_is_better else max
    if len(curve) < patience or best(curve) in curve[-patience:]:
        return False
    print(curve, best(curve), curve[-patience:])
    return True

