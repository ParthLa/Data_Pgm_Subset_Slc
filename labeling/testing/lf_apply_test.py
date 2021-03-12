from labeling.lf import *
from labeling.apply import *
from labeling.preprocess import *
from labeling.continuous_scoring import *
import numpy as np

@preprocessor()
def square(x):
    return {"value":x*x}

@continuous_scorer()
def score(x):
    return np.exp(-1*np.linalg.norm(x['value']))

@labeling_function(pre=[square], cont_scorer=score)
def lf1(x):
    if np.linalg.norm(x['value']) < 1:
        return 0
    else:
        return 1

@labeling_function(pre=[square])                # no continuous scorer specified
def lf2(x):
    if np.linalg.norm(x['value']) < 1:
        return 1
    else:
        return 0

lfs = [lf1, lf2]
data = np.random.rand(5,4)
applier = LFApplier(lfs=lfs)
L,S=applier.apply(data)
print(L)        # output labels
print(S)        # confidence                    # -1.0 where scorer not specified
