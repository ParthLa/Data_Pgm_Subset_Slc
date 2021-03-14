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
data = np.array([[0.0983969  0.52830115 0.90600643 0.24581662]
 [0.80224391 0.69694779 0.2144578  0.56402219]
 [0.25178871 0.08541089 0.38441611 0.71337658]
 [0.62578016 0.88521213 0.57809485 0.61731398]
 [0.15089138 0.44804339 0.88242663 0.48004503]])

applier = LFApplier(lfs=lfs)
L,S=applier.apply(data)
Lc=np.array([[0 1]
 [0 1]
 [0 1]
 [1 0]
 [0 1]])
 Sc=np.array([[ 0.41930488 -1.        ]
 [ 0.41977896 -1.        ]
 [ 0.58639824 -1.        ]
 [ 0.36346583 -1.        ]
 [ 0.43308815 -1.        ]])
if (np.array_equal(S,Sc) and np.array_equal(L,Lc)):
    print("works fine")
else:
    print("something went wrong")

