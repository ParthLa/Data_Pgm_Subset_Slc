from labeling.lf import *
from labeling.apply import *
from labeling.preprocess import *
from labeling.continuous_scoring import *
import numpy as np

pre_resources={"r0":1.0}

@preprocessor(resources=pre_resources)
def square(x,**kwargs):
    return {"value":x*x*kwargs["r0"]}

cf_resources={"r1":4, "r2":8, "len1":4}
lf_resources={"r3":4, "len2":5}

@continuous_scorer(resources=cf_resources)
def score(x, **kwargs):
    t1=np.exp(-1*np.linalg.norm(x['value']))
    t2=(kwargs["r1"]+kwargs["r2"])/(kwargs["len1"]*kwargs["len1"])
    t3=kwargs["r2"]/kwargs["len2"]
    return t1*t2*t3

@labeling_function(pre=[square], resources=lf_resources, cont_scorer=score)
def lf1(x, **kwargs):
    if np.linalg.norm(x['value']) < 1 and kwargs["r3"]==4:
        return 0
    return -1

@labeling_function(pre=[square])                # no continuous scorer specified
def lf2(x):
    if np.linalg.norm(x['value']) < 1:
        return 1
    return -1

lfs = [lf1, lf2]
data = np.array([[0.74245789, 0.42154025, 0.30051336, 0.35684219],
[0.0303558,  0.74972649, 0.99098186, 0.56246529]])

applier = LFApplier(lfs=lfs)
L,S=applier.apply(data)

Lc=np.array([[0,  1], [-1, -1]])
Sc=np.array([[ 0.65867513, -1.],[-1.,-1.]])

if (np.allclose(S,Sc) and np.allclose(L,Lc)):
    print("works fine")
else:
    print("something went wrong")

