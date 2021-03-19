from labeling.apply import *
from labeling.noisy_labels import *
from lfs import LFS
from utils import load_data_to_numpy
import re

X, Y = load_data_to_numpy()
applier = LFApplier(lfs = LFS)

L,S = applier.apply(X)


sms_noisy_labels = NoisyLabels("sms",X,Y,LFS)
L,S = sms_noisy_labels.get_labels()
sms_noisy_labels.generate_pickle()

