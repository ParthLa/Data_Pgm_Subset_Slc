from labeling.apply import *

from lfs import LFS
from utils import load_data_to_numpy
import re

X, Y = load_data_to_numpy()
applier = LFApplier(lfs = LFS)

L,S = applier.apply(X)

    