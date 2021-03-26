from labeling.apply import *
from labeling.noisy_labels import *
from lfs import rules
from utils import load_data_to_numpy
import re

X, Y = load_data_to_numpy()

sms_noisy_labels = NoisyLabels("sms",X,Y,rules)
L,S = sms_noisy_labels.get_labels()
sms_noisy_labels.generate_pickle()

