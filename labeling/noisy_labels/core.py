from typing import Any, Callable, List, Mapping, Optional
from labeling.types import DataPoint, DataPoints
from labeling.apply import *
from labeling.lf import *
from labeling.lf_set import *
from labeling.utils.pickle import *

import pickle
import numpy as np

class NoisyLabels:
    def __init__(
        self,
        name: str,
        data: DataPoints,
        gold_labels: Optional[DataPoints],
        rules: LFSet,
    ) -> None:
        self.name = name
        self._data = data
        self._gold_labels = gold_labels
        self._rules = rules
        self._L = None
        self._S = None

    def get_labels(self):
        if self._L is None or self._S is none:
            applier = LFApplier(lf_set = self._rules)
            L,S = applier.apply(self._data)
            self._L = L
            self._S = S
        return self._L, self._S

    def generate_pickle(self, filename=None):
        if filename is None:
            filename = self.name+"_pickle"
        
        if (self._L is None or self._S is None):
            applier = LFApplier(lf_set = self._rules)
            L,S = applier.apply(self._data)
            self._L = L
            self._S = S

        num_inst=self._data.shape[0]
        num_rules=self._L.shape[0]

        x=self._data
        l=self._L

        m=(self._L>-1).astype(int)                                              # lf covers example or not 
        L=self._gold_labels                                                     # true labels
        d=np.ones((num_inst, 1))                                                # belongs to labeled data or not
        r=np.zeros((num_inst, num_rules))                                       # exemplars

        s=self._S                                                               # continuous scores
        n=np.array([lf._is_cont for lf in self._rules.get_lfs()], dtype=int)    # lf continuous or not
        k=np.array([lf._label for lf in self._rules.get_lfs()], dtype=int)      # lf associated to which class

        to_dump = [x,l,m,L,d,r,s,n,k]
        dump_to_pickle(filename, to_dump)






