from typing import Any, Callable, List, Mapping, Optional
from labeling.types import DataPoint, DataPoints
from labeling.apply import *
from labeling.lf import *

import pickle
import numpy as np

class NoisyLabels:
    def __init__(
        self,
        name: str,
        data: DataPoints,
        truelabels,
        lfs: List[LabelingFunction],
    ) -> None:
        self.name = name
        self._data = data
        self._truelabels = truelabels
        self._lfs = lfs

        applier = LFApplier(lfs = lfs)
        L,S = applier.apply(self._data)
        self._L = L
        self._S = S
    

    def get_labels(self):
        return self._L, self._S

    def generate_pickle(self, filename=None):
        if filename is None:
            filename = self.name+"_pickle"
        
        num_inst=self._data.shape[0]
        num_rules=self._L.shape[0]

        x=self._data
        l=self._L

        m=np.zeros((num_inst, num_rules))
        L=self._truelabels
        d=np.ones((num_inst, 1))
        r=np.zeros((num_inst, num_rules))

        s=self._S
        n=np.array([lf._is_cont for lf in self._lfs], dtype=int)
        k=np.max(self._L, 1)                                        # nope

        to_dump = [x,l,m,L,d,r,s,n,k]
        f=open(filename, "wb")
        for item in to_dump:
            pickle.dump(item,f)






