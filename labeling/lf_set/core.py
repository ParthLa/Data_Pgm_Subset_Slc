from typing import Any, Callable, List, Set, Mapping, Optional
from labeling.types import DataPoint, DataPoints
from labeling.apply import *
from labeling.lf import *
import warnings

import pickle
import numpy as np

class LFSet:              # LFSet
    def __init__(
        self,
        name: str,
        lfs: List[LabelingFunction] = [],
    ) -> None:
        self.name = name
        self._lfs = set(lfs)

    def get_lfs(
        self,
    ) -> Set[LabelingFunction]:
        return self._lfs
        
    def add_lf(
        self,
        lf: LabelingFunction,
    ) -> None:
        if lf in self._lfs:
            warnings.warn('Attempting to add duplicate LF to LFset') 
        else:
            self._lfs.add(lf)
    
    def add_lf_list(
        self,
        lf_list: List[LabelingFunction],
    ) -> None:
        if len(self._lfs.intersection(lf_list))>0:
            warnings.warn('Attempting to add duplicate LF to LFset') 
        self._lfs = self._lfs.union(lf_list)
    
    def remove_lf(
        self,
        lf: LabelingFunction
    ) -> None:
        if lf in self._lfs:
            self._lfs.remove(lf)
        else:
            raise ValueError("Trying to remove an LF not in this LF set!")
    
