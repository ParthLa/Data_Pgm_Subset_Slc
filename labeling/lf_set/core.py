from typing import Any, Callable, List, Set, Mapping, Optional
from labeling.types import DataPoint, DataPoints
from labeling.apply import *
from labeling.lf import *
import warnings

import pickle
import numpy as np

class LFSet:              # LFSet
    """Class for Set of Labeling Functions
    """
    def __init__(
        self,
        name: str,
        lfs: List[LabelingFunction] = [],
    ) -> None:
        """Instantiates LFSet class with list of labeling functions

        Args:
            name (str): Name for this LFset.
            lfs (List[LabelingFunction], optional): List of LFs to add to this object. Defaults to [].
        """
        self.name = name
        self._lfs = set(lfs)

    def get_lfs(
        self,
    ) -> Set[LabelingFunction]:
        """Returns LFs contained in this LFSet object

        Returns:
            Set[LabelingFunction]: LFs in this LFSet
        """
        return self._lfs
        
    def add_lf(
        self,
        lf: LabelingFunction,
    ) -> None:
        """Adds single LF to this LFSet

        Args:
            lf (LabelingFunction): LF to add
        """
        if lf in self._lfs:
            warnings.warn('Attempting to add duplicate LF to LFset') 
        else:
            self._lfs.add(lf)
    
    def add_lf_list(
        self,
        lf_list: List[LabelingFunction],
    ) -> None:
        """Adds a list of LFs to this LFSet

        Args:
            lf_list (List[LabelingFunction]): List of LFs to add to this LFSet
        """
        if len(self._lfs.intersection(lf_list))>0:
            warnings.warn('Attempting to add duplicate LF to LFset') 
        self._lfs = self._lfs.union(lf_list)
    
    def remove_lf(
        self,
        lf: LabelingFunction
    ) -> None:
        """Removes a LF from this set

        Args:
            lf (LabelingFunction): LF to remove from this set

        Raises:
            ValueError: If LF not already in LFset
        """
        if lf in self._lfs:
            self._lfs.remove(lf)
        else:
            raise ValueError("Trying to remove an LF not in this LF set!")
    
