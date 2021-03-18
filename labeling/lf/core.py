from typing import Any, Callable, List, Mapping, Optional
from labeling.types import DataPoint
from labeling.preprocess import BasePreprocessor
from labeling.continuous_scoring import BaseContinuousScorer
class LabelingFunction:
    def __init__(
        self,
        name: str,
        f: Callable[..., int],
        resources: Optional[Mapping[str, Any]] = None,
        pre: Optional[List[BasePreprocessor]] = None,
        cont_scorer: Optional[BaseContinuousScorer] = None,
    ) -> None:
        self.name = name
        self._f = f
        self._resources = resources or {}
        self._pre = pre or []
        self._cont_scorer = cont_scorer

    def _preprocess_data_point(self, x: DataPoint) -> DataPoint:
        for preprocessor in self._pre:
            x = preprocessor(x)
            if x is None:
                raise ValueError("Preprocessor should not return None")
        return x

    def __call__(self, x: DataPoint) -> (int, float):
        x = self._preprocess_data_point(x)
        if self._cont_scorer is None:
            return self._f(x,**self._resources), -1.0            
        return self._f(x,**self._resources), self._cont_scorer(x,**self._resources) 

    def __repr__(self) -> str:
        preprocessor_str = f", Preprocessors: {self._pre}"
        return f"{type(self).__name__} {self.name}{preprocessor_str}"


class labeling_function:
    def __init__(
        self,
        name: Optional[str] = None,
        resources: Optional[Mapping[str, Any]] = None,
        pre: Optional[List[BasePreprocessor]] = None,
        cont_scorer: Optional[BaseContinuousScorer] = None,
    ) -> None:
        if callable(name):
            raise ValueError("Looks like this decorator is missing parentheses!")
        self.name = name
        self.resources = resources
        self.pre = pre
        self.cont_scorer = cont_scorer

    def __call__(self, f: Callable[..., int]) -> LabelingFunction:
        name = self.name or f.__name__
        return LabelingFunction(name=name, resources=self.resources, f=f, pre=self.pre, cont_scorer=self.cont_scorer)
