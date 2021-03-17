from typing import Any, Callable, List, Mapping, Optional
from labeling.types import DataPoint


class BaseContinuousScorer:
    def __init__(
        self,
        name: str,
        cf: Callable[..., int],
        resources: Optional[Mapping[str, Any]] = None,
        # pre: Optional[List[BasePreprocessor]] = None,
    ) -> None:
        self.name = name
        self._cf = cf
        self._resources = resources or {}
    
    def __call__(self, x: DataPoint, **kwargs) -> float:
        return self._cf(x,**self._resources, **kwargs)

    def __repr__(self) -> str:
        return f"{type(self).__name__} {self.name}"

class continuous_scorer:
    def __init__(
        self,
        name: Optional[str] = None,
        resources: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if callable(name):
            raise ValueError("Looks like this decorator is missing parentheses!")
        self.name = name
        self.resources = resources

    def __call__(self, cf: Callable[..., int]) -> BaseContinuousScorer:
        name = self.name or cf.__name__
        return BaseContinuousScorer(name=name, resources=self.resources, cf=cf)
