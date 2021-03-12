from typing import Any, Callable, List, Mapping, Optional
from labeling.types import DataPoint


class BaseContinuousScorer:
    def __init__(
        self,
        name: str,
        f: Callable[..., int],
        resources: Optional[Mapping[str, Any]] = None,
        # pre: Optional[List[BasePreprocessor]] = None,
    ) -> None:
        self.name = name
        self._f = f
        self._resources = resources or {}
    
    def __call__(self, x: DataPoint) -> float:
        return self._f(x,**self._resources)

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

    def __call__(self, f: Callable[..., int]) -> BaseContinuousScorer:
        name = self.name or f.__name__
        return BaseContinuousScorer(name=name, resources=self.resources, f=f)
