from abc import ABC, abstractmethod
from enum import Enum


class ProblemKind(Enum):
    LINEAR = "linear"
    NONLINEAR = "nonlinear"
    TRANSIENT = "transient"


class Problem(ABC):
    """Mirrors include/problem.hpp. Only the method matching `kind` gets
    called by runner.py, same convention as main.cpp's switch."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def kind(self) -> ProblemKind: ...

    def assemble_linear(self):
        raise NotImplementedError(f"{self.name} does not implement assemble_linear")

    def assemble_nonlinear(self):
        raise NotImplementedError(f"{self.name} does not implement assemble_nonlinear")

    def assemble_transient(self):
        raise NotImplementedError(f"{self.name} does not implement assemble_transient")

    def transient_l2_error(self, x, time: float):
        return None
