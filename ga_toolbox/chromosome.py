from abc import ABC, abstractmethod
from typing import Any, Optional


class Chromosome(ABC):
    """Abstract Base Class representing a single candidate solution in the population."""

    length: int
    genes: list[Any]
    fitness: Optional[float]
    is_feasible: bool

    def __init__(self, length: int) -> None:
        pass

    @abstractmethod
    def initialize_random(self) -> None:
        """Populates genes randomly within valid representation constraints."""
        pass

    @abstractmethod
    def clone(self) -> "Chromosome":
        """Returns a deep copy of the chromosome."""
        pass


class BinaryChromosome(Chromosome):
    """Chromosome representation where genes are binary values (0 or 1)."""

    def __init__(self, length: int) -> None:
        pass

    def initialize_random(self) -> None:
        pass

    def clone(self) -> "BinaryChromosome":
        pass


class IntegerChromosome(Chromosome):
    """Chromosome representation where genes are integers or permutations."""

    lower_bound: int
    upper_bound: int
    is_permutation: bool

    def __init__(
        self,
        length: int,
        lower_bound: int,
        upper_bound: int,
        is_permutation: bool = False,
    ) -> None:
        pass

    def initialize_random(self) -> None:
        pass

    def clone(self) -> "IntegerChromosome":
        pass


class FloatingPointChromosome(Chromosome):
    """Chromosome representation where genes are continuous floating-point numbers."""

    lower_bound: float
    upper_bound: float

    def __init__(
        self,
        length: int,
        lower_bound: float,
        upper_bound: float,
    ) -> None:
        pass

    def initialize_random(self) -> None:
        pass

    def clone(self) -> "FloatingPointChromosome":
        pass
