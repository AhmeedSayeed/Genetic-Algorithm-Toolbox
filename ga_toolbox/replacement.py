from abc import ABC, abstractmethod
from .chromosome import Chromosome


class ReplacementStrategy(ABC):
    """Abstract Base Class for population replacement (survivor selection) algorithms."""

    @abstractmethod
    def replace(
        self,
        current_pop: list[Chromosome],
        offspring_pop: list[Chromosome],
        maximize: bool,
    ) -> list[Chromosome]:
        """Determines which individuals from current and offspring populations survive."""
        pass


class GenerationalReplacement(ReplacementStrategy):
    """Generational replacement strategy where offspring replace the parent population."""

    def __init__(self) -> None:
        pass

    def replace(
        self,
        current_pop: list[Chromosome],
        offspring_pop: list[Chromosome],
        maximize: bool,
    ) -> list[Chromosome]:
        pass


class ElitistReplacement(ReplacementStrategy):
    """Elitist replacement strategy preserving top individuals across generations."""

    elitism_count: int

    def __init__(self, elitism_count: int = 1) -> None:
        pass

    def replace(
        self,
        current_pop: list[Chromosome],
        offspring_pop: list[Chromosome],
        maximize: bool,
    ) -> list[Chromosome]:
        pass
