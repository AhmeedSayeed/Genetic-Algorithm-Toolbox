from abc import ABC, abstractmethod
from .chromosome import Chromosome


class SelectionStrategy(ABC):
    """Abstract Base Class for parent selection algorithms."""

    @abstractmethod
    def select(
        self,
        population: list[Chromosome],
        num_parents: int,
        maximize: bool,
    ) -> list[Chromosome]:
        """Selects parent chromosomes from the current population based on fitness."""
        pass


class RouletteWheelSelection(SelectionStrategy):
    """Fitness-proportionate (Roulette Wheel) parent selection strategy."""

    def __init__(self) -> None:
        pass

    def select(
        self,
        population: list[Chromosome],
        num_parents: int,
        maximize: bool,
    ) -> list[Chromosome]:
        pass


class TournamentSelection(SelectionStrategy):
    """Tournament-based parent selection strategy."""

    k: int

    def __init__(self, k: int = 3) -> None:
        pass

    def select(
        self,
        population: list[Chromosome],
        num_parents: int,
        maximize: bool,
    ) -> list[Chromosome]:
        pass
