from abc import ABC, abstractmethod
from .chromosome import Chromosome


class CrossoverStrategy(ABC):
    """Abstract Base Class for genetic recombination (crossover) algorithms."""

    @abstractmethod
    def crossover(
        self,
        parent1: Chromosome,
        parent2: Chromosome,
    ) -> tuple[Chromosome, Chromosome]:
        """Recombines two parent chromosomes into two offspring chromosomes."""
        pass


class MultiPointCrossover(CrossoverStrategy):
    """N-point crossover strategy for binary, integer, and floating-point chromosomes."""

    num_points: int

    def __init__(self, num_points: int = 2) -> None:
        pass

    def crossover(
        self,
        parent1: Chromosome,
        parent2: Chromosome,
    ) -> tuple[Chromosome, Chromosome]:
        pass


class OrderCrossover(CrossoverStrategy):
    """Order Crossover (OX) strategy for permutation-based integer chromosomes."""

    def __init__(self) -> None:
        pass

    def crossover(
        self,
        parent1: Chromosome,
        parent2: Chromosome,
    ) -> tuple[Chromosome, Chromosome]:
        pass
