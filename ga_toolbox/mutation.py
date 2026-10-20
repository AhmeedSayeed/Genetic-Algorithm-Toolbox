from abc import ABC, abstractmethod
from .chromosome import Chromosome


class MutationStrategy(ABC):
    """Abstract Base Class for genetic mutation algorithms."""

    @abstractmethod
    def mutate(
        self,
        chromosome: Chromosome,
        mutation_rate: float,
    ) -> Chromosome:
        """Applies random genetic variations to a chromosome according to the mutation rate."""
        pass


class BitFlipMutation(MutationStrategy):
    """Bit-flip mutation strategy for binary chromosomes."""

    def __init__(self) -> None:
        pass

    def mutate(
        self,
        chromosome: Chromosome,
        mutation_rate: float,
    ) -> Chromosome:
        pass


class SwapMutation(MutationStrategy):
    """Swap mutation strategy for integer and permutation chromosomes."""

    def __init__(self) -> None:
        pass

    def mutate(
        self,
        chromosome: Chromosome,
        mutation_rate: float,
    ) -> Chromosome:
        pass


class NonUniformMutation(MutationStrategy):
    """Non-uniform mutation strategy for floating-point chromosomes."""

    def __init__(self, b: float = 5.0, max_generations: int = 200) -> None:
        pass

    def mutate(
        self,
        chromosome: Chromosome,
        mutation_rate: float,
    ) -> Chromosome:
        pass
