from abc import ABC, abstractmethod
from functools import cmp_to_key
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
        super().__init__()

    def replace(
        self,
        current_pop: list[Chromosome],
        offspring_pop: list[Chromosome],
        maximize: bool,
    ) -> list[Chromosome]:
        if not current_pop:
            raise ValueError("current_pop must not be empty.")
        if len(offspring_pop) != len(current_pop):
            raise ValueError(
                "offspring_pop must have the same size as current_pop."
            )
        return list(offspring_pop)


class ElitistReplacement(ReplacementStrategy):
    """Elitist replacement strategy preserving top individuals across generations."""

    elitism_count: int

    def __init__(self, elitism_count: int = 1) -> None:
        if elitism_count < 0:
            raise ValueError("elitism_count must be non-negative.")
        self.elitism_count = elitism_count

    def replace(
        self,
        current_pop: list[Chromosome],
        offspring_pop: list[Chromosome],
        maximize: bool,
    ) -> list[Chromosome]:
        if not current_pop:
            raise ValueError("current_pop must not be empty.")
        if len(offspring_pop) != len(current_pop):
            raise ValueError(
                "offspring_pop must have the same size as current_pop."
            )

        elite_count = min(self.elitism_count, len(current_pop))
        ranked_current = _ranked(current_pop, maximize)
        elites = ranked_current[:elite_count]

        remaining_candidates = offspring_pop + ranked_current[elite_count:]
        ranked_remaining = _ranked(remaining_candidates, maximize)
        return elites + ranked_remaining[: len(current_pop) - elite_count]


def _ranked(
    population: list[Chromosome], maximize: bool
) -> list[Chromosome]:
    if any(chrom.fitness is None for chrom in population):
        raise ValueError("All chromosomes must be evaluated before replacement.")

    def compare(left: Chromosome, right: Chromosome) -> int:
        if left.is_better_than(right, maximize):
            return -1
        if right.is_better_than(left, maximize):
            return 1
        return 0

    return sorted(population, key=cmp_to_key(compare))
