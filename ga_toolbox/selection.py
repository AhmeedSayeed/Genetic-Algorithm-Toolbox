from abc import ABC, abstractmethod
import math
import random
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
        super().__init__()

    def select(
        self,
        population: list[Chromosome],
        num_parents: int,
        maximize: bool,
    ) -> list[Chromosome]:
        _validate_selection_inputs(population, num_parents)
        if num_parents == 0:
            return []

        if any(chrom.fitness is None for chrom in population):
            raise ValueError("All chromosomes must be evaluated before selection.")
        if any(
            not math.isfinite(chrom.fitness)
            for chrom in population
            if chrom.fitness is not None
        ):
            raise ValueError("Chromosome fitness values must be finite.")

        feasible_exists = any(chrom.is_feasible for chrom in population)
        eligible = (
            [chrom for chrom in population if chrom.is_feasible]
            if feasible_exists
            else population
        )
        oriented_fitness = [
            chrom.fitness if maximize else -chrom.fitness
            for chrom in eligible
        ]
        minimum = min(oriented_fitness)
        weights = [value - minimum + 1.0 for value in oriented_fitness]

        selected = random.choices(eligible, weights=weights, k=num_parents)
        return selected


class TournamentSelection(SelectionStrategy):
    """Tournament-based parent selection strategy."""

    k: int

    def __init__(self, k: int = 3) -> None:
        if k < 1:
            raise ValueError("k must be at least 1.")
        self.k = k

    def select(
        self,
        population: list[Chromosome],
        num_parents: int,
        maximize: bool,
    ) -> list[Chromosome]:
        _validate_selection_inputs(population, num_parents)
        if num_parents == 0:
            return []

        selected: list[Chromosome] = []
        tournament_size = min(self.k, len(population))
        for _ in range(num_parents):
            tournament = random.sample(population, tournament_size)
            winner = tournament[0]
            for candidate in tournament[1:]:
                if candidate.is_better_than(winner, maximize):
                    winner = candidate
            selected.append(winner)
        return selected


def _validate_selection_inputs(
    population: list[Chromosome], num_parents: int
) -> None:
    if not population and num_parents > 0:
        raise ValueError("population must not be empty when selecting parents.")
    if num_parents < 0:
        raise ValueError("num_parents must be non-negative.")
