from typing import Callable, Optional
from .chromosome import Chromosome
from .selection import SelectionStrategy
from .crossover import CrossoverStrategy
from .mutation import MutationStrategy
from .replacement import ReplacementStrategy


class GAToolbox:
    """High-level Genetic Algorithm engine orchestrating chromosomes, strategies, and evaluation."""

    chromosome_prototype: Chromosome
    fitness_fn: Callable[[Chromosome], float]
    feasibility_fn: Callable[[Chromosome], bool]
    selection_strategy: SelectionStrategy
    crossover_strategy: CrossoverStrategy
    mutation_strategy: MutationStrategy
    replacement_strategy: ReplacementStrategy
    population_size: int
    max_generations: int
    crossover_rate: float
    mutation_rate: float
    maximize: bool

    def __init__(
        self,
        chromosome_prototype: Chromosome,
        fitness_fn: Callable[[Chromosome], float],
        feasibility_fn: Optional[Callable[[Chromosome], bool]] = None,
        selection_strategy: Optional[SelectionStrategy] = None,
        crossover_strategy: Optional[CrossoverStrategy] = None,
        mutation_strategy: Optional[MutationStrategy] = None,
        replacement_strategy: Optional[ReplacementStrategy] = None,
        population_size: int = 100,
        max_generations: int = 200,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.05,
        maximize: bool = True,
    ) -> None:
        pass

    def initialize_population(self) -> list[Chromosome]:
        """Creates the initial population using the chromosome prototype."""
        pass

    def evaluate_population(self, population: list[Chromosome]) -> None:
        """Evaluates fitness and feasibility for chromosomes in the population."""
        pass

    def run(self) -> Chromosome:
        """Executes the genetic algorithm evolutionary loop and returns the best chromosome."""
        pass
