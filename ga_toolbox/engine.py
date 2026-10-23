import random
from typing import Callable, Optional
from .chromosome import (
    BinaryChromosome,
    Chromosome,
    FloatingPointChromosome,
    IntegerChromosome,
)
from .crossover import CrossoverStrategy, MultiPointCrossover, OrderCrossover
from .mutation import (
    BitFlipMutation,
    MutationStrategy,
    NonUniformMutation,
    SwapMutation,
)
from .replacement import ElitistReplacement, ReplacementStrategy
from .selection import SelectionStrategy, TournamentSelection


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
    infeasibility_penalty: float
    seed: Optional[int]
    history: list[float]

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
        infeasibility_penalty: float = 0.0,
        seed: Optional[int] = None,
    ) -> None:
        if not isinstance(chromosome_prototype, Chromosome):
            raise TypeError("chromosome_prototype must be an instance of Chromosome.")
        if not callable(fitness_fn):
            raise TypeError("fitness_fn must be callable.")
        if feasibility_fn is not None and not callable(feasibility_fn):
            raise TypeError("feasibility_fn must be callable when provided.")
        if population_size < 2:
            raise ValueError("population_size must be at least 2.")
        if max_generations < 1:
            raise ValueError("max_generations must be at least 1.")
        if not 0.0 <= crossover_rate <= 1.0:
            raise ValueError("crossover_rate must be between 0.0 and 1.0.")
        if not 0.0 <= mutation_rate <= 1.0:
            raise ValueError("mutation_rate must be between 0.0 and 1.0.")
        if infeasibility_penalty < 0.0:
            raise ValueError("infeasibility_penalty must be non-negative.")

        self.chromosome_prototype = chromosome_prototype
        self.fitness_fn = fitness_fn
        self.feasibility_fn = (
            feasibility_fn if feasibility_fn is not None else lambda c: True
        )
        self.population_size = population_size
        self.max_generations = max_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.maximize = maximize
        self.infeasibility_penalty = infeasibility_penalty
        self.seed = seed
        self.history = []

        self.selection_strategy = (
            selection_strategy
            if selection_strategy is not None
            else TournamentSelection(k=min(3, population_size))
        )
        self.crossover_strategy = (
            crossover_strategy
            if crossover_strategy is not None
            else self._default_crossover_strategy(chromosome_prototype)
        )
        self.mutation_strategy = (
            mutation_strategy
            if mutation_strategy is not None
            else self._default_mutation_strategy(
                chromosome_prototype, max_generations
            )
        )
        if isinstance(self.mutation_strategy, NonUniformMutation):
            self.mutation_strategy.max_generations = self.max_generations

        self.replacement_strategy = (
            replacement_strategy
            if replacement_strategy is not None
            else ElitistReplacement(elitism_count=1)
        )

    @staticmethod
    def _default_crossover_strategy(prototype: Chromosome) -> CrossoverStrategy:
        if isinstance(prototype, IntegerChromosome) and prototype.is_permutation:
            return OrderCrossover()
        num_points = max(1, min(2, prototype.length - 1))
        return MultiPointCrossover(num_points=num_points)

    @staticmethod
    def _default_mutation_strategy(
        prototype: Chromosome, max_generations: int
    ) -> MutationStrategy:
        if isinstance(prototype, BinaryChromosome):
            return BitFlipMutation()
        if isinstance(prototype, IntegerChromosome):
            return SwapMutation()
        if isinstance(prototype, FloatingPointChromosome):
            return NonUniformMutation(max_generations=max_generations)
        return SwapMutation()

    def initialize_population(self) -> list[Chromosome]:
        """Creates the initial population using the chromosome prototype."""
        population: list[Chromosome] = []
        for _ in range(self.population_size):
            individual = self.chromosome_prototype.clone()
            individual.initialize_random()
            population.append(individual)
        return population

    def evaluate_population(self, population: list[Chromosome]) -> None:
        """Evaluates fitness and feasibility for chromosomes in the population."""
        for chrom in population:
            is_valid = chrom.validate()
            is_prob_feasible = bool(self.feasibility_fn(chrom)) if is_valid else False
            chrom.is_feasible = is_valid and is_prob_feasible

            raw_fitness = float(self.fitness_fn(chrom))
            if not chrom.is_feasible and self.infeasibility_penalty > 0.0:
                if self.maximize:
                    raw_fitness -= self.infeasibility_penalty
                else:
                    raw_fitness += self.infeasibility_penalty
            chrom.fitness = raw_fitness

    def _get_best(self, population: list[Chromosome]) -> Chromosome:
        """Returns the best chromosome in the population using Deb's feasibility-first rule."""
        if not population:
            raise ValueError("Cannot select the best chromosome from an empty population.")
        best = population[0]
        for candidate in population[1:]:
            if candidate.is_better_than(best, self.maximize):
                best = candidate
        return best

    def run(self) -> Chromosome:
        """Executes the genetic algorithm evolutionary loop and returns the best chromosome."""
        if self.seed is not None:
            random.seed(self.seed)

        self.history = []
        population = self.initialize_population()
        self.evaluate_population(population)

        best_chromosome = self._get_best(population).clone()
        assert best_chromosome.fitness is not None
        self.history.append(best_chromosome.fitness)

        for gen in range(self.max_generations):
            parents = self.selection_strategy.select(
                population, self.population_size, self.maximize
            )
            if not parents or len(parents) != self.population_size:
                raise RuntimeError(
                    f"{self.selection_strategy.__class__.__name__}.select() must return "
                    f"a list of {self.population_size} parents."
                )

            offspring_pop: list[Chromosome] = []
            for i in range(0, len(parents), 2):
                if i + 1 < len(parents):
                    if random.random() < self.crossover_rate:
                        child1, child2 = self.crossover_strategy.crossover(
                            parents[i], parents[i + 1]
                        )
                    else:
                        child1 = parents[i].clone()
                        child2 = parents[i + 1].clone()
                        child1.invalidate()
                        child2.invalidate()
                    offspring_pop.extend([child1, child2])
                else:
                    child1 = parents[i].clone()
                    child1.invalidate()
                    offspring_pop.append(child1)

            offspring_pop = offspring_pop[: self.population_size]
            offspring_pop = [
                self.mutation_strategy.mutate(
                    child, self.mutation_rate, current_generation=gen
                )
                for child in offspring_pop
            ]

            self.evaluate_population(offspring_pop)
            population = self.replacement_strategy.replace(
                population, offspring_pop, self.maximize
            )
            if not population or len(population) != self.population_size:
                raise RuntimeError(
                    f"{self.replacement_strategy.__class__.__name__}.replace() must return "
                    f"a population of size {self.population_size}."
                )

            current_best = self._get_best(population)
            if current_best.is_better_than(best_chromosome, self.maximize):
                best_chromosome = current_best.clone()

            assert best_chromosome.fitness is not None
            self.history.append(best_chromosome.fitness)

        return best_chromosome
