import random
from abc import ABC, abstractmethod
from .chromosome import Chromosome, BinaryChromosome, IntegerChromosome, FloatingPointChromosome


class MutationStrategy(ABC):
    """Abstract Base Class for genetic mutation algorithms."""

    @abstractmethod
    def mutate(
        self,
        chromosome: Chromosome,
        mutation_rate: float,
        current_generation: int = 0,
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
        current_generation: int = 0,
    ) -> Chromosome:
        if not isinstance(chromosome, BinaryChromosome):
            raise TypeError("BitFlipMutation requires a BinaryChromosome.")

        if not 0.0 <= mutation_rate <= 1.0:
            raise ValueError("mutation_rate must be between 0 and 1.")

        mutated = chromosome.clone()

        for i in range(mutated.length):
            if random.random() < mutation_rate:
                mutated.genes[i] = 1 - mutated.genes[i]

        mutated.invalidate()
        mutated.is_feasible = mutated.validate()

        return mutated


class SwapMutation(MutationStrategy):
    """Swap mutation strategy for integer and permutation chromosomes."""

    def __init__(self) -> None:
        pass

    def mutate(
        self,
        chromosome: Chromosome,
        mutation_rate: float,
        current_generation: int = 0,
    ) -> Chromosome:

        if not isinstance(chromosome, IntegerChromosome):
            raise TypeError("SwapMutation requires an IntegerChromosome.")

        if not 0.0 <= mutation_rate <= 1.0:
            raise ValueError("mutation_rate must be between 0 and 1.")

        mutated = chromosome.clone()

        if mutated.length > 1:
            for i in range(mutated.length):
                if random.random() < mutation_rate:
                    j = random.randrange(mutated.length)
                    while j == i:
                        j = random.randrange(mutated.length)

                    mutated.genes[i], mutated.genes[j] = mutated.genes[j], mutated.genes[i]

        mutated.invalidate()
        mutated.is_feasible = mutated.validate()

        return mutated


class NonUniformMutation(MutationStrategy):
    """Non-uniform mutation strategy for floating-point chromosomes."""

    def __init__(self, b: float = 5.0, max_generations: int = 200) -> None:
        if b <= 0:
            raise ValueError("b must be greater than 0.")

        if max_generations <= 0:
            raise ValueError("max_generations must be greater than 0.")

        self.b = b
        self.max_generations = max_generations

    def mutate(
        self,
        chromosome: Chromosome,
        mutation_rate: float,
        current_generation: int = 0,
    ) -> Chromosome:

        if not isinstance(chromosome, FloatingPointChromosome):
            raise TypeError(
                "NonUniformMutation requires a FloatingPointChromosome."
            )

        if not 0.0 <= mutation_rate <= 1.0:
            raise ValueError("mutation_rate must be between 0 and 1.")

        if not 0 <= current_generation <= self.max_generations:
            raise ValueError(
                "current_generation must be between 0 and max_generations."
            )

        exponent = (1 - current_generation / self.max_generations) ** self.b

        mutated = chromosome.clone()

        for i in range(mutated.length):

            if random.random() < mutation_rate:
                if random.random() <= 0.5:
                    y = mutated.upper_bound - mutated.genes[i]
                    direction = 1
                else:
                    y = mutated.genes[i] - mutated.lower_bound
                    direction = -1

                r = random.random()

                delta = y * (1 - r ** exponent)

                mutated.genes[i] += direction * delta

        mutated.invalidate()
        mutated.is_feasible = mutated.validate()

        return mutated