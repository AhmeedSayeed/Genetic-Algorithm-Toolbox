import random
from abc import ABC, abstractmethod
from typing import Any, Optional


class Chromosome(ABC):
    """Abstract Base Class representing a single candidate solution in the population."""

    length: int
    genes: list[Any]
    fitness: Optional[float]
    is_feasible: bool

    def __init__(self, length: int) -> None:
        if length <= 0:
            raise ValueError("Chromosome length must be a positive integer.")
        self.length = length
        self.genes = []
        self.fitness = None
        self.is_feasible = True

    def invalidate(self) -> None:
        """Resets cached fitness and feasibility after genes are modified."""
        self.fitness = None
        self.is_feasible = True

    def is_better_than(self, other: "Chromosome", maximize: bool = True) -> bool:
        """Compares two evaluated chromosomes using Deb's feasibility-first rule."""
        if self.fitness is None or other.fitness is None:
            raise ValueError("Both chromosomes must be evaluated before comparison.")
        if self.is_feasible != other.is_feasible:
            return self.is_feasible
        return (
            self.fitness > other.fitness
            if maximize
            else self.fitness < other.fitness
        )

    @abstractmethod
    def initialize_random(self) -> None:
        """Populates genes randomly within valid representation constraints."""
        pass

    @abstractmethod
    def clone(self) -> "Chromosome":
        """Returns a deep copy of the chromosome."""
        pass

    @abstractmethod
    def validate(self) -> bool:
        """Returns True if genes satisfy all representation and bound constraints."""
        pass

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(genes={self.genes}, "
            f"fitness={self.fitness}, is_feasible={self.is_feasible})"
        )


class BinaryChromosome(Chromosome):
    """Chromosome representation where genes are binary values (0 or 1)."""

    def __init__(self, length: int) -> None:
        super().__init__(length)

    def initialize_random(self) -> None:
        self.genes = [random.randint(0, 1) for _ in range(self.length)]
        self.invalidate()

    def clone(self) -> "BinaryChromosome":
        copy_chrom = BinaryChromosome(self.length)
        copy_chrom.genes = self.genes.copy()
        copy_chrom.fitness = self.fitness
        copy_chrom.is_feasible = self.is_feasible
        return copy_chrom

    def validate(self) -> bool:
        if len(self.genes) != self.length:
            return False
        return all(
            isinstance(g, int) and not isinstance(g, bool) and g in (0, 1)
            for g in self.genes
        )


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
        super().__init__(length)
        if lower_bound > upper_bound:
            raise ValueError("lower_bound cannot be greater than upper_bound.")
        if is_permutation and (upper_bound - lower_bound + 1) != length:
            raise ValueError(
                "For a permutation chromosome, (upper_bound - lower_bound + 1) must equal length."
            )
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.is_permutation = is_permutation

    def initialize_random(self) -> None:
        if self.is_permutation:
            self.genes = random.sample(
                range(self.lower_bound, self.upper_bound + 1), self.length
            )
        else:
            self.genes = [
                random.randint(self.lower_bound, self.upper_bound)
                for _ in range(self.length)
            ]
        self.invalidate()

    def clone(self) -> "IntegerChromosome":
        copy_chrom = IntegerChromosome(
            self.length, self.lower_bound, self.upper_bound, self.is_permutation
        )
        copy_chrom.genes = self.genes.copy()
        copy_chrom.fitness = self.fitness
        copy_chrom.is_feasible = self.is_feasible
        return copy_chrom

    def validate(self) -> bool:
        if len(self.genes) != self.length:
            return False
        in_bounds = all(
            isinstance(g, int)
            and not isinstance(g, bool)
            and self.lower_bound <= g <= self.upper_bound
            for g in self.genes
        )
        if not in_bounds:
            return False
        if self.is_permutation and len(set(self.genes)) != self.length:
            return False
        return True


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
        super().__init__(length)
        if lower_bound > upper_bound:
            raise ValueError("lower_bound cannot be greater than upper_bound.")
        self.lower_bound = float(lower_bound)
        self.upper_bound = float(upper_bound)

    def initialize_random(self) -> None:
        self.genes = [
            random.uniform(self.lower_bound, self.upper_bound)
            for _ in range(self.length)
        ]
        self.invalidate()

    def clone(self) -> "FloatingPointChromosome":
        copy_chrom = FloatingPointChromosome(
            self.length, self.lower_bound, self.upper_bound
        )
        copy_chrom.genes = self.genes.copy()
        copy_chrom.fitness = self.fitness
        copy_chrom.is_feasible = self.is_feasible
        return copy_chrom

    def validate(self) -> bool:
        if len(self.genes) != self.length:
            return False
        return all(
            isinstance(g, (int, float))
            and not isinstance(g, bool)
            and self.lower_bound <= float(g) <= self.upper_bound
            for g in self.genes
        )
