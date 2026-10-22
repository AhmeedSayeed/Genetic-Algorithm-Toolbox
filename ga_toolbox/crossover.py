import random
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
        if num_points < 1:
            raise ValueError("num_points must be at least 1.")
        self.num_points = num_points

    def crossover(
        self,
        parent1: Chromosome,
        parent2: Chromosome,
    ) -> tuple[Chromosome, Chromosome]:
        if len(parent1.genes) != len(parent2.genes):
            raise ValueError("Parent chromosomes must have equal length.")

        chrom_len = len(parent1.genes)
        if chrom_len < 2:
            return parent1.clone(), parent2.clone()

        if self.num_points >= chrom_len:
            raise ValueError(
                f"num_points ({self.num_points}) must be strictly less than chromosome length ({chrom_len})."
            )

        crossover_points = random.sample(range(1, chrom_len), self.num_points)
        crossover_points.sort()

        points_before = []
        count = 0
        for i in range(chrom_len):
            if i in crossover_points:
                count += 1
            points_before.append(count)

        offspring1_genes = []
        offspring2_genes = []
        for i in range(chrom_len):
            if points_before[i] % 2 == 0:
                offspring1_genes.append(parent1.genes[i])
                offspring2_genes.append(parent2.genes[i])
            else:
                offspring1_genes.append(parent2.genes[i])
                offspring2_genes.append(parent1.genes[i])

        child1 = parent1.clone()
        child1.genes = offspring1_genes
        child1.invalidate()

        child2 = parent2.clone()
        child2.genes = offspring2_genes
        child2.invalidate()

        return child1, child2


class OrderCrossover(CrossoverStrategy):
    """Order Crossover (OX) strategy for permutation-based integer chromosomes."""

    def __init__(self) -> None:
        pass

    def crossover(
        self,
        parent1: Chromosome,
        parent2: Chromosome,
    ) -> tuple[Chromosome, Chromosome]:
        if len(parent1.genes) != len(parent2.genes):
            raise ValueError("Parent chromosomes must have equal length.")

        chrom_len = len(parent1.genes)
        if chrom_len < 2:
            return parent1.clone(), parent2.clone()

        if len(set(parent1.genes)) != chrom_len or len(set(parent2.genes)) != chrom_len:
            raise ValueError(
                "Order crossover requires permutation chromosomes with unique genes."
            )

        # Select two distinct cut points: 0 <= pt1 < pt2 < chrom_len
        pt1, pt2 = sorted(random.sample(range(chrom_len), 2))

        parents = [parent1.genes, parent2.genes]
        offspring_genes = [[None] * chrom_len for _ in range(2)]

        for l in range(2):
            primary = parents[l]
            donor = parents[1 - l]

            copied_segment = primary[pt1 : pt2 + 1]
            offspring_genes[l][pt1 : pt2 + 1] = copied_segment
            copied_set = set(copied_segment)

            child_idx = (pt2 + 1) % chrom_len
            donor_idx = (pt2 + 1) % chrom_len
            remaining = chrom_len - len(copied_segment)

            for _ in range(remaining):
                while donor[donor_idx] in copied_set:
                    donor_idx = (donor_idx + 1) % chrom_len

                offspring_genes[l][child_idx] = donor[donor_idx]
                child_idx = (child_idx + 1) % chrom_len
                donor_idx = (donor_idx + 1) % chrom_len

        child1 = parent1.clone()
        child1.genes = offspring_genes[0]
        child1.invalidate()

        child2 = parent2.clone()
        child2.genes = offspring_genes[1]
        child2.invalidate()

        return child1, child2
