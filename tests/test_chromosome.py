"""Tests for chromosome representations: construction, initialization, cloning,
validation, invalidation, and feasibility-first comparison."""

import random
import unittest

from ga_toolbox import BinaryChromosome, FloatingPointChromosome, IntegerChromosome


def evaluated(chrom, fitness, feasible=True):
    chrom.fitness = fitness
    chrom.is_feasible = feasible
    return chrom


class TestConstruction(unittest.TestCase):
    def test_rejects_non_positive_length(self):
        with self.assertRaises(ValueError):
            BinaryChromosome(0)

    def test_rejects_inverted_bounds(self):
        with self.assertRaises(ValueError):
            IntegerChromosome(3, 5, 1)
        with self.assertRaises(ValueError):
            FloatingPointChromosome(3, 1.0, -1.0)

    def test_permutation_range_must_equal_length(self):
        with self.assertRaises(ValueError):
            IntegerChromosome(5, 1, 10, is_permutation=True)
        IntegerChromosome(5, 1, 5, is_permutation=True)  # exact range is fine

    def test_new_chromosome_is_unevaluated(self):
        chrom = BinaryChromosome(4)
        self.assertIsNone(chrom.fitness)
        self.assertTrue(chrom.is_feasible)


class TestInitialization(unittest.TestCase):
    def setUp(self):
        random.seed(0)

    def test_binary_genes_are_bits(self):
        for _ in range(200):
            chrom = BinaryChromosome(12)
            chrom.initialize_random()
            self.assertEqual(len(chrom.genes), 12)
            self.assertTrue(set(chrom.genes) <= {0, 1})

    def test_integer_genes_within_bounds(self):
        for _ in range(200):
            chrom = IntegerChromosome(10, -3, 4)
            chrom.initialize_random()
            self.assertTrue(all(-3 <= g <= 4 for g in chrom.genes))

    def test_permutation_contains_every_value_once(self):
        for _ in range(200):
            chrom = IntegerChromosome(8, 1, 8, is_permutation=True)
            chrom.initialize_random()
            self.assertEqual(sorted(chrom.genes), list(range(1, 9)))

    def test_float_genes_within_bounds(self):
        for _ in range(200):
            chrom = FloatingPointChromosome(6, -2.5, 2.5)
            chrom.initialize_random()
            self.assertTrue(all(-2.5 <= g <= 2.5 for g in chrom.genes))

    def test_initialize_resets_fitness(self):
        chrom = evaluated(BinaryChromosome(4), 3.0, feasible=False)
        chrom.initialize_random()
        self.assertIsNone(chrom.fitness)
        self.assertTrue(chrom.is_feasible)


class TestCloneAndInvalidate(unittest.TestCase):
    def _assert_independent_clone(self, chrom):
        chrom.initialize_random()
        evaluated(chrom, 7.0, feasible=False)
        clone = chrom.clone()

        self.assertIs(type(clone), type(chrom))
        self.assertEqual(clone.genes, chrom.genes)
        self.assertIsNot(clone.genes, chrom.genes)
        self.assertEqual(clone.fitness, 7.0)
        self.assertFalse(clone.is_feasible)

        original_genes = list(chrom.genes)
        clone.genes[0] = clone.genes[1]
        self.assertEqual(chrom.genes, original_genes)

    def test_binary_clone_is_deep(self):
        self._assert_independent_clone(BinaryChromosome(6))

    def test_integer_clone_is_deep_and_keeps_config(self):
        chrom = IntegerChromosome(6, 1, 6, is_permutation=True)
        self._assert_independent_clone(chrom)
        clone = chrom.clone()
        self.assertEqual(
            (clone.lower_bound, clone.upper_bound, clone.is_permutation),
            (1, 6, True),
        )

    def test_float_clone_is_deep_and_keeps_bounds(self):
        chrom = FloatingPointChromosome(6, -1.0, 1.0)
        self._assert_independent_clone(chrom)
        clone = chrom.clone()
        self.assertEqual((clone.lower_bound, clone.upper_bound), (-1.0, 1.0))

    def test_invalidate_resets_cached_state(self):
        chrom = evaluated(BinaryChromosome(4), 9.0, feasible=False)
        chrom.invalidate()
        self.assertIsNone(chrom.fitness)
        self.assertTrue(chrom.is_feasible)


class TestValidate(unittest.TestCase):
    def test_binary_rejects_non_bits_and_bools(self):
        chrom = BinaryChromosome(3)
        chrom.genes = [0, 1, 2]
        self.assertFalse(chrom.validate())
        chrom.genes = [0, True, 1]
        self.assertFalse(chrom.validate())
        chrom.genes = [0, 1, 1]
        self.assertTrue(chrom.validate())

    def test_rejects_wrong_length(self):
        chrom = BinaryChromosome(3)
        chrom.genes = [0, 1]
        self.assertFalse(chrom.validate())

    def test_integer_rejects_out_of_bounds(self):
        chrom = IntegerChromosome(3, 0, 5)
        chrom.genes = [0, 5, 6]
        self.assertFalse(chrom.validate())

    def test_permutation_rejects_duplicates(self):
        chrom = IntegerChromosome(4, 1, 4, is_permutation=True)
        chrom.genes = [1, 2, 2, 4]
        self.assertFalse(chrom.validate())
        chrom.genes = [4, 2, 3, 1]
        self.assertTrue(chrom.validate())

    def test_float_rejects_out_of_bounds(self):
        chrom = FloatingPointChromosome(2, -1.0, 1.0)
        chrom.genes = [0.5, 1.0001]
        self.assertFalse(chrom.validate())


class TestIsBetterThan(unittest.TestCase):
    def test_feasible_beats_infeasible_regardless_of_fitness(self):
        feasible = evaluated(BinaryChromosome(2), 1.0, feasible=True)
        infeasible = evaluated(BinaryChromosome(2), 1000.0, feasible=False)
        for maximize in (True, False):
            self.assertTrue(feasible.is_better_than(infeasible, maximize))
            self.assertFalse(infeasible.is_better_than(feasible, maximize))

    def test_fitness_direction_respects_maximize(self):
        low = evaluated(BinaryChromosome(2), 1.0)
        high = evaluated(BinaryChromosome(2), 5.0)
        self.assertTrue(high.is_better_than(low, maximize=True))
        self.assertTrue(low.is_better_than(high, maximize=False))

    def test_equal_chromosomes_are_not_better(self):
        a = evaluated(BinaryChromosome(2), 3.0)
        b = evaluated(BinaryChromosome(2), 3.0)
        self.assertFalse(a.is_better_than(b, maximize=True))
        self.assertFalse(b.is_better_than(a, maximize=True))

    def test_unevaluated_comparison_raises(self):
        a = evaluated(BinaryChromosome(2), 3.0)
        b = BinaryChromosome(2)
        with self.assertRaises(ValueError):
            a.is_better_than(b)


if __name__ == "__main__":
    unittest.main()
