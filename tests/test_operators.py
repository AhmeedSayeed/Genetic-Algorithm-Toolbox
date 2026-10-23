"""Behavioral tests for selection, crossover, mutation, and replacement operators."""

import random
import unittest
from unittest.mock import patch

from ga_toolbox import (
    BinaryChromosome,
    BitFlipMutation,
    ElitistReplacement,
    FloatingPointChromosome,
    GenerationalReplacement,
    IntegerChromosome,
    MultiPointCrossover,
    NonUniformMutation,
    OrderCrossover,
    RouletteWheelSelection,
    SwapMutation,
    TournamentSelection,
)


def make_binary(genes, fitness=None, feasible=True):
    chrom = BinaryChromosome(len(genes))
    chrom.genes = list(genes)
    chrom.fitness = fitness
    chrom.is_feasible = feasible
    return chrom


def make_perm(genes):
    chrom = IntegerChromosome(len(genes), 1, len(genes), is_permutation=True)
    chrom.genes = list(genes)
    return chrom


def population(fitness_feasible_pairs):
    return [make_binary([0, 0], f, feas) for f, feas in fitness_feasible_pairs]


# ---------------------------------------------------------------- selection
class TestTournamentSelection(unittest.TestCase):
    def setUp(self):
        random.seed(0)

    def test_returns_requested_number_of_parents(self):
        pop = population([(1.0, True), (2.0, True), (3.0, True)])
        self.assertEqual(len(TournamentSelection(k=2).select(pop, 7, True)), 7)

    def test_full_tournament_always_picks_best(self):
        pop = population([(1.0, True), (5.0, True), (3.0, True)])
        picked = TournamentSelection(k=3).select(pop, 20, maximize=True)
        self.assertTrue(all(c.fitness == 5.0 for c in picked))
        picked = TournamentSelection(k=3).select(pop, 20, maximize=False)
        self.assertTrue(all(c.fitness == 1.0 for c in picked))

    def test_prefers_feasible_over_fitter_infeasible(self):
        pop = population([(1.0, True), (100.0, False)])
        picked = TournamentSelection(k=2).select(pop, 20, maximize=True)
        self.assertTrue(all(c.is_feasible for c in picked))

    def test_rejects_invalid_k(self):
        with self.assertRaises(ValueError):
            TournamentSelection(k=0)


class TestRouletteWheelSelection(unittest.TestCase):
    def setUp(self):
        random.seed(0)

    def test_only_feasible_selected_when_any_exist(self):
        pop = population([(1.0, True), (2.0, True), (999.0, False)])
        picked = RouletteWheelSelection().select(pop, 200, maximize=True)
        self.assertTrue(all(c.is_feasible for c in picked))

    def test_falls_back_to_whole_population_when_none_feasible(self):
        pop = population([(1.0, False), (2.0, False)])
        picked = RouletteWheelSelection().select(pop, 50, maximize=True)
        self.assertEqual(len(picked), 50)

    def test_fitter_individual_selected_more_often_when_maximizing(self):
        pop = population([(1.0, True), (100.0, True)])
        picked = RouletteWheelSelection().select(pop, 1000, maximize=True)
        share = sum(c.fitness == 100.0 for c in picked) / 1000
        self.assertGreater(share, 0.9)

    def test_lower_fitness_selected_more_often_when_minimizing(self):
        pop = population([(1.0, True), (100.0, True)])
        picked = RouletteWheelSelection().select(pop, 1000, maximize=False)
        share = sum(c.fitness == 1.0 for c in picked) / 1000
        self.assertGreater(share, 0.9)

    def test_handles_negative_fitness(self):
        pop = population([(-50.0, True), (-10.0, True)])
        picked = RouletteWheelSelection().select(pop, 100, maximize=True)
        self.assertEqual(len(picked), 100)

    def test_rejects_unevaluated_population(self):
        pop = population([(1.0, True)]) + [make_binary([0, 1])]
        with self.assertRaises(ValueError):
            RouletteWheelSelection().select(pop, 2, maximize=True)


# ---------------------------------------------------------------- crossover
class TestMultiPointCrossover(unittest.TestCase):
    def setUp(self):
        random.seed(0)

    @staticmethod
    def _switches(genes):
        return sum(genes[i] != genes[i - 1] for i in range(1, len(genes)))

    def test_children_are_complementary_and_switch_num_points_times(self):
        p1, p2 = make_binary([0] * 10), make_binary([1] * 10)
        for num_points in (1, 2, 3):
            for _ in range(50):
                c1, c2 = MultiPointCrossover(num_points).crossover(p1, p2)
                self.assertTrue(all(a + b == 1 for a, b in zip(c1.genes, c2.genes)))
                self.assertEqual(self._switches(c1.genes), num_points)

    def test_does_not_modify_parents_and_invalidates_children(self):
        p1 = make_binary([0] * 6, fitness=1.0)
        p2 = make_binary([1] * 6, fitness=2.0)
        c1, c2 = MultiPointCrossover(2).crossover(p1, p2)
        self.assertEqual(p1.genes, [0] * 6)
        self.assertEqual(p2.genes, [1] * 6)
        self.assertIsNone(c1.fitness)
        self.assertIsNone(c2.fitness)

    def test_works_on_float_chromosomes(self):
        p1 = FloatingPointChromosome(5, 0.0, 1.0)
        p2 = FloatingPointChromosome(5, 0.0, 1.0)
        p1.genes, p2.genes = [0.0] * 5, [1.0] * 5
        c1, _ = MultiPointCrossover(1).crossover(p1, p2)
        self.assertTrue(c1.validate())
        self.assertEqual(set(c1.genes), {0.0, 1.0})

    def test_rejects_too_many_points(self):
        with self.assertRaises(ValueError):
            MultiPointCrossover(5).crossover(make_binary([0] * 5), make_binary([1] * 5))


class TestOrderCrossover(unittest.TestCase):
    def test_known_example_with_fixed_cut_points(self):
        p1 = make_perm([1, 2, 3, 4, 5, 6, 7, 8])
        p2 = make_perm([3, 7, 5, 1, 6, 8, 2, 4])
        # Force cut points 2..5 (inclusive).
        with patch("ga_toolbox.crossover.random.sample", return_value=[5, 2]):
            c1, c2 = OrderCrossover().crossover(p1, p2)
        # child1 keeps p1[2:6] = [3,4,5,6]; fills from p2 starting after pt2.
        self.assertEqual(c1.genes, [1, 8, 3, 4, 5, 6, 2, 7])
        # child2 keeps p2[2:6] = [5,1,6,8]; fills from p1 starting after pt2.
        self.assertEqual(c2.genes, [3, 4, 5, 1, 6, 8, 7, 2])

    def test_children_are_always_valid_permutations(self):
        random.seed(0)
        for _ in range(300):
            p1 = IntegerChromosome(9, 1, 9, is_permutation=True)
            p2 = IntegerChromosome(9, 1, 9, is_permutation=True)
            p1.initialize_random()
            p2.initialize_random()
            c1, c2 = OrderCrossover().crossover(p1, p2)
            self.assertEqual(sorted(c1.genes), list(range(1, 10)))
            self.assertEqual(sorted(c2.genes), list(range(1, 10)))

    def test_rejects_non_permutation_parents(self):
        with self.assertRaises(ValueError):
            OrderCrossover().crossover(make_binary([0, 1, 1]), make_binary([1, 0, 0]))


# ----------------------------------------------------------------- mutation
class TestBitFlipMutation(unittest.TestCase):
    def test_rate_zero_changes_nothing(self):
        chrom = make_binary([0, 1, 0, 1, 1])
        self.assertEqual(BitFlipMutation().mutate(chrom, 0.0).genes, [0, 1, 0, 1, 1])

    def test_rate_one_flips_every_bit_without_touching_original(self):
        chrom = make_binary([0, 1, 0, 1, 1], fitness=3.0)
        mutated = BitFlipMutation().mutate(chrom, 1.0)
        self.assertEqual(mutated.genes, [1, 0, 1, 0, 0])
        self.assertEqual(chrom.genes, [0, 1, 0, 1, 1])
        self.assertIsNone(mutated.fitness)

    def test_rejects_wrong_chromosome_type(self):
        with self.assertRaises(TypeError):
            BitFlipMutation().mutate(make_perm([1, 2, 3]), 0.5)


class TestSwapMutation(unittest.TestCase):
    def test_rate_zero_changes_nothing(self):
        self.assertEqual(SwapMutation().mutate(make_perm([1, 2, 3, 4]), 0.0).genes, [1, 2, 3, 4])

    def test_preserves_permutation_and_changes_order(self):
        random.seed(0)
        chrom = make_perm(list(range(1, 11)))
        mutated = SwapMutation().mutate(chrom, 1.0)
        self.assertEqual(sorted(mutated.genes), list(range(1, 11)))
        self.assertNotEqual(mutated.genes, chrom.genes)
        self.assertEqual(chrom.genes, list(range(1, 11)))

    def test_rejects_wrong_chromosome_type(self):
        with self.assertRaises(TypeError):
            SwapMutation().mutate(make_binary([0, 1]), 0.5)


class TestNonUniformMutation(unittest.TestCase):
    def _float_chrom(self, genes, lb=-5.0, ub=5.0):
        chrom = FloatingPointChromosome(len(genes), lb, ub)
        chrom.genes = list(genes)
        return chrom

    def test_rate_zero_changes_nothing(self):
        chrom = self._float_chrom([1.0, -2.0])
        mutated = NonUniformMutation(max_generations=10).mutate(chrom, 0.0, 0)
        self.assertEqual(mutated.genes, [1.0, -2.0])

    def test_always_stays_within_bounds(self):
        random.seed(0)
        mutation = NonUniformMutation(b=2.0, max_generations=100)
        for gen in (0, 25, 50, 99):
            for _ in range(500):
                chrom = self._float_chrom([random.uniform(-5, 5) for _ in range(4)])
                mutated = mutation.mutate(chrom, 1.0, current_generation=gen)
                self.assertTrue(mutated.validate(), mutated.genes)

    def test_step_size_shrinks_over_generations(self):
        random.seed(0)
        mutation = NonUniformMutation(b=5.0, max_generations=100)

        def mean_step(gen):
            total = 0.0
            for _ in range(2000):
                chrom = self._float_chrom([0.0])
                total += abs(mutation.mutate(chrom, 1.0, gen).genes[0])
            return total / 2000

        early, late = mean_step(0), mean_step(90)
        self.assertGreater(early, 10 * late)

    def test_no_change_at_final_generation(self):
        chrom = self._float_chrom([0.5, -0.5])
        mutated = NonUniformMutation(max_generations=50).mutate(chrom, 1.0, 50)
        self.assertEqual(mutated.genes, [0.5, -0.5])

    def test_rejects_generation_beyond_max(self):
        with self.assertRaises(ValueError):
            NonUniformMutation(max_generations=10).mutate(self._float_chrom([0.0]), 0.5, 11)

    def test_rejects_wrong_chromosome_type(self):
        with self.assertRaises(TypeError):
            NonUniformMutation().mutate(make_binary([0, 1]), 0.5)


# -------------------------------------------------------------- replacement
class TestGenerationalReplacement(unittest.TestCase):
    def test_returns_exactly_the_offspring(self):
        current = population([(9.0, True)] * 3)
        offspring = population([(1.0, True)] * 3)
        survivors = GenerationalReplacement().replace(current, offspring, True)
        self.assertEqual([id(c) for c in survivors], [id(c) for c in offspring])

    def test_rejects_size_mismatch(self):
        with self.assertRaises(ValueError):
            GenerationalReplacement().replace(population([(1.0, True)] * 3), population([(1.0, True)] * 2), True)


class TestElitistReplacement(unittest.TestCase):
    def test_elite_survives_and_rest_is_best_of_combined_pool(self):
        current = population([(0.0, True), (1.0, True), (2.0, True), (3.0, True)])
        offspring = population([(10.0, True), (11.0, True), (12.0, True), (13.0, True)])
        survivors = ElitistReplacement(elitism_count=1).replace(current, offspring, True)
        self.assertEqual([c.fitness for c in survivors], [3.0, 13.0, 12.0, 11.0])

    def test_minimization_keeps_lowest_fitness(self):
        current = population([(5.0, True), (9.0, True)])
        offspring = population([(7.0, True), (8.0, True)])
        survivors = ElitistReplacement(elitism_count=1).replace(current, offspring, False)
        self.assertEqual([c.fitness for c in survivors], [5.0, 7.0])

    def test_feasible_survives_over_fitter_infeasible(self):
        current = population([(1.0, True), (2.0, True)])
        offspring = population([(500.0, False), (600.0, False)])
        survivors = ElitistReplacement(elitism_count=1).replace(current, offspring, True)
        self.assertTrue(all(c.is_feasible for c in survivors))

    def test_preserves_population_size(self):
        current = population([(float(i), True) for i in range(7)])
        offspring = population([(float(i), True) for i in range(7)])
        self.assertEqual(len(ElitistReplacement(3).replace(current, offspring, True)), 7)

    def test_rejects_negative_elitism(self):
        with self.assertRaises(ValueError):
            ElitistReplacement(elitism_count=-1)


if __name__ == "__main__":
    unittest.main()
