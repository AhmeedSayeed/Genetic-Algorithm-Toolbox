"""Tests for the GAToolbox engine: validation, defaults, evaluation, the
evolutionary loop contract, reproducibility, and end-to-end optimization."""

import unittest

from ga_toolbox import (
    BinaryChromosome,
    BitFlipMutation,
    ElitistReplacement,
    FloatingPointChromosome,
    GAToolbox,
    IntegerChromosome,
    MultiPointCrossover,
    MutationStrategy,
    NonUniformMutation,
    OrderCrossover,
    ReplacementStrategy,
    SelectionStrategy,
    SwapMutation,
    TournamentSelection,
)


def onemax(chrom):
    return float(sum(chrom.genes))


def make_engine(**overrides):
    params = dict(
        chromosome_prototype=BinaryChromosome(8),
        fitness_fn=onemax,
        population_size=10,
        max_generations=5,
        seed=1,
    )
    params.update(overrides)
    return GAToolbox(**params)


# ------------------------------------------------------------ test doubles
class BrokenSelection(SelectionStrategy):
    def select(self, population, num_parents, maximize):
        return None


class ShortReplacement(ReplacementStrategy):
    def replace(self, current_pop, offspring_pop, maximize):
        return offspring_pop[:-1]


class RecordingMutation(MutationStrategy):
    """Wraps BitFlipMutation and records the generation passed on each call."""

    def __init__(self):
        self.inner = BitFlipMutation()
        self.generations = []

    def mutate(self, chromosome, mutation_rate, current_generation=0):
        self.generations.append(current_generation)
        return self.inner.mutate(chromosome, mutation_rate, current_generation)


class RecordingReplacement(ReplacementStrategy):
    """Wraps ElitistReplacement and records offspring sizes and evaluation state."""

    def __init__(self):
        self.inner = ElitistReplacement(elitism_count=1)
        self.offspring_sizes = []
        self.all_evaluated = True

    def replace(self, current_pop, offspring_pop, maximize):
        self.offspring_sizes.append(len(offspring_pop))
        if any(c.fitness is None for c in offspring_pop):
            self.all_evaluated = False
        return self.inner.replace(current_pop, offspring_pop, maximize)


# ------------------------------------------------------------------- tests
class TestValidation(unittest.TestCase):
    def test_rejects_invalid_hyperparameters(self):
        invalid = [
            dict(population_size=1),
            dict(max_generations=0),
            dict(crossover_rate=1.5),
            dict(crossover_rate=-0.1),
            dict(mutation_rate=1.1),
            dict(infeasibility_penalty=-1.0),
        ]
        for overrides in invalid:
            with self.subTest(**overrides), self.assertRaises(ValueError):
                make_engine(**overrides)

    def test_rejects_invalid_types(self):
        with self.assertRaises(TypeError):
            make_engine(chromosome_prototype=[0, 1, 0])
        with self.assertRaises(TypeError):
            make_engine(fitness_fn=42)
        with self.assertRaises(TypeError):
            make_engine(feasibility_fn="not callable")


class TestDefaultStrategies(unittest.TestCase):
    def test_binary_defaults(self):
        engine = make_engine(chromosome_prototype=BinaryChromosome(8))
        self.assertIsInstance(engine.crossover_strategy, MultiPointCrossover)
        self.assertIsInstance(engine.mutation_strategy, BitFlipMutation)
        self.assertIsInstance(engine.selection_strategy, TournamentSelection)
        self.assertIsInstance(engine.replacement_strategy, ElitistReplacement)

    def test_permutation_defaults(self):
        engine = make_engine(
            chromosome_prototype=IntegerChromosome(6, 1, 6, is_permutation=True)
        )
        self.assertIsInstance(engine.crossover_strategy, OrderCrossover)
        self.assertIsInstance(engine.mutation_strategy, SwapMutation)

    def test_plain_integer_defaults(self):
        engine = make_engine(chromosome_prototype=IntegerChromosome(6, 0, 9))
        self.assertIsInstance(engine.crossover_strategy, MultiPointCrossover)
        self.assertIsInstance(engine.mutation_strategy, SwapMutation)

    def test_float_defaults_sync_max_generations(self):
        engine = make_engine(
            chromosome_prototype=FloatingPointChromosome(3, -1.0, 1.0),
            max_generations=77,
        )
        self.assertIsInstance(engine.mutation_strategy, NonUniformMutation)
        self.assertEqual(engine.mutation_strategy.max_generations, 77)

    def test_injected_non_uniform_mutation_is_synced(self):
        mutation = NonUniformMutation(max_generations=200)
        engine = make_engine(
            chromosome_prototype=FloatingPointChromosome(3, -1.0, 1.0),
            mutation_strategy=mutation,
            max_generations=500,
        )
        self.assertEqual(mutation.max_generations, 500)
        engine.run()  # must not raise for generations > 200

    def test_short_chromosome_gets_safe_crossover_points(self):
        engine = make_engine(chromosome_prototype=BinaryChromosome(2))
        self.assertEqual(engine.crossover_strategy.num_points, 1)
        engine.run()


class TestEvaluatePopulation(unittest.TestCase):
    def _chrom(self, genes):
        chrom = BinaryChromosome(len(genes))
        chrom.genes = list(genes)
        return chrom

    def test_sets_fitness_and_feasibility(self):
        engine = make_engine(feasibility_fn=lambda c: sum(c.genes) <= 2)
        a, b = self._chrom([1, 1, 0, 0]), self._chrom([1, 1, 1, 1])
        engine.evaluate_population([a, b])
        self.assertEqual((a.fitness, a.is_feasible), (2.0, True))
        self.assertEqual((b.fitness, b.is_feasible), (4.0, False))

    def test_penalty_direction_follows_maximize(self):
        never_feasible = lambda c: False
        chrom = self._chrom([1, 1, 1, 0])

        make_engine(
            feasibility_fn=never_feasible, infeasibility_penalty=100.0, maximize=True
        ).evaluate_population([chrom])
        self.assertEqual(chrom.fitness, 3.0 - 100.0)

        make_engine(
            feasibility_fn=never_feasible, infeasibility_penalty=100.0, maximize=False
        ).evaluate_population([chrom])
        self.assertEqual(chrom.fitness, 3.0 + 100.0)

    def test_no_penalty_by_default(self):
        chrom = self._chrom([1, 0, 1, 0])
        make_engine(feasibility_fn=lambda c: False).evaluate_population([chrom])
        self.assertEqual(chrom.fitness, 2.0)
        self.assertFalse(chrom.is_feasible)

    def test_invalid_genes_are_infeasible_without_calling_feasibility_fn(self):
        calls = []
        engine = make_engine(feasibility_fn=lambda c: calls.append(c) or True)
        chrom = self._chrom([0, 1, 2, 0])  # 2 is not a valid bit
        engine.evaluate_population([chrom])
        self.assertFalse(chrom.is_feasible)
        self.assertEqual(calls, [])


class TestRunLoop(unittest.TestCase):
    def test_history_has_one_entry_per_generation_plus_initial(self):
        engine = make_engine(max_generations=12)
        engine.run()
        self.assertEqual(len(engine.history), 13)

    def test_best_so_far_history_never_gets_worse(self):
        engine = make_engine(max_generations=30)
        engine.run()
        self.assertEqual(engine.history, sorted(engine.history))
        engine = make_engine(fitness_fn=lambda c: -onemax(c), maximize=False, max_generations=30)
        engine.run()
        self.assertEqual(engine.history, sorted(engine.history, reverse=True))

    def test_same_seed_gives_same_result(self):
        first = make_engine(max_generations=20, seed=123)
        second = make_engine(max_generations=20, seed=123)
        best1, best2 = first.run(), second.run()
        self.assertEqual(best1.genes, best2.genes)
        self.assertEqual(first.history, second.history)

    def test_mutation_receives_every_generation_index(self):
        mutation = RecordingMutation()
        make_engine(mutation_strategy=mutation, population_size=6, max_generations=4).run()
        self.assertEqual(sorted(set(mutation.generations)), [0, 1, 2, 3])
        self.assertEqual(len(mutation.generations), 6 * 4)

    def test_odd_population_size_is_preserved(self):
        replacement = RecordingReplacement()
        make_engine(replacement_strategy=replacement, population_size=7, max_generations=5).run()
        self.assertEqual(replacement.offspring_sizes, [7] * 5)

    def test_offspring_are_evaluated_before_replacement(self):
        replacement = RecordingReplacement()
        make_engine(replacement_strategy=replacement).run()
        self.assertTrue(replacement.all_evaluated)

    def test_broken_selection_raises_clear_error(self):
        with self.assertRaises(RuntimeError) as ctx:
            make_engine(selection_strategy=BrokenSelection()).run()
        self.assertIn("BrokenSelection", str(ctx.exception))

    def test_wrong_size_replacement_raises_clear_error(self):
        with self.assertRaises(RuntimeError) as ctx:
            make_engine(replacement_strategy=ShortReplacement()).run()
        self.assertIn("ShortReplacement", str(ctx.exception))


class TestEndToEnd(unittest.TestCase):
    def test_constrained_binary_finds_feasible_optimum(self):
        engine = make_engine(
            chromosome_prototype=BinaryChromosome(10),
            feasibility_fn=lambda c: sum(c.genes) <= 7,
            population_size=30,
            max_generations=40,
            seed=42,
        )
        best = engine.run()
        self.assertTrue(best.is_feasible)
        self.assertEqual(best.fitness, 7.0)

    def test_float_sphere_minimization_converges(self):
        engine = make_engine(
            chromosome_prototype=FloatingPointChromosome(3, -5.0, 5.0),
            fitness_fn=lambda c: sum(x * x for x in c.genes),
            population_size=40,
            max_generations=60,
            maximize=False,
            seed=42,
        )
        best = engine.run()
        self.assertLess(best.fitness, 0.2)

    def test_permutation_sorting_reaches_identity(self):
        engine = make_engine(
            chromosome_prototype=IntegerChromosome(6, 1, 6, is_permutation=True),
            fitness_fn=lambda c: sum(abs(g - (i + 1)) for i, g in enumerate(c.genes)),
            population_size=30,
            max_generations=60,
            maximize=False,
            seed=42,
        )
        best = engine.run()
        self.assertEqual(best.genes, [1, 2, 3, 4, 5, 6])


if __name__ == "__main__":
    unittest.main()
