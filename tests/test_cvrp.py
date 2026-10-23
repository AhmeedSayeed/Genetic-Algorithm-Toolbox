"""Tests for the CVRP application: loading, decoding, feasibility, fitness,
output formatting, and an end-to-end seeded solve."""

import random
import unittest
from pathlib import Path

from cvrp import CVRPProblem
from ga_toolbox import IntegerChromosome

INPUT_PATH = Path(__file__).resolve().parent.parent / "cvrp" / "cvrp_input.json"

# Routes from the Google OR-Tools reference solution (total distance 6208m).
REFERENCE_ROUTES = [[13, 15, 11, 12], [5, 2, 6, 8], [10, 16, 14, 9], [7, 1, 3, 4]]
REFERENCE_DISTANCE = 6208


def permutation(genes):
    chrom = IntegerChromosome(16, 1, 16, is_permutation=True)
    chrom.genes = list(genes)
    return chrom


class TestLoading(unittest.TestCase):
    def test_loads_instance_from_json(self):
        problem = CVRPProblem.from_json(INPUT_PATH)
        self.assertEqual(problem.num_customers, 16)
        self.assertEqual(problem.num_vehicles, 4)
        self.assertEqual(problem.vehicle_capacities, [15, 15, 15, 15])
        self.assertEqual(sum(problem.demands), 60)

    def test_rejects_inconsistent_parameters(self):
        with self.assertRaises(ValueError):
            CVRPProblem(
                distance_matrix=[[0, 1], [1, 0]],
                demands=[0],  # wrong length
                vehicle_capacities=[5],
            )


class TestDecodingAndEvaluation(unittest.TestCase):
    def setUp(self):
        self.problem = CVRPProblem.from_json(INPUT_PATH)

    def test_reference_solution_decodes_to_reference_routes(self):
        chrom = permutation([c for route in REFERENCE_ROUTES for c in route])
        routes, unassigned = self.problem.decode_routes(chrom)
        self.assertEqual(routes, REFERENCE_ROUTES)
        self.assertEqual(unassigned, [])

    def test_reference_solution_is_feasible_with_reference_distance(self):
        chrom = permutation([c for route in REFERENCE_ROUTES for c in route])
        self.assertTrue(self.problem.is_feasible(chrom))
        self.assertEqual(self.problem.evaluate_fitness(chrom), REFERENCE_DISTANCE)

    def test_every_route_respects_capacity(self):
        random.seed(0)
        for _ in range(200):
            chrom = permutation(random.sample(range(1, 17), 16))
            routes, _ = self.problem.decode_routes(chrom)
            for route in routes:
                self.assertLessEqual(sum(self.problem.demands[c] for c in route), 15)

    def test_wasted_capacity_makes_solution_infeasible_and_penalized(self):
        # 7 and 15 both demand 8: they cannot share a vehicle, wasting capacity
        # until the last vehicle overflows and customers are left unassigned.
        rest = [c for c in range(1, 17) if c not in (7, 15, 8, 16)]
        chrom = permutation([7, 15, 8, 16] + rest)
        _, unassigned = self.problem.decode_routes(chrom)
        self.assertGreater(len(unassigned), 0)
        self.assertFalse(self.problem.is_feasible(chrom))
        self.assertGreater(self.problem.evaluate_fitness(chrom), REFERENCE_DISTANCE + 5000)

    def test_format_solution_matches_or_tools_style(self):
        chrom = permutation([c for route in REFERENCE_ROUTES for c in route])
        text = self.problem.format_solution(chrom)
        self.assertIn("Route for vehicle 0:", text)
        self.assertIn(" 0 Load(0) ->  13 Load(4)", text)
        self.assertIn(f"Total distance of all routes: {REFERENCE_DISTANCE}m", text)
        self.assertIn("Total load of all routes: 60", text)


class TestSolve(unittest.TestCase):
    def test_seeded_ga_reaches_feasible_near_reference_solution(self):
        random.seed(0)
        problem = CVRPProblem.from_json(INPUT_PATH)
        best = problem.optimize(
            population_size=150,
            max_generations=500,
            crossover_rate=0.85,
            mutation_rate=0.05,
            tournament_k=3,
            elitism_count=2,
        )
        self.assertTrue(best.is_feasible)
        self.assertLessEqual(best.fitness, REFERENCE_DISTANCE * 1.05)


if __name__ == "__main__":
    unittest.main()
