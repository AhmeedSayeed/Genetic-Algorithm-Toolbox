"""Entry point for testing and running the Genetic Algorithm Toolbox on CVRP."""

from pathlib import Path
from cvrp import CVRPProblem


def main() -> None:
    base_dir = Path(__file__).parent / "cvrp"
    input_path = base_dir / "cvrp_input.json"
    output_path = base_dir / "cvrp_output.txt"

    problem = CVRPProblem.from_json(input_path)
    best_chromosome = problem.optimize(
        population_size=150,
        max_generations=500,
        crossover_rate=0.85,
        mutation_rate=0.05,
        tournament_k=3,
        elitism_count=2,
    )

    if best_chromosome is not None:
        report = problem.save_solution_to_txt(output_path, best_chromosome)
        print(report)
        print(f"\nSolution saved to: {output_path}")
    else:
        print(
            "Note: GAToolbox.run() returned None (waiting for team's engine implementation)."
        )


if __name__ == "__main__":
    main()
