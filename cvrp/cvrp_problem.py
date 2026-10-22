import json
from pathlib import Path
from typing import Optional

from ga_toolbox import (
    Chromosome,
    ElitistReplacement,
    GAToolbox,
    IntegerChromosome,
    OrderCrossover,
    SwapMutation,
    TournamentSelection,
)


class CVRPProblem:
    """Capacitated Vehicle Routing Problem (CVRP) solver using the GA Toolbox."""

    distance_matrix: list[list[int]]
    demands: list[int]
    vehicle_capacities: list[int]
    num_vehicles: int
    depot: int
    num_customers: int
    best_solution: Optional[Chromosome]

    def __init__(
        self,
        distance_matrix: Optional[list[list[int]]] = None,
        demands: Optional[list[int]] = None,
        vehicle_capacities: Optional[list[int]] = None,
        num_vehicles: Optional[int] = None,
        depot: int = 0,
    ) -> None:
        self.distance_matrix = distance_matrix or []
        self.demands = demands or []
        self.vehicle_capacities = vehicle_capacities or []
        self.num_vehicles = (
            num_vehicles
            if num_vehicles is not None
            else len(self.vehicle_capacities)
        )
        self.depot = depot
        self.num_customers = (
            len(self.distance_matrix) - 1 if self.distance_matrix else 0
        )
        self.best_solution = None

        if self.distance_matrix:
            self._validate_parameters()

    @classmethod
    def from_json(cls, json_path: str | Path) -> "CVRPProblem":
        """Creates and initializes a CVRPProblem instance directly from a JSON file."""
        problem = cls()
        problem.load_from_json(json_path)
        return problem

    def load_from_json(self, json_path: str | Path) -> None:
        """Reads CVRP problem parameters from a JSON file."""
        path = Path(json_path)
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        self.distance_matrix = data["distance_matrix"]
        self.demands = data["demands"]
        self.vehicle_capacities = data["vehicle_capacities"]
        self.num_vehicles = data.get("num_vehicles", len(self.vehicle_capacities))
        self.depot = data.get("depot", 0)
        self.num_customers = len(self.distance_matrix) - 1
        self.best_solution = None

        self._validate_parameters()

    def _validate_parameters(self) -> None:
        """Validates consistency of loaded CVRP parameters."""
        num_nodes = len(self.distance_matrix)
        if num_nodes < 2:
            raise ValueError("distance_matrix must contain at least a depot and 1 customer.")
        if any(len(row) != num_nodes for row in self.distance_matrix):
            raise ValueError("distance_matrix must be a square 2D matrix.")
        if len(self.demands) != num_nodes:
            raise ValueError("Length of demands must match distance_matrix dimensions.")
        if self.num_vehicles <= 0:
            raise ValueError("num_vehicles must be a positive integer.")
        if len(self.vehicle_capacities) != self.num_vehicles:
            raise ValueError("Length of vehicle_capacities must equal num_vehicles.")
        if self.depot != 0:
            raise ValueError("Standard depot index must be 0.")

    def decode_routes(
        self, chromosome: Chromosome
    ) -> tuple[list[list[int]], list[int]]:
        """Traverses chromosome genes and assigns cities to vehicles based on capacity.

        Returns:
            A tuple (routes, unassigned_cities) where:
            - routes: list of city lists for each vehicle (0 to num_vehicles - 1).
            - unassigned_cities: cities remaining after all vehicles run out of capacity.
        """
        routes: list[list[int]] = [[] for _ in range(self.num_vehicles)]
        unassigned_cities: list[int] = []

        vehicle_idx = 0
        remaining_capacity = self.vehicle_capacities[0]

        for city in chromosome.genes:
            city_demand = self.demands[city]

            # Advance to the next vehicle if the current vehicle cannot fit this city
            while (
                vehicle_idx < self.num_vehicles
                and remaining_capacity < city_demand
            ):
                vehicle_idx += 1
                if vehicle_idx < self.num_vehicles:
                    remaining_capacity = self.vehicle_capacities[vehicle_idx]

            if vehicle_idx < self.num_vehicles:
                routes[vehicle_idx].append(city)
                remaining_capacity -= city_demand
            else:
                unassigned_cities.append(city)

        return routes, unassigned_cities

    def is_feasible(self, chromosome: Chromosome) -> bool:
        """Checks if all cities can be assigned to the available vehicles within capacity."""
        if not chromosome.validate():
            return False

        _, unassigned_cities = self.decode_routes(chromosome)
        return len(unassigned_cities) == 0

    def evaluate_fitness(self, chromosome: Chromosome) -> float:
        """Calculates total travel distance across all vehicles (plus penalty if infeasible)."""
        routes, unassigned_cities = self.decode_routes(chromosome)
        total_distance = 0.0

        for route in routes:
            if not route:
                continue

            # Distance from depot to the first city in the route
            last_city = route[0]
            total_distance += self.distance_matrix[self.depot][last_city]

            # Distance between consecutive cities in the route
            for current_city in route[1:]:
                total_distance += self.distance_matrix[last_city][current_city]
                last_city = current_city

            # Return distance from the last city back to the depot
            total_distance += self.distance_matrix[last_city][self.depot]

        # If all vehicles ran out and cities remain unassigned, penalize the chromosome
        if unassigned_cities:
            last_city = self.depot
            overflow_demand = 0
            for city in unassigned_cities:
                total_distance += self.distance_matrix[last_city][city]
                overflow_demand += self.demands[city]
                last_city = city
            total_distance += self.distance_matrix[last_city][self.depot]

            # Penalty ensures infeasible chromosomes always have higher cost than feasible ones
            total_distance += overflow_demand * 2000.0 + len(unassigned_cities) * 5000.0

        return total_distance

    def optimize(
        self,
        population_size: int = 100,
        max_generations: int = 500,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.05,
        tournament_k: int = 3,
        elitism_count: int = 2,
    ) -> Chromosome:
        """Configures and runs the GAToolbox engine to solve the CVRP instance."""
        if not self.distance_matrix:
            raise ValueError("Problem parameters are not loaded. Call load_from_json() first.")

        chromosome_prototype = IntegerChromosome(
            length=self.num_customers,
            lower_bound=1,
            upper_bound=self.num_customers,
            is_permutation=True,
        )

        engine = GAToolbox(
            chromosome_prototype=chromosome_prototype,
            fitness_fn=self.evaluate_fitness,
            feasibility_fn=self.is_feasible,
            selection_strategy=TournamentSelection(k=tournament_k),
            crossover_strategy=OrderCrossover(),
            mutation_strategy=SwapMutation(),
            replacement_strategy=ElitistReplacement(elitism_count=elitism_count),
            population_size=population_size,
            max_generations=max_generations,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            maximize=False,
        )

        self.best_solution = engine.run()
        return self.best_solution

    def format_solution(self, chromosome: Optional[Chromosome] = None) -> str:
        """Formats the decoded routes into the Google OR-Tools CVRP output text format."""
        target = chromosome or self.best_solution
        if target is None:
            raise ValueError("No solution chromosome provided or stored from optimize().")

        routes, _ = self.decode_routes(target)
        lines: list[str] = []
        total_distance = 0
        total_load = 0

        for vehicle_id, route in enumerate(routes):
            lines.append(f"Route for vehicle {vehicle_id}:")
            route_distance = 0
            route_load = 0

            steps = [f" {self.depot} Load(0)"]
            prev_node = self.depot

            for city in route:
                route_load += self.demands[city]
                route_distance += self.distance_matrix[prev_node][city]
                steps.append(f" {city} Load({route_load})")
                prev_node = city

            route_distance += self.distance_matrix[prev_node][self.depot]
            steps.append(f" {self.depot} Load({route_load})")

            lines.append(" -> ".join(steps))
            lines.append(f"Distance of the route: {route_distance}m")
            lines.append(f"Load of the route: {route_load}")
            lines.append("")

            total_distance += route_distance
            total_load += route_load

        lines.append(f"Total distance of all routes: {total_distance}m")
        lines.append(f"Total load of all routes: {total_load}")

        return "\n".join(lines)

    def save_solution_to_txt(
        self,
        output_path: str | Path,
        chromosome: Optional[Chromosome] = None,
    ) -> str:
        """Writes the formatted CVRP solution to a .txt file and returns the formatted text."""
        formatted_output = self.format_solution(chromosome)
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(formatted_output + "\n", encoding="utf-8")
        return formatted_output
