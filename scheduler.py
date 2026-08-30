"""Genetic-algorithm scheduler for flexible parallel manufacturing machines."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


class AdvancedManufacturingScheduler:
    """Optimize flexible parallel-machine schedules by minimizing makespan.

    Each job must be processed exactly once on one compatible machine. Processing
    times can differ by machine. Jobs are independent: there are no release dates,
    precedence constraints, setup times, or preemption.
    """

    def __init__(self, num_jobs: int, num_machines: int, seed: int = 42) -> None:
        if num_jobs <= 0:
            raise ValueError("num_jobs must be greater than 0")
        if num_machines <= 0:
            raise ValueError("num_machines must be greater than 0")

        self.num_jobs = num_jobs
        self.num_machines = num_machines
        self.seed = seed
        self.rng = random.Random(seed)

        self.processing_times: Dict[Tuple[int, int], int] = {}
        self.machine_compatibility: Dict[int, List[int]] = defaultdict(list)

        for job in range(num_jobs):
            num_compatible = self.rng.randint(1, num_machines)
            compatible_machines = self.rng.sample(range(num_machines), num_compatible)

            for machine in compatible_machines:
                if self.rng.random() < 0.3:
                    duration = self.rng.randint(40, 60)
                else:
                    duration = self.rng.randint(15, 39)

                self.processing_times[(job, machine)] = duration
                self.machine_compatibility[job].append(machine)

    def genetic_algorithm(
        self,
        population_size: int = 50,
        generations: int = 100,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.2,
    ):
        """Run the genetic algorithm and return ``(schedule, makespan)``."""
        if population_size <= 0:
            raise ValueError("population_size must be greater than 0")
        if generations < 0:
            raise ValueError("generations cannot be negative")
        if not 0.0 <= crossover_rate <= 1.0:
            raise ValueError("crossover_rate must be between 0 and 1")
        if not 0.0 <= mutation_rate <= 1.0:
            raise ValueError("mutation_rate must be between 0 and 1")

        population = [self._create_individual() for _ in range(population_size)]
        self._evaluate_population(population)

        for _ in range(generations):
            population.sort(key=lambda individual: individual["fitness"])

            elite_size = max(1, population_size // 10)
            new_population = [self._clone_individual(individual) for individual in population[:elite_size]]

            while len(new_population) < population_size:
                parent1 = self._tournament_selection(population)
                parent2 = self._tournament_selection(population)

                if self.rng.random() < crossover_rate:
                    child = self._crossover(parent1, parent2)
                else:
                    child = self._clone_individual(parent1)

                if self.rng.random() < mutation_rate:
                    child = self._mutate(child)

                new_population.append(child)

            population = new_population
            # Critical: offspring do not inherit a valid fitness value.
            self._evaluate_population(population)

        best_solution = min(population, key=lambda individual: individual["fitness"])
        return self._create_schedule(best_solution), best_solution["fitness"]

    def _create_individual(self):
        sequence = list(range(self.num_jobs))
        self.rng.shuffle(sequence)

        # Use the locally fastest compatible machine as the initial assignment.
        assignments = {
            job: min(
                self.machine_compatibility[job],
                key=lambda machine: self.processing_times[(job, machine)],
            )
            for job in sequence
        }
        return {"sequence": sequence, "assignments": assignments}

    @staticmethod
    def _clone_individual(individual):
        # Do not copy fitness because sequence/assignment changes can invalidate it.
        return {
            "sequence": individual["sequence"].copy(),
            "assignments": individual["assignments"].copy(),
        }

    def _evaluate_population(self, population) -> None:
        for individual in population:
            individual["fitness"] = self._calculate_fitness(individual)

    def _tournament_selection(self, population, tournament_size: int = 3):
        if not population:
            raise ValueError("population cannot be empty")
        effective_size = min(tournament_size, len(population))
        tournament = self.rng.sample(population, effective_size)
        return min(tournament, key=lambda individual: individual["fitness"])

    def _crossover(self, parent1, parent2):
        child = self._clone_individual(parent1)
        length = len(parent1["sequence"])
        if length <= 1:
            return child

        start = self.rng.randint(0, length - 1)
        end = self.rng.randint(start, length - 1)
        segment = parent2["sequence"][start : end + 1]
        segment_set = set(segment)

        child_sequence = [-1] * length
        child_sequence[start : end + 1] = segment

        remaining = (job for job in parent1["sequence"] if job not in segment_set)
        for index in range(length):
            if child_sequence[index] == -1:
                child_sequence[index] = next(remaining)

        child["sequence"] = child_sequence

        for job in child_sequence:
            if self.rng.random() < 0.5:
                child["assignments"][job] = parent2["assignments"][job]

        return child

    def _mutate(self, individual):
        mutated = self._clone_individual(individual)

        if len(mutated["sequence"]) >= 2 and self.rng.random() < 0.5:
            first, second = self.rng.sample(range(len(mutated["sequence"])), 2)
            mutated["sequence"][first], mutated["sequence"][second] = (
                mutated["sequence"][second],
                mutated["sequence"][first],
            )

        if self.rng.random() < 0.5:
            job = self.rng.choice(mutated["sequence"])
            compatible_machines = self.machine_compatibility[job]
            if len(compatible_machines) > 1:
                current_machine = mutated["assignments"][job]
                alternatives = [machine for machine in compatible_machines if machine != current_machine]
                mutated["assignments"][job] = self.rng.choice(alternatives)

        return mutated

    def _calculate_fitness(self, individual) -> int:
        machine_completion_times = [0] * self.num_machines

        for job in individual["sequence"]:
            machine = individual["assignments"][job]
            if machine not in self.machine_compatibility[job]:
                raise ValueError(f"job {job} is assigned to incompatible machine {machine}")
            machine_completion_times[machine] += self.processing_times[(job, machine)]

        return max(machine_completion_times)

    def _create_schedule(self, individual):
        machine_completion_times = [0] * self.num_machines
        schedule = []

        for job in individual["sequence"]:
            machine = individual["assignments"][job]
            processing_time = self.processing_times[(job, machine)]
            start_time = machine_completion_times[machine]
            end_time = start_time + processing_time

            schedule.append(
                {
                    "job": job,
                    "machine": machine,
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration": processing_time,
                }
            )
            machine_completion_times[machine] = end_time

        return schedule

    def plot_schedule(self, schedule, title: str = "Optimized Manufacturing Schedule"):
        """Create a Gantt chart for a non-empty schedule."""
        if not schedule:
            raise ValueError("schedule cannot be empty")

        fig, ax = plt.subplots(figsize=(15, 8))
        colors = plt.cm.tab10.colors + plt.cm.Set3.colors

        for machine in range(self.num_machines):
            machine_jobs = sorted(
                (job for job in schedule if job["machine"] == machine),
                key=lambda job: job["start_time"],
            )

            for job in machine_jobs:
                ax.barh(
                    machine,
                    job["duration"],
                    left=job["start_time"],
                    color=colors[job["job"] % len(colors)],
                    edgecolor="black",
                    alpha=0.8,
                )
                ax.text(
                    job["start_time"] + job["duration"] / 2,
                    machine,
                    f"Job {job['job']}",
                    ha="center",
                    va="center",
                    color="black",
                    fontweight="bold",
                )

        ax.set_yticks(range(self.num_machines))
        ax.set_yticklabels([f"Machine {machine}" for machine in range(self.num_machines)])
        ax.set_xlabel("Time (minutes)")
        ax.set_ylabel("Machine")
        ax.set_title(title)
        ax.grid(True, axis="x", linestyle="--", alpha=0.7)
        makespan = max(job["end_time"] for job in schedule)
        ax.set_xlim(0, makespan * 1.05)
        fig.tight_layout()
        return fig, ax


if __name__ == "__main__":
    scheduler = AdvancedManufacturingScheduler(num_jobs=10, num_machines=3)
    schedule, makespan = scheduler.genetic_algorithm(population_size=50, generations=100)

    print(f"Optimized Makespan: {makespan} minutes")
    for item in sorted(schedule, key=lambda x: (x["machine"], x["start_time"])):
        print(
            f"Job {item['job']} on Machine {item['machine']}: "
            f"start={item['start_time']}, end={item['end_time']}"
        )

    scheduler.plot_schedule(schedule)
    plt.savefig("optimized_manufacturing_schedule.png", dpi=300, bbox_inches="tight")
