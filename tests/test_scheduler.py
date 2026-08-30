import matplotlib
import pytest

matplotlib.use("Agg")

from scheduler import AdvancedManufacturingScheduler


def assert_valid_schedule(scheduler, schedule, makespan):
    assert len(schedule) == scheduler.num_jobs
    assert sorted(item["job"] for item in schedule) == list(range(scheduler.num_jobs))

    by_machine = {machine: [] for machine in range(scheduler.num_machines)}
    for item in schedule:
        job = item["job"]
        machine = item["machine"]
        assert machine in scheduler.machine_compatibility[job]
        assert item["duration"] == scheduler.processing_times[(job, machine)]
        assert item["end_time"] == item["start_time"] + item["duration"]
        by_machine[machine].append(item)

    for machine_jobs in by_machine.values():
        machine_jobs.sort(key=lambda item: item["start_time"])
        for previous, current in zip(machine_jobs, machine_jobs[1:]):
            assert previous["end_time"] <= current["start_time"]

    assert makespan == max(item["end_time"] for item in schedule)


def test_standard_ga_run_returns_valid_schedule():
    scheduler = AdvancedManufacturingScheduler(10, 3, seed=42)
    schedule, makespan = scheduler.genetic_algorithm(population_size=30, generations=20)
    assert_valid_schedule(scheduler, schedule, makespan)


def test_final_generation_is_evaluated():
    scheduler = AdvancedManufacturingScheduler(8, 3, seed=7)
    schedule, makespan = scheduler.genetic_algorithm(population_size=10, generations=1)
    assert_valid_schedule(scheduler, schedule, makespan)


@pytest.mark.parametrize("population_size", [1, 2, 3])
def test_small_populations_do_not_break_tournament_selection(population_size):
    scheduler = AdvancedManufacturingScheduler(5, 2, seed=4)
    schedule, makespan = scheduler.genetic_algorithm(population_size=population_size, generations=3)
    assert_valid_schedule(scheduler, schedule, makespan)


def test_single_job_is_safe_for_mutation_and_crossover():
    scheduler = AdvancedManufacturingScheduler(1, 3, seed=3)
    schedule, makespan = scheduler.genetic_algorithm(
        population_size=5,
        generations=10,
        crossover_rate=1.0,
        mutation_rate=1.0,
    )
    assert_valid_schedule(scheduler, schedule, makespan)


def test_zero_generations_returns_evaluated_initial_population():
    scheduler = AdvancedManufacturingScheduler(6, 3, seed=11)
    schedule, makespan = scheduler.genetic_algorithm(population_size=8, generations=0)
    assert_valid_schedule(scheduler, schedule, makespan)


def test_seed_reproduces_generated_problem_instance():
    left = AdvancedManufacturingScheduler(10, 4, seed=99)
    right = AdvancedManufacturingScheduler(10, 4, seed=99)
    assert left.processing_times == right.processing_times
    assert dict(left.machine_compatibility) == dict(right.machine_compatibility)


@pytest.mark.parametrize(
    "jobs,machines",
    [(0, 2), (-1, 2), (2, 0), (2, -1)],
)
def test_invalid_problem_dimensions_raise(jobs, machines):
    with pytest.raises(ValueError):
        AdvancedManufacturingScheduler(jobs, machines)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"population_size": 0},
        {"population_size": -1},
        {"generations": -1},
        {"crossover_rate": -0.1},
        {"crossover_rate": 1.1},
        {"mutation_rate": -0.1},
        {"mutation_rate": 1.1},
    ],
)
def test_invalid_ga_parameters_raise(kwargs):
    scheduler = AdvancedManufacturingScheduler(4, 2)
    with pytest.raises(ValueError):
        scheduler.genetic_algorithm(**kwargs)


def test_incompatible_assignment_is_rejected():
    scheduler = AdvancedManufacturingScheduler(1, 2, seed=12)
    compatible = set(scheduler.machine_compatibility[0])
    incompatible = next((m for m in range(2) if m not in compatible), None)
    if incompatible is None:
        pytest.skip("seed generated compatibility with every machine")

    individual = {"sequence": [0], "assignments": {0: incompatible}}
    with pytest.raises(ValueError):
        scheduler._calculate_fitness(individual)


def test_plot_schedule_returns_figure_and_axes():
    scheduler = AdvancedManufacturingScheduler(4, 2, seed=5)
    schedule, _ = scheduler.genetic_algorithm(population_size=5, generations=2)
    fig, ax = scheduler.plot_schedule(schedule)
    assert fig is not None
    assert ax is not None
    fig.clf()


def test_plot_rejects_empty_schedule():
    scheduler = AdvancedManufacturingScheduler(2, 2)
    with pytest.raises(ValueError):
        scheduler.plot_schedule([])
