# Flexible Manufacturing Scheduling with a Genetic Algorithm

A reproducible Python implementation of a genetic algorithm for flexible parallel-machine manufacturing scheduling. Each job is processed exactly once on one compatible machine, processing times may vary by machine, and the objective is to minimize makespan.

## What was fixed

This version hardens the original implementation against several failure modes:

- evaluates every newly generated population before selecting the final best individual;
- supports population sizes smaller than the tournament size;
- safely handles single-job crossover and mutation;
- supports `generations=0` by evaluating the initial population;
- validates problem dimensions and GA hyperparameters;
- uses an instance-local seeded random generator for reproducibility;
- rejects incompatible machine assignments;
- validates empty schedules before plotting.

## Requirements

- Python 3.10+
- matplotlib
- pytest (for tests)

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run

```bash
python scheduler.py
```

The example optimizes 10 jobs on 3 machines and writes:

```text
optimized_manufacturing_schedule.png
```

## Use as a module

```python
from scheduler import AdvancedManufacturingScheduler

scheduler = AdvancedManufacturingScheduler(
    num_jobs=20,
    num_machines=5,
    seed=42,
)

schedule, makespan = scheduler.genetic_algorithm(
    population_size=100,
    generations=200,
)

print(makespan)
```

## Tests

```bash
pytest -q
```

The regression suite covers schedule feasibility, makespan consistency, the former final-generation fitness bug, small populations, single-job instances, zero generations, invalid parameters, reproducibility, incompatible assignments, and plotting behavior.

## Continuous integration

GitHub Actions executes the test suite on Python 3.10, 3.11, and 3.12 for pushes to `main` and for pull requests.

## Model scope

The implemented problem is a flexible parallel-machine scheduling model. It currently assumes independent jobs without release dates, precedence constraints, sequence-dependent setup times, machine downtime, or preemption.

One important modeling detail is that, under these assumptions, job sequence does not change makespan for a fixed set of machine assignments. The genetic representation retains a sequence because it is useful for constructing and displaying the resulting machine schedules, while makespan optimization is primarily driven by the machine-assignment genes.
