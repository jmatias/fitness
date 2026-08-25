# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

- Install dependencies: `poetry install`
- Run the main script: `poetry run python fitness/main.py`

There are no tests or linters configured in this project.

## Architecture

`fitness/main.py` is a single-script pandas pipeline (uses `# %%` cell markers, meant to be run interactively e.g. in PyCharm/Jupyter) that reconciles daily body-weight readings from two sources into one series:

1. Reads `data_files/fitness_agg.csv` (Fitbit export) and `data_files/weight.csv` (Withings export), each keeping only `Date` and a weight column.
2. Concatenates both sources, averages multiple readings per day (`calculate_mean_weight_per_day`), fills in any missing calendar days (`insert_missing_days`), and linearly interpolates gaps (`interpolate_missing_weights`).
3. Writes the result to `data_files/weight_interpolated.csv`, which `data_files/Fitness.twb`/`Fitness.tflx` (Tableau) consume for visualization.

`data_files/` is gitignored entirely (see `.gitignore`) since it holds personal exports and `credentials.json` — never remove it from `.gitignore` or commit its contents.
