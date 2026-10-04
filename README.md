# quant-numerics

![ci](https://github.com/veuksik/quant-numerics/actions/workflows/ci.yml/badge.svg)

Numerical methods for quantitative finance: Monte Carlo simulation, time series
and pricing, implemented from scratch and tested against analytical results.

## Setup

    uv sync
    uv run pytest

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

## Principles

- Every algorithm is tested against a known analytical result.
- All simulations are seedable and reproducible.
- Dependencies are pinned in `uv.lock`.

## Layout

- `src/quant_numerics/`: library code
- `tests/`: tests
- `notebooks/`: experiments and verification notebooks
- `puzzles/`: probability and trading-intuition problem log

## Status

Work in progress. Results and benchmarks will be added as the modules are built.
