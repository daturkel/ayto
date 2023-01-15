# AYTO

[![PyPI version](https://badge.fury.io/py/ayto.svg)](https://badge.fury.io/py/ayto) ![Python version](https://img.shields.io/pypi/pyversions/ayto) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A Python library for calculating couple probabilities for the TV show [Are You the One?](https://en.wikipedia.org/wiki/Are_You_the_One%3F).

## Installation

```
pip install ayto
```

## Usage

```python
from ayto import AYTO

guys = ["Al", "Bill", "Carl"]
girls = ["Daisy", "Emily", "Faith"]

season = AYTO(guys, girls)

# "there are 6 possible scenarios"
print(f"there are {len(season.scenarios)} possible scenarios")

# Al and Daphne go to the Truth Booth and get "NO MATCH"
season.apply_truth_booth("Al", "Daphne", False)

# "4 scenarios remain"
print(f"{len(season.scenarios)} scenarios remain")

# A matchup ceremony with 1 beam
season.apply_matchup_ceremony(
    [("Al", "Emily"), ("Bill", "Daisy"), ("Carl", "Faith")], beams=1
)

# "2 scenarios remain"
print(f"{len(season.scenarios)} scenarios remain")

# Calculate couple probabilities
season.calc_probs()

print(season.probs)
#         Albert  Billy  Carl
# Daphne     0.0    0.5   0.5
# Emily      0.5    0.0   0.5
# Faith      0.5    0.5   0.0
```
