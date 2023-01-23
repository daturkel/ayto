from __future__ import annotations

from ayto import AYTO


def test_benchmark_initialize(benchmark, names_long: list[str]):
    benchmark(AYTO, names_long, names_long)


def test_benchmark_truth_booth(benchmark, names_long: list[str]):
    def setup():
        ayto_instance = AYTO(names_long, names_long)
        return ((ayto_instance,), {})

    benchmark.pedantic(
        lambda x: x.apply_truth_booth(
            names_long[0], names_long[0], False, calc_probs=False
        ),
        setup=setup,
        rounds=5,
    )


def test_benchmark_matchup_ceremony(benchmark, names_long: list[str]):
    def setup():
        ayto_instance = AYTO(names_long, names_long)
        return ((ayto_instance,), {})

    matchup = [(name, name) for name in names_long]

    benchmark.pedantic(
        lambda x: x.apply_matchup_ceremony(matchup, 3, calc_probs=False),
        setup=setup,
        rounds=5,
    )


def test_benchmark_calc_probs(benchmark, names_long: list[str]):
    def setup():
        ayto_instance = AYTO(names_long, names_long)
        ayto_instance.apply_truth_booth(
            names_long[0], names_long[0], False, calc_probs=False
        )
        return ((ayto_instance,), {})

    benchmark.pedantic(lambda x: x.calculate_probabilities(), setup=setup, rounds=5)
