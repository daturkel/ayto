from __future__ import annotations

import re

import numpy as np
import pytest

from ayto import AYTO


def test_num_scenarios(ayto_instance: AYTO):
    assert ayto_instance.num_scenarios == 120


def test_initial_probabilities(ayto_instance: AYTO):
    assert np.allclose(ayto_instance.probabilities, 1 / 5)


def test_truth_booth(ayto_instance: AYTO):
    num_remaining = ayto_instance.apply_truth_booth("Albert", "Gina", True)
    assert num_remaining == 24


def test_contradiction(ayto_instance: AYTO):
    ayto_instance.apply_truth_booth("Albert", "Gina", True)
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Impossible scenario provided. Did you enter contradictory data?"
        ),
    ):
        ayto_instance.apply_truth_booth("Albert", "Gina", False)


class TestMatchup:
    def test_matchup(self, ayto_instance: AYTO):
        num_remaining = ayto_instance.apply_matchup_ceremony(
            [
                ("Albert", "Faith"),
                ("Bob", "Gina"),
                ("Charles", "Heather"),
                ("Devin", "Ingrid"),
                ("Eli", "Joy"),
            ],
            2,
            calc_probs=False,
        )
        assert num_remaining == 20

    def test_results(self, ayto_instance: AYTO):
        ayto_instance.calculate_probabilities()
        assert np.allclose(
            ayto_instance.probabilities.values,
            [
                [0.4, 0.15, 0.15, 0.15, 0.15],
                [0.15, 0.4, 0.15, 0.15, 0.15],
                [0.15, 0.15, 0.4, 0.15, 0.15],
                [0.15, 0.15, 0.15, 0.4, 0.15],
                [0.15, 0.15, 0.15, 0.15, 0.4],
            ],
        )


def test_missing_name_guy(ayto_instance: AYTO):
    with pytest.raises(
        ValueError,
        match=re.escape(
            r"Unknown name XYZ, must be one of ['Albert', 'Bob', 'Charles', 'Devin', 'Eli']"
        ),
    ):
        ayto_instance.apply_matchup_ceremony([("XYZ", "ABC")], False)


def test_missing_name_girl(ayto_instance: AYTO):
    with pytest.raises(
        ValueError,
        match=re.escape(
            r"Unknown name ABC, must be one of ['Faith', 'Gina', 'Heather', 'Ingrid', 'Joy']"
        ),
    ):
        ayto_instance.apply_matchup_ceremony([("Albert", "ABC")], False)


class TestTryPartial:
    def test_try_partial_matches(self, ayto_instance: AYTO):
        scenarios, p, probs = ayto_instance.try_partial_scenario(
            matches=[("Albert", "Faith"), ("Bob", "Gina")]
        )
        probs_close = np.allclose(
            probs.values,
            [
                [1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 1 / 3, 1 / 3, 1 / 3],
                [0, 0, 1 / 3, 1 / 3, 1 / 3],
                [0, 0, 1 / 3, 1 / 3, 1 / 3],
            ],
        )
        assert (scenarios, p, probs_close) == (6, 0.05, True)

    def test_try_partial_non_matches(self, ayto_instance: AYTO):
        scenarios, p, probs = ayto_instance.try_partial_scenario(
            non_matches=[("Albert", "Faith"), ("Bob", "Gina")]
        )
        probs_close = np.allclose(
            probs.values,
            [
                [0, 4 / 13, 3 / 13, 3 / 13, 3 / 13],
                [4 / 13, 0, 3 / 13, 3 / 13, 3 / 13],
                [3 / 13, 3 / 13, 7 / 39, 7 / 39, 7 / 39],
                [3 / 13, 3 / 13, 7 / 39, 7 / 39, 7 / 39],
                [3 / 13, 3 / 13, 7 / 39, 7 / 39, 7 / 39],
            ],
        )
        assert (scenarios, p, probs_close) == (78, 0.65, True)

    def test_try_partial_both(self, ayto_instance: AYTO):
        scenarios, p, probs = ayto_instance.try_partial_scenario(
            matches=[("Albert", "Faith")],
            non_matches=[("Bob", "Gina"), ("Charles", "Heather")],
        )
        probs_close = np.allclose(
            probs.values,
            [
                [1, 0, 0, 0, 0],
                [0, 0, 3 / 7, 2 / 7, 2 / 7],
                [0, 3 / 7, 0, 2 / 7, 2 / 7],
                [0, 2 / 7, 2 / 7, 3 / 14, 3 / 14],
                [0, 2 / 7, 2 / 7, 3 / 14, 3 / 14],
            ],
        )
        assert (scenarios, p, probs_close) == (14, 14 / 120, True)
