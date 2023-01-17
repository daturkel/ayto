from __future__ import annotations

import re

import numpy as np
import pytest

from ayto import AYTO


def test_num_scenarios(ayto_instance: AYTO):
    assert ayto_instance.num_scenarios == 120


def test_truth_booth(ayto_instance: AYTO):
    num_remaining = ayto_instance.apply_truth_booth("Albert", "Gina", True)
    assert num_remaining == 24


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
            r"Unknown name XYZ, must be one of ['Faith', 'Gina', 'Heather', 'Ingrid', 'Joy']"
        ),
    ):
        ayto_instance.apply_matchup_ceremony([("Albert", "ABC")], False)
