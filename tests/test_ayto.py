from __future__ import annotations

import json
from pathlib import Path
import re

import numpy as np
import pytest

from ayto import AYTO


def test_num_scenarios(ayto_instance: AYTO):
    assert ayto_instance.num_scenarios == 120


def test_initial_probabilities(ayto_instance: AYTO):
    assert np.allclose(ayto_instance.probabilities, 1 / 5)


def test_contradiction(ayto_instance: AYTO):
    ayto_instance.apply_truth_booth("Albert", "Gina", True)
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Impossible scenario provided. Did you enter contradictory data?"
        ),
    ):
        ayto_instance.apply_truth_booth("Albert", "Gina", False)


class TestTruthBooth:
    def test_truth_booth(self, ayto_instance: AYTO):
        num_remaining = ayto_instance.apply_truth_booth("Albert", "Gina", True)
        assert num_remaining == 24

    def test_truth_booth_history(self, ayto_instance: AYTO):
        assert ayto_instance.history == [
            {"type": "truth_booth", "guy": "Albert", "girl": "Gina", "match": True}
        ]

    def test_truth_booth_results(self, ayto_instance: AYTO):
        assert np.allclose(
            ayto_instance.probabilities.values,
            np.array(
                [
                    [0, 0.25, 0.25, 0.25, 0.25],
                    [1, 0, 0, 0, 0],
                    [0, 0.25, 0.25, 0.25, 0.25],
                    [0, 0.25, 0.25, 0.25, 0.25],
                    [0, 0.25, 0.25, 0.25, 0.25],
                ]
            ),
        )


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

    def test_matchup_history(self, ayto_instance: AYTO):
        assert ayto_instance.history == [
            {
                "type": "matchup_ceremony",
                "matchup": [
                    ("Albert", "Faith"),
                    ("Bob", "Gina"),
                    ("Charles", "Heather"),
                    ("Devin", "Ingrid"),
                    ("Eli", "Joy"),
                ],
                "beams": 2,
            }
        ]

    def test_matchup_results(self, ayto_instance: AYTO):
        ayto_instance.calculate_probabilities()
        assert np.allclose(
            ayto_instance.probabilities.values,
            np.array(
                [
                    [0.4, 0.15, 0.15, 0.15, 0.15],
                    [0.15, 0.4, 0.15, 0.15, 0.15],
                    [0.15, 0.15, 0.4, 0.15, 0.15],
                    [0.15, 0.15, 0.15, 0.4, 0.15],
                    [0.15, 0.15, 0.15, 0.15, 0.4],
                ]
            ),
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
            np.array(
                [
                    [1, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0],
                    [0, 0, 1 / 3, 1 / 3, 1 / 3],
                    [0, 0, 1 / 3, 1 / 3, 1 / 3],
                    [0, 0, 1 / 3, 1 / 3, 1 / 3],
                ]
            ),
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
            np.array(
                [
                    [1, 0, 0, 0, 0],
                    [0, 0, 3 / 7, 2 / 7, 2 / 7],
                    [0, 3 / 7, 0, 2 / 7, 2 / 7],
                    [0, 2 / 7, 2 / 7, 3 / 14, 3 / 14],
                    [0, 2 / 7, 2 / 7, 3 / 14, 3 / 14],
                ]
            ),
        )
        assert (scenarios, p, probs_close) == (14, 14 / 120, True)

    def test_partial_is_temporary(self, ayto_instance: AYTO):
        assert all(
            [
                ayto_instance.history == [],
                ayto_instance.num_scenarios == 120,
                np.allclose(ayto_instance.probabilities, 1 / 5),
            ]
        )

    def test_try_partial_error(self, ayto_instance: AYTO):
        with pytest.raises(
            ValueError, match=r"Either matches, non_matches, or both must be provided."
        ):
            ayto_instance.try_partial_scenario()


class TestSaveAndLoad:
    def test_serialize(self, ayto_instance: AYTO):
        matchup = [
            ("Albert", "Faith"),
            ("Bob", "Gina"),
            ("Charles", "Heather"),
            ("Devin", "Ingrid"),
            ("Eli", "Joy"),
        ]
        ayto_instance.apply_truth_booth("Albert", "Faith", True)
        ayto_instance.apply_matchup_ceremony(
            matchup,
            2,
        )
        assert ayto_instance._serialize() == {
            "guys": ["Albert", "Bob", "Charles", "Devin", "Eli"],
            "girls": ["Faith", "Gina", "Heather", "Ingrid", "Joy"],
            "history": [
                {
                    "type": "truth_booth",
                    "guy": "Albert",
                    "girl": "Faith",
                    "match": True,
                },
                {"type": "matchup_ceremony", "matchup": matchup, "beams": 2},
            ],
        }

    def test_save(self, ayto_instance: AYTO, path: Path):
        ayto_instance.save(path / "instance")
        with open(path / "instance", "r") as f:
            assert json.load(f) == {
                "guys": ["Albert", "Bob", "Charles", "Devin", "Eli"],
                "girls": ["Faith", "Gina", "Heather", "Ingrid", "Joy"],
                "history": [
                    {
                        "type": "truth_booth",
                        "guy": "Albert",
                        "girl": "Faith",
                        "match": True,
                    },
                    {
                        "type": "matchup_ceremony",
                        "matchup": [
                            ["Albert", "Faith"],
                            ["Bob", "Gina"],
                            ["Charles", "Heather"],
                            ["Devin", "Ingrid"],
                            ["Eli", "Joy"],
                        ],
                        "beams": 2,
                    },
                ],
            }

    def test_load(self, ayto_instance: AYTO, path: Path):
        new_instance = AYTO.load(path / "instance")
        assert all(
            [
                new_instance.guys == ayto_instance.guys,
                new_instance.girls == ayto_instance.girls,
                np.allclose(new_instance._scenarios, ayto_instance._scenarios),
                np.allclose(
                    new_instance.probabilities.values,
                    ayto_instance.probabilities.values,
                ),
                new_instance.history == ayto_instance.history,
            ]
        )
