from ayto import AYTO


def test_num_scenarios(ayto_instance: AYTO):
    assert ayto_instance.num_scenarios == 24


def test_truth_booth(ayto_instance: AYTO):
    num_remaining = ayto_instance.apply_truth_booth("Albert", "Emily", True)
    assert num_remaining == 6


def test_matchup(ayto_instance: AYTO):
    num_remaining = ayto_instance.apply_matchup_ceremony(
        [
            ("Albert", "Emily"),
            ("Bob", "Faith"),
            ("Charles", "Gina"),
            ("Devin", "Heather"),
        ],
        2,
    )
    assert num_remaining == 6
