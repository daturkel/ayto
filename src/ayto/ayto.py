from __future__ import annotations

from collections import defaultdict
from itertools import permutations
import logging
import pickle

import numpy as np
from numpy.typing import NDArray
import pandas as pd

logger = logging.getLogger()
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
logger.addHandler(handler)


class AYTO:
    """A class to calculate couple probabilities for a season of Are You the One."""

    def __init__(self, guys: list[str], girls: list[str]):
        """This class tracks a season of Are You the One through Truth Booths and Matchup Ceremonies.

        Parameters
        ----------
        guys
            A list of the names of the guys. Order doesn't matter but duplicate names should
            be disambiguated (e.g. ["Joe B.", "Joe C."])
        girls
            A list of the names of the girls. Order doesn't matter but duplicate names should
            be disambiguated (e.g. ["Jane B.", "Jane C."])

        """
        self.guys = guys
        self.girls = girls
        self.n = len(self.guys)
        self.guy_ids, self.girl_ids = self._generate_maps()
        self._scenarios = self._generate_scenarios()
        self._probs: defaultdict | None = None

    @property
    def probabilities(self) -> pd.DataFrame:
        """A pandas dataframe of couple probabilities."""
        if self._probs is None:
            self.calculate_probabilities()
        return pd.DataFrame(self._probs)

    @property
    def num_scenarios(self) -> int:
        """How many scenarios remain possible."""
        return self._scenarios.shape[0]

    def _generate_maps(self) -> tuple[dict[str, int], dict[str, int]]:
        guy_ids = {name: id_ for id_, name in enumerate(self.guys)}
        girl_ids = {name: id_ for id_, name in enumerate(self.girls)}

        return guy_ids, girl_ids

    def _generate_scenarios(self) -> NDArray:
        girl_ids = [self.girl_ids[girl] for girl in self.girls]
        scenarios = list(permutations(girl_ids))

        return np.array(scenarios)

    def _get_beams(self, scenario: tuple[int], matchup: list[int]) -> int:
        beams = 0
        for i, person in enumerate(matchup):
            if person == scenario[i]:
                beams += 1

        return beams

    def apply_truth_booth(
        self, guy: str, girl: str, match: bool, calc_probs=True
    ) -> int:
        """Update scenarios to reflect a Truth Booth outcome.

        Parameters
        ----------
        guy
            Name of the guy in the Truth Booth
        girl
            Name of the girl in the Truth Booth
        match
            Whether or not the pair is a match
        calc_probs
            If `True`, recalculate couple probabilities after applying the truth booth
            (default is `True`)

        Returns
        -------
        int
            Number of scenarios remaining

        """
        self.apply_matchup_ceremony([(guy, girl)], int(match), calc_probs)

        return self.num_scenarios

    def apply_matchup_ceremony(
        self, matchup: list[tuple[str, str]], beams: int, calc_probs=True
    ) -> int:
        """Update scenarios to reflect a Matchup Ceremony.

        Parameters
        ----------
        matchup
            The couples seated together for the Matchup Ceremony as a list of tuples of
            names, e.g. [("Joe", "Sally"), ("Tim", "Jane")]
        beams
            How many beams (correct pairs) the ceremony generated
        calc_probs
            If `True`, recalculate couple probabilities after applying the matchup ceremony
            (default is `True`)

        Returns
        -------
        int
            Number of scenarios remaining

        """
        # ensure all names are known
        for guy, girl in matchup:
            if guy not in self.guy_ids:
                valid = list(self.guy_ids.keys())
                raise ValueError(f"Unknown name {guy}, must be one of {valid}")
            elif girl not in self.girl_ids:
                valid = list(self.girl_ids.keys())
                raise ValueError(f"Unknown name {girl}, must be one of {valid}")

        # put matchup in format understood by _apply_matchup_ceremony
        matchup_ints = [-1] * self.n
        for guy, girl in matchup:
            guy_id = self.guy_ids[guy]
            girl_id = self.girl_ids[girl]
            matchup_ints[guy_id] = girl_id

        self._apply_matchup_ceremony(matchup_ints, beams)

        if calc_probs:
            self.calculate_probabilities()

        return self.num_scenarios

    def _apply_matchup_ceremony(self, matchup: list[int], beams: int):
        # count number of matches between matchup and each scenario
        sums = (self._scenarios == matchup).sum(axis=1)
        # true if number of matches is the number of beams, else false
        idx = sums == beams
        self._scenarios = self._scenarios[idx]

    def calculate_probabilities(self):
        """Update the probabilities for each couple.

        This method does not return anything. To see the probabilities, access the `probabilities`
        attribute.

        This method is only necessary if Truth Booths and Matchup Ceremonies are applied
        with `calc_prob=False` (e.g. if you are applying multiple in a row and don't want
        to calculate the probabilities until the end).

        """
        counter = defaultdict(lambda: defaultdict(int))
        for scenario in self._scenarios:
            for guy, girl in enumerate(scenario):
                counter[guy][girl] += 1

        self._probs = defaultdict(lambda: defaultdict(float))
        for guy_idx, guy in enumerate(self.guys):
            total = sum(counter[guy_idx].values())
            for girl in self.girls:
                girl_idx = self.girl_ids[girl]
                self._probs[guy][girl] = counter[guy_idx][girl_idx] / total

    def save(self, path: str):
        """Save results to a file.

        Parameters
        ----------
        path
            Filepath to save the class to (e.g. "season_5.pickle")

        """
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> AYTO:
        """Load results from a file.

        Parameters
        ----------
        path
            Filepath to load the class from (e.g. "season_5.pickle")

        Returns
        -------
        AYTO
            An AYTO class loaded from the previously saved results.

        """
        with open(path, "rb") as f:
            return pickle.load(f)
