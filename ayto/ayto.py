from __future__ import annotations

from collections import defaultdict
from itertools import permutations
import logging
import pickle
from time import perf_counter

import pandas as pd

logger = logging.getLogger()
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
logger.addHandler(handler)


class AYTO:
    """A class to calculate couple probabilities for a season of Are You the One."""

    def __init__(self, guys: list[str], girls: list[str], verbose=False):
        """This class tracks a season of Are You the One through Truth Booths and Matchup Ceremonies.

        Parameters
        ----------
        guys
            A list of the names of the guys. Order doesn't matter but duplicate names should
            be disambiguated (e.g. ["Joe B.", "Joe C."])
        girls
            A list of the names of the girls. Order doesn't matter but duplicate names should
            be disambiguated (e.g. ["Jane B.", "Jane C."])
        verbose
            If `True`, the class will log messages with performance information and number
            of scenarios remaining (default is `False`)

        """
        self.guys = guys
        self.girls = girls
        self.n = len(self.guys)
        self.verbose = verbose
        self.guy_ids, self.girl_ids = self._generate_maps()
        self.scenarios = self._generate_scenarios()
        self.num_scenarios = len(self.scenarios)
        self._probs: defaultdict | None = None

    @property
    def probs(self) -> pd.DataFrame:
        """A pandas dataframe of couple probabilities."""
        if self._probs is None:
            self.calc_probs()
        return pd.DataFrame(self._probs)

    def _generate_maps(self) -> tuple[dict[str, int], dict[str, int]]:
        guy_ids = {name: id_ for id_, name in enumerate(self.guys)}
        girl_ids = {name: id_ for id_, name in enumerate(self.girls)}

        return guy_ids, girl_ids

    def _generate_scenarios(self):
        tic = perf_counter()
        girl_ids = [self.girl_ids[girl] for girl in self.girls]
        scenarios = list(permutations(girl_ids))
        toc = perf_counter()
        if self.verbose:
            sec = round(toc - tic, 1)
            logger.info(f"Generated {len(scenarios)} scenarios in {sec}s")

        return scenarios

    def _get_beams(self, scenario: tuple[int], matchup: list[int]) -> int:
        beams = 0
        for i, person in enumerate(matchup):
            if person == scenario[i]:
                beams += 1

        return beams

    def apply_truth_booth(self, guy: str, girl: str, match: bool) -> int:
        """Update scenarios to reflect a Truth Booth outcome.

        Parameters
        ----------
        guy
            Name of the guy in the Truth Booth
        girl
            Name of the girl in the Truth Booth
        match
            Whether or not the pair is a match

        Returns
        -------
        int
            Number of scenarios remaining

        """
        self.apply_matchup_ceremony([(guy, girl)], int(match))

        return self.num_scenarios

    def apply_matchup_ceremony(self, matchup: list[tuple[str, str]], beams: int) -> int:
        """Update scenarios to reflect a Matchup Ceremony.

        Parameters
        ----------
        matchup
            The couples seated together for the Matchup Ceremony as a list of tuples of
            names, e.g. [("Joe", "Sally"), ("Tim", "Jane")]
        beams
            How many beams (correct pairs) the ceremony generated

        Returns
        -------
        int
            Number of scenarios remaining

        """
        tic = perf_counter()
        matchup_ints = [-1] * self.n
        for guy, girl in matchup:
            guy_id = self.guy_ids[guy]
            girl_id = self.girl_ids[girl]
            matchup_ints[guy_id] = girl_id

        self._apply_matchup_ceremony(matchup_ints, beams)
        toc = perf_counter()
        if self.verbose:
            sec = round(toc - tic, 1)
            logger.info(f"Applied in {sec}s, {self.num_scenarios} scenarios remain")

        return self.num_scenarios

    def _apply_matchup_ceremony(self, matchup: list[int], beams: int):
        self.scenarios = [
            scenario
            for scenario in self.scenarios
            if self._get_beams(scenario, matchup) == beams
        ]
        self.num_scenarios = len(self.scenarios)

    def calc_probs(self):
        """Update the probabilities for each couple.

        This method does not return anything. To see the probabilities, see the `prob`
        attribute.

        """
        tic = perf_counter()
        counter = defaultdict(lambda: defaultdict(int))
        for scenario in self.scenarios:
            for guy, girl in enumerate(scenario):
                counter[guy][girl] += 1

        self._probs = defaultdict(lambda: defaultdict(float))
        for guy_idx, guy in enumerate(self.guys):
            total = sum(counter[guy_idx].values())
            for girl in self.girls:
                girl_idx = self.girl_ids[girl]
                self._probs[guy][girl] = counter[guy_idx][girl_idx] / total

        toc = perf_counter()
        if self.verbose:
            sec = round(toc - tic, 1)
            logger.info(f"Calculated probabilities in {sec}s")

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
