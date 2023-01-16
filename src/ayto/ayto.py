from __future__ import annotations

from collections import defaultdict
import pickle

import numpy as np
from numpy.typing import NDArray
import pandas as pd

from .utils import faster_permutations


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
        self._initialize_probs()

    @property
    def num_scenarios(self) -> int:
        """How many scenarios remain possible."""
        return self._scenarios.shape[0]

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
            names, with the guy name first in each pair, e.g. [("Joe", "Sally"), ("Tim", "Jane")]
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
        # put matchup in format understood by _apply_matchup_ceremony (will raise an exception
        # if there are any unknown names)
        matchup_ints = self._parse_matchup(matchup)

        idx = self._get_matchup_idx(matchup_ints, beams)
        self._scenarios = self._scenarios[idx]

        if calc_probs:
            self.calculate_probabilities()

        return self.num_scenarios

    def _try_partial_scenario(
        self, matchup: list[tuple[str, str]]
    ) -> tuple[int, float, pd.DataFrame]:
        """EXPERIMENTAL Get results for a hypothetical partial scenario.

        For example, if you want to know what happens if Al and Kate are a match and Joe
        and Cindy are a match, pass `matchup=[("Al", "Kate"), ("Joe", "Cindy")]`. The results
        are how many scenarios match this partial scenario, the probability of this partial
        scenario, and the couple probabilities if this partial scenario is true.

        Parameters
        ----------
        matchup
            A list of tuples of couples (with the guy first in each pair) that are matches
            in this partial scenario.

        Returns
        -------
        int
            How many scenarios match this partial scenario.
        float
            The probability of this partial scenario out of all the currently possible
            scenarios.
        pd.DataFrame
            The couple probabilities if the partial scenario is true.

        """
        matchup_ints = self._parse_matchup(matchup)
        idx = self._get_matchup_idx(matchup_ints, len(matchup_ints))
        scenarios = self._scenarios[idx]
        num_scenarios = scenarios.shape[0]
        probabilities = self._calculate_probabilities(scenarios)

        return num_scenarios, num_scenarios / self.num_scenarios, probabilities

    def calculate_probabilities(self):
        """Update the probabilities for each couple.

        This method does not return anything. To see the probabilities, access the `probabilities`
        attribute.

        This method is only necessary if Truth Booths and Matchup Ceremonies are applied
        with `calc_prob=False` (e.g. if you are applying multiple in a row and don't want
        to calculate the probabilities until the end).

        """
        self.probabilities = self._calculate_probabilities()

    def _parse_matchup(self, matchup: list[tuple[str, str]]):
        # ensure all names are known
        for guy, girl in matchup:
            if guy not in self.guy_ids:
                valid = list(self.guy_ids.keys())
                raise ValueError(f"Unknown name {guy}, must be one of {valid}")
            elif girl not in self.girl_ids:
                valid = list(self.girl_ids.keys())
                raise ValueError(f"Unknown name {girl}, must be one of {valid}")

        matchup_ints = [-1] * self.n
        for guy, girl in matchup:
            guy_id = self.guy_ids[guy]
            girl_id = self.girl_ids[girl]
            matchup_ints[guy_id] = girl_id

        return matchup_ints

    def _generate_maps(self) -> tuple[dict[str, int], dict[str, int]]:
        guy_ids = {name: id_ for id_, name in enumerate(self.guys)}
        girl_ids = {name: id_ for id_, name in enumerate(self.girls)}

        return guy_ids, girl_ids

    def _generate_scenarios(self) -> NDArray:
        scenarios = faster_permutations(self.n)

        return np.array(scenarios)

    def _get_matchup_idx(self, matchup: list[int], beams: int) -> NDArray:
        # count number of matches between matchup and each scenario
        sums = (self._scenarios == matchup).sum(axis=1)
        # true if number of matches is the number of beams, else false
        idx = sums == beams
        return idx

    def _initialize_probs(self):
        probs = defaultdict(lambda: defaultdict(float))
        for guy in self.guys:
            for girl in self.girls:
                probs[guy][girl] = 1 / self.n

        self.probabilities = pd.DataFrame(probs)

    def _calculate_probabilities(self, scenarios: NDArray | None = None):
        if scenarios is None:
            these_scenarios = self._scenarios
        else:
            these_scenarios = scenarios

        num_scenarios = these_scenarios.shape[0]

        probs = defaultdict(lambda: defaultdict(float))

        # iterate over guys
        for guy_idx, guy in enumerate(self.guys):
            # get that guy's column of the scenarios dataframe
            col = these_scenarios[:, guy_idx]
            # construct a dictionary counting how many times each girl_idx appears (it's
            # marginally faster than doing `count = (col == girl_idx).sum()` for each girl)
            unique, counts = np.unique(col, return_counts=True)
            count_dict = dict(zip(unique, counts))
            for girl_idx, girl in enumerate(self.girls):
                # populate probs dict with probaility = counts / num_scenarios
                probs[guy][girl] = count_dict.get(girl_idx, 0) / num_scenarios

        return pd.DataFrame(probs)

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
