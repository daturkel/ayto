from __future__ import annotations

from collections import defaultdict
import json

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
        self.guy_ids, self.girl_ids = self._initialize_maps()
        self._scenarios = self._initialize_scenarios()
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

    def try_partial_scenario(
        self,
        matches: list[tuple[str, str]] | None = None,
        non_matches: list[tuple[str, str]] | None = None,
    ) -> tuple[int, float, pd.DataFrame]:
        """Get results for a hypothetical partial scenario.

        For example, if you want to see what happens if Al and Kate are a match but Joe
        and Cindy are *not* a match, pass `matches=[("Al", "Kate")], non_matches=[("Joe",
        "Cindy")]`. The results are how many scenarios match this partial scenario, the
        probability of this partial scenario, and the couple probabilities if this partial
        scenario is true.

        Parameters
        ----------
        matches
            A list of tuples of couples (with the guy first in each pair) that are matches
            in this partial scenario. If `None`, only `non_matches` will be considered.
        non_matches
            A list of tuples of couples (with guy first in each pair) that are *not* matches
            in this partial scenario. If `None`, only `matches` will be considered.

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
        if (matches is None) and (non_matches is None):
            raise ValueError("Either matches, non_matches, or both must be provided.")

        idx = np.array([True] * self.num_scenarios)

        # index = index & (all matches are true)
        if matches:
            match_ints = self._parse_matchup(matches)
            match_idx = self._get_matchup_idx(match_ints, len(matches))
            idx = idx & match_idx

        if non_matches:
            # for each nonmatch
            for non_match in non_matches:
                # index = index & this match is not true
                non_match_ints = self._parse_matchup([non_match])
                non_match_idx = self._get_matchup_idx(non_match_ints, 1)
                idx = idx & ~non_match_idx

        scenarios = self._scenarios[idx]
        num_hyp_scenarios = scenarios.shape[0]
        probabilities = self._calculate_probabilities(scenarios)

        return num_hyp_scenarios, num_hyp_scenarios / self.num_scenarios, probabilities

    def calculate_probabilities(self):
        """Update the probabilities for each couple.

        This method does not return anything. To see the probabilities, access the `probabilities`
        attribute.

        This method is only necessary if Truth Booths and Matchup Ceremonies are applied
        with `calc_prob=False` (e.g. if you are applying multiple in a row and don't want
        to calculate the probabilities until the end).

        """
        self.probabilities = self._calculate_probabilities()

    def _parse_matchup(self, matchup: list[tuple[str, str]]) -> list[int]:
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

    def _initialize_maps(self) -> tuple[dict[str, int], dict[str, int]]:
        guy_ids = {name: id_ for id_, name in enumerate(self.guys)}
        girl_ids = {name: id_ for id_, name in enumerate(self.girls)}

        return guy_ids, girl_ids

    def _initialize_scenarios(self) -> NDArray:
        scenarios = faster_permutations(self.n)

        return np.array(scenarios)

    def _initialize_probs(self):
        probs = defaultdict(lambda: defaultdict(float))
        for guy in self.guys:
            for girl in self.girls:
                probs[guy][girl] = 1 / self.n

        self.probabilities = pd.DataFrame(probs)

    def _get_matchup_idx(self, matchup: list[int], beams: int) -> NDArray:
        # count number of matches between matchup and each scenario
        sums = (self._scenarios == matchup).sum(axis=1)
        # true if number of matches is the number of beams, else false
        idx = sums == beams
        return idx

    def _calculate_probabilities(self, scenarios: NDArray | None = None):
        if scenarios is None:
            these_scenarios = self._scenarios
        else:
            these_scenarios = scenarios

        num_scenarios = these_scenarios.shape[0]
        if num_scenarios == 0:
            raise ValueError(
                "Impossible scenario provided. Did you enter contradictory data?"
            )

        probs: defaultdict[str, defaultdict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )

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
        data = {
            "guys": self.guys,
            "girls": self.girls,
            "scenarios": self._scenarios.tolist(),
            "probabilities": self.probabilities.values.tolist(),
        }
        with open(path, "w") as f:
            json.dump(data, f)

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
        with open(path, "r") as f:
            data = json.load(f)

        instance = cls(data["guys"], data["girls"])
        instance._scenarios = np.array(data["scenarios"], dtype=np.uint8, order="F")
        instance.probabilities.loc[:, :] = data["probabilities"]

        return instance
