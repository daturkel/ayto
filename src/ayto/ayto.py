from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

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
        self.history: list[dict] = []

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
        ids = []
        for name, lookup in [(guy, self.guy_ids), (girl, self.girl_ids)]:
            try:
                ids.append(lookup[name])
            except KeyError:
                valid = list(lookup.keys())
                raise ValueError(f"Unknown name {name}, must be one of {valid}")

        guy_idx, girl_idx = ids

        idx = self._get_truth_booth_idx(guy_idx, girl_idx, match)
        self._scenarios = self._scenarios[idx]

        if calc_probs:
            self.calculate_probabilities()

        self.history.append(
            {"type": "truth_booth", "guy": guy, "girl": girl, "match": match}
        )

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
        idx = self._get_matchup_idx(matchup, beams)
        self._scenarios = self._scenarios[idx]

        if calc_probs:
            self.calculate_probabilities()

        self.history.append(
            {"type": "matchup_ceremony", "matchup": matchup, "beams": beams}
        )

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
            match_idx = self._get_matchup_idx(matches, len(matches))
            idx = idx & match_idx

        if non_matches:
            # for each nonmatch
            for non_match in non_matches:
                # index = index & this match is not true
                non_match_idx = self._get_matchup_idx([non_match], 1)
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

    def _get_matchup_idx(self, matchup: list[tuple[str, str]], beams: int) -> NDArray:
        matchup_list = self._parse_matchup(matchup)
        # count number of matches between matchup and each scenario
        sums = (self._scenarios == matchup_list).sum(axis=1)
        # true if number of matches is the number of beams, else false
        idx = sums == beams
        return idx

    def _get_truth_booth_idx(self, guy_idx: int, girl_idx: int, match: bool) -> NDArray:
        idx = self._scenarios[:, guy_idx] == girl_idx
        if not match:
            idx = ~idx

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

    def _serialize(self) -> dict:
        return {"guys": self.guys, "girls": self.girls, "history": self.history}

    def save(self, path: Path | str):
        """Save results to a file.

        Parameters
        ----------
        path
            Filepath to save the class to (e.g. "season_5.json")

        """
        with open(path, "w") as f:
            json.dump(self._serialize(), f)

    @classmethod
    def load(cls, path: Path | str) -> AYTO:
        """Load results from a file.

        Parameters
        ----------
        path
            Filepath to load the class from (e.g. "season_5.json")

        Returns
        -------
        AYTO
            An AYTO class loaded from the previously saved results.

        """
        with open(path, "r") as f:
            data = json.load(f)

        # json doesn't support tuples, so we re-tuple-ize the matchup ceremonies
        for event in data["history"]:
            if event["type"] == "matchup_ceremony":
                event["matchup"] = [tuple(pair) for pair in event["matchup"]]

        instance = cls(data["guys"], data["girls"])
        for event in data["history"]:
            event_type = event.pop("type")
            if event_type == "truth_booth":
                instance.apply_truth_booth(**event, calc_probs=False)
            elif event_type == "matchup_ceremony":
                instance.apply_matchup_ceremony(**event, calc_probs=False)

        instance.calculate_probabilities()

        return instance
