import pytest

from ayto import AYTO

GUYS = ["Albert", "Bob", "Charles", "Devin", "Eli"]
GIRLS = ["Faith", "Gina", "Heather", "Ingrid", "Joy"]


@pytest.fixture()
def names_long():
    return [str(i) for i in range(11)]


@pytest.fixture(scope="class")
def ayto_instance():
    return AYTO(GUYS, GIRLS)
