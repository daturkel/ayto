import pytest

from ayto import AYTO

GUYS = ["Albert", "Bob", "Charles", "Devin", "Eli"]
GIRLS = ["Faith", "Gina", "Heather", "Ingrid", "Joy"]


@pytest.fixture()
def guys():
    return GUYS


@pytest.fixture()
def girls():
    return GIRLS


@pytest.fixture(scope="class")
def ayto_instance():
    return AYTO(GUYS, GIRLS)
