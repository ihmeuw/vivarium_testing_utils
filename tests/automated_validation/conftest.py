import pytest
from pathlib import Path


@pytest.fixture
def sim_result_dir():
    return Path(__file__).parent / "data/sim_outputs"
