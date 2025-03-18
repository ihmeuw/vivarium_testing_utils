from pathlib import Path

import pytest


@pytest.fixture
def sim_result_dir():
    return Path(__file__).parent / "data/sim_outputs"
