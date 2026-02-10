from typing import Generator

import pandas as pd
import pytest
from _pytest.config import Config, argparsing
from _pytest.logging import LogCaptureFixture
from _pytest.python import Function
from loguru import logger


def pytest_addoption(parser: argparsing.Parser) -> None:
    parser.addoption("--runslow", action="store_true", default=False, help="run slow tests")


def pytest_configure(config: Config) -> None:
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config: Config, items: list[Function]) -> None:
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture
def caplog(caplog: LogCaptureFixture) -> Generator[LogCaptureFixture, None, None]:
    handler_id = logger.add(
        caplog.handler,
        format="{message}",
        level=0,
        filter=lambda record: record["level"].no >= caplog.handler.level,
        enqueue=False,  # Set to 'True' if your test is spawning child processes.
    )
    yield caplog
    logger.remove(handler_id)


@pytest.fixture
def simple_demographic_index() -> pd.MultiIndex:
    return pd.MultiIndex.from_tuples(
        [
            ("Male", 5, 10),
            ("Male", 10, 15),
            ("Female", 5, 10),
            ("Female", 10, 15),
        ],
        names=["sex", "age_start", "age_end"],
    )


@pytest.fixture
def observed_proportion_dataframe(simple_demographic_index: pd.MultiIndex) -> pd.DataFrame:
    return pd.DataFrame(
        {"value": [0.10, 0.25, 0.50, 0.75]},
        index=simple_demographic_index,
    )
