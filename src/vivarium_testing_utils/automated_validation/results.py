from collections import defaultdict
from dataclasses import dataclass, field
from typing import DefaultDict

from .comparison import Comparison


@dataclass
class TestResults:
    """
    Stores passing and failing test results as nested defaultdicts.
    Structure: Dict[str, Dict[str, Comparison]]
    """

    passing: DefaultDict[str, DefaultDict[str, Comparison]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(Comparison))
    )
    failing: DefaultDict[str, DefaultDict[str, Comparison]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(Comparison))
    )
