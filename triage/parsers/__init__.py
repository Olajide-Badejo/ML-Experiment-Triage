"""Log format parsers, all producing the same `Experiment` model.

`DEFAULT_PARSERS` is the order `discover_runs` tries. TensorBoard comes first
because an event directory is unambiguous, and the JSON parser comes last
because it is the most permissive about what it will accept.
"""

from triage.parsers.base import ParseError, Parser, discover_runs
from triage.parsers.csv_parser import CsvParser
from triage.parsers.json_parser import JsonlParser
from triage.parsers.tensorboard_parser import TensorBoardParser

DEFAULT_PARSERS: list[Parser] = [TensorBoardParser(), CsvParser(), JsonlParser()]

__all__ = [
    "DEFAULT_PARSERS",
    "CsvParser",
    "JsonlParser",
    "ParseError",
    "Parser",
    "TensorBoardParser",
    "discover_runs",
]
