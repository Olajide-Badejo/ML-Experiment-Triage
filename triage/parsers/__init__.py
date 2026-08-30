"""Log format parsers, all producing one of the two shared models.

`DEFAULT_PARSERS` is the order `discover_runs` tries. TensorBoard comes first
because an event directory is unambiguous, and the JSON parser comes last
because it is the most permissive about what it will accept.

`OutcomesParser` sits immediately before it, and the order is the whole point:
`JsonlParser.can_parse` says yes to any `.jsonl` at all and then raises on a file
with no step column, so a step free evaluation file placed after it would never
be looked at. In its strict default the outcomes parser claims only a file that
declares a record schema and carries no step on any record, which no training log
does. `--outcomes` relaxes that to any step free JSONL, and `parsers_for` is
where the flag turns into a parser list.
"""

from triage.parsers.base import ParseError, Parser, discover_runs
from triage.parsers.csv_parser import CsvParser
from triage.parsers.json_parser import JsonlParser
from triage.parsers.outcomes_parser import OutcomesParser
from triage.parsers.tensorboard_parser import TensorBoardParser

DEFAULT_PARSERS: list[Parser] = [
    TensorBoardParser(),
    CsvParser(),
    OutcomesParser(),
    JsonlParser(),
]


def parsers_for(outcomes: bool = False) -> list[Parser]:
    """The parser order to use, with step free JSONL read generically or not.

    `outcomes=True` is `triage ingest --outcomes`: an explicit instruction from
    somebody who knows the files are evaluation rows, which is the only basis on
    which reading an undeclared step free JSONL as outcomes is reasonable.
    """
    return [
        TensorBoardParser(),
        CsvParser(),
        OutcomesParser(strict=not outcomes),
        JsonlParser(),
    ]


__all__ = [
    "DEFAULT_PARSERS",
    "CsvParser",
    "JsonlParser",
    "OutcomesParser",
    "ParseError",
    "Parser",
    "TensorBoardParser",
    "discover_runs",
    "parsers_for",
]
