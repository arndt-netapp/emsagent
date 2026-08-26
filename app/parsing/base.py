from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Iterator, List, Optional, Protocol


@dataclass
class ParsedEvent:
    event_time: Optional[datetime]
    node: Optional[str]
    event_name: str
    severity: Optional[str]
    message: str
    raw_line: str
    parse_confidence: str = "high"


class EmsFormatParser(Protocol):
    name: str

    def detect(self, sample_lines: List[str]) -> bool:
        ...

    def parse(
        self, lines: Iterable[str], reference_time: Optional[datetime] = None
    ) -> Iterator[ParsedEvent]:
        """`reference_time` exists for formats whose timestamps are incomplete:
        the autosupport EMS log carries no year, so it has to be anchored to
        something outside the file (see autosupport_format._assign_years).
        Formats whose timestamps are self-contained accept and ignore it.

        It is an explicit argument rather than a wall-clock read inside the
        parser so that tests are deterministic — an implicit `now()` makes a
        year-inference test that passes all year fail on New Year's Eve."""
        ...
