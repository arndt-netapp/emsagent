import re
from datetime import datetime
from typing import Iterable, Iterator, List, Optional

from dateutil import parser as dateutil_parser

from app.parsing.base import ParsedEvent

# Matches the line shape emitted by app.ontap.converter.event_to_text_line:
#   <time> [<node>: <event.name>:<severity>]: <log_message>
#
# Note `^(?P<time>\S+)`: this deliberately cannot span the space-separated
# ctime timestamp of an autosupport EMS log, which is why that format falls
# through to autosupport_format.py rather than being half-parsed here (its
# three-segment bracket would otherwise land the process name in event_name).
LINE_RE = re.compile(
    r"^(?P<time>\S+) \[(?P<node>[^:]+): (?P<event_name>[^:\]]+):(?P<severity>[^\]]+)\]: (?P<message>.*)$"
)

DETECT_THRESHOLD = 0.5


class EmsTextFormatParser:
    name = "ems_text"

    def detect(self, sample_lines: List[str]) -> bool:
        if not sample_lines:
            return False
        matches = sum(1 for line in sample_lines if LINE_RE.match(line))
        return (matches / len(sample_lines)) >= DETECT_THRESHOLD

    def parse(
        self, lines: Iterable[str], reference_time: Optional[datetime] = None
    ) -> Iterator[ParsedEvent]:
        # reference_time is accepted and ignored: these timestamps are full
        # ISO-8601 and carry their own year. See EmsFormatParser.parse.
        for line in lines:
            if not line.strip():
                continue
            match = LINE_RE.match(line)
            if not match:
                yield ParsedEvent(
                    event_time=None,
                    node=None,
                    event_name="unparsed.line",
                    severity=None,
                    message=line.strip(),
                    raw_line=line.rstrip("\n"),
                    parse_confidence="low",
                )
                continue
            fields = match.groupdict()
            try:
                event_time = dateutil_parser.isoparse(fields["time"])
            except (ValueError, TypeError):
                event_time = None
            yield ParsedEvent(
                event_time=event_time,
                node=fields["node"].strip(),
                event_name=fields["event_name"].strip(),
                severity=fields["severity"].strip(),
                message=fields["message"].strip(),
                raw_line=line.rstrip("\n"),
                parse_confidence="high",
            )
