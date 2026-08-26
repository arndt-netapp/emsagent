"""Parser for the EMS log file carried inside an ONTAP autosupport bundle
(`EMS-LOG-FILE.txt`, downloadable per-cluster from activeiq.netapp.com).

This is the *historical* path into the app: a bundle exists for clusters this
tool can never reach over REST, and covers windows a live fetch can no longer
return. It is read-only and, unlike `ems_text_format.py`, is NOT the inverse of
`app/ontap/converter.py` — nothing in this codebase ever writes this shape.

    Tue Aug 25 00:08:22 -0700 [cluster2-n03: dense_ads_monitor: sis.auto.session.change:notice]: ADS: ...

Two things make it different from the fetched format, and both are handled here:

- **There is no year in the timestamp.** It has to be reconstructed from a
  reference time; see `_assign_years` for the rule and its failure mode.
- **There is an extra bracket segment** naming the process/thread that emitted
  the event (`dense_ads_monitor`, `secd`, `wafl_exempt14`).
"""

import re
from datetime import datetime, timedelta, timezone
from typing import Iterable, Iterator, List, Optional, Tuple

from app.parsing.base import ParsedEvent

# Matches one autosupport EMS line:
#   <ctime + offset> [<node>: <process>: <event.name>:<severity>]: <log_message>
#
# `source` (the emitting process/thread) is captured only to keep the bracket's
# group boundaries unambiguous — it is deliberately NOT carried onto ParsedEvent
# and not stored as a column. The full original text survives on `raw_line`, and
# the process is largely implied by the event name.
#
# Two details are load-bearing:
#   * `[ \d]?\d` for the day — ctime pads a single-digit day with a space
#     ("Aug  5"), so a fixed two-digit pattern silently drops nine days a month.
#   * `severity` is `[^:\]]+`, tighter than ems_text_format's `[^\]]+`. That
#     tightening is what keeps the two formats disjoint: with colons allowed,
#     this regex's `source`/`event_name`/`severity` groups would also swallow
#     ems_text's two-segment bracket. `test_the_two_formats_do_not_overlap`
#     pins the disjointness in both directions.
LINE_RE = re.compile(
    r"^(?P<time>[A-Z][a-z]{2} [A-Z][a-z]{2} [ \d]?\d \d{2}:\d{2}:\d{2} [-+]\d{4}) "
    r"\[(?P<node>[^:\]]+): (?P<source>[^:\]]+): (?P<event_name>[^:\]]+):(?P<severity>[^:\]]+)\]: "
    r"(?P<message>.*)$"
)

# Same threshold as ems_text_format: a majority of the sampled lines.
DETECT_THRESHOLD = 0.5

# The timestamp minus its year, e.g. "Tue Aug 25 00:08:22 -0700". The weekday
# is parsed and discarded rather than skipped: strptime insists on consuming
# the whole string, and a weekday that contradicts the date is not worth
# second-guessing a support bundle over. Runs of whitespace in a format string
# match runs of whitespace in the input, so "Aug  5" needs nothing special.
_TIME_FMT = "%a %b %d %H:%M:%S %z"

# How far ahead of the reference time the newest entry may sit before we
# conclude the file is from the previous year rather than this one. Generous
# enough to absorb the file's own UTC offset (up to ~14h) plus a clock skew,
# and far short of the multi-day gaps a real year rollover produces.
_FUTURE_SLACK = timedelta(hours=25)

# How much a backwards step may increase before it is read as a rollover past
# Jan 1 rather than ordinary out-of-order jitter. EMS lines are emitted in
# rough, not strict, time order; a couple of days of tolerance separates
# "these two lines are shuffled" from "this file crossed a New Year".
_ROLLOVER_TOLERANCE = timedelta(days=2)


def _parse_without_year(time_text: str) -> Optional[Tuple[int, int, int, int, int, timezone]]:
    """Split a yearless timestamp into its parts, or None if it won't parse.

    Returns the pieces rather than a datetime because a datetime cannot be
    built until the year is known — which is the whole problem this module has.
    A leap year is used for the parse so "Feb 29" survives it."""
    try:
        stamp = datetime.strptime(f"2024 {time_text}", f"%Y {_TIME_FMT}")
    except ValueError:
        return None
    return (stamp.month, stamp.day, stamp.hour, stamp.minute, stamp.second, stamp.tzinfo)


def _build(parts, year: int) -> Optional[datetime]:
    month, day, hour, minute, second, tz = parts
    try:
        return datetime(year, month, day, hour, minute, second, tzinfo=tz)
    except ValueError:
        # Feb 29 in a non-leap year. The date itself is real, so the year is
        # what's wrong: the caller steps back and tries again.
        return None


def _assign_years(stamps: List, reference_time: datetime) -> List[Optional[datetime]]:
    """Reconstruct the year for each yearless timestamp, in file order.

    The file is walked BACKWARDS from a reference time (the file's mtime; see
    `ems_parser.parse_file`), because the newest entry is the only one whose
    year can be anchored to anything external:

    1. The last entry gets the reference year, or the year before it if that
       would place it in the future.
    2. Stepping backwards, an entry that lands more than `_ROLLOVER_TOLERANCE`
       LATER than the entry after it has rolled back past Jan 1, so the year
       decrements. A file spanning a New Year therefore carries two years.

    Known failure mode, stated plainly because it cannot be fixed from inside
    the file: if the newest entry is more than a year older than the reference
    time, every year comes out low by a whole number of years. The result is
    internally consistent — ordering, adjacency, compaction and rate baselines
    all still hold — but absolute dates in findings are shifted. The upload
    path is where this bites hardest, since it writes the file itself and its
    mtime is therefore the upload time; a `cp -p` into the watch directory
    preserves a genuinely old mtime and infers correctly."""
    years: List[Optional[datetime]] = [None] * len(stamps)
    year = reference_time.year
    later: Optional[datetime] = None

    for i in range(len(stamps) - 1, -1, -1):
        parts = stamps[i]
        if parts is None:
            continue

        current = None
        for _ in range(5):  # bounded: at most one leap year in any 4 attempts
            current = _build(parts, year)
            if current is None:  # Feb 29 of a non-leap year
                year -= 1
                continue
            if later is None:
                # The newest entry: anchor it against the reference time.
                if current > reference_time + _FUTURE_SLACK:
                    year -= 1
                    continue
            elif current > later + _ROLLOVER_TOLERANCE:
                # Walking backwards moved us forward in the calendar — the
                # sequence crossed Jan 1.
                year -= 1
                continue
            break

        years[i] = current
        if current is not None:
            later = current
    return years


class AutosupportEmsFormatParser:
    name = "autosupport_ems"

    def detect(self, sample_lines: List[str]) -> bool:
        if not sample_lines:
            return False
        matches = sum(1 for line in sample_lines if LINE_RE.match(line))
        return (matches / len(sample_lines)) >= DETECT_THRESHOLD

    def parse(
        self, lines: Iterable[str], reference_time: Optional[datetime] = None
    ) -> Iterator[ParsedEvent]:
        # Materialized because year reconstruction needs the whole file before
        # it can date any single line. parse_file already reads the file into
        # memory, so this costs nothing new.
        lines = [line for line in lines if line.strip()]
        reference_time = reference_time or datetime.now(timezone.utc)

        matches = [LINE_RE.match(line) for line in lines]
        stamps = [
            _parse_without_year(m.group("time")) if m is not None else None for m in matches
        ]
        times = _assign_years(stamps, reference_time)

        for line, match, event_time in zip(lines, matches, times):
            if match is None:
                # Same fallback as ems_text_format: a line we can't read is
                # surfaced as a low-confidence row rather than dropped or
                # folded into its neighbour.
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
            yield ParsedEvent(
                event_time=event_time,
                node=fields["node"].strip(),
                event_name=fields["event_name"].strip(),
                severity=fields["severity"].strip(),
                message=fields["message"].strip(),
                raw_line=line.rstrip("\n"),
                parse_confidence="high",
            )
