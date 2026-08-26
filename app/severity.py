"""EMS severity ranking, and the two vocabularies it has to span.

Severity reaches this app as free text, from sources that do not agree on the
words:

  * ONTAP's REST API (`message.severity`) and the shipped message catalog use
    emergency / alert / error / notice / informational / debug.
  * An autosupport EMS-LOG-FILE writes `info` where the catalog writes
    `informational` (see `samples/sample_autosupport.log`), and EMS text logs
    also carry syslog's `warning` and `critical`, which ONTAP's REST enum has
    no member for at all.
  * `ontap/converter.py` writes the literal "unknown" for a record with no
    severity, and both parsers leave severity None on an `unparsed.line` row.

Hence two lists here rather than one, and confusing them breaks something in
each direction:

  * `_RANKS` is the WIDE one, used to filter rows already in hand. It has to
    recognize every spelling any source might have written, because a name it
    does not recognize is KEPT (see `meets_minimum`) — a missing alias reads as
    "the filter silently did less than the file row claims it did".
  * `ONTAP_SEVERITIES` is the NARROW one: the only values that may appear in a
    REST query. `message.severity=warning` is not a filter that matches
    nothing, it is an invalid enum member that 400s the entire fetch.

Ranks follow syslog: lower is more severe, and every threshold is "this rank
or lower". `critical` and `warning` sit between ONTAP's own names, which is
what lets them work as thresholds anyway — `at_or_above` filters ONTAP's list
by rank, so a `warning` floor expands to emergency|alert|error and notice
falls below it, exactly as the local filter would decide.
"""

from typing import List, Optional, Tuple

_RANKS = {
    "emergency": 0,
    "alert": 1,
    "critical": 2,
    "error": 3,
    "warning": 4,
    "notice": 5,
    "informational": 6,
    # Autosupport bundles' spelling of `informational`. Confirmed in a real
    # sanitized bundle, so this alias is load-bearing rather than defensive:
    # without it every `info` line survives a notice-and-higher filter.
    "info": 6,
    "debug": 7,
}

# The severity enum ONTAP's REST API actually accepts in a query.
ONTAP_SEVERITIES = ("emergency", "alert", "error", "notice", "informational", "debug")

# The floor applied when nobody chooses one — on the fetch form, the upload
# form, and a file discovered in the watch directory. Notice-and-higher drops
# informational and debug, which are the bulk of raw EMS volume and the least
# likely to carry a storage risk, and keeps notice, where a lot of genuine
# operational signal (takeover/giveback, callhome) actually lives.
#
# This is a real default, not a suggestion: it changes which rows reach the
# database, so it is recorded on `files.severity_filter` at ingestion. See the
# "Severity filtering" section of CLAUDE.md for why that record matters.
DEFAULT_MIN_SEVERITY = "notice"

# Values meaning "no floor at all". Spelled words rather than "" or None
# because api.js's `apiUpload` drops empty-string form fields entirely, so an
# empty value would reach the server as an ABSENT field and pick up
# DEFAULT_MIN_SEVERITY — i.e. choosing "All severities" would silently filter.
_NO_FILTER = {"", "all", "any", "none"}


def canonical(severity: Optional[str]) -> Optional[str]:
    """The canonical name for a severity as written by any source, or None if
    it isn't one we know. Case-insensitive: the catalog is queried
    case-insensitively for the same reason — the wording travels through text
    logs that nobody normalized."""
    if severity is None:
        return None
    name = severity.strip().lower()
    if name not in _RANKS:
        return None
    # `info` and `informational` share a rank; return the name ONTAP uses, so
    # a canonical value is always safe to put in a REST query.
    return "informational" if name == "info" else name


def normalize_minimum(value: Optional[str]) -> Optional[str]:
    """Validate a caller-supplied severity floor, returning None for "no
    filter" and raising ValueError on a name we can't rank.

    Rejecting an unknown name rather than ignoring it is deliberate on the
    fetch path: a typo'd floor that fell through to "no filter" would pull
    every severity while the file row recorded the filter the user asked for."""
    if value is None:
        return None
    name = value.strip().lower()
    if name in _NO_FILTER:
        return None
    resolved = canonical(name)
    if resolved is None:
        raise ValueError(
            f"unknown severity '{value}'; expected one of: {', '.join(sorted(_RANKS))}, or 'all'"
        )
    return resolved


def at_or_above(minimum: Optional[str]) -> List[str]:
    """The ONTAP severity names at or above `minimum`, most severe first.

    Filtered from ONTAP_SEVERITIES rather than from `_RANKS`, so a floor of
    `warning` or `critical` — names ONTAP's enum lacks — still yields a legal
    query rather than one the cluster rejects."""
    if minimum is None:
        return list(ONTAP_SEVERITIES)
    floor = _RANKS[minimum]
    return [s for s in ONTAP_SEVERITIES if _RANKS[s] <= floor]


def ontap_query_value(minimum: Optional[str]) -> Optional[str]:
    """`minimum` as a value for the `message.severity` query parameter, or None
    when no filter applies.

    ONTAP expresses OR in a query value as `a|b|c`. `requests` percent-encodes
    the pipe to %7C, which the cluster decodes before matching. NOTE: this
    syntax has NOT been verified against a live cluster in this repo — the
    ontap fakes assert only that the string reaches `params`, so a syntax
    ONTAP rejects passes every test here and 400s in production. A rejected
    fetch surfaces as an OntapClientError carrying ONTAP's own response body,
    not as silently-unfiltered data."""
    if minimum is None:
        return None
    return "|".join(at_or_above(minimum))


def meets_minimum(severity: Optional[str], minimum: Optional[str]) -> bool:
    """Whether one event's severity clears the floor.

    Unrecognized severities — None on an `unparsed.line` row, the "unknown"
    that `converter.py` writes for a record with no severity, or a spelling
    predating this table — are KEPT. Failing open matters more than it looks:
    an unparsed line is a line the parser could not read, so dropping it would
    discard exactly the rows a human most needs to see, and it would do so
    under the heading of removing low-severity noise."""
    if minimum is None:
        return True
    rank = _RANKS.get((severity or "").strip().lower())
    if rank is None:
        return True
    return rank <= _RANKS[minimum]


def partition(events, minimum: Optional[str]) -> Tuple[list, int]:
    """Split parsed events into (kept, number dropped) by severity floor.

    Applied to ParsedEvent objects BEFORE they become event dicts, so
    `sequence_num` is assigned densely over the events that survive —
    compaction orders by (event_time, sequence_num) and numbering around holes
    would leave gaps that mean nothing."""
    if minimum is None:
        return list(events), 0
    kept = [e for e in events if meets_minimum(e.severity, minimum)]
    return kept, len(events) - len(kept)
