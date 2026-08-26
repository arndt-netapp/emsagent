"""The severity floor: ranking, the two vocabularies, and what it refuses to
drop. See app/severity.py for why there are two lists rather than one."""

import pytest

from app.severity import (
    DEFAULT_MIN_SEVERITY,
    ONTAP_SEVERITIES,
    at_or_above,
    canonical,
    meets_minimum,
    normalize_minimum,
    ontap_query_value,
    partition,
)
from app.parsing.base import ParsedEvent


def _event(severity):
    return ParsedEvent(
        event_time=None,
        node="node1",
        event_name="a.b.c",
        severity=severity,
        message="m",
        raw_line="raw",
        parse_confidence="high",
    )


def test_the_default_floor_keeps_notice_and_drops_informational_and_debug():
    """The whole justification for defaulting to a filter: informational and
    debug are the bulk of raw EMS volume, while notice is where takeover,
    giveback and callhome events live."""
    assert meets_minimum("notice", DEFAULT_MIN_SEVERITY) is True
    assert meets_minimum("error", DEFAULT_MIN_SEVERITY) is True
    assert meets_minimum("emergency", DEFAULT_MIN_SEVERITY) is True
    assert meets_minimum("informational", DEFAULT_MIN_SEVERITY) is False
    assert meets_minimum("debug", DEFAULT_MIN_SEVERITY) is False


def test_autosupport_spells_informational_as_info_and_the_floor_knows_it():
    """`samples/sample_autosupport.log` writes `wafl.vol.snap_create.done:info`
    where the catalog says `informational`. Without the alias every info line
    in a bundle survives a notice floor, because an unrecognized severity is
    deliberately kept."""
    assert canonical("info") == "informational"
    assert meets_minimum("info", "notice") is False
    assert meets_minimum("INFO", "notice") is False


def test_severity_names_are_matched_case_insensitively():
    assert meets_minimum("Debug", "notice") is False
    assert meets_minimum("ERROR", "notice") is True


def test_syslog_names_ontap_does_not_use_still_rank():
    """EMS text logs carry `warning` (see samples/sample_ems_events.log), which
    ONTAP's REST enum has no member for. It has to rank, or every warning line
    in an uploaded log would be kept by a floor that should drop it — and, at
    error-and-higher, dropped by one that should keep it."""
    assert meets_minimum("warning", "notice") is True
    assert meets_minimum("warning", "error") is False
    assert meets_minimum("critical", "error") is True


def test_an_unrecognized_severity_is_kept_rather_than_dropped():
    """Failing open is the point: an `unparsed.line` row has severity None and
    is a line the parser could not read, so dropping it would discard exactly
    what a human most needs to see — under the heading of removing noise.
    `converter.py` writes the literal "unknown" for the same reason."""
    assert meets_minimum(None, "emergency") is True
    assert meets_minimum("unknown", "emergency") is True
    assert meets_minimum("something-ontap-invents-in-2030", "emergency") is True


def test_no_floor_keeps_everything():
    assert meets_minimum("debug", None) is True
    assert partition([_event("debug"), _event("info")], None) == (
        [_event("debug"), _event("info")],
        0,
    )


def test_partition_reports_how_many_it_dropped():
    events = [_event("emergency"), _event("info"), _event("debug"), _event("notice")]
    kept, dropped = partition(events, "notice")
    assert [e.severity for e in kept] == ["emergency", "notice"]
    assert dropped == 2


def test_the_ontap_query_lists_severities_at_or_above_the_floor():
    assert ontap_query_value("notice") == "emergency|alert|error|notice"
    assert ontap_query_value("error") == "emergency|alert|error"
    assert ontap_query_value("emergency") == "emergency"
    assert ontap_query_value(None) is None


def test_the_ontap_query_never_names_a_severity_outside_ontaps_enum():
    """`message.severity=warning` is not a filter that matches nothing, it is
    an invalid enum member that 400s the entire fetch. A floor of `warning` or
    `critical` is legal locally, so the expansion has to drop the name itself
    while still applying its rank."""
    assert ontap_query_value("warning") == "emergency|alert|error"
    assert ontap_query_value("critical") == "emergency|alert"
    for floor in ("emergency", "alert", "critical", "error", "warning", "notice", "debug"):
        assert set(at_or_above(floor)) <= set(ONTAP_SEVERITIES)


def test_all_and_none_mean_no_floor_but_a_typo_is_an_error():
    """A typo'd floor that fell through to "no filter" would pull every
    severity while the file row recorded the filter the user asked for."""
    assert normalize_minimum("all") is None
    assert normalize_minimum("") is None
    assert normalize_minimum(None) is None
    assert normalize_minimum("  ERROR ") == "error"
    assert normalize_minimum("info") == "informational"
    with pytest.raises(ValueError):
        normalize_minimum("noticee")
    with pytest.raises(ValueError):
        normalize_minimum("high")
