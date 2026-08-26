from datetime import datetime, timezone
from pathlib import Path

from app.parsing.ems_parser import parse_file
from app.parsing.formats.autosupport_format import AutosupportEmsFormatParser
from app.parsing.formats.ems_text_format import EmsTextFormatParser

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"

# An autosupport EMS line carries no year, so every test that cares about the
# resulting datetime passes an explicit reference time. Never rely on the
# implicit wall-clock default here: that makes these tests pass all year and
# fail on New Year's Eve.
AUTOSUPPORT_LINE = (
    "Tue Aug 25 00:08:22 -0700 "
    "[cluster2-n03: dense_ads_monitor: sis.auto.session.change:notice]: "
    "ADS: Number of auto sessions changed from 3 to 4"
)
EMS_TEXT_LINE = "2026-08-10T00:00:12-04:00 [node1: callhome.reboot:notice]: callhome.reboot: System rebooted."


def test_ems_text_format_parses_well_formed_line():
    line = "2026-08-10T01:15:33-04:00 [node1: disk.predictiveFailure:alert]: disk.predictiveFailure: Disk 0a.00.3 has been predictively failed and will be replaced."
    parser = EmsTextFormatParser()
    [event] = list(parser.parse([line]))

    assert event.node == "node1"
    assert event.event_name == "disk.predictiveFailure"
    assert event.severity == "alert"
    assert event.message == "disk.predictiveFailure: Disk 0a.00.3 has been predictively failed and will be replaced."
    assert event.event_time is not None
    assert event.event_time.year == 2026
    assert event.parse_confidence == "high"


def test_ems_text_format_falls_back_on_malformed_line():
    parser = EmsTextFormatParser()
    [event] = list(parser.parse(["### corrupted log line: buffer truncated mid-write ###"]))

    assert event.event_name == "unparsed.line"
    assert event.parse_confidence == "low"
    assert event.node is None


def test_ems_text_format_detect_requires_majority_match():
    parser = EmsTextFormatParser()
    good_line = "2026-08-10T00:00:12-04:00 [node1: callhome.reboot:notice]: callhome.reboot: System rebooted."
    assert parser.detect([good_line, good_line, "garbage"]) is True
    assert parser.detect(["garbage", "garbage", good_line]) is False


def test_parse_file_detects_ems_text_format_and_extracts_fields(tmp_path):
    content = (
        "2026-08-10T00:00:12-04:00 [node1: callhome.reboot:notice]: callhome.reboot: System rebooted.\n"
        "2026-08-10T00:01:03-04:00 [node2: wafl.vol.autoSize.done:notice]: wafl.vol.autoSize.done: done.\n"
    )
    path = tmp_path / "sample.log"
    path.write_text(content)

    format_name, events = parse_file(path)

    assert format_name == "ems_text"
    assert len(events) == 2
    assert events[0].node == "node1"
    assert events[1].event_name == "wafl.vol.autoSize.done"
    assert all(e.parse_confidence == "high" for e in events)


def test_parse_file_falls_back_to_raw_when_no_format_matches(tmp_path):
    path = tmp_path / "unstructured.txt"
    path.write_text("just some plain text\nwith no structure at all\n")

    format_name, events = parse_file(path)

    assert format_name == "raw_fallback"
    assert len(events) == 2
    assert all(e.parse_confidence == "low" for e in events)
    assert all(e.event_name == "unparsed.line" for e in events)


def test_parse_checked_in_sample_log():
    format_name, events = parse_file(SAMPLES_DIR / "sample_ems_events.log")

    assert format_name == "ems_text"
    high_confidence = [e for e in events if e.parse_confidence == "high"]
    low_confidence = [e for e in events if e.parse_confidence == "low"]

    assert len(high_confidence) == 15
    assert len(low_confidence) == 2

    nodes = {e.node for e in high_confidence}
    assert nodes == {"node1", "node2", "node3"}

    predictive = [e for e in high_confidence if e.event_name == "disk.predictiveFailure"]
    assert len(predictive) == 1
    assert predictive[0].severity == "alert"

    performance = [e for e in high_confidence if e.event_name == "wafl.vvol.offline"]
    assert len(performance) == 1
    assert performance[0].node == "node2"


# --- autosupport EMS log (EMS-LOG-FILE.txt) ---------------------------------


def test_autosupport_format_parses_well_formed_line():
    parser = AutosupportEmsFormatParser()
    [event] = list(parser.parse([AUTOSUPPORT_LINE], datetime(2026, 8, 25, tzinfo=timezone.utc)))

    assert event.node == "cluster2-n03"
    assert event.event_name == "sis.auto.session.change"
    assert event.severity == "notice"
    assert event.message == "ADS: Number of auto sessions changed from 3 to 4"
    assert event.parse_confidence == "high"


def test_autosupport_format_drops_the_process_but_keeps_it_in_raw_line():
    """The emitting process/thread is parsed out to keep the bracket's group
    boundaries unambiguous, then deliberately discarded — it is not a column
    and must not leak into any structured field. raw_line is where it lives."""
    parser = AutosupportEmsFormatParser()
    [event] = list(parser.parse([AUTOSUPPORT_LINE], datetime(2026, 8, 25, tzinfo=timezone.utc)))

    assert "dense_ads_monitor" not in (event.event_name, event.node, event.severity)
    assert "dense_ads_monitor" not in event.message
    assert "dense_ads_monitor" in event.raw_line


def test_the_two_formats_do_not_overlap():
    """Neither parser may claim the other's lines.

    This is the property that makes registry.PARSERS order irrelevant. It is
    not free: ems_text's severity group would swallow the autosupport bracket's
    trailing segments if it weren't for the timestamp anchor, and the
    autosupport severity group is `[^:\\]]+` rather than `[^\\]]+` precisely so
    it can't match ems_text's two-segment bracket."""
    ems_text, autosupport = EmsTextFormatParser(), AutosupportEmsFormatParser()

    assert ems_text.detect([EMS_TEXT_LINE]) is True
    assert ems_text.detect([AUTOSUPPORT_LINE]) is False
    assert autosupport.detect([AUTOSUPPORT_LINE]) is True
    assert autosupport.detect([EMS_TEXT_LINE]) is False


def test_autosupport_format_falls_back_on_malformed_line():
    parser = AutosupportEmsFormatParser()
    [event] = list(
        parser.parse(["### banner: sanitized bundle ###"], datetime(2026, 8, 25, tzinfo=timezone.utc))
    )

    assert event.event_name == "unparsed.line"
    assert event.parse_confidence == "low"
    assert event.event_time is None


def test_autosupport_format_handles_space_padded_day():
    """ctime pads a single-digit day with a space ("Aug  5"). A fixed
    two-digit pattern silently drops nine days out of every month."""
    parser = AutosupportEmsFormatParser()
    line = "Tue Aug  5 00:08:22 -0700 [n1: proc: cf.fm.noPartner:error]: no partner"
    [event] = list(parser.parse([line], datetime(2026, 8, 25, tzinfo=timezone.utc)))

    assert event.event_time is not None
    assert (event.event_time.month, event.event_time.day) == (8, 5)


def test_autosupport_year_is_taken_from_the_reference_time():
    parser = AutosupportEmsFormatParser()
    [event] = list(parser.parse([AUTOSUPPORT_LINE], datetime(2026, 8, 25, 18, tzinfo=timezone.utc)))

    assert event.event_time.year == 2026


def test_autosupport_year_steps_back_rather_than_dating_a_file_in_the_future():
    """A December file read in January is last year's, not next year's."""
    parser = AutosupportEmsFormatParser()
    line = "Sun Dec 07 12:00:00 -0000 [n1: proc: cf.fm.noPartner:error]: no partner"
    [event] = list(parser.parse([line], datetime(2026, 1, 5, tzinfo=timezone.utc)))

    assert (event.event_time.year, event.event_time.month) == (2025, 12)


def test_autosupport_file_spanning_new_year_carries_two_years():
    parser = AutosupportEmsFormatParser()
    lines = [
        "Wed Dec 31 23:59:58 -0700 [n1: proc: a.b.c:info]: last of the year",
        "Thu Jan 01 00:00:04 -0700 [n1: proc: a.b.c:info]: first of the year",
        "Mon Jan 05 00:00:04 -0700 [n1: proc: a.b.c:info]: later",
    ]
    events = list(parser.parse(lines, datetime(2026, 1, 5, 12, tzinfo=timezone.utc)))

    assert [e.event_time.year for e in events] == [2025, 2026, 2026]
    # And the reconstructed sequence is still monotonic, which is the whole
    # point: compaction and the last_24h anchor both order by event_time.
    assert events[0].event_time < events[1].event_time < events[2].event_time


def test_autosupport_out_of_order_jitter_does_not_roll_the_year_back():
    """EMS lines are emitted in rough, not strict, time order. A few seconds of
    shuffle must not read as a rollover past Jan 1 — which would date the
    earlier line a full year before its neighbour."""
    parser = AutosupportEmsFormatParser()
    lines = [
        "Tue Aug 25 00:00:05 -0700 [n1: proc: a.b.c:info]: later, listed first",
        "Tue Aug 25 00:00:01 -0700 [n1: proc: a.b.c:info]: earlier, listed second",
    ]
    events = list(parser.parse(lines, datetime(2026, 8, 26, tzinfo=timezone.utc)))

    assert {e.event_time.year for e in events} == {2026}


def test_autosupport_leap_day_resolves_to_a_leap_year():
    """Feb 29 cannot be built in the reference year, and the date is real — so
    the year is what's wrong. Stepping back finds the most recent leap year."""
    parser = AutosupportEmsFormatParser()
    line = "Thu Feb 29 12:00:00 -0000 [n1: proc: a.b.c:info]: leap day"
    [event] = list(parser.parse([line], datetime(2025, 6, 1, tzinfo=timezone.utc)))

    assert (event.event_time.year, event.event_time.month, event.event_time.day) == (2024, 2, 29)


def test_parse_checked_in_autosupport_sample(tmp_path):
    # Copied so the reference time (the file's mtime) is deterministic rather
    # than whenever the repo happened to be checked out.
    path = tmp_path / "EMS-LOG-FILE.txt"
    path.write_bytes((SAMPLES_DIR / "sample_autosupport.log").read_bytes())

    format_name, events = parse_file(path)

    assert format_name == "autosupport_ems"
    assert len(events) == 9
    assert all(e.parse_confidence == "high" for e in events)
    assert {e.node for e in events} == {"cluster2-n03"}
    assert all(e.event_time is not None for e in events)
    # Real ONTAP event names, all of which resolve against data/ems_catalog.json.gz.
    assert "sis.auto.session.change" in {e.event_name for e in events}
    assert {e.severity for e in events} >= {"notice", "error", "info", "debug"}
