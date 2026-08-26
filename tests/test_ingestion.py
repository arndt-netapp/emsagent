from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from app.db import repo_events, repo_files
from app.db.session import session as db_session
from app.ingestion.watcher import (
    RECOGNIZED_EXTENSIONS,
    discover_new_files,
    safe_upload_name,
    scan_directory,
)
from app.parsing.base import ParsedEvent
from app.services.ingestion_service import _to_event_dicts, ingest_pending_files


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "test.db"


@pytest.fixture
def watch_dir(tmp_path):
    d = tmp_path / "watch"
    d.mkdir()
    return d


def test_scan_directory_only_picks_recognized_extensions(watch_dir):
    (watch_dir / "a.log").write_text("line one\n")
    (watch_dir / "b.txt").write_text("line one\n")
    (watch_dir / "c.csv").write_text("a,b\n")
    (watch_dir / "d.bin").write_bytes(b"\x00\x01")
    (watch_dir / "e").mkdir()

    found = {p.name for p in scan_directory(watch_dir)}
    assert found == {"a.log", "b.txt", "c.csv"}
    assert all(Path(f).suffix in RECOGNIZED_EXTENSIONS for f in found)


def test_discover_new_files_registers_pending_files(db_path, watch_dir):
    (watch_dir / "sample.log").write_text("hello\nworld\n")
    with db_session(db_path) as conn:
        discovered = discover_new_files(conn, watch_dir)
        assert len(discovered) == 1
        assert discovered[0].status == "pending"
        assert discovered[0].filename == "sample.log"


def test_discover_new_files_dedups_identical_content_on_rescan(db_path, watch_dir):
    (watch_dir / "sample.log").write_text("hello\nworld\n")
    with db_session(db_path) as conn:
        first = discover_new_files(conn, watch_dir)
        assert len(first) == 1
        second = discover_new_files(conn, watch_dir)
        assert len(second) == 0
        assert len(repo_files.list_files(conn)) == 1


def test_discover_new_files_dedups_renamed_identical_content(db_path, watch_dir):
    (watch_dir / "original.log").write_text("same content\n")
    with db_session(db_path) as conn:
        discover_new_files(conn, watch_dir)

    (watch_dir / "original.log").rename(watch_dir / "renamed.log")
    with db_session(db_path) as conn:
        second = discover_new_files(conn, watch_dir)
        assert len(second) == 0


def test_ingest_pending_files_marks_processed_with_event_count(db_path, watch_dir):
    (watch_dir / "sample.log").write_text(
        "2026-08-10T00:00:12-04:00 [node1: callhome.reboot:notice]: callhome.reboot: System rebooted.\n"
    )
    with db_session(db_path) as conn:
        discover_new_files(conn, watch_dir)
        result = ingest_pending_files(conn)

        assert result == {"processed": 1, "failed": 0, "event_count": 1}
        files = repo_files.list_files(conn)
        assert files[0].status == "processed"
        assert files[0].detected_format == "ems_text"
        assert files[0].event_count == 1


def test_ingest_pending_files_respects_file_ids_filter(db_path, watch_dir):
    (watch_dir / "a.log").write_text("2026-08-10T00:00:12-04:00 [node1: x.y:notice]: x.y: hi\n")
    (watch_dir / "b.log").write_text("2026-08-10T00:00:12-04:00 [node2: x.y:notice]: x.y: hi\n")
    with db_session(db_path) as conn:
        discovered = discover_new_files(conn, watch_dir)
        target_id = discovered[0].id

        result = ingest_pending_files(conn, file_ids=[target_id])
        assert result["processed"] == 1

        remaining_pending = repo_files.get_pending(conn)
        assert len(remaining_pending) == 1
        assert remaining_pending[0].id != target_id


@pytest.mark.parametrize(
    "supplied,expected",
    [
        # The two escapes an uploaded filename gets you. The second is the one
        # that reads as safe and isn't: pathlib DISCARDS the left operand when
        # the right side is absolute, so no ".." is needed to leave the watch
        # directory.
        ("../../pwned.log", "pwned.log"),
        ("/etc/cron.d/pwned", "pwned"),
        ("..\\..\\pwned.log", ".._.._pwned.log"),
        # Nothing usable is left, so the name is invented rather than the
        # upload rejected — `..` is not a filename anyone typed on purpose.
        ("..", "upload.log"),
        (".", "upload.log"),
        ("", "upload.log"),
        (None, "upload.log"),
        # Everything outside [A-Za-z0-9._-] is replaced, so no separator can be
        # reintroduced by a character the header decoder already unescaped.
        ("a b;rm -rf.log", "a_b_rm_-rf.log"),
        # The ordinary cases, including the fetch-output convention that
        # cluster_from_filename has to keep recovering, pass through untouched.
        ("EMS-LOG-FILE.txt", "EMS-LOG-FILE.txt"),
        ("ems_fetch_prod-a_20260819T101500Z.log", "ems_fetch_prod-a_20260819T101500Z.log"),
    ],
)
def test_safe_upload_name_reduces_a_filename_to_one_safe_component(supplied, expected):
    assert safe_upload_name(supplied) == expected


def test_event_times_are_normalized_to_utc_at_ingestion():
    """Every event_time comparison downstream is a TEXT comparison — ORDER BY,
    the scope window, the dedup BETWEEN — and lexicographic order over ISO
    strings is chronological only if they share an offset. A cluster fetch
    (Z) and an autosupport bundle (-0700) do not, so compaction's adjacency
    pass would run over a wrong ordering."""
    local = datetime(2026, 8, 25, 0, 8, 22, tzinfo=timezone(timedelta(hours=-7)))
    dicts = _to_event_dicts([ParsedEvent(local, "node1", "a.b", "notice", "m", "raw")])

    assert dicts[0]["event_time"] == "2026-08-25T07:08:22+00:00"


def test_a_naive_event_time_is_left_exactly_as_parsed():
    """The guard that makes the normalization above safe: astimezone() on a
    naive datetime assumes the MACHINE's local zone, so converting one would
    shift it by whatever the developer's box is set to. No offset in, no
    offset invented."""
    naive = datetime(2026, 8, 25, 0, 8, 22)
    dicts = _to_event_dicts([ParsedEvent(naive, "node1", "a.b", "notice", "m", "raw")])

    assert dicts[0]["event_time"] == "2026-08-25T00:08:22"


def test_mixed_offset_files_are_stored_in_a_comparable_order(db_path, watch_dir):
    """End-to-end version of the above, with offsets chosen so that string
    order and real order actually DISAGREE — the whole point.

    Untouched, these two sort by their leading date ("...08-25T20" before
    "...08-26T01") and come out backwards, which is what compaction would then
    compute node adjacency over."""
    # 20:00 -07:00 == 03:00 UTC on the 26th: the LATER of the two.
    (watch_dir / "bundle.log").write_text(
        "2026-08-25T20:00:00-07:00 [node1: disk.smart.error:warning]: second\n"
    )
    # 01:00 UTC on the 26th: the earlier one, despite the larger date string.
    (watch_dir / "fetch.log").write_text(
        "2026-08-26T01:00:00+00:00 [node1: disk.smart.error:warning]: first\n"
    )
    with db_session(db_path) as conn:
        discover_new_files(conn, watch_dir)
        ingest_pending_files(conn)

        ordered = [e.message for e in repo_events.get_all_events_ordered(conn)]

    assert ordered == ["first", "second"]


def test_ingest_file_marks_failed_on_parser_exception(db_path, watch_dir):
    (watch_dir / "bad.log").write_text("irrelevant\n")
    with db_session(db_path) as conn:
        discover_new_files(conn, watch_dir)
        with patch("app.services.ingestion_service.parse_file", side_effect=RuntimeError("boom")):
            result = ingest_pending_files(conn)

        assert result == {"processed": 0, "failed": 1, "event_count": 0}
        files = repo_files.list_files(conn)
        assert files[0].status == "failed"
        assert "boom" in files[0].error_message
