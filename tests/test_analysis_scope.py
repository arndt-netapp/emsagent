"""Run scoping: which events one analysis run covers.

The invariant under test throughout is that a run sees exactly one cluster.
Before scoping existed, every run analyzed the entire events table, which both
grew without bound across runs and silently correlated unrelated clusters.
"""

import sqlite3
from pathlib import Path

import pytest

from app.agent import compaction
from app.db import repo_analysis_runs, repo_events, repo_files
from app.services import analysis_service

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "app" / "db" / "schema.sql"


@pytest.fixture
def conn():
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.executescript(SCHEMA_PATH.read_text())
    return connection


def _ingest(conn, cluster, filename, events, ingested_at="2026-08-18T00:00:00+00:00"):
    """One 'pull': a file plus its events, as the fetch path would create."""
    record = repo_files.insert_file(conn, filename, f"seed://{filename}", filename, 0, cluster=cluster)
    repo_events.bulk_insert_events(
        conn,
        record.id,
        [
            {
                "raw_line": "raw",
                "event_time": t,
                "node": node,
                "event_name": name,
                "severity": "error",
                "message": "m",
                "sequence_num": i + 1,
            }
            for i, (t, node, name) in enumerate(events)
        ],
        cluster=cluster,
    )
    conn.execute(
        "UPDATE files SET status = 'processed', event_count = ?, ingested_at = ? WHERE id = ?",
        (len(events), ingested_at, record.id),
    )
    conn.commit()
    return record


# Two clusters that share node names, which is realistic: ONTAP names nodes
# <cluster>-01/-02 and the simulator defaults to "cluster1".
CLUSTER_A_EVENTS = [
    ("2026-08-17T10:00:00+00:00", "cluster1-01", "disk.smart.error"),
    ("2026-08-18T10:00:00+00:00", "cluster1-01", "disk.predictiveFailure"),
]
CLUSTER_B_EVENTS = [
    ("2026-08-18T10:00:05+00:00", "cluster1-01", "cf.fsm.takeoverOfPartnerDisabled"),
]


def test_scope_isolates_clusters(conn):
    _ingest(conn, "prod-a", "a.log", CLUSTER_A_EVENTS)
    _ingest(conn, "prod-b", "b.log", CLUSTER_B_EVENTS)

    scope = analysis_service.resolve_scope(conn, "last_24h", "prod-a")
    events = repo_events.get_all_events_ordered(conn, scope)
    assert {e.cluster for e in events} == {"prod-a"}


def test_compaction_does_not_merge_same_named_nodes_across_clusters(conn):
    """Both clusters have a node called cluster1-01 firing at nearly the same
    moment. Keyed on node alone, compaction would fold them into one group with
    a fabricated count and time span — a pattern that never happened."""
    _ingest(conn, "prod-a", "a.log", [("2026-08-18T10:00:00+00:00", "cluster1-01", "disk.smart.error")])
    _ingest(conn, "prod-b", "b.log", [("2026-08-18T10:00:01+00:00", "cluster1-01", "disk.smart.error")])

    groups = compaction.build_compact_corpus(conn)
    assert len(groups) == 2
    assert all(g.count == 1 for g in groups)


def test_last_24h_anchors_to_newest_event_not_wall_clock(conn):
    """The corpus is historical log data. A wall-clock window would return
    nothing for any file collected more than a day ago and look broken."""
    _ingest(
        conn,
        "prod-a",
        "a.log",
        [
            # Well outside a 24h window from the newest event, and years before
            # wall-clock "now" — so this passing proves the anchor is the data,
            # not the clock.
            ("2024-11-20T10:00:00+00:00", "n1", "old.event"),
            ("2024-11-26T09:00:00+00:00", "n1", "recent.event"),
        ],
    )
    scope = analysis_service.resolve_scope(conn, "last_24h", "prod-a")
    names = {e.event_name for e in repo_events.get_all_events_ordered(conn, scope)}
    assert names == {"recent.event"}


def test_recent_pull_scopes_to_one_file(conn):
    _ingest(conn, "prod-a", "old.log", CLUSTER_A_EVENTS, ingested_at="2026-08-18T00:00:00+00:00")
    newest = _ingest(
        conn,
        "prod-a",
        "new.log",
        [("2026-08-18T11:00:00+00:00", "n2", "new.event")],
        ingested_at="2026-08-18T05:00:00+00:00",
    )
    scope = analysis_service.resolve_scope(conn, "recent_pull", "prod-a")
    assert scope.file_id == newest.id
    events = repo_events.get_all_events_ordered(conn, scope)
    assert [e.event_name for e in events] == ["new.event"]


def test_unspecified_cluster_is_addressable(conn):
    """Dropped log files carry no cluster identity, so they land in the NULL
    pseudo-cluster. `cluster IS ?` (not `= ?`) is what makes it selectable —
    `= NULL` matches nothing and would silently analyze zero events."""
    _ingest(conn, None, "dropped.log", CLUSTER_A_EVENTS)
    scope = analysis_service.resolve_scope(conn, "recent_pull", None)
    assert repo_events.count_events(conn, scope) == len(CLUSTER_A_EVENTS)


def test_scope_summary_describes_only_the_scoped_events(conn):
    """This text is what the model is told its corpus consists of; a global
    summary next to a scoped corpus would describe nodes that aren't there."""
    _ingest(conn, "prod-a", "a.log", CLUSTER_A_EVENTS)
    _ingest(conn, "prod-b", "b.log", [("2026-08-18T10:00:05+00:00", "other-node", "x.y")])

    scope = analysis_service.resolve_scope(conn, "last_24h", "prod-a")
    stats = repo_events.get_scope_summary_stats(conn, scope)
    assert "other-node" not in stats["nodes"]
    assert stats["files"] == ["a.log"]
    assert stats["cluster"] == "prod-a"


def test_available_scopes_offers_two_options_per_cluster(conn):
    _ingest(conn, "prod-a", "a.log", CLUSTER_A_EVENTS)
    _ingest(conn, "prod-b", "b.log", CLUSTER_B_EVENTS)

    options = analysis_service.available_scopes(conn)
    assert {(o["mode"], o["cluster"]) for o in options} == {
        ("recent_pull", "prod-a"),
        ("last_24h", "prod-a"),
        ("recent_pull", "prod-b"),
        ("last_24h", "prod-b"),
    }
    assert all(o["event_count"] > 0 for o in options)


def test_rerun_refusal_is_per_scope_not_global(conn):
    """Analyzing cluster B must not make cluster A look 'already analyzed', and
    vice versa — a global event-count comparison would do exactly that."""
    _ingest(conn, "prod-a", "a.log", CLUSTER_A_EVENTS)
    _ingest(conn, "prod-b", "b.log", CLUSTER_B_EVENTS)

    run_id = analysis_service.trigger(conn, mode="last_24h", cluster="prod-a")
    scope = repo_analysis_runs.get_scope_for_run(conn, run_id)
    repo_analysis_runs.complete_run(
        conn,
        run_id,
        events_considered=repo_events.count_events(conn, scope),
        iterations=1,
        candidates_generated=0,
        candidates_auto_suppressed=0,
        input_tokens=0,
        output_tokens=0,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
        scope_json="{}",
    )
    conn.commit()

    # Same scope again: nothing changed, so refuse.
    with pytest.raises(analysis_service.NoNewDataError):
        analysis_service.trigger(conn, mode="last_24h", cluster="prod-a")

    # A different cluster is a different corpus and must be allowed.
    assert analysis_service.trigger(conn, mode="last_24h", cluster="prod-b")


def test_scope_survives_the_round_trip_to_the_background_task(conn):
    """execute_run receives only a run_id, so the scope has to come back off
    the row or the background task would silently analyze everything."""
    _ingest(conn, "prod-a", "a.log", CLUSTER_A_EVENTS)
    run_id = analysis_service.trigger(conn, mode="recent_pull", cluster="prod-a")

    restored = repo_analysis_runs.get_scope_for_run(conn, run_id)
    assert restored.mode == "recent_pull"
    assert restored.cluster == "prod-a"
    assert restored.file_id is not None


def test_legacy_runs_without_scope_resolve_to_none(conn):
    """Runs predating scoping analyzed the whole table; None reproduces that
    rather than crashing or inventing a scope."""
    run = repo_analysis_runs.start_run(conn)
    assert repo_analysis_runs.get_scope_for_run(conn, run.id) is None


def test_unknown_mode_is_rejected(conn):
    _ingest(conn, "prod-a", "a.log", CLUSTER_A_EVENTS)
    with pytest.raises(analysis_service.InvalidScopeError):
        analysis_service.resolve_scope(conn, "everything", "prod-a")


def test_scope_with_no_events_is_rejected(conn):
    _ingest(conn, "prod-a", "a.log", CLUSTER_A_EVENTS)
    with pytest.raises(analysis_service.InvalidScopeError):
        analysis_service.trigger(conn, mode="recent_pull", cluster="does-not-exist")


def test_file_mode_can_analyze_an_older_bundle(conn):
    """The Analyze button on the Files page. `recent_pull` only ever resolves to
    the NEWEST file for a cluster, so without a file mode an older bundle could
    not be analyzed at all."""
    older = _ingest(conn, "prod-a", "old.log", CLUSTER_A_EVENTS, ingested_at="2026-08-18T00:00:00+00:00")
    _ingest(conn, "prod-a", "new.log", [("2026-08-18T11:00:00+00:00", "n2", "new.event")],
            ingested_at="2026-08-18T05:00:00+00:00")

    scope = analysis_service.resolve_scope(conn, "file", None, file_id=older.id)
    assert scope.file_id == older.id
    assert {e.event_name for e in repo_events.get_all_events_ordered(conn, scope)} == {
        e[2] for e in CLUSTER_A_EVENTS
    }


def test_file_mode_takes_the_cluster_from_the_file(conn):
    """The file is the authority on where its events came from. Honoring a
    caller-supplied cluster would let a scope be built whose filter matches
    nothing."""
    record = _ingest(conn, "prod-b", "b.log", CLUSTER_B_EVENTS)
    scope = analysis_service.resolve_scope(conn, "file", "prod-a", file_id=record.id)
    assert scope.cluster == "prod-b"
    assert repo_events.count_events(conn, scope) == len(CLUSTER_B_EVENTS)


def test_file_mode_requires_a_file_id(conn):
    _ingest(conn, "prod-a", "a.log", CLUSTER_A_EVENTS)
    with pytest.raises(analysis_service.InvalidScopeError):
        analysis_service.resolve_scope(conn, "file", "prod-a")


def test_file_mode_rejects_an_unknown_file(conn):
    with pytest.raises(analysis_service.InvalidScopeError):
        analysis_service.resolve_scope(conn, "file", None, file_id=999)


def test_rerun_refusal_distinguishes_two_files(conn):
    """Two runs over two different files are different scopes. Keyed only on
    (mode, cluster), the second would be refused as a pointless repeat."""
    first = _ingest(conn, "prod-a", "a.log", CLUSTER_A_EVENTS, ingested_at="2026-08-18T00:00:00+00:00")
    second = _ingest(conn, "prod-a", "b.log", [("2026-08-18T12:00:00+00:00", "n9", "other.event")],
                     ingested_at="2026-08-18T06:00:00+00:00")

    run_id = analysis_service.trigger(conn, mode="file", file_id=first.id)
    scope = repo_analysis_runs.get_scope_for_run(conn, run_id)
    repo_analysis_runs.complete_run(
        conn, run_id, events_considered=repo_events.count_events(conn, scope), iterations=1,
        candidates_generated=0, candidates_auto_suppressed=0, input_tokens=0, output_tokens=0,
        cache_creation_input_tokens=0, cache_read_input_tokens=0, scope_json="{}",
    )
    conn.commit()

    with pytest.raises(analysis_service.NoNewDataError):
        analysis_service.trigger(conn, mode="file", file_id=first.id)

    assert analysis_service.trigger(conn, mode="file", file_id=second.id)
