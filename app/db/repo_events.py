import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from app.db.models import EventRecord, EventScope
from dateutil import parser as dateutil_parser


# Sentinel meaning "don't filter by cluster at all", as distinct from
# `cluster=None`, which is the *unspecified* pseudo-cluster (events from
# uploaded log files) and must match itself via `IS NULL`. A plain
# `Optional[str] = None` default cannot express both, and the difference is not
# cosmetic: conflating them either leaks another cluster's events into an
# investigation or hides the uploaded-logs pool from one.
ANY_CLUSTER = object()


def _cluster_clause(cluster: Any, alias: str = "") -> Tuple[List[str], List[Any]]:
    """`(clauses, params)` restricting a query to one cluster, or nothing at all
    when the caller passed ANY_CLUSTER."""
    if cluster is ANY_CLUSTER:
        return [], []
    prefix = f"{alias}." if alias else ""
    return [f"{prefix}cluster IS ?"], [cluster]


def _row_to_event(row: sqlite3.Row) -> EventRecord:
    return EventRecord(
        id=row["id"],
        file_id=row["file_id"],
        cluster=row["cluster"],
        raw_line=row["raw_line"],
        event_time=row["event_time"],
        node=row["node"],
        event_name=row["event_name"],
        severity=row["severity"],
        message=row["message"],
        sequence_num=row["sequence_num"],
        parse_confidence=row["parse_confidence"],
        created_at=row["created_at"],
    )


def bulk_insert_events(
    conn: sqlite3.Connection, file_id: int, events: Iterable[Dict[str, Any]], cluster: Optional[str] = None
) -> int:
    """`cluster` is denormalized onto every event row from the parent file, the
    same way `node` already lives on the row rather than being joined for. It
    is what every analysis run scopes by."""
    now = datetime.now(timezone.utc).isoformat()
    rows = [
        (
            file_id,
            cluster,
            e["raw_line"],
            e.get("event_time"),
            e.get("node"),
            e["event_name"],
            e.get("severity"),
            e.get("message"),
            e["sequence_num"],
            e.get("parse_confidence", "high"),
            now,
        )
        for e in events
    ]
    if not rows:
        return 0
    conn.executemany(
        """
        INSERT INTO events
            (file_id, cluster, raw_line, event_time, node, event_name, severity, message,
             sequence_num, parse_confidence, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    return len(rows)


def existing_event_keys(
    conn: sqlite3.Connection,
    cluster: Optional[str],
    start_time: Optional[str],
    end_time: Optional[str],
) -> Set[Tuple]:
    """Natural keys of events already stored for this cluster in a time range,
    used to drop events a previous fetch already ingested.

    EMS events carry no durable id through the text log format — the ONTAP REST
    response's per-node `index` is dropped by converter.py — so identity has to
    be reconstructed from the fields that survive: node, timestamp, event name,
    severity, and message text.

    Bounded by the incoming batch's time range rather than scanning the whole
    cluster: only events overlapping that window can collide, and an unbounded
    key set would grow with total history."""
    query = "SELECT node, event_time, event_name, severity, message FROM events WHERE cluster IS ?"
    params: List[Any] = [cluster]
    if start_time and end_time:
        # Events with no parsed timestamp can't be range-filtered, so they are
        # always included rather than silently treated as non-colliding.
        query += " AND (event_time IS NULL OR event_time BETWEEN ? AND ?)"
        params.extend([start_time, end_time])
    rows = conn.execute(query, params).fetchall()
    return {
        (r["node"], r["event_time"], r["event_name"], r["severity"], r["message"]) for r in rows
    }


def query_events(
    conn: sqlite3.Connection,
    node: Optional[str] = None,
    severity: Optional[str] = None,
    event_name_contains: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    since_file_id: Optional[int] = None,
    limit: int = 100,
    cluster: Any = ANY_CLUSTER,
) -> List[EventRecord]:
    """`cluster` defaults to ANY_CLUSTER for callers that genuinely want every
    cluster; Stage 2's tools always pass the investigation's own, because ONTAP
    names nodes <cluster>-01/-02 and two clusters can present the same node
    name (see EventScope)."""
    clauses, params = _cluster_clause(cluster)
    if node:
        clauses.append("node = ?")
        params.append(node)
    if severity:
        clauses.append("severity = ?")
        params.append(severity)
    if event_name_contains:
        clauses.append("event_name LIKE ?")
        params.append(f"%{event_name_contains}%")
    if start_time:
        clauses.append("event_time >= ?")
        params.append(start_time)
    if end_time:
        clauses.append("event_time <= ?")
        params.append(end_time)
    if since_file_id is not None:
        clauses.append("file_id >= ?")
        params.append(since_file_id)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    params.append(limit)
    rows = conn.execute(
        f"SELECT * FROM events {where} ORDER BY event_time, sequence_num LIMIT ?", params
    ).fetchall()
    return [_row_to_event(r) for r in rows]


def get_event(conn: sqlite3.Connection, event_id: int) -> Optional[EventRecord]:
    row = conn.execute("SELECT * FROM events WHERE id = ?", (event_id,)).fetchone()
    return _row_to_event(row) if row else None


def get_events_near(
    conn: sqlite3.Connection,
    event_id: int,
    window_minutes: int = 30,
    limit: Optional[int] = None,
    cluster: Any = ANY_CLUSTER,
) -> List[EventRecord]:
    """Events around `event_id` on the same node.

    "Same node" is only unambiguous within one cluster — node names collide
    across clusters — so an unscoped call here can present another cluster's
    events as context for this one. Callers that know their cluster pass it."""
    anchor = get_event(conn, event_id)
    if anchor is None or not anchor.event_time:
        return []
    try:
        anchor_dt = dateutil_parser.isoparse(anchor.event_time)
    except (ValueError, TypeError):
        return []
    window = timedelta(minutes=window_minutes)
    start = (anchor_dt - window).isoformat()
    end = (anchor_dt + window).isoformat()
    cluster_clauses, cluster_params = _cluster_clause(cluster)
    query = f"""
        SELECT * FROM events
        WHERE node = ? AND event_time BETWEEN ? AND ? AND id != ?
        {"".join(f" AND {c}" for c in cluster_clauses)}
        ORDER BY event_time, sequence_num
        """
    params: List[Any] = [anchor.node, start, end, event_id, *cluster_params]
    if limit is not None:
        query += " LIMIT ?"
        params.append(limit)
    rows = conn.execute(query, params).fetchall()
    return [_row_to_event(r) for r in rows]


def count_events(conn: sqlite3.Connection, scope: Optional[EventScope] = None) -> int:
    where, params = _scope_clause(scope)
    return conn.execute(f"SELECT COUNT(*) AS c FROM events{where}", params).fetchone()["c"]


def _scope_clause(scope: Optional[EventScope], alias: str = "") -> Tuple[str, List[Any]]:
    """Turn an EventScope into a SQL fragment. `None` means unscoped (every
    event ever ingested) — which is what the Stage 2 investigation tools use,
    since they search for corroborating evidence rather than defining a run's
    corpus.

    `alias` prefixes the columns (e.g. "e") for queries that join `files`, where
    a bare `cluster` would be ambiguous — both tables have that column.

    Note `cluster IS ?` rather than `cluster = ?`: the unspecified pseudo-
    cluster is stored as NULL, and `= NULL` matches nothing in SQL. Getting
    this wrong would make dropped-log-file runs silently analyze zero events."""
    if scope is None:
        return "", []
    prefix = f"{alias}." if alias else ""
    clauses = [f"{prefix}cluster IS ?"]
    params: List[Any] = [scope.cluster]
    if scope.file_id is not None:
        clauses.append(f"{prefix}file_id = ?")
        params.append(scope.file_id)
    if scope.since is not None:
        clauses.append(f"{prefix}event_time >= ?")
        params.append(scope.since)
    return " WHERE " + " AND ".join(clauses), params


def list_clusters(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    """Every cluster present in the ingested data, with enough detail to build
    the run-scope selector. NULL sorts last as the "unspecified" bucket."""
    rows = conn.execute(
        """
        SELECT cluster,
               COUNT(*) AS event_count,
               MIN(event_time) AS min_time,
               MAX(event_time) AS max_time
        FROM events
        GROUP BY cluster
        ORDER BY (cluster IS NULL), cluster
        """
    ).fetchall()
    return [
        {
            "cluster": r["cluster"],
            "event_count": r["event_count"],
            "min_time": r["min_time"],
            "max_time": r["max_time"],
        }
        for r in rows
    ]


def get_latest_event_time(conn: sqlite3.Connection, cluster: Optional[str]) -> Optional[str]:
    """Newest event_time for one cluster — the anchor for the 24-hour window.

    Deliberately not `now()`: this tool is normally pointed at log files that
    were collected earlier, sometimes much earlier. Anchoring a "last 24 hours"
    window to wall-clock time would return zero events for any historical log
    and make the feature look broken. Anchored here, it always means the most
    recent 24 hours of that cluster's activity."""
    row = conn.execute(
        "SELECT MAX(event_time) AS t FROM events WHERE cluster IS ? AND event_time IS NOT NULL",
        (cluster,),
    ).fetchone()
    return row["t"] if row else None


def get_all_events_ordered(conn: sqlite3.Connection, scope: Optional[EventScope] = None) -> List[EventRecord]:
    """Every event in `scope`, no limit, ordered for compaction's chronological
    dedup pass. Distinct from query_events, which always applies a LIMIT.

    Unscoped it returns the whole table, which is what it did unconditionally
    before run scoping existed — that made every run re-analyze all history,
    growing without bound across runs."""
    where, params = _scope_clause(scope)
    rows = conn.execute(f"SELECT * FROM events{where} ORDER BY event_time, sequence_num", params).fetchall()
    return [_row_to_event(r) for r in rows]


def get_event_rate_baseline(
    conn: sqlite3.Connection,
    event_name: str,
    node: Optional[str] = None,
    cluster: Any = ANY_CLUSTER,
) -> Dict[str, Any]:
    """How often this event fires across the whole ingested history, so
    "unusual" can be arithmetic instead of the model's intuition.

    Returns total occurrences, the days they span, a mean per-day rate, and the
    busiest single day — plus the same figures for the rest of the cluster when
    `node` is given, which is what makes "this node is an outlier" a checkable
    claim rather than an assertion.

    Honest caveat, worth knowing before trusting the number: the baseline is
    only over what has been *ingested*, not over the cluster's real history. A
    single day's log makes every event look like it fires every day.

    All four queries below take the same `cluster` filter, deliberately: a
    per-node count scoped to one cluster next to a `corpus_days` denominator
    spanning every cluster would produce a rate that is arithmetically wrong,
    and `elsewhere_in_cluster` describing other clusters' nodes would be a
    plainly false statement rather than a loose one."""
    cluster_clauses, cluster_params = _cluster_clause(cluster)
    cluster_sql = "".join(f" AND {c}" for c in cluster_clauses)

    row = conn.execute(
        f"""
        SELECT COUNT(*) AS total,
               MIN(event_time) AS first_seen,
               MAX(event_time) AS last_seen,
               COUNT(DISTINCT substr(event_time, 1, 10)) AS distinct_days
        FROM events
        WHERE event_name = ? AND (? IS NULL OR node = ?){cluster_sql}
        """,
        (event_name, node, node, *cluster_params),
    ).fetchone()

    busiest = conn.execute(
        f"""
        SELECT substr(event_time, 1, 10) AS day, COUNT(*) AS c
        FROM events
        WHERE event_name = ? AND (? IS NULL OR node = ?) AND event_time IS NOT NULL{cluster_sql}
        GROUP BY day ORDER BY c DESC LIMIT 1
        """,
        (event_name, node, node, *cluster_params),
    ).fetchone()

    # Denominator is the corpus-wide day span, not this event's own day span:
    # dividing an event's count by the days it appeared on would report every
    # event as firing at least once a day by construction.
    corpus = conn.execute(
        f"SELECT COUNT(DISTINCT substr(event_time, 1, 10)) AS days FROM events "
        f"WHERE event_time IS NOT NULL{cluster_sql}",
        tuple(cluster_params),
    ).fetchone()
    corpus_days = corpus["days"] or 0
    total = row["total"] or 0

    result: Dict[str, Any] = {
        "event_name": event_name,
        "node": node,
        "total_occurrences": total,
        "first_seen": row["first_seen"],
        "last_seen": row["last_seen"],
        "days_seen_on": row["distinct_days"] or 0,
        "corpus_days": corpus_days,
        "mean_per_day": round(total / corpus_days, 2) if corpus_days else None,
        "busiest_day": {"day": busiest["day"], "count": busiest["c"]} if busiest else None,
    }

    if node:
        other = conn.execute(
            f"SELECT COUNT(*) AS c, COUNT(DISTINCT node) AS nodes FROM events "
            f"WHERE event_name = ? AND node != ?{cluster_sql}",
            (event_name, node, *cluster_params),
        ).fetchone()
        result["elsewhere_in_cluster"] = {
            "total_occurrences": other["c"] or 0,
            "distinct_other_nodes": other["nodes"] or 0,
        }
    return result


def get_scope_summary_stats(conn: sqlite3.Connection, scope: Optional[EventScope] = None) -> Dict[str, Any]:
    """Snapshot of the events in `scope`: used both to describe the scope to the
    agent at the start of a run and to record, per completed run, exactly which
    events/files/nodes it covered (see analysis_runs.scope_json) so that's
    visible later even after further ingestion.

    Scoped rather than global, and that distinction is load-bearing: this is the
    text the model is told its corpus consists of. A global summary next to a
    cluster-scoped corpus would tell the agent it is looking at nodes and time
    ranges that aren't in front of it."""
    where, params = _scope_clause(scope)
    total_row = conn.execute(
        f"SELECT COUNT(*) AS total, MIN(event_time) AS min_time, MAX(event_time) AS max_time FROM events{where}",
        params,
    ).fetchone()
    node_where = f"{where} AND node IS NOT NULL" if where else " WHERE node IS NOT NULL"
    nodes = [r["node"] for r in conn.execute(f"SELECT DISTINCT node FROM events{node_where} ORDER BY node", params)]
    severity_where = f"{where} AND severity IS NOT NULL" if where else " WHERE severity IS NOT NULL"
    severity_rows = conn.execute(
        f"SELECT severity, COUNT(*) AS c FROM events{severity_where} GROUP BY severity", params
    ).fetchall()
    file_where, _ = _scope_clause(scope, alias="e")
    files = [
        r["filename"]
        for r in conn.execute(
            f"""
            SELECT DISTINCT f.filename FROM files f
            JOIN events e ON e.file_id = f.id{file_where}
            ORDER BY f.filename
            """,
            params,
        )
    ]
    return {
        "total": total_row["total"],
        "min_time": total_row["min_time"],
        "max_time": total_row["max_time"],
        "nodes": nodes,
        "severity_counts": {r["severity"]: r["c"] for r in severity_rows},
        "files": files,
        "cluster": scope.cluster if scope else None,
        "scope_label": scope.label if scope else "all ingested events",
    }
