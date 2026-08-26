import gzip
import json
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

from app.config import REPO_ROOT

if TYPE_CHECKING:  # pragma: no cover - typing only
    from app.db.models import EventScope

# The committed catalog fixture: every EMS message ONTAP can emit, pulled via
# GET /api/support/ems/messages?fields=* and stripped of _links. This is
# product data, not customer data, which is why it can live in the repo — and
# it's what lets the agent explain an event with no cluster attached.
CATALOG_FIXTURE = REPO_ROOT / "data" / "ems_catalog.json.gz"

# Ceiling on how many definitions one lookup can return. Descriptions plus
# corrective actions average ~290 characters and run to ~3800 at the tail, so
# an unbounded lookup is the same hole through STAGE2_COST_CAP_USD that
# unbounded event queries were (see tools.MAX_TOOL_RESULT_EVENTS).
MAX_LOOKUP_RESULTS = 15


def _row_to_dict(row: sqlite3.Row) -> Dict[str, object]:
    return {
        "name": row["name"],
        "severity": row["severity"],
        "description": row["description"],
        "corrective_action": row["corrective_action"],
        "snmp_trap_type": row["snmp_trap_type"],
        "deprecated": bool(row["deprecated"]),
    }


def count_catalog(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT COUNT(*) AS c FROM ems_catalog").fetchone()["c"]


def load_catalog(conn: sqlite3.Connection, path: Optional[Path] = None, force: bool = False) -> int:
    """Populate ems_catalog from the gzipped fixture. A no-op when the table is
    already populated (the common case — this runs on every init_db) unless
    `force`, so the 8000-row insert happens once per database rather than once
    per connection. Returns the number of rows loaded, 0 if skipped."""
    if not force and count_catalog(conn) > 0:
        return 0
    path = path or CATALOG_FIXTURE
    if not path.exists():
        # Not fatal: the app is fully usable without the catalog, the agent
        # just loses its event definitions. Ingestion/parsing/UI don't touch it.
        return 0
    with gzip.open(path, "rt", encoding="utf-8") as f:
        payload = json.load(f)
    records = payload.get("records", [])
    conn.executemany(
        """
        INSERT OR REPLACE INTO ems_catalog
            (name, severity, description, corrective_action, snmp_trap_type, deprecated)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [
            (
                r["name"],
                r.get("severity"),
                r.get("description"),
                r.get("corrective_action"),
                r.get("snmp_trap_type"),
                1 if r.get("deprecated") else 0,
            )
            for r in records
            if r.get("name")
        ],
    )
    return len(records)


def lookup(conn: sqlite3.Connection, names: List[str]) -> List[Dict[str, object]]:
    """Exact (case-insensitive) definition lookup for a list of event names.

    Case-insensitive because the catalog's own casing ("AccessCache.NearLimits")
    is not always how the name appears in a log line or a cluster's event feed,
    and a case-sensitive miss looks identical to "this event doesn't exist" —
    which would teach the model the catalog is useless."""
    if not names:
        return []
    deduped = list(dict.fromkeys(names))[:MAX_LOOKUP_RESULTS]
    placeholders = ",".join("?" for _ in deduped)
    rows = conn.execute(
        f"SELECT * FROM ems_catalog WHERE name COLLATE NOCASE IN ({placeholders}) ORDER BY name",
        deduped,
    ).fetchall()
    return [_row_to_dict(r) for r in rows]


def search(conn: sqlite3.Connection, pattern: str, limit: int = MAX_LOOKUP_RESULTS) -> List[Dict[str, object]]:
    """Substring search over catalog names, for when the agent knows the
    subsystem ('takeover', 'disk.') but not the exact event name."""
    rows = conn.execute(
        "SELECT * FROM ems_catalog WHERE name LIKE ? ORDER BY name LIMIT ?",
        (f"%{pattern}%", max(1, min(limit, MAX_LOOKUP_RESULTS))),
    ).fetchall()
    return [_row_to_dict(r) for r in rows]


def glossary_for_events(
    conn: sqlite3.Connection,
    scope: Optional["EventScope"] = None,
    max_entries: int = 200,
    max_chars: int = 200,
) -> List[Dict[str, str]]:
    """Definitions for exactly the event names present in the run's corpus,
    trimmed to one short line each — Stage 1's grounding block.

    Scoped to distinct names actually in `events` (typically tens, not the
    catalog's 8065) so this stays bounded by corpus *variety*, not corpus size,
    and truncated because Stage 1 needs "what is this event" to judge whether a
    pattern matters; the full text and corrective action are a Stage 2 tool
    call away, priced under the cost cap.

    Takes the run's scope so a cluster-scoped run doesn't get definitions for
    another cluster's events — those would be pure token cost for names the
    model will never see in its corpus."""
    # Imported lazily to keep repo_catalog free of a module-level dependency on
    # repo_events, which imports models the other way around.
    from app.db.repo_events import _scope_clause

    inner_where, params = _scope_clause(scope)
    name_filter = f"{inner_where} AND event_name IS NOT NULL" if inner_where else " WHERE event_name IS NOT NULL"
    rows = conn.execute(
        f"""
        SELECT c.name, c.severity, c.description
        FROM ems_catalog c
        WHERE c.name COLLATE NOCASE IN (
            SELECT DISTINCT event_name FROM events{name_filter}
        )
        ORDER BY c.name
        LIMIT ?
        """,
        (*params, max_entries),
    ).fetchall()
    entries = []
    for r in rows:
        description = (r["description"] or "").strip()
        if len(description) > max_chars:
            description = description[: max_chars - 1].rstrip() + "…"
        entries.append({"name": r["name"], "severity": r["severity"] or "", "description": description})
    return entries
