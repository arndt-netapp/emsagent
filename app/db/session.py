import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

from app.config import settings

SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"


def _connect(database_path: Path) -> sqlite3.Connection:
    # check_same_thread=False: each connection is still scoped to a single
    # request/call and used sequentially, never concurrently, from more than
    # one thread — but FastAPI's threadpool-backed yield dependencies can run
    # a generator's setup and its post-yield teardown on two different pool
    # threads, which sqlite3's default same-thread check rejects outright.
    conn = sqlite3.connect(str(database_path), timeout=10, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 10000")
    return conn


def _migrate(conn: sqlite3.Connection) -> None:
    """Add/rename columns on tables that already existed before this column
    was introduced. Must run BEFORE schema.sql's executescript: schema.sql's
    CREATE INDEX statements assume columns like findings.candidate_id
    already exist, but CREATE TABLE IF NOT EXISTS is a no-op on a table
    that's already there, so an existing table only gets new columns via
    these explicit, idempotent ALTER TABLE calls. Brand-new tables (e.g.
    `candidates`) need no migration step — CREATE TABLE IF NOT EXISTS
    handles them. A no-op on a fresh database (every PRAGMA table_info
    below returns empty, so every guard is skipped)."""
    run_columns = {row["name"] for row in conn.execute("PRAGMA table_info(analysis_runs)")}
    if run_columns:
        for column in ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"):
            if column not in run_columns:
                conn.execute(f"ALTER TABLE analysis_runs ADD COLUMN {column} INTEGER DEFAULT 0")
        # Run scoping. Pre-existing runs keep NULL, which reads as "unscoped" —
        # accurate, since they analyzed every event in the database.
        for column in ("scope_mode", "scope_cluster", "scope_since", "scope_label"):
            if column not in run_columns:
                conn.execute(f"ALTER TABLE analysis_runs ADD COLUMN {column} TEXT")
        if "scope_file_id" not in run_columns:
            conn.execute("ALTER TABLE analysis_runs ADD COLUMN scope_file_id INTEGER")
        renames = [("findings_created", "candidates_generated"), ("findings_suppressed", "candidates_auto_suppressed")]
        for old, new in renames:
            if old in run_columns and new not in run_columns:
                conn.execute(f"ALTER TABLE analysis_runs RENAME COLUMN {old} TO {new}")

    finding_columns = {row["name"] for row in conn.execute("PRAGMA table_info(findings)")}
    if finding_columns and "candidate_id" not in finding_columns:
        conn.execute("ALTER TABLE findings ADD COLUMN candidate_id INTEGER")

    # Cluster scoping. Pre-existing rows keep NULL, which reads as the
    # "unspecified" pseudo-cluster — correct for dropped log files, and the
    # honest answer for cluster pulls ingested before the column existed,
    # since their cluster identity was never recorded anywhere but the filename.
    for table in ("files", "events"):
        columns = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
        if columns and "cluster" not in columns:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN cluster TEXT")

    file_columns = {row["name"] for row in conn.execute("PRAGMA table_info(files)")}
    if file_columns:
        if "duplicates_skipped" not in file_columns:
            conn.execute("ALTER TABLE files ADD COLUMN duplicates_skipped INTEGER DEFAULT 0")
        # Severity filtering. Pre-existing rows keep NULL, which reads as "no
        # floor" — accurate, since they were ingested before one existed. That
        # is also exactly the mixed-population case the column exists to make
        # visible, so it must not be backfilled with today's default.
        if "severity_filter" not in file_columns:
            conn.execute("ALTER TABLE files ADD COLUMN severity_filter TEXT")
        if "severity_skipped" not in file_columns:
            conn.execute("ALTER TABLE files ADD COLUMN severity_skipped INTEGER DEFAULT 0")

    candidate_columns = {row["name"] for row in conn.execute("PRAGMA table_info(candidates)")}
    if candidate_columns:
        for column in ("investigation_cache_creation_input_tokens", "investigation_cache_read_input_tokens"):
            if column not in candidate_columns:
                conn.execute(f"ALTER TABLE candidates ADD COLUMN {column} INTEGER")
        if "leads" not in candidate_columns:
            # Left NULL for pre-existing rows: their refs can no longer be
            # resolved reliably, so graph.load_context falls back to the
            # legacy re-derivation path for those and only those.
            conn.execute("ALTER TABLE candidates ADD COLUMN leads TEXT")


def init_db(database_path: Optional[Path] = None) -> None:
    conn = _connect(database_path or settings.database_path)
    try:
        _migrate(conn)
        conn.executescript(SCHEMA_PATH.read_text())
        # Imported here rather than at module scope: repo_catalog reads the
        # fixture path off app.config, and session.py is imported by nearly
        # everything, so a top-level import would widen the import graph for
        # no benefit. Idempotent — returns immediately once the table is
        # populated, which is every call after the first.
        from app.db import repo_catalog

        repo_catalog.load_catalog(conn)
        conn.commit()
    finally:
        conn.close()


def get_connection(database_path: Optional[Path] = None) -> sqlite3.Connection:
    database_path = database_path or settings.database_path
    init_db(database_path)
    return _connect(database_path)


@contextmanager
def session(database_path: Optional[Path] = None):
    conn = get_connection(database_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
