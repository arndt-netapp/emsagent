import hashlib
import logging
import sqlite3
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import get_db
from app.api.schemas import ClusterFetchRequest, FileOut
from app.config import settings
from app.ontap.client import OntapClientError, fetch_ems_events
from app.ontap.converter import event_to_text_line
from app.services.ingestion_service import ingest_file
from app.severity import normalize_minimum, ontap_query_value
from app.db import repo_files

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/clusters", tags=["clusters"])


@router.post("/fetch-events", response_model=FileOut)
def fetch_events(body: ClusterFetchRequest, conn: sqlite3.Connection = Depends(get_db)):
    """Pull recent EMS events straight from a cluster's REST API, write them
    into the watch directory, and ingest them immediately. Credentials are
    used for this request only and are never persisted."""
    # Rejected before anything is fetched or written: an unrankable floor must
    # not reach ONTAP as a query parameter, and must not be recorded on a file
    # row as a filter that was never applied.
    try:
        severity_filter = normalize_minimum(body.min_severity)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Microsecond precision, not seconds: filepath is UNIQUE, and two fetches
    # of the same cluster within one second produced the same filename — the
    # second overwrote the first's file on disk and then failed the constraint.
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    safe_cluster = body.cluster.replace("/", "_").replace(":", "_")
    output_path = settings.ems_watch_dir / f"ems_fetch_{safe_cluster}_{timestamp}.log"

    fetched = 0
    truncated_by = None
    try:
        with output_path.open("w", encoding="utf-8") as f:
            for record in fetch_ems_events(
                host=body.cluster,
                user=body.username,
                password=body.password,
                count=body.count,
                # Filtering cluster-side is what makes the floor worth having
                # on this path: `count` caps events RETURNED, so dropping
                # informational/debug at the cluster spends the same 500-event
                # budget on a wider window of the events worth analyzing rather
                # than on a shorter one of everything. ingest_file applies the
                # same floor again locally; see its comment.
                severity=ontap_query_value(severity_filter),
                log_message=body.log_message,
                verify_tls=body.verify_tls,
            ):
                f.write(event_to_text_line(record) + "\n")
                fetched += 1
    except OntapClientError as exc:
        # A large fetch is a long walk of paginated requests, and the events
        # already written are perfectly good — throwing away 9,900 of them
        # because request 100 of 100 failed would make a big fetch strictly
        # worse than a small one. Keep what arrived, ingest it, and record why
        # it stopped short; only a fetch that got nothing at all is an error.
        if fetched == 0:
            output_path.unlink(missing_ok=True)
            raise HTTPException(status_code=502, detail=f"Failed to fetch events from cluster: {exc}")
        truncated_by = str(exc)
        logger.warning(
            "cluster %s: fetch stopped after %s of %s requested events: %s",
            body.cluster,
            fetched,
            body.count,
            exc,
        )
    except Exception:
        # Anything unexpected leaves no half-written file behind in the watch
        # directory, where the next scan would pick it up as a new log.
        output_path.unlink(missing_ok=True)
        raise

    if fetched == 0:
        output_path.unlink(missing_ok=True)
        raise HTTPException(status_code=502, detail="Cluster returned no events for the given filters")

    file_hash = hashlib.sha256(output_path.read_bytes()).hexdigest()
    # The watch-directory path has always skipped files whose hash is already
    # known (watcher.discover_new_files); this path never did, so re-fetching a
    # cluster that had produced no new events since the last fetch hit the
    # UNIQUE constraint on files.file_hash and returned a 500. Identical bytes
    # mean zero new events, so there is nothing to do — 409 matches how
    # analysis_service.trigger already reports "nothing new".
    existing = repo_files.find_by_hash(conn, file_hash)
    if existing is not None:
        output_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=409,
            detail=(
                f"Cluster returned the same {fetched} events already ingested as "
                f"'{existing.filename}' — nothing new to add."
            ),
        )

    file_record = repo_files.insert_file(
        conn,
        filename=output_path.name,
        filepath=str(output_path.resolve()),
        file_hash=file_hash,
        file_size_bytes=output_path.stat().st_size,
        # The one place a cluster's identity is captured. Without it these
        # events join the "unspecified" bucket and become uncorrelatable with
        # (or worse, correlated against) another cluster's events.
        cluster=body.cluster,
        severity_filter=severity_filter,
    )
    ingested = ingest_file(conn, file_record)

    # Surfaced on the file row rather than swallowed: a user who asked for
    # 10,000 events and silently got 6,300 would read the shortfall as "the
    # cluster only had that many". ingest_file clears error_message on its own
    # success path, so this has to be written after it, not before.
    if truncated_by and ingested.status == "processed":
        repo_files.mark_status(
            conn,
            ingested.id,
            ingested.status,
            error_message=(
                f"Fetch stopped after {fetched} of the {body.count} events requested "
                f"({truncated_by}). The events above were kept."
            ),
        )
        ingested = repo_files.get_file(conn, ingested.id)

    return FileOut.model_validate(ingested, from_attributes=True)
