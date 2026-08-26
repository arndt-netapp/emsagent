import sqlite3
import uuid
from typing import List, Optional

from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile

from app.api.deps import get_db
from app.api.schemas import FileOut, IngestRequest, IngestResult
from app.config import settings
from app.db import repo_files
from app.ingestion.watcher import (
    cluster_from_filename,
    compute_file_hash,
    discover_new_files,
    safe_upload_name,
)
from app.ontap.client import OntapClientError, validate_host
from app.services.ingestion_service import ingest_file, ingest_pending_files
from app.severity import DEFAULT_MIN_SEVERITY, normalize_minimum

router = APIRouter(prefix="/api/files", tags=["files"])


@router.get("", response_model=List[FileOut])
def list_files(limit: int = 100, offset: int = 0, conn: sqlite3.Connection = Depends(get_db)):
    return [FileOut.model_validate(f, from_attributes=True) for f in repo_files.list_files(conn, limit, offset)]


@router.post("/upload", response_model=FileOut)
def upload_file(
    file: UploadFile,
    cluster: Optional[str] = Form(None),
    min_severity: Optional[str] = Form(DEFAULT_MIN_SEVERITY),
    conn: sqlite3.Connection = Depends(get_db),
):
    """`cluster` is optional and names the cluster this file's events came from.

    `min_severity` is the severity floor for this file's events, defaulting to
    notice-and-higher — an autosupport bundle is mostly informational and debug
    lines, and every one of them is paid for again in Stage 1, whose cost is
    linear in corpus size. Pass "all" to keep every severity. It is recorded on
    the file row, so a database holding both filtered and unfiltered files says
    so; see app/severity.py.

    It exists because an autosupport EMS log (`EMS-LOG-FILE.txt`) carries no
    cluster identity anywhere — not in its lines, not in its name — so the
    filename convention has nothing to recover and every such upload would join
    the "unspecified" pool, where a second cluster's bundle would then be
    correlated against the first. A human uploading the bundle knows which
    cluster it is; this is the only honest place to get that from.

    It is validated as a hostname rather than accepted as a free-form label,
    because it is not a label: the cluster value IS the REST host the Stage 2
    agent later queries, with the configured ONTAP credentials attached. A
    typo'd label would merely mislabel data; an attacker-chosen one would send
    those credentials wherever it pointed. See ontap.client.validate_host."""
    # Sync def: this does blocking file + DB I/O, so FastAPI runs it in its
    # worker threadpool rather than on the event loop.
    #
    # Both checks happen before anything is written, so a rejected upload
    # leaves nothing behind in the watch directory.
    cluster = (cluster or "").strip() or None
    if cluster is not None:
        try:
            validate_host(cluster)
        except OntapClientError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        severity_filter = normalize_minimum(min_severity)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # NEVER `settings.ems_watch_dir / file.filename`: that name is
    # attacker-controlled and escapes the directory both by traversal and by
    # being absolute (pathlib drops the left operand for an absolute right
    # one). safe_upload_name is the guarantee; the containment check below is
    # the assertion that it held.
    dest = settings.ems_watch_dir / safe_upload_name(file.filename)
    if dest.exists():
        stem, suffix = dest.stem, dest.suffix
        dest = settings.ems_watch_dir / f"{stem}_{uuid.uuid4().hex[:8]}{suffix}"
    watch_dir = settings.ems_watch_dir.resolve()
    if watch_dir != dest.resolve().parent:
        raise HTTPException(status_code=400, detail="Invalid upload filename")
    contents = file.file.read()
    dest.write_bytes(contents)

    # Registers exactly the file that was just uploaded, rather than scanning
    # the watch directory. discover_new_files would also register every other
    # unknown file sitting there as `pending` while ingesting only this one —
    # and with no "Ingest pending files" button any more, those rows could never
    # be processed.
    file_hash = compute_file_hash(dest)
    existing = repo_files.find_by_hash(conn, file_hash)
    if existing is not None:
        dest.unlink(missing_ok=True)
        raise HTTPException(
            status_code=409,
            detail=f"An identical file has already been uploaded as '{existing.filename}'",
        )

    record = repo_files.insert_file(
        conn,
        filename=dest.name,
        filepath=str(dest.resolve()),
        file_hash=file_hash,
        file_size_bytes=dest.stat().st_size,
        # An explicitly stated cluster wins; otherwise fall back to recovering
        # it from this app's own fetch-output naming convention, so a bundle
        # produced by scripts/fetch_ems_events.py keeps the cluster it came
        # from instead of joining the unspecified pool where two clusters'
        # events would be correlated together. Blank stays None, which is the
        # honest answer for a log file of unknown provenance.
        cluster=cluster or cluster_from_filename(dest.name),
        severity_filter=severity_filter,
    )

    # Ingest immediately, as the cluster-fetch path already does. Upload used to
    # leave the file `pending` for a separate "Ingest pending files" step, so
    # removing that button without this would strand every upload at zero
    # events. One action, one outcome.
    ingested = ingest_file(conn, record)
    return FileOut.model_validate(ingested, from_attributes=True)


@router.post("/scan", response_model=List[FileOut])
def scan(conn: sqlite3.Connection = Depends(get_db)):
    discovered = discover_new_files(conn)
    return [FileOut.model_validate(f, from_attributes=True) for f in discovered]


@router.post("/ingest", response_model=IngestResult)
def ingest(body: IngestRequest, conn: sqlite3.Connection = Depends(get_db)):
    result = ingest_pending_files(conn, body.file_ids)
    return IngestResult(**result)


@router.get("/{file_id}", response_model=FileOut)
def get_file(file_id: int, conn: sqlite3.Connection = Depends(get_db)):
    record = repo_files.get_file(conn, file_id)
    if record is None:
        raise HTTPException(status_code=404, detail="File not found")
    return FileOut.model_validate(record, from_attributes=True)
