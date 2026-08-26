import hashlib
import re
import sqlite3
from pathlib import Path
from typing import List, Optional

from app.config import settings
from app.db.models import FileRecord
from app.db.repo_files import find_by_hash, insert_file
from app.severity import DEFAULT_MIN_SEVERITY

RECOGNIZED_EXTENSIONS = {".log", ".txt", ".csv"}

# Both the fetch form (routes_clusters) and scripts/fetch_ems_events.py write
# ems_fetch_<cluster>_<UTC timestamp>.log, so a file produced by either carries
# the cluster it came from in its own name. Recovering it matters because every
# analysis run is scoped to one cluster: without it, script output uploaded
# through the browser lands in the "unspecified" bucket, where two different
# clusters' events would sit together and be correlated against each other.
#
# The timestamp is 8 digits, T, then 6 (script) or 12 (fetch form, microseconds)
# digits and Z. The cluster segment is greedy-free so a cluster name containing
# underscores still resolves.
_FETCH_FILENAME = re.compile(r"^ems_fetch_(?P<cluster>.+)_\d{8}T\d{6}(?:\d{6})?Z$")


# Characters allowed to survive into a watch-directory filename. Everything
# else is replaced, which is what makes the result path-component-safe.
_UNSAFE_NAME_CHARS = re.compile(r"[^A-Za-z0-9._-]")

# Used when an upload's name reduces to nothing usable.
_FALLBACK_UPLOAD_NAME = "upload.log"


def safe_upload_name(filename: Optional[str]) -> str:
    """Reduce a client-supplied upload filename to a safe watch-directory name.

    An uploaded filename arrives in the `Content-Disposition` header and is
    entirely attacker-controlled. Joined onto the watch directory unmodified it
    escapes in two different ways, and both were reachable:

    * `../../x.log` traverses upward out of the directory;
    * an ABSOLUTE name escapes without any traversal at all, because
      `Path("/data/logs") / "/etc/cron.d/x"` is `/etc/cron.d/x` — pathlib
      discards the left operand entirely. This one is easy to miss when
      reviewing for "..".

    So: take the basename (which drops every directory component and both
    escapes), then replace anything outside `[A-Za-z0-9._-]` so no separator
    can be reintroduced by an encoding the header decoder handled for us.
    Names that reduce to a traversal or to nothing (``""``, ``"."``, ``".."``,
    all-dots) become `upload.log` rather than being rejected — the file's
    contents are what the user cares about, and `..` as a "name" is not a
    filename anyone typed on purpose.

    The caller still asserts containment after joining. That is deliberate
    belt-and-braces: this function is the guarantee, and the assertion is what
    turns a future regression in it into an error instead of a write."""
    name = Path(filename or "").name
    name = _UNSAFE_NAME_CHARS.sub("_", name)
    if not name.strip("."):
        return _FALLBACK_UPLOAD_NAME
    return name


def cluster_from_filename(filename: str) -> Optional[str]:
    """Recover the cluster from this app's own fetch-output naming convention.

    Returns None for any other filename, which is the honest answer: a log file
    of unknown provenance has no cluster identity, and guessing one would be
    worse than leaving it unspecified.

    Note `/` and `:` were replaced with `_` when the name was built, so a
    cluster identified by a URL rather than a host or IP won't round-trip
    exactly. Hostnames and IPs, which is what the fetch paths take, do."""
    match = _FETCH_FILENAME.match(Path(filename).stem)
    return match.group("cluster") if match else None


def scan_directory(watch_dir: Optional[Path] = None) -> List[Path]:
    watch_dir = watch_dir or settings.ems_watch_dir
    if not watch_dir.exists():
        return []
    return sorted(
        p for p in watch_dir.iterdir() if p.is_file() and p.suffix.lower() in RECOGNIZED_EXTENSIONS
    )


def compute_file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_new_files(
    conn: sqlite3.Connection,
    watch_dir: Optional[Path] = None,
    min_severity: Optional[str] = DEFAULT_MIN_SEVERITY,
) -> List[FileRecord]:
    """Register every unknown file in the watch directory as `pending`.

    `min_severity` is the floor those files will be ingested under. It carries
    the same default as the upload and fetch forms even though nobody chose it
    here — a file that appeared in the directory has no human attached to ask,
    and a discovery path that quietly kept every severity would be the one way
    to get unfiltered rows into a database whose other files are filtered,
    which is precisely the mix `files.severity_filter` exists to prevent."""
    discovered: List[FileRecord] = []
    for path in scan_directory(watch_dir):
        file_hash = compute_file_hash(path)
        if find_by_hash(conn, file_hash) is not None:
            continue
        record = insert_file(
            conn,
            filename=path.name,
            filepath=str(path.resolve()),
            file_hash=file_hash,
            cluster=cluster_from_filename(path.name),
            file_size_bytes=path.stat().st_size,
            severity_filter=min_severity,
        )
        discovered.append(record)
    return discovered
