from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, List, Tuple

from app.parsing.base import ParsedEvent
from app.parsing.formats.registry import detect_format

SAMPLE_SIZE = 50


def _raw_fallback_parse(lines: Iterable[str]) -> Iterator[ParsedEvent]:
    for line in lines:
        if not line.strip():
            continue
        yield ParsedEvent(
            event_time=None,
            node=None,
            event_name="unparsed.line",
            severity=None,
            message=line.strip(),
            raw_line=line.rstrip("\n"),
            parse_confidence="low",
        )


def parse_file(path: Path) -> Tuple[str, List[ParsedEvent]]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    non_blank = [l for l in lines if l.strip()]
    sample = non_blank[:SAMPLE_SIZE]

    parser = detect_format(sample)
    if parser is not None:
        # The file's own mtime is the anchor for formats whose timestamps carry
        # no year (autosupport). It is the best available answer, not a great
        # one: the upload path writes the file itself, so there mtime IS upload
        # time. A file copied into the watch directory with `cp -p` keeps a
        # genuinely old mtime and dates correctly. See
        # autosupport_format._assign_years for what goes wrong when it's off.
        reference_time = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        return parser.name, list(parser.parse(lines, reference_time))
    return "raw_fallback", list(_raw_fallback_parse(lines))
