from typing import Any, Dict


def event_to_text_line(record: Dict[str, Any]) -> str:
    """Convert one GET /api/support/ems/events record into the text line
    app/parsing/formats/ems_text_format.py parses. Keeping both directions of
    this round trip in one place keeps them from drifting out of sync."""
    time = record.get("time", "unknown-time")
    node = record.get("node", {}).get("name", "unknown-node")
    message = record.get("message", {})
    event_name = message.get("name", "unknown.event")
    severity = message.get("severity", "unknown")
    log_message = record.get("log_message", "").replace("\n", " ").strip()

    return f"{time} [{node}: {event_name}:{severity}]: {log_message}"
