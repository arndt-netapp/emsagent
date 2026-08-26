#!/usr/bin/env python3
"""Fetch EMS log events from an ONTAP cluster's REST API and dump them to a
text log file in the format app.parsing.formats.ems_text_format understands.

Usable standalone (to gather sample/test log files) or imported by the web
API's cluster-fetch endpoint. Credentials are never written to disk.

Examples:
    python -m scripts.fetch_ems_events --cluster 10.0.0.5 --user admin --count 200
    EMS_CLUSTER_PASSWORD=secret python -m scripts.fetch_ems_events -c cluster.example.com -u admin -n 50 --min-severity error

Fetches notice-and-higher by default, matching the web UI. The file this writes
carries no record of that floor — uploading it applies the upload form's own
floor to whatever is in it — so a file fetched with --min-severity all and then
uploaded under the default is filtered at upload time instead.
"""
import argparse
import getpass
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402
from app.ontap.client import fetch_ems_events  # noqa: E402
from app.ontap.converter import event_to_text_line  # noqa: E402
from app.severity import DEFAULT_MIN_SEVERITY, normalize_minimum, ontap_query_value  # noqa: E402


def _default_output_path(cluster: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_cluster = cluster.replace("/", "_").replace(":", "_")
    return settings.ems_watch_dir / f"ems_fetch_{safe_cluster}_{timestamp}.log"


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cluster", "-c", required=True, help="Cluster management hostname or IP")
    parser.add_argument("--user", "-u", default=os.environ.get("EMS_CLUSTER_USER"), help="Cluster username")
    parser.add_argument("--count", "-n", type=int, default=500, help="Number of events to fetch (default: 500)")
    parser.add_argument("--output", "-o", type=Path, default=None, help="Output file path (default: watch dir)")
    parser.add_argument(
        "--min-severity",
        default=DEFAULT_MIN_SEVERITY,
        help=(
            "Lowest severity to fetch, e.g. error (default: %(default)s). "
            "Pass 'all' for every severity."
        ),
    )
    parser.add_argument("--log-message", default=None, help="Filter by log_message text (e.g. *disk*)")
    parser.add_argument("--page-size", type=int, default=500, help="Records per REST page (default: 500)")
    parser.add_argument(
        "--verify-tls", action="store_true", help="Verify the cluster's TLS certificate (default: skip verification)"
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)

    # Before the password prompt: a typo'd severity should not cost the user a
    # credential entry before it is reported. Exits rather than raising, so a
    # misspelled floor reads as a usage error instead of a traceback.
    try:
        min_severity = normalize_minimum(args.min_severity)
    except ValueError as exc:
        sys.exit(f"error: {exc}")

    if not args.user:
        args.user = input("Cluster username: ")
    password = os.environ.get("EMS_CLUSTER_PASSWORD") or getpass.getpass("Cluster password: ")

    output_path = args.output or _default_output_path(args.cluster)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    events = fetch_ems_events(
        host=args.cluster,
        user=args.user,
        password=password,
        count=args.count,
        severity=ontap_query_value(min_severity),
        log_message=args.log_message,
        verify_tls=args.verify_tls,
        page_size=args.page_size,
    )

    fetched = 0
    with output_path.open("w", encoding="utf-8") as f:
        for record in events:
            f.write(event_to_text_line(record) + "\n")
            fetched += 1

    print(f"fetched {fetched} events from {args.cluster} -> {output_path}")


if __name__ == "__main__":
    main()
