#!/usr/bin/env python3
"""Insert a synthetic batch of EMS events directly into the database,
bypassing ingestion/parsing entirely, so the agent loop can be exercised
without depending on parser correctness.

Usage:
    python -m scripts.seed_events
"""
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db import repo_events, repo_files  # noqa: E402
from app.db.session import session  # noqa: E402


def _events(base_time: datetime):
    def t(minutes: int) -> str:
        return (base_time + timedelta(minutes=minutes)).isoformat()

    events = []

    # Baseline noise, should not trigger any finding on its own.
    events.append({"event_time": t(0), "node": "node-c", "event_name": "callhome.reboot", "severity": "notice", "message": "System rebooted."})
    events.append({"event_time": t(5), "node": "node-c", "event_name": "cf.fm.partnerDataFine", "severity": "notice", "message": "Cluster failover partner data is fine."})

    # Escalating predictive-failure pattern on node-a.
    events.append({"event_time": t(10), "node": "node-a", "event_name": "disk.smart.error", "severity": "warning", "message": "Disk 1a.00.5 reported a SMART health warning."})
    events.append({"event_time": t(40), "node": "node-a", "event_name": "disk.smart.error", "severity": "warning", "message": "Disk 1a.00.5 reported a SMART health warning."})
    events.append({"event_time": t(70), "node": "node-a", "event_name": "raid.rg.recons.perf.degraded", "severity": "error", "message": "RAID reconstruction on disk 1a.00.5 is degraded."})
    events.append({"event_time": t(100), "node": "node-a", "event_name": "disk.predictiveFailure", "severity": "alert", "message": "Disk 1a.00.5 has been predictively failed and will be replaced."})
    events.append({"event_time": t(101), "node": "node-a", "event_name": "raid.disk.spare.needed", "severity": "alert", "message": "A spare disk is needed to replace disk 1a.00.5 on node-a."})

    # Performance-degradation burst on node-b.
    events.append({"event_time": t(200), "node": "node-b", "event_name": "qos.monitor.latency", "severity": "warning", "message": "Workload vol9_wid latency exceeded threshold of 20ms."})
    events.append({"event_time": t(215), "node": "node-b", "event_name": "qos.monitor.latency", "severity": "warning", "message": "Workload vol9_wid latency exceeded threshold of 20ms."})
    events.append({"event_time": t(230), "node": "node-b", "event_name": "monitor.disk.util.high", "severity": "error", "message": "Aggregate aggr2 disk utilization sustained above 95 percent."})
    events.append({"event_time": t(240), "node": "node-b", "event_name": "wafl.vvol.offline", "severity": "alert", "message": "Volume vol9 taken offline due to sustained high latency."})

    for i, e in enumerate(events, start=1):
        e["sequence_num"] = i
        e["raw_line"] = f"seeded: {e['event_name']} on {e['node']} at {e['event_time']}"
        e["parse_confidence"] = "high"
    return events


def main() -> None:
    base_time = datetime.now(timezone.utc)
    with session() as conn:
        file_record = repo_files.insert_file(
            conn,
            filename=f"seed-{base_time.strftime('%Y%m%dT%H%M%S')}.synthetic",
            filepath=f"seed://{uuid.uuid4()}",
            file_hash=uuid.uuid4().hex,
            file_size_bytes=0,
        )
        events = _events(base_time)
        repo_events.bulk_insert_events(conn, file_record.id, events)
        repo_files.mark_status(conn, file_record.id, "processed", detected_format="seed", event_count=len(events))
        print(f"seeded {len(events)} events into file_id={file_record.id}")


if __name__ == "__main__":
    main()
