import argparse

from app.db.session import session
from app.ingestion.watcher import discover_new_files
from app.services.ingestion_service import ingest_pending_files


def main() -> None:
    parser = argparse.ArgumentParser(description="EMS log ingestion CLI")
    parser.add_argument("command", choices=["scan", "ingest"])
    args = parser.parse_args()

    with session() as conn:
        if args.command == "scan":
            discovered = discover_new_files(conn)
            print(f"discovered {len(discovered)} new file(s)")
            for f in discovered:
                print(f"  [{f.id}] {f.filename}")
        elif args.command == "ingest":
            discovered = discover_new_files(conn)
            result = ingest_pending_files(conn)
            print(f"discovered {len(discovered)} new file(s)")
            print(
                f"ingested: processed={result['processed']} failed={result['failed']} "
                f"events={result['event_count']}"
            )


if __name__ == "__main__":
    main()
