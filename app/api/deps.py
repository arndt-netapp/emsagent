import sqlite3
from typing import Iterator

from app.db.session import session as db_session


def get_db() -> Iterator[sqlite3.Connection]:
    with db_session() as conn:
        yield conn
