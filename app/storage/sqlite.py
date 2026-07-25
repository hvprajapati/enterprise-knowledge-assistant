import sqlite3
from pathlib import Path


class SQLiteConnection:
    """Creates and manages the SQLite connection.

    Enables WAL journal mode by default so that concurrent readers
    are not blocked by writers.
    """

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection
