"""Database initialisation and session dependency."""

import os
import sqlite3
import sys
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.pool import StaticPool

_kime_db_path = os.getenv("KIME_DB_PATH")
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"sqlite:///{_kime_db_path}" if _kime_db_path else "sqlite:///./kime.db",
)

# connect_args is SQLite-specific: allow the same connection to be used across
# threads (needed for FastAPI's sync dependency injection).
if DATABASE_URL == "sqlite:///:memory:":
    # Use a single shared connection so that tables created by the test fixture
    # are visible to the request-handling sessions.
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
else:
    connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
    engine = create_engine(DATABASE_URL, connect_args=connect_args)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# Preserve the Base class across module reloads so that model classes remain
# registered on the same MetaData object even when this module is reloaded
# (e.g. by the test fixture that updates KIME_DB_PATH).  Without this guard a
# reload would create a fresh Base whose metadata has no tables, breaking
# Base.metadata.create_all().
_prior = sys.modules.get(__name__)
if _prior is not None and isinstance(getattr(_prior, "Base", None), type):
    Base: type = _prior.Base  # type: ignore[assignment]
else:
    class Base(DeclarativeBase):  # type: ignore[no-redef]
        """Shared declarative base for all ORM models."""


def init_db(db_path: str | None = None) -> None:
    """Create all tables that don't exist yet.

    When *db_path* is supplied a lightweight sqlite3 schema used by
    :func:`backend.worker.process_job` is created at that path.  When
    *db_path* is ``None`` the SQLAlchemy ORM schema is created on the
    module-level ``engine`` (used outside Alembic by the main app).
    """
    if db_path is not None:
        conn = get_connection(db_path)
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                technique TEXT
            );
            CREATE TABLE IF NOT EXISTS scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL REFERENCES jobs(id),
                feedback TEXT,
                criteria TEXT NOT NULL,
                overall_score REAL
            );
            """
        )
        conn.commit()
        conn.close()
    else:
        # Import models so their metadata is registered on Base before create_all.
        import backend.models  # noqa: F401

        Base.metadata.create_all(bind=engine)


def get_connection(db_path: str) -> sqlite3.Connection:
    """Return a :class:`sqlite3.Connection` for the database at *db_path*.

    The connection uses :attr:`sqlite3.Row` as its row factory so that
    results can be accessed by column name.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency that yields a DB session and closes it afterwards."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
