"""Database initialisation and session dependency."""

import os
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.pool import StaticPool

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./kime.db")

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


class Base(DeclarativeBase):
    """Shared declarative base for all ORM models."""


def init_db() -> None:
    """Create all tables that don't exist yet (used outside Alembic)."""
    # Import models so their metadata is registered on Base before create_all.
    import backend.models  # noqa: F401

    Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency that yields a DB session and closes it afterwards."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
