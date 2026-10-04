"""Database connection. Uses the DATABASE_URL from your .env file, or a local SQLite file if none."""
import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URL = f"sqlite:///{ROOT / 'idap.db'}"


def get_engine(url: str | None = None):
    return create_engine(url or os.getenv("DATABASE_URL", DEFAULT_URL))


engine = get_engine()
SessionLocal = sessionmaker(bind=engine)
