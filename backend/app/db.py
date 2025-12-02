from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from .config import settings


def get_database_url() -> str:
    return (
        f"postgresql+psycopg2://{settings.db_user}:"
        f"{settings.db_password}@{settings.db_host}:"
        f"{settings.db_port}/{settings.db_name}"
    )


engine = create_engine(get_database_url(), echo=False, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db() -> None:
    """Create the aggregates table if it doesn't exist."""
    with engine.connect() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS aggregates (
                    id SERIAL PRIMARY KEY,
                    country VARCHAR(2) NOT NULL,
                    category VARCHAR(50) NOT NULL,
                    total_amount NUMERIC(18, 2) NOT NULL,
                    txn_count BIGINT NOT NULL
                );
                """
            )
        )
        conn.commit()


