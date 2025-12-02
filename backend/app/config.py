import os


class Settings:
    # AWS / LocalStack
    aws_access_key_id: str = os.getenv("AWS_ACCESS_KEY_ID", "test")
    aws_secret_access_key: str = os.getenv("AWS_SECRET_ACCESS_KEY", "test")
    aws_region: str = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    s3_endpoint_url: str = os.getenv("S3_ENDPOINT_URL", "http://localhost:4566")
    s3_bucket_name: str = os.getenv("S3_BUCKET_NAME", "etl-bucket")

    # Postgres
    db_host: str = os.getenv("POSTGRES_HOST", "localhost")
    db_port: int = int(os.getenv("POSTGRES_PORT", "5432"))
    db_user: str = os.getenv("POSTGRES_USER", "etl_user")
    db_password: str = os.getenv("POSTGRES_PASSWORD", "etl_password")
    db_name: str = os.getenv("POSTGRES_DB", "etl_db")

    # Data / ETL config - Tối ưu cho máy 8GB RAM
    total_rows: int = int(os.getenv("ETL_TOTAL_ROWS", "2000000"))  # 2M by default (phù hợp 8GB RAM)
    chunk_size: int = int(os.getenv("ETL_CHUNK_SIZE", "200000"))  # 200k rows/chunk (tối ưu memory)
    s3_raw_prefix: str = "raw/"
    s3_processed_prefix: str = "processed/"


settings = Settings()


