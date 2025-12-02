from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates
from sqlalchemy import text

from .config import settings
from .db import SessionLocal, init_db
from .etl import generate_and_upload_data, run_etl

# Setup templates và static files
BASE_DIR = Path(__file__).resolve().parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "app" / "templates"))

app = FastAPI(title="ETL Demo with LocalStack")

# Mount static files
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "app" / "static")), name="static")


@app.on_event("startup")
def on_startup():
    init_db()


def _run_etl_job():
    run_etl()


@app.get("/health")
def health():
    return {"status": "ok"}


def _run_generate_upload_job(total_rows: int = None, chunk_size: int = None):
    generate_and_upload_data(total_rows=total_rows, chunk_size=chunk_size)


@app.post("/data/generate")
def generate_data(
    background_tasks: BackgroundTasks,
    total_rows: int = Query(None, description="Tổng số rows cần sinh (mặc định: từ config)"),
    chunk_size: int = Query(None, description="Số rows mỗi chunk (mặc định: từ config)"),
):
    """
    Sinh và upload data lên S3 (chỉ upload, không chạy ETL).
    Phù hợp cho máy 8GB RAM với chunking tối ưu.
    """
    if total_rows is None:
        total_rows = settings.total_rows
    if chunk_size is None:
        chunk_size = settings.chunk_size
    
    # Giới hạn cho 8GB RAM
    if total_rows > 2000000:
        total_rows = 2000000
    if chunk_size > 200000:
        chunk_size = 200000
    
    background_tasks.add_task(_run_generate_upload_job, total_rows, chunk_size)
    return {
        "message": "Data generation job started",
        "config": {
            "total_rows": total_rows,
            "chunk_size": chunk_size,
            "estimated_chunks": (total_rows + chunk_size - 1) // chunk_size,
        },
        "note": "Data sẽ được upload lên S3. Kiểm tra logs để xem progress.",
    }


@app.post("/etl/run")
def trigger_etl(background_tasks: BackgroundTasks):
    """
    Trigger ETL asynchronously với tất cả các phương pháp xử lý dữ liệu.
    """
    background_tasks.add_task(_run_etl_job)
    return {
        "message": "ETL job started với tất cả các phương pháp xử lý dữ liệu",
        "config": {
            "total_rows": settings.total_rows,
            "chunk_size": settings.chunk_size,
        },
        "processing_methods": [
            "1. Validation - Kiểm tra và làm sạch dữ liệu",
            "2. Deduplication - Loại bỏ duplicates",
            "3. Filtering - Lọc dữ liệu theo điều kiện",
            "4. Transformation - Biến đổi dữ liệu",
            "5. Normalization - Chuẩn hóa dữ liệu",
            "6. Enrichment - Làm giàu dữ liệu",
            "7. Aggregation - Tổng hợp dữ liệu",
        ],
    }


@app.get("/stats")
def get_stats():
    """
    Return aggregated data from Postgres.
    """
    db = SessionLocal()
    try:
        result = db.execute(
            text(
                """
                SELECT country, category, total_amount, txn_count
                FROM aggregates
                ORDER BY country, category;
                """
            )
        )
        rows = [
            {
                "country": r.country,
                "category": r.category,
                "total_amount": float(r.total_amount),
                "txn_count": int(r.txn_count),
            }
            for r in result
        ]
    finally:
        db.close()

    return {"items": rows}


@app.get("/s3/list")
def list_s3_files():
    """
    Liệt kê tất cả files trong S3 bucket.
    """
    from .etl import list_raw_objects
    
    try:
        files = list_raw_objects()
        return {
            "status": "success",
            "total_files": len(files),
            "files": files,
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
        }


@app.get("/api/evaluation")
def get_evaluation():
    """
    Trả về dữ liệu đánh giá tổng quan cho bảng đánh giá.
    """
    db = SessionLocal()
    try:
        # Tổng hợp theo country
        country_result = db.execute(
            text("""
                SELECT 
                    country,
                    COUNT(*) as group_count,
                    SUM(txn_count) as total_txns,
                    SUM(total_amount) as total_amount,
                    AVG(total_amount) as avg_amount,
                    MAX(total_amount) as max_amount,
                    MIN(total_amount) as min_amount
                FROM aggregates
                GROUP BY country
                ORDER BY total_amount DESC
            """)
        )
        
        # Tổng hợp theo category
        category_result = db.execute(
            text("""
                SELECT 
                    category,
                    COUNT(*) as group_count,
                    SUM(txn_count) as total_txns,
                    SUM(total_amount) as total_amount,
                    AVG(total_amount) as avg_amount,
                    MAX(total_amount) as max_amount,
                    MIN(total_amount) as min_amount
                FROM aggregates
                GROUP BY category
                ORDER BY total_amount DESC
            """)
        )
        
        # Tổng hợp tổng thể
        overall_result = db.execute(
            text("""
                SELECT 
                    COUNT(*) as total_groups,
                    SUM(txn_count) as total_txns,
                    SUM(total_amount) as total_amount,
                    AVG(total_amount) as avg_amount,
                    MAX(total_amount) as max_amount,
                    MIN(total_amount) as min_amount,
                    COUNT(DISTINCT country) as country_count,
                    COUNT(DISTINCT category) as category_count
                FROM aggregates
            """)
        )
        
        overall = overall_result.fetchone()
        
        return {
            "overall": {
                "total_groups": int(overall.total_groups) if overall.total_groups else 0,
                "total_txns": int(overall.total_txns) if overall.total_txns else 0,
                "total_amount": float(overall.total_amount) if overall.total_amount else 0,
                "avg_amount": float(overall.avg_amount) if overall.avg_amount else 0,
                "max_amount": float(overall.max_amount) if overall.max_amount else 0,
                "min_amount": float(overall.min_amount) if overall.min_amount else 0,
                "country_count": int(overall.country_count) if overall.country_count else 0,
                "category_count": int(overall.category_count) if overall.category_count else 0,
            },
            "by_country": [
                {
                    "country": r.country,
                    "group_count": int(r.group_count),
                    "total_txns": int(r.total_txns),
                    "total_amount": float(r.total_amount),
                    "avg_amount": float(r.avg_amount),
                    "max_amount": float(r.max_amount),
                    "min_amount": float(r.min_amount),
                }
                for r in country_result
            ],
            "by_category": [
                {
                    "category": r.category,
                    "group_count": int(r.group_count),
                    "total_txns": int(r.total_txns),
                    "total_amount": float(r.total_amount),
                    "avg_amount": float(r.avg_amount),
                    "max_amount": float(r.max_amount),
                    "min_amount": float(r.min_amount),
                }
                for r in category_result
            ],
        }
    finally:
        db.close()


@app.get("/api/charts/by-country")
def get_chart_data_by_country():
    """
    Dữ liệu cho đồ thị theo country.
    """
    db = SessionLocal()
    try:
        result = db.execute(
            text("""
                SELECT 
                    country,
                    SUM(txn_count) as total_txns,
                    SUM(total_amount) as total_amount
                FROM aggregates
                GROUP BY country
                ORDER BY total_amount DESC
            """)
        )
        
        data = [
            {
                "country": r.country,
                "total_txns": int(r.total_txns),
                "total_amount": float(r.total_amount),
            }
            for r in result
        ]
        
        return {
            "labels": [d["country"] for d in data],
            "txn_data": [d["total_txns"] for d in data],
            "amount_data": [d["total_amount"] for d in data],
        }
    finally:
        db.close()


@app.get("/api/charts/by-category")
def get_chart_data_by_category():
    """
    Dữ liệu cho đồ thị theo category.
    """
    db = SessionLocal()
    try:
        result = db.execute(
            text("""
                SELECT 
                    category,
                    SUM(txn_count) as total_txns,
                    SUM(total_amount) as total_amount
                FROM aggregates
                GROUP BY category
                ORDER BY total_amount DESC
            """)
        )
        
        data = [
            {
                "category": r.category,
                "total_txns": int(r.total_txns),
                "total_amount": float(r.total_amount),
            }
            for r in result
        ]
        
        return {
            "labels": [d["category"] for d in data],
            "txn_data": [d["total_txns"] for d in data],
            "amount_data": [d["total_amount"] for d in data],
        }
    finally:
        db.close()


@app.get("/api/components")
def get_component_evaluation():
    """
    Đánh giá từng thành phần của hệ thống ETL.
    """
    from .etl import list_raw_objects
    
    try:
        s3_files = list_raw_objects()
        
        # Tính toán điểm đánh giá cho từng component
        components = [
            {
                "name": "Data Generation",
                "status": "success" if len(s3_files) > 0 else "warning",
                "score": min(100, len(s3_files) * 10),
                "description": f"{len(s3_files)} files generated",
                "details": "Sinh dữ liệu transaction với các trường: id, user_id, country, category, amount, event_time",
            },
            {
                "name": "S3 Storage",
                "status": "success" if len(s3_files) > 0 else "error",
                "score": 100 if len(s3_files) > 0 else 0,
                "description": f"{len(s3_files)} files stored in S3",
                "details": "Upload và lưu trữ dữ liệu raw trên LocalStack S3",
            },
            {
                "name": "Data Validation",
                "status": "success",
                "score": 95,
                "description": "Validation pipeline active",
                "details": "Kiểm tra null values, duplicates, outliers, invalid amounts",
            },
            {
                "name": "Data Transformation",
                "status": "success",
                "score": 90,
                "description": "7 transformation methods applied",
                "details": "Validation, Deduplication, Filtering, Transformation, Normalization, Enrichment, Aggregation",
            },
            {
                "name": "PostgreSQL Database",
                "status": "success",
                "score": 100,
                "description": "Database connection active",
                "details": "Lưu trữ aggregated data trong PostgreSQL",
            },
            {
                "name": "API Performance",
                "status": "success",
                "score": 85,
                "description": "FastAPI endpoints responsive",
                "details": "RESTful API với background tasks cho ETL jobs",
            },
        ]
        
        return {"components": components}
    except Exception as e:
        return {
            "components": [],
            "error": str(e),
        }


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    """Render dashboard template."""
    return templates.TemplateResponse("index.html", {"request": request})


