# ETL Demo Project

ETL end‑to‑end cho 2–3 triệu bản ghi synthetic, chạy hoàn toàn bằng Docker/LocalStack/Postgres/FastAPI.

## Thành phần chính
- `docker-compose.yml`: dựng LocalStack (S3), Postgres, backend FastAPI.
- `backend/`: chứa code Python (ETL + REST API + dashboard).
- `run_all.sh`: script build + chạy toàn bộ stack và trigger ETL.

## Chuẩn bị
1. Cài Docker & Docker Compose.
2. Trên Windows nên dùng Git Bash hoặc WSL để chạy script `.sh`.

## Cách chạy nhanh
```bash
bash run_all.sh
```
Script sẽ:
- build image backend;
- `docker compose up -d` cho localstack, postgres, backend;
- health check backend (`/health`);
- gọi `POST /etl/run` để bắt đầu ETL (~2M rows mặc định).

Mở trình duyệt tại `http://localhost:8000/` để xem dashboard:
- nút **Run ETL** (có thể chạy lại);
- thống kê tổng số nhóm, tổng giao dịch, tổng tiền;
- bảng chi tiết theo `country` + `category`.

## Tuỳ chỉnh dữ liệu
Số dòng và kích thước chunk đọc từ biến môi trường:
```yaml
environment:
  - ETL_TOTAL_ROWS=3000000
  - ETL_CHUNK_SIZE=300000
```
Sửa trực tiếp trong `docker-compose.yml` (service `backend`) hoặc đặt biến khi chạy.

## Thư mục chính
- `backend/app/config.py`: cấu hình chung.
- `backend/app/db.py`: kết nối & khởi tạo bảng `aggregates`.
- `backend/app/etl.py`: pipeline generate → upload S3 → aggregate → load Postgres.
- `backend/app/main.py`: FastAPI API + dashboard HTML.


