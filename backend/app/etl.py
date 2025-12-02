import io
import logging
import math
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import boto3
import numpy as np
import pandas as pd
from botocore.client import Config
from sqlalchemy import text

from .config import settings
from .db import engine

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=settings.s3_endpoint_url,
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key,
        region_name=settings.aws_region,
        config=Config(s3={"addressing_style": "path"}),
    )


def ensure_bucket():
    s3 = get_s3_client()
    buckets = s3.list_buckets().get("Buckets", [])
    if not any(b["Name"] == settings.s3_bucket_name for b in buckets):
        s3.create_bucket(Bucket=settings.s3_bucket_name)


def is_double_day(date: datetime) -> bool:
    """Kiểm tra xem có phải ngày đôi không (1/1, 2/2, ..., 12/12)"""
    return date.month == date.day and date.month <= 12


def is_sale_event(date: datetime) -> bool:
    """Kiểm tra xem có phải ngày sale không"""
    # Black Friday (thứ 6 sau Lễ Tạ ơn - thường là ngày 23-29 tháng 11)
    if date.month == 11 and 23 <= date.day <= 29 and date.weekday() == 4:
        return True
    # Cyber Monday (thứ 2 sau Black Friday)
    if date.month == 11 and 26 <= date.day <= 30 and date.weekday() == 0:
        return True
    # End of month sale (3 ngày cuối tháng)
    if date.day >= 28:
        return True
    # Mid-month sale (ngày 15-17)
    if 15 <= date.day <= 17:
        return True
    return False


def generate_chunk(start_id: int, size: int) -> pd.DataFrame:
    """
    Generate a chunk of synthetic e-commerce sales data với patterns tự nhiên.
    
    Features:
    - E-commerce product categories
    - Ngày đôi (1/1, 2/2, ..., 12/12) có lượng mua nhiều
    - Sale events (Black Friday, Cyber Monday, end of month, etc.)
    - Country distribution và category preferences
    - User behavior patterns
    - Amount distribution theo product category
    """
    logger.info(f"Generating chunk: start_id={start_id}, size={size}")
    
    ids = np.arange(start_id, start_id + size, dtype="int64")
    
    # 1. Country distribution không đều (US và VN phổ biến hơn)
    country_weights = {
        "US": 0.25,  # 25%
        "VN": 0.20,  # 20%
        "JP": 0.12,  # 12%
        "DE": 0.10,  # 10%
        "FR": 0.10,  # 10%
        "GB": 0.10,  # 10%
        "SG": 0.08,  # 8%
        "AU": 0.05,  # 5%
    }
    countries_list = list(country_weights.keys())
    countries_probs = list(country_weights.values())
    countries = np.random.choice(countries_list, size=size, p=countries_probs)
    
    # 2. User IDs với power users (một số user xuất hiện nhiều hơn)
    # 80% users là regular, 20% là power users (xuất hiện 3-5 lần nhiều hơn)
    regular_users = np.random.randint(1, 800000, size=int(size * 0.8))
    power_users = np.random.choice(
        np.arange(800000, 1000000), 
        size=int(size * 0.2),
        replace=True
    )
    user_ids = np.concatenate([regular_users, power_users])
    np.random.shuffle(user_ids)
    user_ids = user_ids[:size].astype("int32")
    
    # 3. E-commerce Product Categories - Distribution phù hợp với bán hàng online
    # Electronics và Fashion chiếm phần lớn, Home & Beauty trung bình, Books & Sports ít hơn
    global_category_distribution = {
        "electronics": 0.30,    # 30% - Nhiều nhất, giá trị cao-trung bình
        "fashion": 0.28,        # 28% - Nhiều thứ 2, giá trị trung bình
        "home": 0.18,          # 18% - Trung bình, giá trị trung bình-cao
        "beauty": 0.12,        # 12% - Trung bình-thấp, giá trị thấp-trung bình
        "books": 0.07,         # 7% - Ít, giá trị thấp
        "sports": 0.05,        # 5% - Ít nhất, giá trị trung bình-cao
    }
    
    # Country-specific adjustments cho e-commerce
    country_category_adjustments = {
        "US": {"electronics": +0.03, "fashion": +0.02, "home": -0.01, "beauty": -0.02, "books": -0.01, "sports": -0.01},
        "VN": {"fashion": +0.05, "electronics": +0.02, "beauty": +0.02, "home": -0.04, "books": -0.03, "sports": -0.02},
        "JP": {"electronics": +0.05, "home": +0.02, "fashion": -0.02, "beauty": -0.02, "books": -0.02, "sports": -0.01},
        "DE": {"electronics": +0.03, "home": +0.03, "fashion": -0.02, "beauty": -0.02, "books": -0.01, "sports": -0.01},
        "FR": {"fashion": +0.04, "beauty": +0.03, "electronics": -0.02, "home": -0.02, "books": -0.02, "sports": -0.01},
        "GB": {"fashion": +0.03, "electronics": +0.02, "home": 0.0, "beauty": -0.02, "books": -0.02, "sports": -0.01},
        "SG": {"electronics": +0.04, "fashion": +0.02, "beauty": +0.01, "home": -0.03, "books": -0.02, "sports": -0.02},
        "AU": {"sports": +0.03, "electronics": +0.02, "home": +0.01, "fashion": -0.02, "beauty": -0.02, "books": -0.02},
    }
    
    categories_list = ["electronics", "fashion", "home", "beauty", "books", "sports"]
    
    # Generate categories với global distribution + country adjustments
    categories = []
    for country in countries:
        # Base distribution
        base_probs = [global_category_distribution[cat] for cat in categories_list]
        
        # Apply country adjustments
        if country in country_category_adjustments:
            adjustments = country_category_adjustments[country]
            adjusted_probs = [
                base_probs[i] + adjustments.get(cat, 0.0)
                for i, cat in enumerate(categories_list)
            ]
            # Normalize để tổng = 1
            total = sum(adjusted_probs)
            adjusted_probs = [p / total for p in adjusted_probs]
        else:
            adjusted_probs = base_probs
        
        # Sample category
        category = np.random.choice(categories_list, p=adjusted_probs)
        categories.append(category)
    
    categories = np.array(categories)
    
    # 4. Amount distribution theo product category - giá sản phẩm e-commerce
    # Electronics: giá cao-trung bình, Home: giá trung bình-cao, Fashion: giá trung bình
    # Beauty: giá thấp-trung bình, Books: giá thấp, Sports: giá trung bình-cao
    category_amount_params = {
        "electronics": {
            "mean": 4.8, "sigma": 1.2, "min": 20, "max": 3000,
            "base_price_range": (50, 2000)  # Điện thoại, laptop, tablet, etc.
        },
        "fashion": {
            "mean": 3.5, "sigma": 1.0, "min": 10, "max": 800,
            "base_price_range": (15, 500)  # Quần áo, giày dép, phụ kiện
        },
        "home": {
            "mean": 4.2, "sigma": 1.1, "min": 25, "max": 2000,
            "base_price_range": (30, 1500)  # Đồ nội thất, trang trí, dụng cụ
        },
        "beauty": {
            "mean": 2.8, "sigma": 0.9, "min": 5, "max": 300,
            "base_price_range": (8, 200)  # Mỹ phẩm, chăm sóc da
        },
        "books": {
            "mean": 2.0, "sigma": 0.7, "min": 3, "max": 100,
            "base_price_range": (5, 50)  # Sách, ebook
        },
        "sports": {
            "mean": 4.0, "sigma": 1.0, "min": 20, "max": 1500,
            "base_price_range": (25, 800)  # Đồ thể thao, dụng cụ tập
        },
    }
    
    # 4b. Generate amounts trước (sẽ được điều chỉnh trong temporal patterns nếu cần)
    amounts = np.zeros(size, dtype="float32")
    for i, cat in enumerate(categories):
        params = category_amount_params[cat]
        
        # Log-normal distribution với bounds
        amount = np.random.lognormal(mean=params["mean"], sigma=params["sigma"])
        amount = np.clip(amount, params["min"], params["max"])
        
        # Electronics và Home có thể có giá trị cao hơn (premium products)
        if cat == "electronics":
            if np.random.random() < 0.20:  # 20% là premium products
                amount *= np.random.uniform(1.5, 3.0)
                amount = min(amount, params["max"])
        elif cat == "home":
            if np.random.random() < 0.15:  # 15% là premium products
                amount *= np.random.uniform(1.4, 2.5)
                amount = min(amount, params["max"])
        
        # Thêm outliers tự nhiên (3-5% tùy category)
        outlier_prob = 0.05 if cat in ["electronics", "home"] else 0.03
        if np.random.random() < outlier_prob:
            amount *= np.random.uniform(1.5, 3.0)
            amount = min(amount, params["max"] * 2)
        
        amounts[i] = np.round(amount, 2)
    
    # 5. Temporal patterns cho e-commerce với ngày đôi và sale events
    now = datetime.utcnow()
    timestamps = []
    
    # Pre-calculate ngày đôi và sale events trong 90 ngày gần đây
    double_days = []
    sale_days = []
    for days_back in range(90):
        check_date = now - timedelta(days=days_back)
        if is_double_day(check_date):
            double_days.append(days_back)
        if is_sale_event(check_date):
            sale_days.append(days_back)
    
    for i in range(size):
        # Base distribution: nhiều transactions gần đây hơn
        days_ago = np.random.exponential(scale=15)
        days_ago = min(days_ago, 90)
        
        # Ngày đôi (1/1, 2/2, ..., 12/12) - tăng lượng mua hàng đáng kể
        # 25% transactions tập trung vào các ngày đôi
        if double_days and np.random.random() < 0.25:
            days_ago = np.random.choice(double_days)
            # Tăng amount trong ngày đôi (mua nhiều hơn)
            amounts[i] *= np.random.uniform(1.2, 2.0)
            amounts[i] = min(amounts[i], category_amount_params[categories[i]]["max"] * 1.5)
        
        # Sale events - tăng lượng mua hàng
        # 20% transactions trong các ngày sale
        elif sale_days and np.random.random() < 0.20:
            days_ago = np.random.choice(sale_days)
            # Tăng amount trong sale (mua nhiều hơn do giảm giá)
            amounts[i] *= np.random.uniform(1.1, 1.8)
            amounts[i] = min(amounts[i], category_amount_params[categories[i]]["max"] * 1.3)
        
        # Weekend boost cho fashion và beauty (30% tăng)
        is_weekend = np.random.random() < 0.3
        if is_weekend and categories[i] in ["fashion", "beauty"]:
            days_ago = min(days_ago, 7)
        
        # End of month boost cho tất cả categories (15% tăng)
        is_end_month = np.random.random() < 0.15
        if is_end_month:
            days_ago = np.random.uniform(0, 3)
        
        # Random time trong ngày - e-commerce peak vào buổi tối (19-22h)
        hour = np.random.normal(loc=20, scale=3)  # Peak vào buổi tối
        hour = np.clip(hour, 0, 23)
        minute = np.random.randint(0, 60)
        second = np.random.randint(0, 60)
        
        timestamp = now - timedelta(
            days=float(days_ago),
            hours=float(23 - hour),
            minutes=float(59 - minute),
            seconds=float(59 - second)
        )
        timestamps.append(timestamp)
    
    # 6. Thêm một số transactions có correlation với user (giữ nguyên distribution)
    # Một số user có spending pattern nhất định (8% users) - giảm để không làm mất distribution
    user_patterns = {}
    unique_user_ids = np.unique(user_ids)
    pattern_users_count = min(int(len(unique_user_ids) * 0.08), 800)
    if pattern_users_count > 0:
        pattern_users = np.random.choice(unique_user_ids, size=pattern_users_count, replace=False)
        
        for user_id in pattern_users:
            # User có preference category - ưu tiên các category phổ biến hơn
            # Để giữ distribution, 70% chọn từ electronics/fashion, 30% từ các category khác
            if np.random.random() < 0.7:
                preferred_cat = np.random.choice(["electronics", "fashion"], p=[0.52, 0.48])
            else:
                preferred_cat = np.random.choice(["home", "beauty", "books", "sports"], p=[0.4, 0.3, 0.2, 0.1])
            
            # User có average spending level dựa trên category preference
            cat_params = category_amount_params[preferred_cat]
            avg_spending = np.random.lognormal(mean=cat_params["mean"], sigma=cat_params["sigma"] * 0.8)
            user_patterns[user_id] = {"category": preferred_cat, "avg_amount": avg_spending}
        
        # Apply patterns cho một số transactions của pattern users (20% thay vì 25%)
        for i, user_id in enumerate(user_ids):
            if user_id in user_patterns and np.random.random() < 0.20:  # 20% transactions follow pattern
                pattern = user_patterns[user_id]
                new_cat = pattern["category"]
                categories[i] = new_cat
                
                # Recalculate amount với category mới
                params = category_amount_params[new_cat]
                amount = np.random.lognormal(mean=np.log(pattern["avg_amount"]), sigma=0.4)
                amount = np.clip(amount, params["min"], params["max"])
                amounts[i] = np.round(amount, 2)
    
    # 7. Tạo DataFrame
    df = pd.DataFrame(
        {
            "id": ids,
            "user_id": user_ids,
            "country": countries,
            "category": categories,
            "amount": amounts,
            "event_time": timestamps,
        }
    )
    
    # Optimize memory usage
    df["country"] = df["country"].astype("category")
    df["category"] = df["category"].astype("category")
    
    logger.info(f"Generated chunk: {len(df)} rows, memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    return df


def upload_chunk_to_s3(df: pd.DataFrame, key: str) -> None:
    """Upload a DataFrame as CSV to S3."""
    s3 = get_s3_client()
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    s3.put_object(
        Bucket=settings.s3_bucket_name, Key=key, Body=csv_buffer.getvalue().encode()
    )


def list_raw_objects() -> List[str]:
    s3 = get_s3_client()
    paginator = s3.get_paginator("list_objects_v2")
    keys: List[str] = []
    for page in paginator.paginate(
        Bucket=settings.s3_bucket_name, Prefix=settings.s3_raw_prefix
    ):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    return keys


def read_csv_from_s3(key: str) -> pd.DataFrame:
    s3 = get_s3_client()
    obj = s3.get_object(Bucket=settings.s3_bucket_name, Key=key)
    return pd.read_csv(io.BytesIO(obj["Body"].read()))


# ============================================================================
# DATA PROCESSING METHODS - Tất cả các phương pháp xử lý dữ liệu
# ============================================================================

def filter_data(df: pd.DataFrame, filters: Optional[Dict] = None) -> pd.DataFrame:
    """
    Phương pháp 1: FILTERING - Lọc dữ liệu theo điều kiện
    
    Filters:
    - min_amount: Lọc giao dịch có amount >= min_amount
    - max_amount: Lọc giao dịch có amount <= max_amount
    - countries: Danh sách countries cần giữ lại
    - categories: Danh sách categories cần giữ lại
    - date_from: Lọc từ ngày này
    - date_to: Lọc đến ngày này
    """
    logger.info("🔍 FILTERING: Bắt đầu lọc dữ liệu...")
    original_count = len(df)
    
    if filters is None:
        filters = {}
    
    filtered_df = df.copy()
    
    # Filter by amount range
    if "min_amount" in filters:
        filtered_df = filtered_df[filtered_df["amount"] >= filters["min_amount"]]
        logger.info(f"  ✓ Lọc theo min_amount >= {filters['min_amount']}: {len(filtered_df)} rows")
    
    if "max_amount" in filters:
        filtered_df = filtered_df[filtered_df["amount"] <= filters["max_amount"]]
        logger.info(f"  ✓ Lọc theo max_amount <= {filters['max_amount']}: {len(filtered_df)} rows")
    
    # Filter by countries
    if "countries" in filters:
        filtered_df = filtered_df[filtered_df["country"].isin(filters["countries"])]
        logger.info(f"  ✓ Lọc theo countries {filters['countries']}: {len(filtered_df)} rows")
    
    # Filter by categories
    if "categories" in filters:
        filtered_df = filtered_df[filtered_df["category"].isin(filters["categories"])]
        logger.info(f"  ✓ Lọc theo categories {filters['categories']}: {len(filtered_df)} rows")
    
    # Filter by date range
    if "date_from" in filters:
        filtered_df = filtered_df[filtered_df["event_time"] >= filters["date_from"]]
        logger.info(f"  ✓ Lọc từ ngày {filters['date_from']}: {len(filtered_df)} rows")
    
    if "date_to" in filters:
        filtered_df = filtered_df[filtered_df["event_time"] <= filters["date_to"]]
        logger.info(f"  ✓ Lọc đến ngày {filters['date_to']}: {len(filtered_df)} rows")
    
    logger.info(f"🔍 FILTERING: Hoàn thành - {original_count} -> {len(filtered_df)} rows ({len(filtered_df)/original_count*100:.1f}%)")
    return filtered_df


def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Phương pháp 2: TRANSFORMATION - Biến đổi dữ liệu
    
    - Chuyển đổi kiểu dữ liệu
    - Tính toán các cột mới
    - Chuẩn hóa dữ liệu
    """
    logger.info("🔄 TRANSFORMATION: Bắt đầu biến đổi dữ liệu...")
    
    transformed_df = df.copy()
    
    # Tính toán cột mới: amount_category (phân loại theo giá trị)
    def categorize_amount(amt):
        if amt < 10:
            return "small"
        elif amt < 100:
            return "medium"
        elif amt < 1000:
            return "large"
        else:
            return "very_large"
    
    transformed_df["amount_category"] = transformed_df["amount"].apply(categorize_amount)
    logger.info("  ✓ Thêm cột amount_category")
    
    # Tính toán cột mới: day_of_week
    transformed_df["day_of_week"] = pd.to_datetime(transformed_df["event_time"]).dt.day_name()
    logger.info("  ✓ Thêm cột day_of_week")
    
    # Tính toán cột mới: month
    transformed_df["month"] = pd.to_datetime(transformed_df["event_time"]).dt.month
    logger.info("  ✓ Thêm cột month")
    
    # Tính toán cột mới: amount_usd (giả sử tỷ giá)
    exchange_rates = {"US": 1.0, "VN": 0.00004, "JP": 0.0067, "DE": 1.08, "FR": 1.08, "GB": 1.27, "SG": 0.74, "AU": 0.66}
    transformed_df["amount_usd"] = transformed_df.apply(
        lambda row: row["amount"] * exchange_rates.get(row["country"], 1.0), axis=1
    )
    logger.info("  ✓ Thêm cột amount_usd (chuyển đổi tiền tệ)")
    
    logger.info("🔄 TRANSFORMATION: Hoàn thành")
    return transformed_df


def validate_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Phương pháp 3: VALIDATION - Kiểm tra và làm sạch dữ liệu
    
    - Kiểm tra null values
    - Kiểm tra duplicate
    - Kiểm tra giá trị bất thường (outliers)
    - Kiểm tra kiểu dữ liệu
    """
    logger.info("✅ VALIDATION: Bắt đầu kiểm tra dữ liệu...")
    
    validation_report = {
        "original_rows": len(df),
        "null_counts": {},
        "duplicates": 0,
        "outliers": 0,
        "invalid_rows": 0,
    }
    
    validated_df = df.copy()
    
    # Kiểm tra null values
    null_counts = validated_df.isnull().sum()
    validation_report["null_counts"] = null_counts[null_counts > 0].to_dict()
    if null_counts.sum() > 0:
        logger.warning(f"  ⚠ Tìm thấy {null_counts.sum()} null values")
        validated_df = validated_df.dropna()
        logger.info(f"  ✓ Đã xóa {len(df) - len(validated_df)} rows có null")
    
    # Kiểm tra duplicates
    duplicates = validated_df.duplicated(subset=["id"]).sum()
    validation_report["duplicates"] = int(duplicates)
    if duplicates > 0:
        logger.warning(f"  ⚠ Tìm thấy {duplicates} duplicates")
        validated_df = validated_df.drop_duplicates(subset=["id"], keep="first")
        logger.info(f"  ✓ Đã xóa {duplicates} duplicates")
    
    # Kiểm tra outliers (sử dụng IQR method)
    Q1 = validated_df["amount"].quantile(0.25)
    Q3 = validated_df["amount"].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 3 * IQR
    upper_bound = Q3 + 3 * IQR
    outliers = ((validated_df["amount"] < lower_bound) | (validated_df["amount"] > upper_bound)).sum()
    validation_report["outliers"] = int(outliers)
    if outliers > 0:
        logger.info(f"  ℹ Tìm thấy {outliers} outliers (giữ lại để phân tích)")
    
    # Kiểm tra giá trị âm (không hợp lệ cho amount)
    invalid_amounts = (validated_df["amount"] < 0).sum()
    if invalid_amounts > 0:
        logger.warning(f"  ⚠ Tìm thấy {invalid_amounts} rows có amount < 0")
        validated_df = validated_df[validated_df["amount"] >= 0]
        validation_report["invalid_rows"] = int(invalid_amounts)
        logger.info(f"  ✓ Đã xóa {invalid_amounts} rows có amount < 0")
    
    validation_report["final_rows"] = len(validated_df)
    logger.info(f"✅ VALIDATION: Hoàn thành - {validation_report['original_rows']} -> {validation_report['final_rows']} rows")
    logger.info(f"  📊 Báo cáo: {validation_report}")
    
    return validated_df, validation_report


def normalize_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Phương pháp 4: NORMALIZATION - Chuẩn hóa dữ liệu
    
    - Chuẩn hóa số liệu (min-max scaling)
    - Chuẩn hóa text (lowercase, trim)
    """
    logger.info("📏 NORMALIZATION: Bắt đầu chuẩn hóa dữ liệu...")
    
    normalized_df = df.copy()
    
    # Chuẩn hóa text fields
    if "country" in normalized_df.columns:
        normalized_df["country"] = normalized_df["country"].str.upper().str.strip()
    if "category" in normalized_df.columns:
        normalized_df["category"] = normalized_df["category"].str.lower().str.strip()
    
    logger.info("  ✓ Chuẩn hóa text fields (uppercase/lowercase, trim)")
    
    # Min-Max scaling cho amount (0-1 range)
    if "amount" in normalized_df.columns:
        min_amount = normalized_df["amount"].min()
        max_amount = normalized_df["amount"].max()
        if max_amount > min_amount:
            normalized_df["amount_normalized"] = (normalized_df["amount"] - min_amount) / (max_amount - min_amount)
            logger.info("  ✓ Chuẩn hóa amount (min-max scaling)")
    
    logger.info("📏 NORMALIZATION: Hoàn thành")
    return normalized_df


def deduplicate_data(df: pd.DataFrame, subset: List[str] = None) -> pd.DataFrame:
    """
    Phương pháp 5: DEDUPLICATION - Loại bỏ dữ liệu trùng lặp
    
    - Loại bỏ duplicate rows
    - Có thể chỉ định các cột để kiểm tra duplicate
    """
    logger.info("🔁 DEDUPLICATION: Bắt đầu loại bỏ duplicates...")
    
    if subset is None:
        subset = ["id"]
    
    original_count = len(df)
    deduplicated_df = df.drop_duplicates(subset=subset, keep="first")
    removed = original_count - len(deduplicated_df)
    
    if removed > 0:
        logger.info(f"  ✓ Đã xóa {removed} duplicate rows ({removed/original_count*100:.2f}%)")
    else:
        logger.info("  ✓ Không có duplicates")
    
    logger.info(f"🔁 DEDUPLICATION: Hoàn thành - {original_count} -> {len(deduplicated_df)} rows")
    return deduplicated_df


def aggregate_data(df: pd.DataFrame, group_by: List[str], agg_funcs: Dict = None) -> pd.DataFrame:
    """
    Phương pháp 6: AGGREGATION - Tổng hợp dữ liệu
    
    - Group by các cột
    - Tính toán các metrics (sum, count, mean, median, etc.)
    """
    logger.info(f"📊 AGGREGATION: Bắt đầu tổng hợp dữ liệu theo {group_by}...")
    
    if agg_funcs is None:
        agg_funcs = {
            "amount": ["sum", "mean", "count", "min", "max"],
        }
    
    agg_df = df.groupby(group_by).agg(agg_funcs).reset_index()
    
    # Flatten column names
    agg_df.columns = ['_'.join(col).strip('_') if col[1] else col[0] for col in agg_df.columns.values]
    
    logger.info(f"  ✓ Tổng hợp thành {len(agg_df)} groups")
    logger.info(f"📊 AGGREGATION: Hoàn thành")
    return agg_df


def enrich_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Phương pháp 7: DATA ENRICHMENT - Làm giàu dữ liệu
    
    - Thêm thông tin từ lookup tables
    - Thêm metadata
    - Thêm calculated fields
    """
    logger.info("💎 DATA ENRICHMENT: Bắt đầu làm giàu dữ liệu...")
    
    enriched_df = df.copy()
    
    # Lookup table cho country names
    country_names = {
        "US": "United States",
        "VN": "Vietnam",
        "JP": "Japan",
        "DE": "Germany",
        "FR": "France",
        "GB": "United Kingdom",
        "SG": "Singapore",
        "AU": "Australia",
    }
    enriched_df["country_name"] = enriched_df["country"].map(country_names)
    logger.info("  ✓ Thêm country_name từ lookup table")
    
    # Thêm metadata
    enriched_df["processed_at"] = datetime.utcnow()
    logger.info("  ✓ Thêm processed_at timestamp")
    
    logger.info("💎 DATA ENRICHMENT: Hoàn thành")
    return enriched_df


def process_data_comprehensive(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Hàm tổng hợp: Áp dụng TẤT CẢ các phương pháp xử lý dữ liệu
    
    Thứ tự xử lý:
    1. Validation (kiểm tra và làm sạch)
    2. Deduplication (loại bỏ duplicates)
    3. Filtering (lọc dữ liệu)
    4. Transformation (biến đổi)
    5. Normalization (chuẩn hóa)
    6. Enrichment (làm giàu)
    7. Aggregation (tổng hợp)
    """
    logger.info("=" * 80)
    logger.info("🚀 BẮT ĐẦU XỬ LÝ DỮ LIỆU TOÀN DIỆN")
    logger.info("=" * 80)
    
    processing_stats = {
        "original_rows": len(df),
        "after_validation": 0,
        "after_deduplication": 0,
        "after_filtering": 0,
        "after_transformation": 0,
        "after_normalization": 0,
        "after_enrichment": 0,
        "final_rows": 0,
    }
    
    # 1. VALIDATION
    df, validation_report = validate_data(df)
    processing_stats["after_validation"] = len(df)
    
    # 2. DEDUPLICATION
    df = deduplicate_data(df, subset=["id"])
    processing_stats["after_deduplication"] = len(df)
    
    # 3. FILTERING (optional - có thể bỏ qua hoặc thêm filters)
    # df = filter_data(df, filters={"min_amount": 0.01})
    processing_stats["after_filtering"] = len(df)
    
    # 4. TRANSFORMATION
    df = transform_data(df)
    processing_stats["after_transformation"] = len(df)
    
    # 5. NORMALIZATION
    df = normalize_data(df)
    processing_stats["after_normalization"] = len(df)
    
    # 6. ENRICHMENT
    df = enrich_data(df)
    processing_stats["after_enrichment"] = len(df)
    
    processing_stats["final_rows"] = len(df)
    
    logger.info("=" * 80)
    logger.info("✅ HOÀN THÀNH XỬ LÝ DỮ LIỆU TOÀN DIỆN")
    logger.info(f"📊 Thống kê: {processing_stats}")
    logger.info("=" * 80)
    
    return df, processing_stats


# ============================================================================
# ETL FUNCTIONS
# ============================================================================

def generate_and_upload_data(total_rows: Optional[int] = None, chunk_size: Optional[int] = None) -> Dict:
    """
    Chỉ sinh và upload data lên S3 (không chạy ETL)
    Phù hợp cho máy 8GB RAM với chunking và memory optimization
    """
    logger.info("=" * 80)
    logger.info("📤 BẮT ĐẦU SINH VÀ UPLOAD DỮ LIỆU LÊN S3")
    logger.info("=" * 80)
    
    ensure_bucket()
    
    if total_rows is None:
        total_rows = settings.total_rows
    if chunk_size is None:
        chunk_size = settings.chunk_size
    
    # Đảm bảo chunk_size phù hợp với 8GB RAM
    # Ước tính: mỗi row ~ 200 bytes, chunk_size 200k rows ~ 40MB
    # Với 8GB RAM, có thể xử lý nhiều chunks nhưng giữ an toàn ở 200k
    if chunk_size > 200000:
        chunk_size = 200000
        logger.info(f"⚠ Chunk size được giới hạn ở 200k rows để phù hợp với 8GB RAM")
    
    total_chunks = math.ceil(total_rows / chunk_size)
    logger.info(f"📊 Cấu hình: {total_rows:,} rows, {chunk_size:,} rows/chunk, {total_chunks} chunks")
    
    current_id = 1
    uploaded_keys: List[str] = []
    total_uploaded = 0
    
    try:
        for chunk_index in range(total_chunks):
            rows = min(chunk_size, total_rows - current_id + 1)
            logger.info(f"📦 Chunk {chunk_index + 1}/{total_chunks}: Sinh {rows:,} rows (ID: {current_id} - {current_id + rows - 1})")
            
            # Generate chunk
            df = generate_chunk(current_id, rows)
            
            # Upload to S3
            key = f"{settings.s3_raw_prefix}transactions_{chunk_index:03d}.csv"
            logger.info(f"  ⬆ Uploading to S3: {key}")
            upload_chunk_to_s3(df, key)
            uploaded_keys.append(key)
            total_uploaded += len(df)
            
            logger.info(f"  ✅ Hoàn thành chunk {chunk_index + 1}: {len(df):,} rows uploaded")
            
            # Giải phóng memory
            del df
            
            current_id += rows
        
        logger.info("=" * 80)
        logger.info(f"✅ HOÀN THÀNH: Đã upload {total_uploaded:,} rows lên S3 trong {len(uploaded_keys)} files")
        logger.info("=" * 80)
        
        return {
            "status": "success",
            "total_rows": int(total_uploaded),
            "total_files": len(uploaded_keys),
            "files": uploaded_keys,
        }
    except Exception as e:
        logger.error(f"❌ Lỗi khi sinh/upload data: {str(e)}")
        raise


def run_etl() -> dict:
    """
    Full ETL pipeline với tất cả các phương pháp xử lý dữ liệu:
    generate -> upload -> process (comprehensive) -> aggregate -> load to Postgres
    """
    logger.info("=" * 80)
    logger.info("🚀 BẮT ĐẦU ETL PIPELINE ĐẦY ĐỦ")
    logger.info("=" * 80)
    
    ensure_bucket()

    total_rows = settings.total_rows
    chunk_size = settings.chunk_size
    total_chunks = math.ceil(total_rows / chunk_size)

    logger.info(f"📊 Cấu hình ETL: {total_rows:,} rows, {chunk_size:,} rows/chunk")

    # Step 1: Generate & upload raw data (nếu chưa có)
    logger.info("\n📤 STEP 1: Generate & Upload Raw Data")
    current_id = 1
    uploaded_keys: List[str] = []
    for chunk_index in range(total_chunks):
        rows = min(chunk_size, total_rows - current_id + 1)
        logger.info(f"  Processing chunk {chunk_index + 1}/{total_chunks}")
        df = generate_chunk(current_id, rows)
        key = f"{settings.s3_raw_prefix}transactions_{chunk_index:03d}.csv"
        upload_chunk_to_s3(df, key)
        uploaded_keys.append(key)
        current_id += rows
        del df  # Giải phóng memory

    # Step 2: Transform với tất cả các phương pháp xử lý
    logger.info("\n🔄 STEP 2: Process Data (Comprehensive)")
    agg_frames: List[pd.DataFrame] = []
    
    for idx, key in enumerate(uploaded_keys):
        logger.info(f"  Processing file {idx + 1}/{len(uploaded_keys)}: {key}")
        df = read_csv_from_s3(key)
        
        # Áp dụng tất cả các phương pháp xử lý
        df_processed, stats = process_data_comprehensive(df)
        
        # Aggregate sau khi xử lý
        agg = (
            df_processed.groupby(["country", "category"])["amount"]
            .agg(["sum", "count"])
            .reset_index()
        )
        agg_frames.append(agg)
        
        # Giải phóng memory
        del df, df_processed

    if not agg_frames:
        return {"rows_generated": 0, "rows_aggregated": 0}

    # Step 3: Final aggregation
    logger.info("\n📊 STEP 3: Final Aggregation")
    all_agg = pd.concat(agg_frames, ignore_index=True)
    final_agg = (
        all_agg.groupby(["country", "category"])
        .agg({"sum": "sum", "count": "sum"})
        .reset_index()
    )
    final_agg.rename(
        columns={"sum": "total_amount", "count": "txn_count"}, inplace=True
    )
    logger.info(f"  ✓ Tổng hợp thành {len(final_agg)} groups")

    # Step 4: Load to Postgres
    logger.info("\n💾 STEP 4: Load to Postgres")
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE aggregates;"))
        logger.info("  ✓ Đã truncate table aggregates")
        
        # Batch insert để tối ưu
        values = [
            {
                "country": row["country"],
                "category": row["category"],
                "total_amount": float(row["total_amount"]),
                "txn_count": int(row["txn_count"]),
            }
            for _, row in final_agg.iterrows()
        ]
        
        # Insert in batches
        batch_size = 100
        for i in range(0, len(values), batch_size):
            batch = values[i:i + batch_size]
            for val in batch:
                conn.execute(
                    text(
                        """
                        INSERT INTO aggregates (country, category, total_amount, txn_count)
                        VALUES (:country, :category, :total_amount, :txn_count)
                        """
                    ),
                    val,
                )
        
        logger.info(f"  ✓ Đã insert {len(values)} records vào Postgres")

    logger.info("=" * 80)
    logger.info("✅ HOÀN THÀNH ETL PIPELINE")
    logger.info("=" * 80)

    return {
        "rows_generated": int(total_rows),
        "rows_aggregated": int(final_agg["txn_count"].sum()),
        "groups": int(len(final_agg)),
    }


