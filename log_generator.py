import csv
import random
import uuid
from datetime import datetime, timedelta

random.seed(42)

NUM_REQUESTS = 10000
NUM_USERS = 120
ANOMALY_RATE = 0.05  # 5% anomalías

SERVICES = ["report-service", "sales-api", "database"]

REPORT_TYPES = [
    "daily_sales",
    "monthly_sales",
    "sales_by_product",
    "sales_by_branch",
    "customer_sales"
]

USER_ROLES = ["admin", "manager", "seller"]

OUTPUT_FILE = "sales_report_system_logs.csv"


def random_timestamp():
    start = datetime(2026, 1, 1, 8, 0, 0)
    end = datetime(2026, 1, 31, 23, 59, 59)
    delta = end - start
    random_seconds = random.randint(0, int(delta.total_seconds()))
    return start + timedelta(seconds=random_seconds)


def generate_normal_values(report_type):
    if report_type == "daily_sales":
        records = random.randint(100, 2000)
    elif report_type == "monthly_sales":
        records = random.randint(2000, 15000)
    elif report_type == "sales_by_product":
        records = random.randint(500, 8000)
    elif report_type == "sales_by_branch":
        records = random.randint(1000, 12000)
    else:
        records = random.randint(300, 6000)

    rows_scanned = records * random.randint(2, 8)
    db_query_time_ms = int(records * random.uniform(0.08, 0.25) + random.randint(100, 800))
    api_time_ms = db_query_time_ms + random.randint(100, 600)
    report_time_ms = api_time_ms + random.randint(500, 2500)

    report_size_kb = int(records * random.uniform(0.15, 0.6))
    cpu_usage = round(random.uniform(20, 65), 2)
    memory_usage_mb = random.randint(250, 1200)

    retry_count = random.choice([0, 0, 0, 1])
    error_count = 0
    status_code = 200

    return {
        "records_returned": records,
        "rows_scanned": rows_scanned,
        "db_query_time_ms": db_query_time_ms,
        "api_response_time_ms": api_time_ms,
        "report_response_time_ms": report_time_ms,
        "report_size_kb": report_size_kb,
        "cpu_usage_percent": cpu_usage,
        "memory_usage_mb": memory_usage_mb,
        "retry_count": retry_count,
        "error_count": error_count,
        "status_code": status_code,
        "anomaly_type": "normal"
    }


def inject_anomaly(values):
    anomaly_type = random.choice([
        "slow_database_query",
        "large_report_export",
        "high_error_rate",
        "high_resource_usage"
    ])

    values = values.copy()

    if anomaly_type == "slow_database_query":
        values["db_query_time_ms"] *= random.randint(8, 20)
        values["api_response_time_ms"] = values["db_query_time_ms"] + random.randint(1000, 4000)
        values["report_response_time_ms"] = values["api_response_time_ms"] + random.randint(2000, 8000)
        values["rows_scanned"] *= random.randint(10, 40)

    elif anomaly_type == "large_report_export":
        values["records_returned"] *= random.randint(10, 30)
        values["report_size_kb"] *= random.randint(15, 50)
        values["report_response_time_ms"] *= random.randint(5, 15)
        values["memory_usage_mb"] *= random.randint(3, 6)

    elif anomaly_type == "high_error_rate":
        values["status_code"] = random.choice([500, 502, 503, 504])
        values["retry_count"] = random.randint(2, 6)
        values["error_count"] = random.randint(1, 5)
        values["report_response_time_ms"] *= random.randint(3, 10)

    elif anomaly_type == "high_resource_usage":
        values["cpu_usage_percent"] = round(random.uniform(85, 99), 2)
        values["memory_usage_mb"] *= random.randint(3, 7)
        values["report_response_time_ms"] *= random.randint(3, 8)

    values["anomaly_type"] = anomaly_type
    return values


def generate_logs():
    rows = []

    for _ in range(NUM_REQUESTS):
        trace_id = f"trace_{uuid.uuid4().hex[:12]}"
        timestamp = random_timestamp()
        user_hash = f"user_{random.randint(1, NUM_USERS):03d}"
        tenant_hash = f"tenant_{random.randint(1, 20):03d}"
        report_type = random.choice(REPORT_TYPES)
        user_role = random.choice(USER_ROLES)

        values = generate_normal_values(report_type)

        is_anomaly = random.random() < ANOMALY_RATE
        if is_anomaly:
            values = inject_anomaly(values)

        # Report Service log
        rows.append({
            "timestamp": timestamp.isoformat(),
            "trace_id": trace_id,
            "service": "report-service",
            "user_hash": user_hash,
            "tenant_hash": tenant_hash,
            "user_role": user_role,
            "report_type": report_type,
            "status_code": values["status_code"],
            "response_time_ms": values["report_response_time_ms"],
            "db_query_time_ms": values["db_query_time_ms"],
            "api_response_time_ms": values["api_response_time_ms"],
            "records_returned": values["records_returned"],
            "rows_scanned": values["rows_scanned"],
            "report_size_kb": values["report_size_kb"],
            "cpu_usage_percent": values["cpu_usage_percent"],
            "memory_usage_mb": values["memory_usage_mb"],
            "retry_count": values["retry_count"],
            "error_count": values["error_count"],
            "is_anomaly": int(is_anomaly),
            "anomaly_type": values["anomaly_type"]
        })

        # Sales API log
        rows.append({
            "timestamp": (timestamp + timedelta(milliseconds=random.randint(50, 300))).isoformat(),
            "trace_id": trace_id,
            "service": "sales-api",
            "user_hash": user_hash,
            "tenant_hash": tenant_hash,
            "user_role": user_role,
            "report_type": report_type,
            "status_code": values["status_code"],
            "response_time_ms": values["api_response_time_ms"],
            "db_query_time_ms": values["db_query_time_ms"],
            "api_response_time_ms": values["api_response_time_ms"],
            "records_returned": values["records_returned"],
            "rows_scanned": values["rows_scanned"],
            "report_size_kb": 0,
            "cpu_usage_percent": round(values["cpu_usage_percent"] * random.uniform(0.7, 1.1), 2),
            "memory_usage_mb": int(values["memory_usage_mb"] * random.uniform(0.5, 0.9)),
            "retry_count": values["retry_count"],
            "error_count": values["error_count"],
            "is_anomaly": int(is_anomaly),
            "anomaly_type": values["anomaly_type"]
        })

        # Database log
        rows.append({
            "timestamp": (timestamp + timedelta(milliseconds=random.randint(100, 600))).isoformat(),
            "trace_id": trace_id,
            "service": "database",
            "user_hash": user_hash,
            "tenant_hash": tenant_hash,
            "user_role": user_role,
            "report_type": report_type,
            "status_code": values["status_code"],
            "response_time_ms": values["db_query_time_ms"],
            "db_query_time_ms": values["db_query_time_ms"],
            "api_response_time_ms": 0,
            "records_returned": values["records_returned"],
            "rows_scanned": values["rows_scanned"],
            "report_size_kb": 0,
            "cpu_usage_percent": round(values["cpu_usage_percent"] * random.uniform(0.8, 1.2), 2),
            "memory_usage_mb": int(values["memory_usage_mb"] * random.uniform(0.6, 1.3)),
            "retry_count": 0,
            "error_count": values["error_count"],
            "is_anomaly": int(is_anomaly),
            "anomaly_type": values["anomaly_type"]
        })

    return rows


def save_csv(rows):
    fieldnames = [
        "timestamp",
        "trace_id",
        "service",
        "user_hash",
        "tenant_hash",
        "user_role",
        "report_type",
        "status_code",
        "response_time_ms",
        "db_query_time_ms",
        "api_response_time_ms",
        "records_returned",
        "rows_scanned",
        "report_size_kb",
        "cpu_usage_percent",
        "memory_usage_mb",
        "retry_count",
        "error_count",
        "is_anomaly",
        "anomaly_type"
    ]

    with open(OUTPUT_FILE, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    logs = generate_logs()
    save_csv(logs)

    print(f"Archivo generado: {OUTPUT_FILE}")
    print(f"Total de logs generados: {len(logs)}")
    print(f"Logs por servicio: {NUM_REQUESTS}")
    print(f"Servicios simulados: {', '.join(SERVICES)}")
    print(f"Usuarios anonimizados: {NUM_USERS}")
    print(f"Porcentaje aproximado de anomalías: {ANOMALY_RATE * 100}%")