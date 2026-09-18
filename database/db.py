import json
import sqlite3
from pathlib import Path

from .models import create_tables

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = BASE_DIR / "storage" / "ecg.db"

_RECORD_FIELDS = (
    "record_id", "时间", "文件名", "风险等级", "风险评分", "总心拍数",
    "异常心拍数", "备注", "报告", "特征", "patient_name", "patient_info",
)


def get_db_path(path=None):
    return Path(path) if path else DEFAULT_DB_PATH


def get_connection(path=None):
    db_path = get_db_path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(db_path))
    connection.row_factory = sqlite3.Row
    create_tables(connection)
    return connection


def initialize_database(path=None):
    with get_connection(path):
        pass


def _serialize_record(record):
    values = dict(record or {})
    values.setdefault("record_id", "")
    values["特征"] = json.dumps(values.get("特征") or {}, ensure_ascii=False)
    values["patient_info"] = json.dumps(values.get("patient_info") or {}, ensure_ascii=False)
    return tuple(values.get(field) for field in _RECORD_FIELDS)


def _deserialize_record(row):
    record = dict(row)
    for field in ("特征", "patient_info"):
        try:
            record[field] = json.loads(record.get(field) or "{}")
        except (TypeError, json.JSONDecodeError):
            record[field] = {}
    return record


def fetch_all_records(path=None):
    with get_connection(path) as connection:
        rows = connection.execute(
            "SELECT * FROM records ORDER BY 时间 DESC, record_id DESC"
        ).fetchall()
    return [_deserialize_record(row) for row in rows]


def insert_record(record, path=None):
    values = _serialize_record(record)
    placeholders = ", ".join("?" for _ in _RECORD_FIELDS)
    columns = ", ".join(f'"{field}"' for field in _RECORD_FIELDS)
    with get_connection(path) as connection:
        connection.execute(
            f"INSERT OR REPLACE INTO records ({columns}) VALUES ({placeholders})",
            values,
        )
        connection.commit()


def upsert_records(records, path=None):
    records = list(records or [])
    placeholders = ", ".join("?" for _ in _RECORD_FIELDS)
    columns = ", ".join(f'"{field}"' for field in _RECORD_FIELDS)
    with get_connection(path) as connection:
        if records:
            connection.executemany(
                f"INSERT OR REPLACE INTO records ({columns}) VALUES ({placeholders})",
                [_serialize_record(record) for record in records],
            )
        connection.commit()


def delete_record(record_id, path=None):
    with get_connection(path) as connection:
        connection.execute("DELETE FROM records WHERE record_id = ?", (record_id,))
        connection.commit()
