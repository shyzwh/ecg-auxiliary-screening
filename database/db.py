import json
import sqlite3
from pathlib import Path

from .models import create_tables

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = BASE_DIR / "storage" / "ecg.db"

_RECORD_FIELDS = (
    "record_id", "时间", "文件名", "风险等级", "风险评分", "总心拍数",
    "异常心拍数", "备注", "报告", "特征", "patient_name", "patient_info", "symptoms",
)


def get_db_path(path=None):
    """返回数据库文件路径，默认使用 storage/ecg.db。"""
    return Path(path) if path else DEFAULT_DB_PATH


def get_connection(path=None):
    """创建 SQLite 连接，并确保表存在。"""
    db_path = get_db_path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(db_path))
    connection.row_factory = sqlite3.Row
    create_tables(connection)
    return connection


def initialize_database(path=None):
    """初始化数据库文件和必要表结构。"""
    with get_connection(path):
        pass


def _serialize_record(record):
    """把历史记录字典转换为 SQLite 可写的元组。"""
    values = dict(record or {})
    values.setdefault("record_id", "")
    values["特征"] = json.dumps(values.get("特征") or {}, ensure_ascii=False)
    values["patient_info"] = json.dumps(values.get("patient_info") or {}, ensure_ascii=False)
    return tuple(values.get(field) for field in _RECORD_FIELDS)


def _deserialize_record(row):
    """从数据库里恢复特征和患者信息字段。"""
    record = dict(row)
    for field in ("特征", "patient_info"):
        try:
            record[field] = json.loads(record.get(field) or "{}")
        except (TypeError, json.JSONDecodeError):
            record[field] = {}
            record["symptoms"] = record.get("symptoms") or ""
    return record


def fetch_all_records(path=None):
    """读取所有历史记录并按时间倒序返回。"""
    with get_connection(path) as connection:
        rows = connection.execute(
            "SELECT * FROM records ORDER BY 时间 DESC, record_id DESC"
        ).fetchall()
    return [_deserialize_record(row) for row in rows]


def insert_record(record, path=None):
    """插入一条单独的历史记录。"""
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
    """批量插入或覆盖历史记录。"""
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
    """删除指定 record_id 的历史记录。"""
    with get_connection(path) as connection:
        connection.execute("DELETE FROM records WHERE record_id = ?", (record_id,))
        connection.commit()
