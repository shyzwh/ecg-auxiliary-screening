import json
from pathlib import Path

from .db import DEFAULT_DB_PATH, upsert_records

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_JSON_PATH = BASE_DIR / "storage" / "records.json"


def migrate(json_path=None, db_path=None):
    source = Path(json_path) if json_path else DEFAULT_JSON_PATH
    if not source.exists():
        return 0
    with source.open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError("records.json 必须包含记录列表")
    upsert_records(records, db_path or DEFAULT_DB_PATH)
    return len(records)


if __name__ == "__main__":
    count = migrate()
    print(f"已迁移 {count} 条历史记录到 SQLite。")
