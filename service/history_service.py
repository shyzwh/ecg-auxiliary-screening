from database.db import DEFAULT_DB_PATH, delete_record, fetch_all_records, upsert_records


def load_history(path=None):
    """从本地数据库读取全部历史筛查记录。"""
    try:
        return fetch_all_records(path or DEFAULT_DB_PATH)
    except Exception:
        return []


def save_history(records, path=None):
    """保存历史记录到数据库。"""
    upsert_records(records, path or DEFAULT_DB_PATH)


def delete_history_record(record_id, path=None):
    """删除指定历史记录。"""
    delete_record(record_id, path or DEFAULT_DB_PATH)
