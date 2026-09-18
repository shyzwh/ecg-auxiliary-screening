from database.db import DEFAULT_DB_PATH, delete_record, fetch_all_records, upsert_records


def load_history(path=None):
    try:
        return fetch_all_records(path or DEFAULT_DB_PATH)
    except Exception:
        return []


def save_history(records, path=None):
    upsert_records(records, path or DEFAULT_DB_PATH)


def delete_history_record(record_id, path=None):
    delete_record(record_id, path or DEFAULT_DB_PATH)
