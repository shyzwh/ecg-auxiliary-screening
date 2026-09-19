"""把历史 records 表中的错误字段名迁移为 文件名。"""

import sqlite3
from pathlib import Path

from .db import DEFAULT_DB_PATH


OLD_COLUMN = "文 件名"
NEW_COLUMN = "文件名"


def migrate_column(db_path=None):
    """按需重命名 records 表字段，并返回迁移状态。"""
    path = Path(db_path) if db_path else DEFAULT_DB_PATH
    with sqlite3.connect(path) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(records)")}
        if OLD_COLUMN in columns and NEW_COLUMN not in columns:
            connection.execute(
                f'ALTER TABLE records RENAME COLUMN "{OLD_COLUMN}" TO "{NEW_COLUMN}"'
            )
            return "renamed"
        if NEW_COLUMN in columns:
            return "already_correct"
        raise RuntimeError("records 表中未找到 文 件名 或 文件名 字段")


if __name__ == "__main__":
    result = migrate_column()
    print(f"filename_column_migration={result}")