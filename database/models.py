# 记录表保存筛查历史、患者信息和结论，便于历史页回顾与导出
CREATE_RECORDS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS records (
    record_id TEXT PRIMARY KEY,
    时间 TEXT,
    文件名 TEXT,
    风险等级 TEXT,
    风险评分 REAL,
    总心拍数 INTEGER,
    异常心拍数 INTEGER,
    备注 TEXT,
    报告 TEXT,
    特征 TEXT,
    patient_name TEXT,
    patient_info TEXT,
    symptoms TEXT DEFAULT ''
)
"""


def create_tables(connection):
    """确保 record 表存在，并补齐新版本新增字段。"""
    connection.execute(CREATE_RECORDS_TABLE_SQL)
    columns = {row[1] for row in connection.execute("PRAGMA table_info(records)").fetchall()}
    if "symptoms" not in columns:
        connection.execute("ALTER TABLE records ADD COLUMN symptoms TEXT DEFAULT ''")
    connection.commit()
