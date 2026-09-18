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
    patient_info TEXT
)
"""


def create_tables(connection):
    connection.execute(CREATE_RECORDS_TABLE_SQL)
    connection.commit()
