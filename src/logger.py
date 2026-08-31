import os
import json
from pathlib import Path
import datetime


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Logger:
    """项目日志系统，负责记录运行过程和错误信息"""

    def __init__(self, log_dir="logs"):
        log_dir = Path(log_dir)
        if not log_dir.is_absolute():
            log_dir = PROJECT_ROOT / log_dir

        # 1. 确保日志文件夹存在
        log_dir.mkdir(parents=True, exist_ok=True)

        # 2. 日志文件名按日期生成
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        self.log_file = os.path.join(log_dir, f"app_{today}.log")

    def info(self, message):
        """记录普通信息"""
        self._write("INFO", message)

    def warning(self, message):
        """记录警告信息"""
        self._write("WARNING", message)

    def error(self, message):
        """记录错误信息"""
        self._write("ERROR", message)

    def _write(self, level, message):
        """把日志写入文件"""
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{now}] [{level}] {message}\n"

        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(line)


# 全局日志对象，其他模块直接 import logger 使用
logger = Logger()