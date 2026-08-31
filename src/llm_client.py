import json
import os

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"), override=False)

GLM_API_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
GLM_MODEL_OPTIONS = [
    {
        "id": "glm-4.7-flash",
        "label": "glm-4.7-flash（完全免费）",
        "description": "混合思考模型，200K上下文，适合复杂筛查报告解读",
        "default": True,
    },
    {
        "id": "glm-4-flash-250414",
        "label": "glm-4-flash-250414（完全免费）",
        "description": "文本生成模型，128K上下文，适合常规指标解读",
        "default": False,
    },
    {
        "id": "glm-4-flash",
        "label": "glm-4-flash（完全免费）",
        "description": "基础免费模型，128K上下文，适合简单问答兜底",
        "default": False,
    },
]


def get_glm_api_key():
    """后台密钥代理模式：优先读取 Streamlit secrets，再回退环境变量。"""
    try:
        secret_value = st.secrets.get("ZHIPU_API_KEY")
        if secret_value and str(secret_value).strip():
            return str(secret_value).strip()
    except Exception:
        pass

    env_value = os.environ.get("ZHIPU_API_KEY")
    if env_value and str(env_value).strip():
        return str(env_value).strip()
    return None


def get_default_glm_model():
    for item in GLM_MODEL_OPTIONS:
        if item.get("default"):
            return item["id"]
    return "glm-4.7-flash"


def test_glm_connection(model_name=None):
    api_key = get_glm_api_key()
    if not api_key:
        return False, "后台服务未配置，AI润色不可用"

    target_model = model_name or get_default_glm_model()
    try:
        _call_glm_api("请回复：连接测试正常。", model_name=target_model)
        return True, "AI润色连接正常，后台代理可用。"
    except Exception as exc:
        return False, f"AI润色连接失败：{exc}"


def _call_glm_api(prompt_text, model_name=None):
    api_key = get_glm_api_key()
    if not api_key:
        raise RuntimeError("后台服务未配置，AI润色不可用")

    request_model = model_name or get_default_glm_model()
    payload = {
        "model": request_model,
        "messages": [
            {
                "role": "system",
                "content": "你是专业的心电筛查报告润色助手。请保持医学事实准确、语气温和、贴近医患沟通，不编造诊断，保留必要提醒。",
            },
            {"role": "user", "content": prompt_text},
        ],
        "temperature": 0.3,
        "top_p": 0.85,
    }

    response = requests.post(
        GLM_API_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        data=json.dumps(payload),
        timeout=30,
    )

    if response.status_code != 200:
        error_text = response.text[:300].replace("\n", " ")
        raise RuntimeError(f"HTTP {response.status_code}: {error_text}")

    data = response.json()
    if "choices" not in data or not data["choices"]:
        raise RuntimeError("AI返回结果为空")

    message = data["choices"][0].get("message", {})
    content = message.get("content", "")
    if isinstance(content, list):
        content = "".join(part.get("text", "") for part in content)
    return str(content).strip()


def polish_report_with_glm(report_text, model_name=None):
    if not report_text or not str(report_text).strip():
        return "离线建议为空，无法进行 AI 润色。"

    prompt = (
        "请将以下离线生成的心电筛查建议润色成更自然、适合医患沟通的中文话术，"
        "保留医学事实与风险提醒，不扩充新的诊断结论，不加入无法确认的内容。"
        "请直接输出润色后的文本，不要解释过程。\n\n"
        f"离线建议：\n{report_text}"
    )
    return _call_glm_api(prompt, model_name=model_name)
