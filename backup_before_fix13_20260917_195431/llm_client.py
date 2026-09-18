import json
import logging
import os
import re

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
LOGGER = logging.getLogger(__name__)
_LAST_AI_ERROR = None


def _set_ai_error(message, exc=None):
    global _LAST_AI_ERROR
    _LAST_AI_ERROR = str(message)
    if exc is not None:
        LOGGER.exception("GLM请求失败: %s", message)
    else:
        LOGGER.error("GLM请求失败: %s", message)


def get_last_ai_error():
    return _LAST_AI_ERROR or "未返回具体错误信息。"


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


def resolve_api_settings(api_key=None, model=None, base_url=None):
    effective_key = (api_key or "").strip() or get_glm_api_key()
    effective_model = (model or "").strip() or get_default_glm_model()
    effective_base_url = (base_url or "").strip() or GLM_API_URL
    return effective_key, effective_model, effective_base_url


def test_glm_connection(model_name=None, api_key=None, base_url=None):
    effective_key, _, effective_base_url = resolve_api_settings(api_key, model_name, base_url)
    if not effective_key:
        return False, "未配置密钥，AI润色不可用。"

    target_model = model_name or get_default_glm_model()
    try:
        _call_glm_api("请回复：连接测试正常。", model_name=target_model, api_key=effective_key, base_url=effective_base_url)
        return True, "AI润色连接正常，后台代理可用。"
    except Exception as exc:
        return False, f"AI润色连接失败：{exc}"


def _call_glm_api(prompt_text, model_name=None, api_key=None, base_url=None, system_prompt=None, timeout=15):
    effective_key, request_model, request_base_url = resolve_api_settings(api_key, model_name, base_url)
    if not effective_key:
        _set_ai_error("未配置ZHIPU_API_KEY或自定义API密钥")
        return None

    payload = {
        "model": request_model,
        "messages": [
            {
                "role": "system",
                "content": system_prompt or "你是专业的心电筛查报告润色助手。请保持医学事实准确、语气温和、贴近医患沟通，不编造诊断，保留必要提醒。",
            },
            {"role": "user", "content": prompt_text},
        ],
        "temperature": 0.3,
        "top_p": 0.85,
    }

    try:
        response = requests.post(
            request_base_url,
            headers={
                "Authorization": f"Bearer {effective_key}",
                "Content-Type": "application/json",
            },
            data=json.dumps(payload),
            timeout=timeout,
        )
    except requests.Timeout as exc:
        _set_ai_error(f"GLM请求超时（当前超时设置为{timeout}秒）", exc)
        raise
    except requests.RequestException as exc:
        _set_ai_error(f"GLM网络请求失败：{exc}", exc)
        raise

    if response.status_code != 200:
        error_text = response.text[:300].replace("\n", " ")
        message = f"HTTP {response.status_code}: {error_text}"
        _set_ai_error(message)
        raise RuntimeError(message)

    data = response.json()
    if "choices" not in data or not data["choices"]:
        _set_ai_error("AI返回结果为空")
        raise RuntimeError("AI返回结果为空")

    message = data["choices"][0].get("message", {})
    content = message.get("content", "")
    if isinstance(content, list):
        content = "".join(part.get("text", "") for part in content)
    return str(content).strip()


def polish_report_with_glm(report_text, api_key=None, model="glm-4-flash", base_url=None):
    if not report_text or not str(report_text).strip():
        return "离线建议为空，无法进行 AI 润色。"

    effective_key, _, _ = resolve_api_settings(api_key, model, base_url)
    if not effective_key:
        return "未配置密钥，AI润色已回退到离线建议。"

    report = str(report_text)
    risk_match = re.search(r"风险分级：([^\n]+)", report)
    risk_level = risk_match.group(1).strip() if risk_match else "暂无明确风险等级"
    abnormal_lines = []
    feature_section = report.split("关键特征判读：", 1)[-1].split("特征增强描述：", 1)[0]
    for line in feature_section.splitlines():
        if "异常" in line and ("轻度" in line or "显著" in line):
            abnormal_lines.append(line.strip().lstrip("- "))
    abnormal_features = "；".join(abnormal_lines) or "未发现需要特别强调的异常特征"
    advice = report.split("综合建议：", 1)[-1].strip() if "综合建议：" in report else report
    prompt = (
        "请把以下心电筛查报告润色为更通俗、简洁、适合医患沟通的版本。\n"
        "只输出四段，段落标题必须依次为：风险结论、病因分析、生活建议、医生话术。"
        "每段不超过3句话。保留风险提醒和原始建议，不增加未提供的诊断。"
        "禁止重复罗列特征名称或数值，禁止抄写12项特征列表。\n\n"
        f"风险等级：{risk_level}\n"
        f"异常特征：{abnormal_features}\n"
        f"原始建议：{advice}"
    )
    return _call_glm_api(
        prompt,
        model_name=model,
        api_key=api_key,
        base_url=base_url,
        system_prompt="你是心内科医生助手。请把以下报告润色为更通俗的版本。",
    )


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            return str(value)
    return value


def _parse_json_response(content):
    if not content:
        return None
    text = str(content).strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    candidate = fenced.group(1).strip() if fenced else text
    try:
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start >= 0 and end > start:
            try:
                parsed = json.loads(candidate[start:end + 1])
                return parsed if isinstance(parsed, dict) else None
            except json.JSONDecodeError:
                return None
    return None


def generate_ai_diagnosis(analysis_result, api_key=None, model=None, base_url=None):
    """Generate a structured, plain-language interpretation of an ECG analysis."""
    global _LAST_AI_ERROR
    _LAST_AI_ERROR = None
    try:
        if not isinstance(analysis_result, dict):
            _set_ai_error("分析结果不是有效字典")
            return None
        payload = {
            "features": analysis_result.get("features", {}),
            "shap_result": analysis_result.get("shap_result", {}),
            "risk_level": analysis_result.get("risk_level"),
            "risk_score": analysis_result.get("score"),
            "risk_probabilities": analysis_result.get("risk_probs", []),
            "cnn_result": {
                "status": analysis_result.get("cnn_status"),
                "confidence": analysis_result.get("cnn_confidence"),
                "abnormal_beat_count": len(analysis_result.get("abnormal_positions", []) or []),
                "total_beat_count": len(analysis_result.get("r_peaks", []) or []),
            },
            "patient_info": analysis_result.get("patient_info", {}),
            "offline_report": analysis_result.get("report_data", {}),
        }
        prompt = (
            "请根据下面完整的心电筛查结果，生成结构化的中文诊断解读。\n"
            "要求：面向基层医护人员和患者，先说结论再解释原因；把数值异常与SHAP贡献分别说明；"
            "不得把辅助筛查结果写成确诊，不得虚构未提供的症状或检查；紧急提示必须明确触发条件。\n"
            "只返回合法JSON对象，不要Markdown代码块，不要额外说明。JSON字段必须严格包含："
            "risk_summary（字符串）、etiology_analysis（字符串）、feature_explanations（数组，"
            "每项含feature、value、explanation）、shap_interpretation（字符串）、"
            "lifestyle_advice（字符串）、emergency_warning（字符串）。\n\n"
            f"分析结果：\n{json.dumps(_json_safe(payload), ensure_ascii=False, indent=2)}"
        )
        content = _call_glm_api(
            prompt,
            model_name=model,
            api_key=api_key,
            base_url=base_url,
            timeout=30,
            system_prompt="你是心内科医生助手，擅长用通俗语言解释心电图结果，避免使用专业术语，面向基层医护人员和患者。",
        )
        parsed = _parse_json_response(content)
        required = {"risk_summary", "etiology_analysis", "feature_explanations", "shap_interpretation", "lifestyle_advice", "emergency_warning"}
        if not parsed:
            _set_ai_error("AI返回内容无法解析为JSON")
            return None
        missing = required.difference(parsed)
        if missing:
            _set_ai_error(f"AI返回JSON缺少字段：{', '.join(sorted(missing))}")
            return None
        return parsed
    except Exception as exc:
        if not get_last_ai_error() or get_last_ai_error() == "未返回具体错误信息。":
            _set_ai_error(f"AI智能解读失败：{exc}", exc)
        return None


def answer_with_rag(question):
    """Answer an ECG disease question using the local knowledge base and GLM."""
    try:
        from knowledge.retriever import search_disease

        if not str(question or "").strip():
            return None
        matches = search_disease(question)
        context = json.dumps(matches, ensure_ascii=False, indent=2)
        prompt = (
            "请基于提供的心电疾病知识回答问题。用通俗中文分点回答，说明可能性和下一步建议；"
            "不要把知识库内容当作个体确诊，不要编造患者没有提供的检查结果。若问题超出知识库，"
            "请明确说明需要医生结合完整病史判断。\n\n"
            f"问题：{question}\n\n相关知识：\n{context}"
        )
        return _call_glm_api(
            prompt,
            timeout=30,
            system_prompt="你是心内科医生助手，负责依据可靠知识向基层医护人员和患者解释常见心电问题。",
        )
    except Exception:
        return None
