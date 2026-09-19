import streamlit as st


def mask_name(name):
    """给姓名做展示层脱敏，保留首字符。"""
    raw = str(name or "").strip()
    if not raw:
        return "未命名患者"
    if len(raw) == 1:
        return raw[0] + "*"
    if len(raw) == 2:
        return raw[0] + "**"
    return raw[0] + "**"


def validate_patient_info(name, age, sex):
    """校验姓名长度和年龄区间，确保用户输入合法。"""
    if name is not None and len(str(name).strip()) > 50:
        return False, "姓名不能超过 50 个字符。"
    try:
        age_value = int(age)
    except Exception:
        return False, "年龄必须为整数。"
    if age_value < 0 or age_value > 120:
        return False, "年龄必须在 0-120 岁之间。"
    return True, "ok"


def get_patient_profile():
    """从 session state 读出当前患者信息。"""
    return {
        "name": st.session_state.get("patient_name", ""),
        "age": int(st.session_state.get("patient_age", 0) or 0),
        "sex": st.session_state.get("patient_sex", "未指定"),
        "height_cm": int(st.session_state.get("patient_height", 170) or 170),
        "weight_kg": float(st.session_state.get("patient_weight", 65) or 65.0),
        "past_history": list(st.session_state.get("patient_history", []) or []),
        "medication": st.session_state.get("patient_medication", ""),
    }


def save_patient_profile():
    """把当前患者信息写回 session state。"""
    st.session_state["patient_profile"] = get_patient_profile()
