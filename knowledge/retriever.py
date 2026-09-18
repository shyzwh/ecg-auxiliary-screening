import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
KNOWLEDGE_PATH = BASE_DIR / "knowledge" / "ecg_diseases.json"


def load_diseases():
    try:
        with KNOWLEDGE_PATH.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data.get("diseases", []) if isinstance(data, dict) else []
    except (OSError, json.JSONDecodeError):
        return []


def search_disease(query_text):
    query = str(query_text or "").strip().lower()
    if not query:
        return []
    query_terms = {query}
    query_terms.update(part for part in query.replace(",", " ").replace("，", " ").split() if part)
    matches = []
    for disease in load_diseases():
        disease_name = str(disease.get("name", "")).lower()
        category = str(disease.get("category", "")).lower()
        searchable = " ".join([
            str(disease.get("id", "")),
            disease_name,
            category,
            str(disease.get("definition", "")),
            " ".join(map(str, disease.get("ecg_features", []))),
            " ".join(map(str, disease.get("causes", []))),
            str(disease.get("clinical_significance", "")),
        ]).lower()
        score = sum(searchable.count(term) for term in query_terms if term)
        if disease_name and disease_name in query:
            score += 5
        if category and category in query:
            score += 2
        if score:
            matches.append((score, disease))
    matches.sort(key=lambda item: item[0], reverse=True)
    return [disease for _, disease in matches[:5]]
