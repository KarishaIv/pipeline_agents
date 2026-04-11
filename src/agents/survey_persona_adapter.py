from __future__ import annotations

import re
from typing import Any, Dict, List

from src.schemas.survey_deliberation_schema import SurveyPersonaModel


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _mid_age(age_group: str) -> float:
    text = str(age_group or "").strip()
    match = re.match(r"(\d+)-(\d+)", text)
    if not match:
        return 40.0
    return (int(match.group(1)) + int(match.group(2))) / 2


def _children_count(children_group: str) -> float:
    text = str(children_group or "").strip()
    if text == "3+":
        return 3.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def _income_score(income_level: str) -> float:
    text = str(income_level or "").lower()
    mapping = [
        ("низкий", 0.10),
        ("выше мрот", 0.25),
        ("средний", 0.45),
        ("выше_среднего", 0.62),
        ("высокий", 0.78),
        ("очень_высокий", 0.92),
        ("ultima", 0.98),
    ]
    for token, score in mapping:
        if token in text:
            return score
    return 0.45


def _education_score(education: str) -> float:
    text = str(education or "").lower()
    if "аспирант" in text:
        return 0.90
    if "магистр" in text:
        return 0.82
    if "бакалавр" in text or "специалитет" in text:
        return 0.72
    if "незакончен" in text:
        return 0.58
    if "среднего профессионального" in text:
        return 0.50
    if "среднее" in text:
        return 0.40
    return 0.50


def _trait_score(text: str, trait: str) -> float:
    value = str(text or "").lower()
    if trait == "openness":
        if "открыт новому опыту" in value:
            return 0.80
        if "умеренно открыт" in value:
            return 0.62
    elif trait == "conscientiousness":
        if "очень организован" in value:
            return 0.92
        if "дисциплинирован" in value:
            return 0.82
        if "обычно ответственен" in value:
            return 0.68
    elif trait == "extraversion":
        if "очень общителен" in value:
            return 0.90
        if "общительный" in value:
            return 0.74
        if "умеренно общителен" in value:
            return 0.52
        if "склонен к одиночеству" in value:
            return 0.20
    elif trait == "agreeableness":
        if "очень эмпатичен" in value:
            return 0.90
        if "альтруист" in value or "эмпатичен" in value:
            return 0.76
    elif trait == "neuroticism":
        if "обычно спокоен" in value:
            return 0.35
    return 0.50


def _top_summary(signals: Dict[str, float]) -> List[str]:
    lines: List[str] = []
    if signals["financial_caution"] >= 0.65:
        lines.append("Склонен к финансовой осторожности и избеганию лишнего риска.")
    elif signals["financial_caution"] <= 0.35:
        lines.append("Скорее допускает риск, если видит в нем полезный шанс.")

    if signals["economic_pressure"] >= 0.65:
        lines.append("Финансовое давление и бытовая устойчивость заметно влияют на ответы.")
    elif signals["economic_pressure"] <= 0.35:
        lines.append("Финансовое давление выражено слабо, поэтому денежные вопросы не доминируют автоматически.")

    if signals["institutional_trust"] >= 0.62:
        lines.append("Склонен доверять институциональным источникам и формальным правилам.")
    elif signals["institutional_trust"] <= 0.40:
        lines.append("Скорее настороженно относится к внешним источникам и институциональным утверждениям.")

    if signals["media_skepticism"] >= 0.62:
        lines.append("К массовым медиа относится критично и не склонен доверять им без дополнительных подтверждений.")
    elif signals["media_skepticism"] <= 0.38:
        lines.append("Скорее допускает доверие к привычным медиа-источникам, если они выглядят официально.")

    if signals["traditionalism"] >= 0.62:
        lines.append("Тяготеет к более традиционным нормам и устойчивым жизненным ролям.")
    elif signals["openness_to_change"] >= 0.68:
        lines.append("Скорее открыт к изменениям и менее привязан к жестким нормам.")

    if signals["social_conformity"] >= 0.62:
        lines.append("Замечает реакцию окружения и склонен учитывать социальные ожидания.")

    if signals["consumer_pragmatism"] >= 0.62:
        lines.append("В потребительских и бытовых вопросах чаще выбирает практичность, а не имидж.")

    if signals["professional_identity_salience"] >= 0.64:
        lines.append("Профессиональная роль сильно влияет на самооценку и ответы по смежным темам.")

    return lines[:6] or [
        "Профиль выглядит умеренным без сильного смещения в одну позицию.",
        "Ответы вероятно будут зависеть от конкретной темы вопроса.",
    ]


class SurveyPersonaAdapter:
    def build(self, profile: Dict[str, Any]) -> SurveyPersonaModel:
        age_mid = _mid_age(profile.get("age_group"))
        children = _children_count(profile.get("children_group"))
        income = _income_score(profile.get("income_level"))
        education = _education_score(profile.get("education"))
        openness = _trait_score(profile.get("openness"), "openness")
        conscientiousness = _trait_score(profile.get("conscientiousness"), "conscientiousness")
        extraversion = _trait_score(profile.get("extraversion"), "extraversion")
        agreeableness = _trait_score(profile.get("agreeableness"), "agreeableness")
        neuroticism = _trait_score(profile.get("neuroticism"), "neuroticism")
        region = str(profile.get("region") or "").lower()
        occupation = str(profile.get("occupation") or "").lower()
        marital_status = str(profile.get("marital_status") or "").lower()

        financial_caution = _clamp(
            0.28 * (1.0 - income)
            + 0.20 * conscientiousness
            + 0.16 * (1.0 - openness)
            + 0.18 * neuroticism
            + 0.10 * min(children / 3.0, 1.0)
            + 0.08 * (1.0 if "бедный" in region else 0.35)
        )
        economic_pressure = _clamp(
            0.38 * (1.0 - income)
            + 0.18 * min(children / 3.0, 1.0)
            + 0.14 * (1.0 if "бедный" in region else 0.35)
            + 0.12 * neuroticism
            + 0.10 * (0.75 if age_mid < 40 else 0.55)
            + 0.08 * (0.65 if marital_status.startswith(("женат", "замуж")) else 0.45)
        )
        institutional_trust = _clamp(
            0.28 * education
            + 0.22 * conscientiousness
            + 0.12 * agreeableness
            + 0.18 * (1.0 - neuroticism)
            + 0.10 * (1.0 if "учител" in occupation or "врач" in occupation or "бухгалтер" in occupation else 0.4)
            + 0.10 * (0.55 if age_mid >= 45 else 0.45)
        )
        media_skepticism = _clamp(
            0.26 * education
            + 0.20 * openness
            + 0.18 * conscientiousness
            + 0.10 * (1.0 - agreeableness)
            + 0.12 * (1.0 if "миллионник" in region else 0.45)
            + 0.08 * (1.0 if "учител" in occupation or "студент" in occupation else 0.35)
            + 0.06 * (1.0 if age_mid < 45 else 0.55)
        )
        traditionalism = _clamp(
            0.30 * (1.0 - openness)
            + 0.18 * conscientiousness
            + 0.15 * (1.0 if age_mid >= 45 else 0.35)
            + 0.17 * min(children / 3.0, 1.0)
            + 0.10 * (1.0 if str(profile.get("marital_status") or "").lower().startswith(("женат", "замуж")) else 0.4)
            + 0.10 * (1.0 if "бедный" in region else 0.45)
        )
        social_conformity = _clamp(
            0.30 * agreeableness
            + 0.20 * conscientiousness
            + 0.18 * extraversion
            + 0.12 * min(children / 3.0, 1.0)
            + 0.10 * (1.0 if str(profile.get("marital_status") or "").lower().startswith(("женат", "замуж")) else 0.35)
            + 0.10 * (0.65 if age_mid >= 40 else 0.45)
        )
        consumer_pragmatism = _clamp(
            0.34 * conscientiousness
            + 0.24 * (1.0 - openness)
            + 0.18 * financial_caution
            + 0.12 * min(children / 3.0, 1.0)
            + 0.12 * (1.0 if "бедный" in region else 0.45)
        )
        openness_to_change = _clamp(
            0.42 * openness
            + 0.18 * extraversion
            + 0.16 * education
            + 0.12 * (1.0 - traditionalism)
            + 0.12 * (0.65 if age_mid < 40 else 0.40)
        )
        professional_identity_salience = _clamp(
            0.24 * conscientiousness
            + 0.16 * education
            + 0.14 * (1.0 if any(token in occupation for token in ["маркет", "учител", "врач", "медсест", "бухгалтер", "юрист", "менедж", "инженер"]) else 0.45)
            + 0.12 * (1.0 if any(token in occupation for token in ["маркет", "учител", "врач", "медсест", "психолог", "дизайнер"]) else 0.35)
            + 0.10 * extraversion
            + 0.10 * agreeableness
            + 0.14 * (0.75 if age_mid >= 30 else 0.45)
        )

        payload = {
            "financial_caution": round(financial_caution, 3),
            "economic_pressure": round(economic_pressure, 3),
            "institutional_trust": round(institutional_trust, 3),
            "media_skepticism": round(media_skepticism, 3),
            "traditionalism": round(traditionalism, 3),
            "social_conformity": round(social_conformity, 3),
            "consumer_pragmatism": round(consumer_pragmatism, 3),
            "openness_to_change": round(openness_to_change, 3),
            "professional_identity_salience": round(professional_identity_salience, 3),
        }
        payload["summary"] = _top_summary(payload)

        if hasattr(SurveyPersonaModel, "model_validate"):
            return SurveyPersonaModel.model_validate(payload)
        return SurveyPersonaModel(**payload)
