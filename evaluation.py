from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from statistics import mean
from typing import Any, Iterable, Optional

# Constants


DEFAULT_LLM_TRUST_COEFFICIENT = 0.7

SCENARIO_TOPIC_FINAL = "topic_final"
SCENARIO_TOPIC_QUESTIONS = "topic_clarification_questions"
SCENARIO_TOPIC_ROUND = "topic_alignment_round"
SCENARIO_TOPIC_PROCESS = "topic_alignment_process"
SCENARIO_DEFENSE_QUESTIONS = "defense_questions"

LLM_CRITERIA = (
    "relevance",
    "completeness",
    "clarity",
    "usefulness",
    "correctness",
)

LLM_CRITERIA_RU = {
    "relevance": "релевантность",
    "completeness": "полнота",
    "clarity": "ясность",
    "usefulness": "полезность",
    "correctness": "корректность",
}

STOP_WORDS_RU = {
    "без", "был", "была", "были", "было", "быть", "вам", "вас", "все",
    "для", "его", "если", "еще", "или", "как", "над", "нее", "них", "они",
    "при", "про", "раз", "так", "там", "тем", "тот", "тут", "уже", "что",
    "эта", "эти", "это", "этот", "такая", "такие", "такой", "данный",
    "данная", "данные", "работа", "работы", "тема", "темы", "задание",
    "задания", "студент", "студента", "преподаватель", "преподавателя",
    "рамках", "рамки", "нужно", "можно", "должен", "должна", "должны",
    "будет", "является", "который", "которая", "которые", "которое",
}

STOP_WORDS_EN = {
    "the", "and", "for", "with", "from", "that", "this", "work", "topic",
    "student", "teacher", "task", "project", "should", "would", "could",
}

STOP_WORDS = STOP_WORDS_RU | STOP_WORDS_EN

GENERIC_TOPIC_PATTERNS = (
    "уточните тему",
    "выполнить работу",
    "разработать проект",
    "подготовить работу",
    "в рамках дисциплины",
    "индивидуальное задание",
    "учебная работа",
)

GENERIC_QUESTION_PATTERNS = (
    "уточните тему",
    "расскажите подробнее",
    "что нужно сделать",
    "какие требования",
    "какая тема",
    "что вы хотите",
    "поясните задание",
)

METHOD_RESULT_KEYWORDS = {
    "метод", "методика", "алгоритм", "модель", "технология", "система",
    "приложение", "прототип", "архитектура", "реализация", "разработка",
    "анализ", "исследование", "оценка", "результат", "критерий",
    "данные", "датасет", "эксперимент", "интерфейс", "модуль",
}

CLARIFICATION_COVERAGE_KEYWORDS = {
    "data": {"данные", "датасет", "источник", "материал", "пример", "выборка", "файл"},
    "methods": {"метод", "алгоритм", "модель", "технология", "подход", "инструмент"},
    "limits": {"границ", "огранич", "объем", "уровень", "формат", "требован"},
    "result": {"результат", "продукт", "отчет", "приложение", "система", "артефакт"},
}

DEFENSE_COVERAGE_KEYWORDS = {
    "theory": {"теор", "понят", "принцип", "обосн", "почему", "чем отличается"},
    "implementation": {"реализ", "код", "архитект", "модуль", "интерфейс", "библиотек"},
    "data": {"данн", "выборк", "материал", "источник", "пример", "датасет"},
    "quality": {"качеств", "метрик", "оцен", "провер", "тест", "критер"},
    "limitations": {"огранич", "недостат", "риск", "улучш", "дальше"},
}

CONFLICT_MARKERS = (
    ("web", "desktop"),
    ("веб", "десктоп"),
    ("мобиль", "десктоп"),
    ("python", "c#"),
    ("java", "python"),
    ("нейросет", "без нейросет"),
    ("машинн", "без машинн"),
)


# Data classes


@dataclass(frozen=True)
class HeuristicMetric:
    """
    Результат одной формальной проверки.

    name    — машинное имя метрики, по которому удобно читать код и отчет;
    value   — измеренное значение или краткое описание найденного признака;
    score   — нормированная оценка от 0.0 до 1.0;
    weight  — вес метрики при расчете общего heuristic_score;
    comment — пояснение для отчета, почему выставлена такая оценка.
    """

    name: str
    value: Any
    score: float
    weight: float = 1.0
    comment: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["score"] = round_float(data["score"])
        data["weight"] = round_float(data["weight"])
        return data


@dataclass(frozen=True)
class LLMScore:
    """
    Структурированная оценка сторонней модели.

    raw_scores содержит исходные оценки 1-5 по критериям:
    relevance, completeness, clarity, usefulness, correctness.
    normalized_score переводит среднюю оценку в диапазон 0.0-1.0.
    """

    raw_scores: dict[str, int]
    raw_average: float
    normalized_score: float
    comment: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_scores": dict(self.raw_scores),
            "raw_average": round_float(self.raw_average),
            "normalized_score": round_float(self.normalized_score),
            "comment": self.comment,
        }


@dataclass(frozen=True)
class EvaluationResult:
    """
    Итог оценки одной генерации.

    final_score считается по формуле:
    final_score = k * llm_score + (1 - k) * heuristic_score.

    Если LLM-оценка не передана, final_score равен heuristic_score,
    а поле evaluation_mode получает значение "heuristic_only".
    """

    scenario_part: str
    method_name: str
    llm_score: Optional[LLMScore]
    heuristic_metrics: list[HeuristicMetric]
    heuristic_score: float
    final_score: float
    k: float
    evaluation_mode: str
    comment: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_part": self.scenario_part,
            "method_name": self.method_name,
            "llm_score": self.llm_score.to_dict() if self.llm_score else None,
            "heuristic_metrics": [metric.to_dict() for metric in self.heuristic_metrics],
            "heuristic_score": round_float(self.heuristic_score),
            "final_score": round_float(self.final_score),
            "k": round_float(self.k),
            "evaluation_mode": self.evaluation_mode,
            "comment": self.comment,
            "extra": self.extra,
        }


@dataclass(frozen=True)
class ProcessEvaluationResult:
    """
    Оценка процесса согласования темы по нескольким раундам.

    scenario_score считается по формуле:
    scenario_score = 0.6 * last_round_score
                   + 0.3 * average_round_score
                   + 0.1 * convergence_score.
    """

    scenario_part: str
    method_name: str
    round_results: list[EvaluationResult]
    last_round_score: float
    average_round_score: float
    convergence_score: float
    scenario_score: float
    finalized: bool
    finalized_round: Optional[int]
    max_rounds: int
    comment: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_part": self.scenario_part,
            "method_name": self.method_name,
            "round_results": [result.to_dict() for result in self.round_results],
            "last_round_score": round_float(self.last_round_score),
            "average_round_score": round_float(self.average_round_score),
            "convergence_score": round_float(self.convergence_score),
            "scenario_score": round_float(self.scenario_score),
            "finalized": self.finalized,
            "finalized_round": self.finalized_round,
            "max_rounds": self.max_rounds,
            "comment": self.comment,
        }


# Generic helpers


def round_float(value: Any, digits: int = 4) -> Any:
    try:
        if value is None:
            return None
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return 0.0
        return round(value, digits)
    except Exception:
        return value


def clamp(value: Any, low: float = 0.0, high: float = 1.0) -> float:
    try:
        number = float(value)
    except Exception:
        number = low
    if math.isnan(number) or math.isinf(number):
        number = low
    return max(low, min(high, number))


def safe_json_dumps(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, indent=2)
    except Exception:
        return str(value)


def normalize_text(text: Any) -> str:
    text = "" if text is None else str(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_for_compare(text: Any) -> str:
    text = normalize_text(text).lower().replace("ё", "е")
    text = re.sub(r"[^a-zа-я0-9\s]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: Any, *, min_len: int = 3, remove_stop_words: bool = True) -> set[str]:
    text = normalize_for_compare(text)
    words = re.findall(r"[a-zа-я0-9]{%d,}" % int(min_len), text)
    if remove_stop_words:
        words = [word for word in words if word not in STOP_WORDS]
    return set(words)


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa = set(a)
    sb = set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))


def list_overlap(a: Iterable[str], b: Iterable[str], limit: int = 20) -> list[str]:
    return sorted(set(a) & set(b))[:limit]


def extract_text_from_payload(payload: Any) -> str:
    """
    Собирает текст из строки, словаря или списка.

    Функция нужна для универсальных тест-кейсов: в разных частях сценария
    вход может быть строкой, JSON-структурой темы, спецификацией задания,
    результатом анализа работы или списком вопросов.
    """
    if payload is None:
        return ""
    if isinstance(payload, str):
        return normalize_text(payload)
    if isinstance(payload, (int, float, bool)):
        return str(payload)
    if isinstance(payload, dict):
        parts: list[str] = []
        preferred_keys = (
            "title", "description", "agreed_title", "agreed_description",
            "summary", "body", "body_text", "context", "text", "question_text",
            "answer_text", "analysis", "analysis_json", "acceptance_criteria",
            "acceptance_criteria_json", "student_topic", "teacher_topic",
            "generated_topic", "generated_output", "submission_text",
        )
        for key in preferred_keys:
            if key in payload:
                parts.append(extract_text_from_payload(payload.get(key)))
        for key, value in payload.items():
            if key not in preferred_keys and not key.endswith("_id"):
                if isinstance(value, (dict, list, tuple)):
                    parts.append(extract_text_from_payload(value))
        return normalize_text("\n".join(part for part in parts if part))
    if isinstance(payload, (list, tuple, set)):
        return normalize_text("\n".join(extract_text_from_payload(item) for item in payload))
    return normalize_text(str(payload))


def split_questions(value: Any) -> list[str]:
    """
    Извлекает список вопросов из строки, списка или JSON-структуры.

    Поддерживаются форматы:
    - ["Вопрос 1?", "Вопрос 2?"];
    - {"questions": [...]} для защиты;
    - {"student_questions": [...], "teacher_questions": [...]} для уточнений;
    - многострочная строка с нумерацией.
    """
    if value is None:
        return []

    if isinstance(value, dict):
        questions: list[str] = []
        labels = ["questions", "student_questions", "teacher_questions", "clarification_questions", "defense_questions"]
        for key in labels:
            if key in value:
                questions.extend(split_questions(value.get(key)))
        if questions:
            return questions
        return split_questions(extract_text_from_payload(value))

    if isinstance(value, (list, tuple, set)):
        result: list[str] = []
        for item in value:
            if isinstance(item, dict):
                candidate = item.get("question") or item.get("text") or item.get("question_text")
                result.extend(split_questions(candidate))
            else:
                result.extend(split_questions(item))
        return [q for q in result if q]

    text = normalize_text(value)
    if not text:
        return []

    lines = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        line = re.sub(r"^[-*•\s]+", "", line)
        line = re.sub(r"^\d+[.)]\s*", "", line)
        line = line.strip()
        if line:
            lines.append(line)

    if len(lines) > 1:
        return [line for line in lines if line]

    if "?" in text:
        chunks = re.findall(r"[^?]+\?", text)
        return [normalize_text(chunk) for chunk in chunks if normalize_text(chunk)]

    return [text] if text else []


def unique_normalized_items(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        norm = normalize_for_compare(item)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        result.append(item)
    return result


def weighted_average(metrics: Iterable[HeuristicMetric]) -> float:
    items = list(metrics)
    total_weight = sum(max(0.0, float(item.weight)) for item in items)
    if total_weight <= 0:
        return 0.0
    value = sum(clamp(item.score) * max(0.0, float(item.weight)) for item in items) / total_weight
    return clamp(value)


# LLM score helpers


def normalize_llm_raw_score(raw_score: Any) -> float:
    """
    Переводит оценку LLM из шкалы 1-5 в шкалу 0.0-1.0.

    Формула:
    normalized_score = (raw_score - 1) / 4.
    """
    try:
        score = float(raw_score)
    except Exception:
        score = 1.0
    return clamp((score - 1.0) / 4.0)


def parse_llm_score(llm_scores: Optional[dict[str, Any]]) -> Optional[LLMScore]:
    """
    Преобразует JSON-ответ LLM-оценщика в LLMScore.

    Ожидаемые поля: relevance, completeness, clarity, usefulness, correctness.
    Допускается вложенный формат {"scores": {...}} или {"raw_scores": {...}}.
    """
    if not llm_scores:
        return None

    source = llm_scores
    if isinstance(llm_scores.get("scores"), dict):
        source = llm_scores["scores"]
    elif isinstance(llm_scores.get("raw_scores"), dict):
        source = llm_scores["raw_scores"]

    raw: dict[str, int] = {}
    for criterion in LLM_CRITERIA:
        value = source.get(criterion)
        if value is None:
            value = source.get(LLM_CRITERIA_RU.get(criterion, ""))
        try:
            numeric = int(round(float(value)))
        except Exception:
            numeric = 1
        raw[criterion] = int(max(1, min(5, numeric)))

    raw_average = mean(raw.values()) if raw else 1.0
    comment = str(llm_scores.get("comment") or llm_scores.get("reason") or llm_scores.get("summary") or "")
    return LLMScore(
        raw_scores=raw,
        raw_average=raw_average,
        normalized_score=normalize_llm_raw_score(raw_average),
        comment=comment,
    )


def combine_llm_and_heuristic_score(
        *,
        llm_score: Optional[float],
        heuristic_score: float,
        k: float = DEFAULT_LLM_TRUST_COEFFICIENT,
) -> tuple[float, str]:
    """
    Объединяет LLM-оценку и формальную оценку.

    final_score = k * llm_score + (1 - k) * heuristic_score.

    Если llm_score отсутствует, возвращается heuristic_score. Это позволяет
    прогонять тесты без внешней модели и отлаживать только формальные метрики.
    """
    heuristic_score = clamp(heuristic_score)
    k = clamp(k)
    if llm_score is None:
        return heuristic_score, "heuristic_only"
    llm_score = clamp(llm_score)
    return clamp(k * llm_score + (1.0 - k) * heuristic_score), "llm_and_heuristic"


def build_evaluation_result(
        *,
        scenario_part: str,
        method_name: str,
        heuristic_metrics: list[HeuristicMetric],
        llm_scores: Optional[dict[str, Any]] = None,
        k: float = DEFAULT_LLM_TRUST_COEFFICIENT,
        comment: str = "",
        extra: Optional[dict[str, Any]] = None,
) -> EvaluationResult:
    llm_score = parse_llm_score(llm_scores)
    heuristic_score = weighted_average(heuristic_metrics)
    final_score, mode = combine_llm_and_heuristic_score(
        llm_score=llm_score.normalized_score if llm_score else None,
        heuristic_score=heuristic_score,
        k=k,
    )
    return EvaluationResult(
        scenario_part=scenario_part,
        method_name=method_name,
        llm_score=llm_score,
        heuristic_metrics=heuristic_metrics,
        heuristic_score=heuristic_score,
        final_score=final_score,
        k=clamp(k),
        evaluation_mode=mode,
        comment=comment,
        extra=extra or {},
    )


# General heuristic metrics


def score_text_length(
        text: Any,
        *,
        min_chars: int,
        max_chars: int,
        ideal_min_chars: Optional[int] = None,
        ideal_max_chars: Optional[int] = None,
        name: str = "text_length",
        weight: float = 1.0,
) -> HeuristicMetric:
    """
    Оценивает достаточность длины сгенерированного текста.

    Слишком короткий текст получает низкую оценку, потому что обычно не содержит
    нужной конкретики. Слишком длинный текст получает штраф, потому что может быть
    раздутым и неудобным для пользователя.
    """
    text = normalize_text(text)
    length = len(text)
    ideal_min = ideal_min_chars if ideal_min_chars is not None else min_chars
    ideal_max = ideal_max_chars if ideal_max_chars is not None else max_chars

    if not text:
        score = 0.0
        comment = "Текст отсутствует."
    elif ideal_min <= length <= ideal_max:
        score = 1.0
        comment = "Длина текста находится в оптимальном диапазоне."
    elif min_chars <= length <= max_chars:
        score = 0.75
        comment = "Длина текста допустима, но не попадает в оптимальный диапазон."
    elif length < min_chars:
        ratio = length / max(1, min_chars)
        score = clamp(0.2 + 0.55 * ratio)
        comment = "Текст короче допустимого диапазона."
    else:
        overflow = (length - max_chars) / max(1, max_chars)
        score = clamp(0.75 - min(0.55, overflow))
        comment = "Текст длиннее допустимого диапазона."

    return HeuristicMetric(name=name, value=length, score=score, weight=weight, comment=comment)


def score_not_empty_and_specific(
        text: Any,
        *,
        generic_patterns: Iterable[str] = GENERIC_TOPIC_PATTERNS,
        name: str = "specificity",
        weight: float = 1.0,
) -> HeuristicMetric:
    """
    Проверяет, что текст не пустой и не состоит из универсальной формулировки.

    Эвристика не пытается понимать смысл глубоко. Она отлавливает пустые ответы,
    слишком короткие формулировки и распространенные шаблоны без предметной области.
    """
    text = normalize_text(text)
    norm = normalize_for_compare(text)
    tokens = tokenize(text)
    found_generic = [pattern for pattern in generic_patterns if pattern in norm]

    if not text:
        score = 0.0
        comment = "Формулировка отсутствует."
    elif len(tokens) < 4:
        score = 0.25
        comment = "Формулировка слишком короткая и выглядит непредметной."
    elif found_generic and len(tokens) < 10:
        score = 0.35
        comment = "Формулировка содержит общий шаблон без достаточной конкретики."
    elif found_generic:
        score = 0.7
        comment = "Есть признаки общей формулировки, но присутствует дополнительная конкретика."
    else:
        score = 1.0
        comment = "Формулировка выглядит предметной по формальным признакам."

    return HeuristicMetric(
        name=name,
        value={"token_count": len(tokens), "generic_patterns": found_generic},
        score=score,
        weight=weight,
        comment=comment,
    )


def score_keyword_overlap(
        generated_text: Any,
        reference_text: Any,
        *,
        name: str,
        label: str,
        min_overlap: int = 2,
        good_overlap: int = 5,
        weight: float = 1.0,
) -> HeuristicMetric:
    """
    Оценивает пересечение ключевых слов с заданным источником.

    Используется отдельно для темы студента и темы преподавателя, чтобы итоговая
    формулировка не потеряла ни одну из сторон согласования.
    """
    generated_tokens = tokenize(generated_text)
    reference_tokens = tokenize(reference_text)
    overlap = list_overlap(generated_tokens, reference_tokens, limit=30)

    if not generated_tokens or not reference_tokens:
        score = 0.0
        comment = f"Недостаточно текста для проверки пересечения с {label}."
    elif len(overlap) >= good_overlap:
        score = 1.0
        comment = f"Есть устойчивое пересечение ключевых слов с {label}."
    elif len(overlap) >= min_overlap:
        score = 0.65 + 0.35 * (len(overlap) - min_overlap) / max(1, good_overlap - min_overlap)
        comment = f"Есть минимально достаточное пересечение с {label}."
    elif len(overlap) == 1:
        score = 0.35
        comment = f"Пересечение с {label} слабое."
    else:
        score = 0.0
        comment = f"Пересечение с {label} не найдено."

    return HeuristicMetric(
        name=name,
        value={"overlap": overlap, "overlap_count": len(overlap)},
        score=score,
        weight=weight,
        comment=comment,
    )


def score_required_keywords(
        text: Any,
        keyword_groups: dict[str, set[str]],
        *,
        name: str,
        min_groups: int = 1,
        weight: float = 1.0,
) -> HeuristicMetric:
    """
    Проверяет покрытие обязательных смысловых групп через словари ключевых слов.

    Пример: для вопросов защиты группы могут соответствовать теории, реализации,
    данным, качеству и ограничениям.
    """
    norm = normalize_for_compare(text)
    matched: dict[str, list[str]] = {}
    for group, keywords in keyword_groups.items():
        found = sorted(keyword for keyword in keywords if keyword in norm)
        if found:
            matched[group] = found

    total_groups = len(keyword_groups) or 1
    matched_count = len(matched)
    if matched_count >= total_groups:
        score = 1.0
        comment = "Покрыты все требуемые группы признаков."
    elif matched_count >= min_groups:
        score = 0.45 + 0.55 * matched_count / total_groups
        comment = "Покрыта часть требуемых групп признаков."
    else:
        score = 0.15 if matched_count else 0.0
        comment = "Требуемые группы признаков почти не покрыты."

    return HeuristicMetric(
        name=name,
        value={"matched_groups": matched, "matched_count": matched_count, "total_groups": total_groups},
        score=score,
        weight=weight,
        comment=comment,
    )


def score_required_terms_presence(
        text: Any,
        terms: set[str],
        *,
        name: str = "method_or_result_presence",
        min_terms: int = 1,
        good_terms: int = 3,
        weight: float = 1.0,
) -> HeuristicMetric:
    """
    Проверяет наличие указания на метод, технологию, результат или артефакт.

    Для согласованной темы это важно, потому что финальная формулировка должна
    описывать не только область, но и проверяемый способ или ожидаемый результат.
    """
    norm = normalize_for_compare(text)
    found = sorted(term for term in terms if term in norm)
    if len(found) >= good_terms:
        score = 1.0
        comment = "Есть несколько признаков метода, технологии или результата."
    elif len(found) >= min_terms:
        score = 0.7
        comment = "Есть минимальный признак метода, технологии или результата."
    else:
        score = 0.0
        comment = "Не найдено явного указания на метод, технологию или результат."
    return HeuristicMetric(name=name, value=found, score=score, weight=weight, comment=comment)


def score_not_copying_one_side(
        generated_text: Any,
        student_text: Any,
        teacher_text: Any,
        *,
        name: str = "not_copying_one_side",
        weight: float = 1.0,
) -> HeuristicMetric:
    """
    Проверяет, что итоговая тема не копирует дословно только одну сторону.

    Метрика сравнивает нормализованный текст с темой студента и темой преподавателя.
    Если итог почти полностью совпадает с одной стороной и слабо связан с другой,
    выставляется штраф.
    """
    generated_norm = normalize_for_compare(generated_text)
    student_norm = normalize_for_compare(student_text)
    teacher_norm = normalize_for_compare(teacher_text)

    if not generated_norm:
        return HeuristicMetric(name=name, value={}, score=0.0, weight=weight,
                               comment="Итоговая формулировка отсутствует.")

    generated_tokens = tokenize(generated_norm)
    student_similarity = jaccard(generated_tokens, tokenize(student_norm))
    teacher_similarity = jaccard(generated_tokens, tokenize(teacher_norm))

    exact_student_copy = generated_norm == student_norm or (student_norm and generated_norm in student_norm)
    exact_teacher_copy = generated_norm == teacher_norm or (teacher_norm and generated_norm in teacher_norm)

    if exact_student_copy and teacher_similarity < 0.25:
        score = 0.25
        comment = "Итоговая формулировка выглядит как копия темы студента без учета преподавателя."
    elif exact_teacher_copy and student_similarity < 0.25:
        score = 0.25
        comment = "Итоговая формулировка выглядит как копия темы преподавателя без учета студента."
    elif max(student_similarity, teacher_similarity) > 0.85 and min(student_similarity, teacher_similarity) < 0.2:
        score = 0.45
        comment = "Итоговая формулировка чрезмерно близка только к одной стороне."
    else:
        score = 1.0
        comment = "Не найдено признаков дословного копирования только одной стороны."

    return HeuristicMetric(
        name=name,
        value={
            "student_similarity": round_float(student_similarity),
            "teacher_similarity": round_float(teacher_similarity),
            "exact_student_copy": exact_student_copy,
            "exact_teacher_copy": exact_teacher_copy,
        },
        score=score,
        weight=weight,
        comment=comment,
    )


def score_domain_preservation(
        generated_text: Any,
        student_text: Any,
        teacher_text: Any,
        *,
        name: str = "domain_preservation",
        weight: float = 1.0,
) -> HeuristicMetric:
    """
    Проверяет сохранение предметной области по пересечению терминов.

    Если итоговая формулировка не содержит терминов ни из темы студента,
    ни из темы преподавателя, предметная область, вероятно, потеряна.
    """
    generated_tokens = tokenize(generated_text)
    source_tokens = tokenize(student_text) | tokenize(teacher_text)
    overlap = list_overlap(generated_tokens, source_tokens, limit=30)

    if not generated_tokens:
        score = 0.0
        comment = "Итоговая формулировка отсутствует."
    elif len(overlap) >= 5:
        score = 1.0
        comment = "Предметная область сохранена по ключевым терминам."
    elif len(overlap) >= 2:
        score = 0.65
        comment = "Предметная область сохранена частично."
    elif len(overlap) == 1:
        score = 0.35
        comment = "Связь с исходной предметной областью слабая."
    else:
        score = 0.0
        comment = "Связь с исходной предметной областью не обнаружена."

    return HeuristicMetric(
        name=name,
        value={"overlap": overlap, "overlap_count": len(overlap)},
        score=score,
        weight=weight,
        comment=comment,
    )


def score_no_obvious_conflict(
        generated_text: Any,
        student_text: Any,
        teacher_text: Any,
        *,
        name: str = "no_obvious_conflict",
        weight: float = 1.0,
) -> HeuristicMetric:
    """
    Ищет грубые формальные конфликты между исходными темами и итогом.

    Метрика намеренно простая: она не заменяет LLM-проверку корректности,
    но помогает обнаружить ситуации, когда в итог попала несовместимая пара
    технологий или форматов.
    """
    combined = normalize_for_compare("\n".join([
        extract_text_from_payload(student_text),
        extract_text_from_payload(teacher_text),
        extract_text_from_payload(generated_text),
    ]))
    found_conflicts: list[tuple[str, str]] = []
    for left, right in CONFLICT_MARKERS:
        if left in combined and right in combined:
            found_conflicts.append((left, right))

    if found_conflicts:
        score = 0.55
        comment = "Найдены возможные формальные конфликты терминов; требуется ручная проверка."
    else:
        score = 1.0
        comment = "Явных формальных конфликтов не найдено."

    return HeuristicMetric(name=name, value=found_conflicts, score=score, weight=weight, comment=comment)


# Question heuristic metrics


def score_question_count(
        questions: Any,
        *,
        min_count: int,
        max_count: int,
        name: str = "question_count",
        weight: float = 1.0,
) -> HeuristicMetric:
    """
    Проверяет, что число вопросов находится в заданном диапазоне.

    Для уточнения темы обычно достаточно 2-6 вопросов, для защиты — 4-8.
    """
    items = split_questions(questions)
    count = len(items)
    if min_count <= count <= max_count:
        score = 1.0
        comment = "Количество вопросов находится в допустимом диапазоне."
    elif count == 0:
        score = 0.0
        comment = "Вопросы отсутствуют."
    elif count < min_count:
        score = clamp(count / max(1, min_count))
        comment = "Вопросов меньше минимально ожидаемого количества."
    else:
        overflow = count - max_count
        score = clamp(1.0 - 0.2 * overflow)
        comment = "Вопросов больше допустимого диапазона."

    return HeuristicMetric(name=name, value=count, score=score, weight=weight, comment=comment)


def score_question_duplicates(
        questions: Any,
        *,
        name: str = "question_duplicates",
        weight: float = 1.0,
) -> HeuristicMetric:
    """
    Проверяет отсутствие повторяющихся или почти одинаковых вопросов.
    """
    items = split_questions(questions)
    if not items:
        return HeuristicMetric(name=name, value={"count": 0, "duplicates": 0}, score=0.0, weight=weight,
                               comment="Вопросы отсутствуют.")

    normalized = [normalize_for_compare(q) for q in items]
    unique = set(normalized)
    duplicate_count = len(normalized) - len(unique)

    near_duplicates = 0
    tokenized = [tokenize(q) for q in items]
    for i in range(len(tokenized)):
        for j in range(i + 1, len(tokenized)):
            if tokenized[i] and tokenized[j] and jaccard(tokenized[i], tokenized[j]) >= 0.75:
                near_duplicates += 1

    total_duplicates = duplicate_count + near_duplicates
    if total_duplicates == 0:
        score = 1.0
        comment = "Повторы вопросов не обнаружены."
    else:
        score = clamp(1.0 - total_duplicates / max(1, len(items)))
        comment = "Обнаружены повторяющиеся или очень похожие вопросы."

    return HeuristicMetric(
        name=name,
        value={"count": len(items), "exact_duplicates": duplicate_count, "near_duplicates": near_duplicates},
        score=score,
        weight=weight,
        comment=comment,
    )


def score_question_format(
        questions: Any,
        *,
        name: str = "question_format",
        weight: float = 1.0,
) -> HeuristicMetric:
    """
    Проверяет, что элементы списка действительно оформлены как вопросы.

    Основной формальный признак — вопросительный знак в конце. Дополнительно
    учитываются вопросительные слова для случаев, когда знак пропущен.
    """
    items = split_questions(questions)
    if not items:
        return HeuristicMetric(name=name, value={"count": 0}, score=0.0, weight=weight, comment="Вопросы отсутствуют.")

    question_words = ("что", "как", "какие", "какой", "какая", "почему", "зачем", "чем", "где", "когда", "насколько")
    good = 0
    weak = 0
    bad_items: list[str] = []
    for q in items:
        norm = normalize_for_compare(q)
        if normalize_text(q).endswith("?"):
            good += 1
        elif norm.startswith(question_words):
            weak += 1
        else:
            bad_items.append(q)

    score = clamp((good + 0.5 * weak) / max(1, len(items)))
    if score >= 0.9:
        comment = "Вопросы корректно оформлены."
    elif score >= 0.6:
        comment = "Часть вопросов оформлена без вопросительного знака."
    else:
        comment = "Существенная часть элементов не выглядит как вопросы."

    return HeuristicMetric(
        name=name,
        value={"count": len(items), "with_question_mark": good, "question_word_without_mark": weak,
               "bad_examples": bad_items[:3]},
        score=score,
        weight=weight,
        comment=comment,
    )


def score_no_generic_questions(
        questions: Any,
        *,
        generic_patterns: Iterable[str] = GENERIC_QUESTION_PATTERNS,
        name: str = "no_generic_questions",
        weight: float = 1.0,
) -> HeuristicMetric:
    """
    Штрафует слишком общие вопросы, которые подходят почти к любой теме.
    """
    items = split_questions(questions)
    if not items:
        return HeuristicMetric(name=name, value={"count": 0}, score=0.0, weight=weight, comment="Вопросы отсутствуют.")

    generic: list[str] = []
    patterns = list(generic_patterns)
    for q in items:
        norm = normalize_for_compare(q)
        if any(pattern in norm for pattern in patterns):
            generic.append(q)

    score = clamp(1.0 - len(generic) / max(1, len(items)))
    if not generic:
        comment = "Слишком общие вопросы не обнаружены."
    else:
        comment = "Есть вопросы с чрезмерно общей формулировкой."

    return HeuristicMetric(
        name=name,
        value={"generic_count": len(generic), "examples": generic[:3]},
        score=score,
        weight=weight,
        comment=comment,
    )


def score_questions_not_already_answered(
        questions: Any,
        context: Any,
        *,
        name: str = "not_already_answered",
        weight: float = 1.0,
) -> HeuristicMetric:
    """
    Проверяет, что вопросы не сводятся к уже явно указанным входным данным.

    Формально ищутся случаи, когда почти все значимые слова вопроса уже находятся
    во входном контексте. Такие вопросы могут быть лишними при согласовании темы.
    """
    items = split_questions(questions)
    context_tokens = tokenize(context)
    if not items:
        return HeuristicMetric(name=name, value={"count": 0}, score=0.0, weight=weight, comment="Вопросы отсутствуют.")
    if not context_tokens:
        return HeuristicMetric(name=name, value={"context_tokens": 0}, score=0.75, weight=weight,
                               comment="Контекст пустой, проверить уже указанные сведения сложно.")

    suspicious: list[str] = []
    for q in items:
        q_tokens = tokenize(q)
        if len(q_tokens) >= 3:
            overlap_ratio = len(q_tokens & context_tokens) / max(1, len(q_tokens))
            if overlap_ratio >= 0.85:
                suspicious.append(q)

    score = clamp(1.0 - len(suspicious) / max(1, len(items)))
    if suspicious:
        comment = "Часть вопросов может дублировать уже указанные входные данные."
    else:
        comment = "Не найдено вопросов, явно повторяющих входные данные."

    return HeuristicMetric(
        name=name,
        value={"suspicious_count": len(suspicious), "examples": suspicious[:3]},
        score=score,
        weight=weight,
        comment=comment,
    )


def score_questions_context_grounding(
        questions: Any,
        context: Any,
        *,
        name: str = "context_grounding",
        min_grounded_ratio: float = 0.35,
        weight: float = 1.0,
) -> HeuristicMetric:
    """
    Проверяет связь вопросов с темой, работой или загруженным контекстом.

    Метрика считает долю вопросов, в которых есть пересечение с ключевыми словами
    контекста. Вопросы без пересечения могут быть слишком общими или посторонними.
    """
    items = split_questions(questions)
    context_tokens = tokenize(context)
    if not items:
        return HeuristicMetric(name=name, value={"count": 0}, score=0.0, weight=weight, comment="Вопросы отсутствуют.")
    if not context_tokens:
        return HeuristicMetric(name=name, value={"context_tokens": 0}, score=0.5, weight=weight,
                               comment="Контекст пустой, связь вопросов с работой проверить сложно.")

    grounded = []
    ungrounded = []
    for q in items:
        q_tokens = tokenize(q)
        if q_tokens & context_tokens:
            grounded.append(q)
        else:
            ungrounded.append(q)

    ratio = len(grounded) / max(1, len(items))
    if ratio >= min_grounded_ratio:
        score = clamp(0.65 + 0.35 * ratio)
        comment = "Вопросы в достаточной степени связаны с контекстом."
    else:
        score = clamp(ratio / max(0.01, min_grounded_ratio) * 0.6)
        comment = "Связь вопросов с контекстом слабая."

    return HeuristicMetric(
        name=name,
        value={"grounded_count": len(grounded), "ungrounded_count": len(ungrounded),
               "ungrounded_examples": ungrounded[:3]},
        score=score,
        weight=weight,
        comment=comment,
    )


def score_understanding_not_retelling(
        questions: Any,
        *,
        name: str = "understanding_not_retelling",
        weight: float = 1.0,
) -> HeuristicMetric:
    """
    Проверяет, что вопросы защиты направлены на понимание, а не только пересказ.

    Признаки понимания: почему, как, чем обосновано, какие ограничения,
    как проверялось, что изменится при другом условии.
    """
    items = split_questions(questions)
    if not items:
        return HeuristicMetric(name=name, value={"count": 0}, score=0.0, weight=weight, comment="Вопросы отсутствуют.")

    understanding_markers = (
        "почему", "зачем", "как", "чем", "обосну", "огранич", "провер",
        "оцен", "сравн", "измен", "если", "какой вывод", "какие выводы",
    )
    matched: list[str] = []
    for q in items:
        norm = normalize_for_compare(q)
        if any(marker in norm for marker in understanding_markers):
            matched.append(q)

    ratio = len(matched) / max(1, len(items))
    score = clamp(0.25 + 0.75 * ratio)
    if ratio >= 0.6:
        comment = "Большая часть вопросов проверяет понимание и обоснование."
    elif ratio >= 0.35:
        comment = "Часть вопросов проверяет понимание, но есть риск пересказа."
    else:
        comment = "Вопросы слабо проверяют понимание и могут сводиться к пересказу."

    return HeuristicMetric(
        name=name,
        value={"understanding_questions": len(matched), "total_questions": len(items), "examples": matched[:3]},
        score=score,
        weight=weight,
        comment=comment,
    )


# Scenario evaluators


def evaluate_final_topic_generation(
        *,
        student_topic: Any,
        teacher_topic: Any,
        generated_topic: Any,
        llm_scores: Optional[dict[str, Any]] = None,
        k: float = DEFAULT_LLM_TRUST_COEFFICIENT,
        method_name: str = "build_agreed_spec",
) -> EvaluationResult:
    """
    Оценивает генерацию итоговой согласованной темы.

    Формальные признаки:
    - длина темы в допустимых пределах;
    - тема не пустая и не слишком общая;
    - есть пересечение с темой студента;
    - есть пересечение с темой преподавателя;
    - сохранена предметная область;
    - нет явного формального конфликта;
    - есть указание на метод, технологию или результат;
    - итог не копирует дословно только одну сторону.
    """
    student_text = extract_text_from_payload(student_topic)
    teacher_text = extract_text_from_payload(teacher_topic)
    generated_text = extract_text_from_payload(generated_topic)

    metrics = [
        score_text_length(
            generated_text,
            min_chars=30,
            max_chars=1200,
            ideal_min_chars=80,
            ideal_max_chars=650,
            name="topic_length",
            weight=0.8,
        ),
        score_not_empty_and_specific(
            generated_text,
            generic_patterns=GENERIC_TOPIC_PATTERNS,
            name="topic_specificity",
            weight=1.2,
        ),
        score_keyword_overlap(
            generated_text,
            student_text,
            name="student_keyword_overlap",
            label="темой студента",
            min_overlap=1,
            good_overlap=4,
            weight=1.1,
        ),
        score_keyword_overlap(
            generated_text,
            teacher_text,
            name="teacher_keyword_overlap",
            label="темой преподавателя",
            min_overlap=1,
            good_overlap=4,
            weight=1.1,
        ),
        score_domain_preservation(
            generated_text,
            student_text,
            teacher_text,
            name="domain_preservation",
            weight=1.0,
        ),
        score_no_obvious_conflict(
            generated_text,
            student_text,
            teacher_text,
            name="no_obvious_conflict",
            weight=0.8,
        ),
        score_required_terms_presence(
            generated_text,
            METHOD_RESULT_KEYWORDS,
            name="method_or_result_presence",
            min_terms=1,
            good_terms=3,
            weight=1.0,
        ),
        score_not_copying_one_side(
            generated_text,
            student_text,
            teacher_text,
            name="not_copying_one_side",
            weight=0.9,
        ),
    ]

    return build_evaluation_result(
        scenario_part=SCENARIO_TOPIC_FINAL,
        method_name=method_name,
        heuristic_metrics=metrics,
        llm_scores=llm_scores,
        k=k,
        comment="Оценка итоговой согласованной темы.",
        extra={
            "student_text_length": len(student_text),
            "teacher_text_length": len(teacher_text),
            "generated_text_length": len(generated_text),
        },
    )


def evaluate_clarification_questions_generation(
        *,
        student_topic: Any,
        teacher_topic: Any,
        generated_questions: Any,
        llm_scores: Optional[dict[str, Any]] = None,
        k: float = DEFAULT_LLM_TRUST_COEFFICIENT,
        method_name: str = "generate_clarification_questions",
        min_questions: int = 2,
        max_questions: int = 6,
) -> EvaluationResult:
    """
    Оценивает генерацию уточняющих вопросов для согласования темы.

    Формальные признаки:
    - количество вопросов в пределах 2-6;
    - вопросы не дублируются;
    - вопросы оформлены как вопросы;
    - вопросы относятся к данным, методам, ограничениям, формату результата;
    - вопросы не спрашивают о том, что уже явно указано во входных данных;
    - нет слишком общих вопросов типа «Уточните тему».
    """
    context = normalize_text("\n".join([
        extract_text_from_payload(student_topic),
        extract_text_from_payload(teacher_topic),
    ]))
    questions = split_questions(generated_questions)
    questions_text = "\n".join(questions)

    metrics = [
        score_question_count(
            questions,
            min_count=min_questions,
            max_count=max_questions,
            name="clarification_question_count",
            weight=1.0,
        ),
        score_question_duplicates(
            questions,
            name="clarification_question_duplicates",
            weight=1.0,
        ),
        score_question_format(
            questions,
            name="clarification_question_format",
            weight=0.8,
        ),
        score_required_keywords(
            questions_text,
            CLARIFICATION_COVERAGE_KEYWORDS,
            name="clarification_coverage",
            min_groups=2,
            weight=1.1,
        ),
        score_questions_not_already_answered(
            questions,
            context,
            name="clarification_not_already_answered",
            weight=0.9,
        ),
        score_no_generic_questions(
            questions,
            generic_patterns=GENERIC_QUESTION_PATTERNS,
            name="clarification_no_generic_questions",
            weight=1.0,
        ),
        score_questions_context_grounding(
            questions,
            context,
            name="clarification_context_grounding",
            min_grounded_ratio=0.25,
            weight=0.8,
        ),
    ]

    return build_evaluation_result(
        scenario_part=SCENARIO_TOPIC_QUESTIONS,
        method_name=method_name,
        heuristic_metrics=metrics,
        llm_scores=llm_scores,
        k=k,
        comment="Оценка уточняющих вопросов для согласования темы.",
        extra={"question_count": len(questions), "questions": questions},
    )


def evaluate_defense_questions_generation(
        *,
        context: Any,
        generated_questions: Any,
        llm_scores: Optional[dict[str, Any]] = None,
        k: float = DEFAULT_LLM_TRUST_COEFFICIENT,
        method_name: str = "build_question_pool",
        min_questions: int = 4,
        max_questions: int = 8,
) -> EvaluationResult:
    """
    Оценивает генерацию вопросов для защиты учебной работы.

    Формальные признаки:
    - количество вопросов в пределах 4-8;
    - вопросы не повторяются;
    - есть вопросы по теории, реализации, данным, качеству, ограничениям;
    - вопросы связаны с темой и загруженной работой;
    - вопросы не требуют полностью внешней информации;
    - вопросы проверяют понимание, а не только пересказ.
    """
    context_text = extract_text_from_payload(context)
    questions = split_questions(generated_questions)
    questions_text = "\n".join(questions)

    metrics = [
        score_question_count(
            questions,
            min_count=min_questions,
            max_count=max_questions,
            name="defense_question_count",
            weight=1.0,
        ),
        score_question_duplicates(
            questions,
            name="defense_question_duplicates",
            weight=1.0,
        ),
        score_question_format(
            questions,
            name="defense_question_format",
            weight=0.7,
        ),
        score_required_keywords(
            questions_text,
            DEFENSE_COVERAGE_KEYWORDS,
            name="defense_aspect_coverage",
            min_groups=3,
            weight=1.3,
        ),
        score_questions_context_grounding(
            questions,
            context_text,
            name="defense_context_grounding",
            min_grounded_ratio=0.35,
            weight=1.1,
        ),
        score_no_generic_questions(
            questions,
            generic_patterns=GENERIC_QUESTION_PATTERNS,
            name="defense_no_generic_questions",
            weight=0.8,
        ),
        score_understanding_not_retelling(
            questions,
            name="defense_understanding_not_retelling",
            weight=1.0,
        ),
    ]

    return build_evaluation_result(
        scenario_part=SCENARIO_DEFENSE_QUESTIONS,
        method_name=method_name,
        heuristic_metrics=metrics,
        llm_scores=llm_scores,
        k=k,
        comment="Оценка вопросов для защиты.",
        extra={"question_count": len(questions), "questions": questions},
    )


def evaluate_topic_alignment_round(
        *,
        student_topic: Any,
        teacher_topic: Any,
        generated_topic: Any = None,
        generated_questions: Any = None,
        llm_scores_topic: Optional[dict[str, Any]] = None,
        llm_scores_questions: Optional[dict[str, Any]] = None,
        k: float = DEFAULT_LLM_TRUST_COEFFICIENT,
        round_no: Optional[int] = None,
) -> EvaluationResult:
    """
    Оценивает один раунд согласования темы.

    Раунд может содержать:
    - итоговую или промежуточную формулировку темы;
    - уточняющие вопросы;
    - оба результата сразу.

    Если есть и тема, и вопросы, итог раунда считается как среднее с весами:
    0.55 для темы и 0.45 для вопросов. Если присутствует только один результат,
    оценка раунда равна оценке этого результата.
    """
    partial_results: list[tuple[str, float, EvaluationResult]] = []

    if generated_topic is not None and extract_text_from_payload(generated_topic):
        topic_result = evaluate_final_topic_generation(
            student_topic=student_topic,
            teacher_topic=teacher_topic,
            generated_topic=generated_topic,
            llm_scores=llm_scores_topic,
            k=k,
            method_name="build_agreed_spec",
        )
        partial_results.append(("topic", 0.55, topic_result))

    if generated_questions is not None and split_questions(generated_questions):
        questions_result = evaluate_clarification_questions_generation(
            student_topic=student_topic,
            teacher_topic=teacher_topic,
            generated_questions=generated_questions,
            llm_scores=llm_scores_questions,
            k=k,
            method_name="generate_clarification_questions",
        )
        partial_results.append(("questions", 0.45, questions_result))

    if not partial_results:
        empty_metric = HeuristicMetric(
            name="round_output_presence",
            value=False,
            score=0.0,
            weight=1.0,
            comment="В раунде нет ни итоговой темы, ни уточняющих вопросов.",
        )
        return build_evaluation_result(
            scenario_part=SCENARIO_TOPIC_ROUND,
            method_name="run_alignment_cycle",
            heuristic_metrics=[empty_metric],
            llm_scores=None,
            k=k,
            comment="Оценка раунда согласования темы.",
            extra={"round_no": round_no, "partial_results": []},
        )

    total_weight = sum(weight for _, weight, _ in partial_results)
    round_score = sum(weight * result.final_score for _, weight, result in partial_results) / max(0.0001, total_weight)
    heuristic_score = sum(weight * result.heuristic_score for _, weight, result in partial_results) / max(0.0001,
                                                                                                          total_weight)

    aggregate_metric = HeuristicMetric(
        name="round_aggregate_score",
        value={kind: result.final_score for kind, _, result in partial_results},
        score=round_score,
        weight=1.0,
        comment="Агрегированная оценка результатов одного раунда согласования.",
    )

    return EvaluationResult(
        scenario_part=SCENARIO_TOPIC_ROUND,
        method_name="run_alignment_cycle",
        llm_score=None,
        heuristic_metrics=[aggregate_metric],
        heuristic_score=heuristic_score,
        final_score=clamp(round_score),
        k=clamp(k),
        evaluation_mode="aggregated_round_score",
        comment="Оценка одного раунда согласования темы.",
        extra={
            "round_no": round_no,
            "partial_results": [
                {"kind": kind, "weight": weight, "result": result.to_dict()}
                for kind, weight, result in partial_results
            ],
        },
    )


def calculate_convergence_score(
        *,
        finalized: bool,
        finalized_round: Optional[int],
        max_rounds: int = 3,
) -> float:
    """
    Оценивает скорость достижения согласованного результата.

    Принятая логика:
    - тема зафиксирована за 1 раунд: 1.0;
    - тема зафиксирована за последний допустимый раунд: 0.5;
    - тема не зафиксирована: 0.0.

    Для промежуточных раундов используется линейное значение между 1.0 и 0.5.
    """
    if not finalized:
        return 0.0
    if finalized_round is None:
        finalized_round = max_rounds
    finalized_round = max(1, int(finalized_round))
    max_rounds = max(1, int(max_rounds))
    if max_rounds == 1:
        return 1.0
    if finalized_round <= 1:
        return 1.0
    if finalized_round >= max_rounds:
        return 0.5
    step = (finalized_round - 1) / max(1, max_rounds - 1)
    return clamp(1.0 - 0.5 * step)


def evaluate_topic_alignment_process(
        *,
        round_results: Optional[list[EvaluationResult]] = None,
        rounds: Optional[list[dict[str, Any]]] = None,
        student_topic: Any = None,
        teacher_topic: Any = None,
        finalized: bool = False,
        finalized_round: Optional[int] = None,
        max_rounds: int = 3,
        k: float = DEFAULT_LLM_TRUST_COEFFICIENT,
) -> ProcessEvaluationResult:
    """
    Оценивает весь процесс согласования темы.

    Можно передать уже готовые round_results или список rounds. В rounds каждый
    элемент может содержать поля:
    - generated_topic;
    - generated_questions;
    - llm_scores_topic;
    - llm_scores_questions;
    - round_no.

    Итоговая формула:
    scenario_score = 0.6 * last_round_score
                   + 0.3 * average_round_score
                   + 0.1 * convergence_score.
    """
    results = list(round_results or [])

    if not results and rounds:
        for index, item in enumerate(rounds, start=1):
            results.append(
                evaluate_topic_alignment_round(
                    student_topic=item.get("student_topic", student_topic),
                    teacher_topic=item.get("teacher_topic", teacher_topic),
                    generated_topic=item.get("generated_topic"),
                    generated_questions=item.get("generated_questions") or item.get("questions"),
                    llm_scores_topic=item.get("llm_scores_topic"),
                    llm_scores_questions=item.get("llm_scores_questions"),
                    k=k,
                    round_no=item.get("round_no", index),
                )
            )

    if not results:
        empty_round = evaluate_topic_alignment_round(
            student_topic=student_topic,
            teacher_topic=teacher_topic,
            generated_topic=None,
            generated_questions=None,
            k=k,
            round_no=None,
        )
        results = [empty_round]

    scores = [clamp(result.final_score) for result in results]
    last_round_score = scores[-1]
    average_round_score = mean(scores) if scores else 0.0

    if finalized and finalized_round is None:
        finalized_round = len(results)

    convergence_score = calculate_convergence_score(
        finalized=finalized,
        finalized_round=finalized_round,
        max_rounds=max_rounds,
    )

    scenario_score = clamp(
        0.6 * last_round_score
        + 0.3 * average_round_score
        + 0.1 * convergence_score
    )

    if finalized:
        comment = "Тема была согласована; итог учитывает последний раунд, среднее качество и скорость согласования."
    else:
        comment = "Тема не была согласована; convergence_score равен 0.0."

    return ProcessEvaluationResult(
        scenario_part=SCENARIO_TOPIC_PROCESS,
        method_name="run_alignment_process",
        round_results=results,
        last_round_score=last_round_score,
        average_round_score=average_round_score,
        convergence_score=convergence_score,
        scenario_score=scenario_score,
        finalized=bool(finalized),
        finalized_round=finalized_round,
        max_rounds=max_rounds,
        comment=comment,
    )


# Evaluation cases and report rows


def evaluate_case(
        case: dict[str, Any],
        *,
        k: float = DEFAULT_LLM_TRUST_COEFFICIENT,
) -> dict[str, Any]:
    """
    Универсальный запуск оценки одного заранее сохраненного кейса.

    Ожидаемый формат case:
    {
        "case_id": "...",
        "scenario_part": "topic_final | topic_clarification_questions | defense_questions | topic_alignment_process",
        "method_name": "...",
        "input_json": {...},
        "generated_output_json": {...},
        "llm_scores_json": {...}
    }
    """
    scenario_part = str(case.get("scenario_part") or "").strip()
    input_json = case.get("input_json") or case.get("input") or {}
    output_json = case.get("generated_output_json") or case.get("generated_output") or case.get("output") or {}
    llm_scores = case.get("llm_scores_json") or case.get("llm_scores")

    if scenario_part == SCENARIO_TOPIC_FINAL:
        result = evaluate_final_topic_generation(
            student_topic=input_json.get("student_topic") or input_json.get("student"),
            teacher_topic=input_json.get("teacher_topic") or input_json.get("teacher"),
            generated_topic=output_json.get("generated_topic") or output_json.get("agreed_spec") or output_json,
            llm_scores=llm_scores,
            k=k,
            method_name=str(case.get("method_name") or "build_agreed_spec"),
        )
        return result.to_dict()

    if scenario_part == SCENARIO_TOPIC_QUESTIONS:
        result = evaluate_clarification_questions_generation(
            student_topic=input_json.get("student_topic") or input_json.get("student"),
            teacher_topic=input_json.get("teacher_topic") or input_json.get("teacher"),
            generated_questions=output_json.get("generated_questions") or output_json,
            llm_scores=llm_scores,
            k=k,
            method_name=str(case.get("method_name") or "generate_clarification_questions"),
        )
        return result.to_dict()

    if scenario_part == SCENARIO_DEFENSE_QUESTIONS:
        result = evaluate_defense_questions_generation(
            context=input_json,
            generated_questions=output_json.get("generated_questions") or output_json.get("questions") or output_json,
            llm_scores=llm_scores,
            k=k,
            method_name=str(case.get("method_name") or "build_question_pool"),
        )
        return result.to_dict()

    if scenario_part == SCENARIO_TOPIC_PROCESS:
        process_result = evaluate_topic_alignment_process(
            rounds=input_json.get("rounds") or output_json.get("rounds") or [],
            student_topic=input_json.get("student_topic") or input_json.get("student"),
            teacher_topic=input_json.get("teacher_topic") or input_json.get("teacher"),
            finalized=bool(input_json.get("finalized", output_json.get("finalized", False))),
            finalized_round=input_json.get("finalized_round") or output_json.get("finalized_round"),
            max_rounds=int(input_json.get("max_rounds") or output_json.get("max_rounds") or 3),
            k=k,
        )
        return process_result.to_dict()

    raise ValueError(f"Неизвестная часть сценария оценки: {scenario_part!r}")


def build_report_row(
        *,
        case_id: str,
        case_title: str = "",
        result: dict[str, Any],
) -> dict[str, Any]:
    """
    Формирует плоскую строку для итоговой таблицы Streamlit/CSV.

    Подробные эвристики остаются в heuristic_metrics_json, чтобы пользователь
    мог раскрыть детализацию без сложной визуализации.
    """
    llm = result.get("llm_score") or {}
    raw_scores = llm.get("raw_scores") or {}

    return {
        "case_id": case_id,
        "case_title": case_title,
        "scenario_part": result.get("scenario_part", ""),
        "method_name": result.get("method_name", ""),
        "llm_relevance": raw_scores.get("relevance"),
        "llm_completeness": raw_scores.get("completeness"),
        "llm_clarity": raw_scores.get("clarity"),
        "llm_usefulness": raw_scores.get("usefulness"),
        "llm_correctness": raw_scores.get("correctness"),
        "llm_score": llm.get("normalized_score"),
        "heuristic_score": result.get("heuristic_score"),
        "final_score": result.get("final_score") or result.get("scenario_score"),
        "evaluation_mode": result.get("evaluation_mode", "process"),
        "comment": result.get("comment", ""),
        "heuristic_metrics_json": safe_json_dumps(result.get("heuristic_metrics", [])),
        "details_json": safe_json_dumps(result),
    }


def build_report_rows(cases: list[dict[str, Any]], *, k: float = DEFAULT_LLM_TRUST_COEFFICIENT) -> list[dict[str, Any]]:
    """
    Прогоняет список сохраненных кейсов и возвращает строки итоговой таблицы.
    """
    rows: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        case_id = str(case.get("case_id") or case.get("id") or f"case_{index}")
        case_title = str(case.get("title") or case.get("case_title") or "")
        result = evaluate_case(case, k=k)
        rows.append(build_report_row(case_id=case_id, case_title=case_title, result=result))
    return rows
