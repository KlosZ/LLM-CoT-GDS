"""
Реализация узких (в смысле ответственности) агентов для системы 
автоматизации защиты лабораторных/практических работ.

- IngestAgent: загрузка материалов преподавателя/работы студента + извлечение текста
- TopicAlignmentAgent — работа с темами и их согласованием;
- MethodicsAgent: генерация методички/рубрики/банка вопросов + калибровка топ-N
- DefenseAgent: анализ работы студента, подбор следующего вопроса, оценка ответа
- FeedbackAgent: обработка обратной связи преподавателя -> обновление policy memory

"Дообучение" реализуется как policy memory: хорошие/плохие 
примеры и заметки преподавателя, которые подмешиваются в промпт.
"""

from __future__ import annotations

import hashlib
import json
import mimetypes
import re
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

try:
    from parsers import (
        build_context_bundle,
        parse_uploaded_files,
    )
except Exception:
    def build_context_bundle(documents, **kwargs):  # type: ignore
        return ""


    def parse_uploaded_files(uploaded_files, **kwargs):  # type: ignore
        return []

try:
    from storage import (
        Storage,
        MATERIAL_ROLE_STUDENT,
        MATERIAL_ROLE_SYSTEM,
        MATERIAL_ROLE_TEACHER,
        MATERIAL_STAGE_GENERAL,
        MATERIAL_STAGE_METHODICS,
        MATERIAL_STAGE_SUBMISSION,
        MATERIAL_STAGE_TOPIC_ALIGNMENT,
        SIDE_STUDENT,
        SIDE_TEACHER,
        SIDE_SYSTEM,
        STATUS_ALIGNED,
        STATUS_DRAFT,
        STATUS_FINALIZED,
        STATUS_NEEDS_CLARIFICATION,
        STATUS_REJECTED,
        WORK_TYPE_OTHER,
    )
except Exception:
    Storage = Any  # type: ignore
    MATERIAL_ROLE_STUDENT = "student"
    MATERIAL_ROLE_SYSTEM = "system"
    MATERIAL_ROLE_TEACHER = "teacher"
    MATERIAL_STAGE_GENERAL = "general"
    MATERIAL_STAGE_METHODICS = "methodics"
    MATERIAL_STAGE_SUBMISSION = "submission"
    MATERIAL_STAGE_TOPIC_ALIGNMENT = "topic_alignment"
    SIDE_STUDENT = "student"
    SIDE_TEACHER = "teacher"
    SIDE_SYSTEM = "system"
    STATUS_ALIGNED = "aligned"
    STATUS_DRAFT = "draft"
    STATUS_FINALIZED = "finalized"
    STATUS_NEEDS_CLARIFICATION = "needs_clarification"
    STATUS_REJECTED = "rejected"
    WORK_TYPE_OTHER = "other"

DEFAULT_UPLOAD_DIR = "uploads"
DEFAULT_METHODICS_FILENAME = "generated_methodics.md"


def _normalize_ws(text: str) -> str:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _safe_filename(name: str) -> str:
    name = Path(name or "uploaded.bin").name
    name = re.sub(r"[^A-Za-zА-Яа-я0-9._ -]+", "_", name)
    return name[:180] or "uploaded.bin"


def _truncate(text: str, limit: int = 1200) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n...[обрезано]"


def _json_dumps(data: Any) -> str:
    try:
        return json.dumps(data, ensure_ascii=False, indent=2)
    except Exception:
        return str(data)


def _json_loads_maybe(text: Any, default: Any = None) -> Any:
    if isinstance(text, dict):
        return text
    if not isinstance(text, str):
        return {} if default is None else default
    try:
        return json.loads(text)
    except Exception:
        return {} if default is None else default


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _tokenize_ru(text: str) -> set[str]:
    words = re.findall(r"[A-Za-zА-Яа-я0-9]{3,}", (text or "").lower())
    stop = {
        "это", "для", "при", "как", "что", "или", "если", "над", "под", "про",
        "the", "and", "for", "with", "from", "that", "this", "work", "topic",
        "тема", "работа", "задание", "дисциплина", "студент", "преподаватель",
        "нужно", "можно", "должен", "должны", "будет", "быть", "рамках", "рамки",
    }
    return {w for w in words if w not in stop}


def _jaccard_set(sa: set[str], sb: set[str]) -> float:
    if not sa and not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb) or 1
    return inter / union


def _count_alignment_answers(turns: list[dict[str, Any]], side: str) -> int:
    return sum(
        1 for turn in turns
        if turn.get("side") == side
        and turn.get("turn_kind") == "answer"
        and (turn.get("answer_text") or "").strip()
    )


def _material_bytes_and_name(uploaded_file: Any) -> tuple[bytes, str, str]:
    if isinstance(uploaded_file, (bytes, bytearray)):
        return bytes(uploaded_file), "uploaded.bin", ""

    if isinstance(uploaded_file, (str, Path)) and Path(uploaded_file).exists():
        path = Path(uploaded_file)
        return path.read_bytes(), path.name, mimetypes.guess_type(str(path))[0] or ""

    if hasattr(uploaded_file, "getvalue"):
        data = uploaded_file.getvalue()
        filename = getattr(uploaded_file, "name", None) or "uploaded.bin"
        mime = getattr(uploaded_file, "type", None) or mimetypes.guess_type(filename)[0] or ""
        return data, filename, mime

    if hasattr(uploaded_file, "read"):
        data = uploaded_file.read()
        if isinstance(data, str):
            data = data.encode("utf-8", errors="ignore")
        filename = getattr(uploaded_file, "name", None) or "uploaded.bin"
        mime = mimetypes.guess_type(str(filename))[0] or ""
        return bytes(data), Path(str(filename)).name, mime

    raise TypeError(f"Неподдерживаемый тип файла: {type(uploaded_file).__name__}")


def _call_llm_json(
        llm_client: Any,
        *,
        system_prompt: str,
        user_prompt: str,
        fallback: dict[str, Any],
        schema_name: Optional[str] = None,
) -> dict[str, Any]:
    if llm_client is None:
        return fallback

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    attempts: list[Callable[[], Any]] = []

    if hasattr(llm_client, "generate_json"):
        attempts.append(lambda: llm_client.generate_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema_name=schema_name,
        ))
        attempts.append(lambda: llm_client.generate_json(
            messages=messages,
            schema_name=schema_name,
        ))

    if hasattr(llm_client, "complete_json"):
        attempts.append(lambda: llm_client.complete_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema_name=schema_name,
        ))

    if hasattr(llm_client, "invoke_json"):
        attempts.append(lambda: llm_client.invoke_json(
            messages=messages,
            schema_name=schema_name,
        ))

    if hasattr(llm_client, "chat_json"):
        attempts.append(lambda: llm_client.chat_json(
            messages=messages,
            schema_name=schema_name,
        ))

    if callable(llm_client):
        attempts.append(lambda: llm_client(
            messages=messages,
            schema_name=schema_name,
        ))

    for attempt in attempts:
        try:
            result = attempt()
            if isinstance(result, dict):
                return result
            if isinstance(result, str):
                parsed = _json_loads_maybe(result, default=None)
                if isinstance(parsed, dict):
                    return parsed
        except Exception:
            continue

    return fallback


def _collect_side_alignment_text(
        side_input: dict[str, Any],
        materials: list[dict[str, Any]],
        turns: list[dict[str, Any]],
        side: str,
) -> str:
    parts = [
        side_input.get("title", ""),
        side_input.get("description", ""),
        _json_dumps(side_input.get("context_json", {})),
    ]

    for material in materials:
        parts.append(material.get("title", ""))
        parts.append(material.get("filename", ""))
        parts.append(material.get("extracted_text", ""))

    for turn in turns:
        if turn.get("side") == side:
            parts.append(turn.get("question_text", ""))
            parts.append(turn.get("answer_text", ""))

    return _normalize_ws("\n".join(part for part in parts if part))


def _heuristic_relation(
        student_payload: dict[str, Any],
        teacher_payload: dict[str, Any],
        student_materials: list[dict[str, Any]],
        teacher_materials: list[dict[str, Any]],
        turns: list[dict[str, Any]],
) -> dict[str, Any]:
    s_text = _collect_side_alignment_text(student_payload, student_materials, turns, SIDE_STUDENT)
    t_text = _collect_side_alignment_text(teacher_payload, teacher_materials, turns, SIDE_TEACHER)

    s_tokens = _tokenize_ru(s_text)
    t_tokens = _tokenize_ru(t_text)

    raw_score = _jaccard_set(s_tokens, t_tokens)
    overlap = sorted(list(s_tokens & t_tokens))[:20]

    keyword_bonus = min(0.25, len(overlap) * 0.025)
    score = round(min(1.0, raw_score + keyword_bonus), 3)

    if score >= 0.35:
        label = "direct"
    elif score >= 0.18:
        label = "partial"
    elif score >= 0.08:
        label = "weak"
    else:
        label = "none"

    student_answers_count = _count_alignment_answers(turns, SIDE_STUDENT)
    teacher_answers_count = _count_alignment_answers(turns, SIDE_TEACHER)

    conflicts = []
    if not overlap:
        conflicts.append("Почти не найдено общих тематических опорных слов.")
    if student_answers_count == 0:
        conflicts.append("Студент пока не дал содержательных уточнений.")
    if teacher_answers_count == 0:
        conflicts.append("Преподаватель пока не дал содержательных уточнений.")

    # Ключевое правило:
    # пока обе стороны не дали хотя бы по одному ответу,
    # первый раунд уточнения обязателен.
    if student_answers_count == 0 or teacher_answers_count == 0:
        needs_clarification = True
        if overlap:
            short_reason = (
                "Темы уже имеют пересечение по ключевым словам и материалам, "
                "но система принудительно запускает хотя бы один раунд уточнения, "
                "чтобы зафиксировать ожидания обеих сторон."
            )
        else:
            short_reason = (
                "Первичная оценка показывает слабое пересечение, поэтому нужен раунд уточнения."
            )
    else:
        needs_clarification = label in {"weak", "none"}
        short_reason = (
            "Эвристическая оценка выполнена по темам, материалам и уже данным ответам обеих сторон."
        )

    return {
        "assessment_mode": "heuristic",
        "relation_score": score,
        "relation_label": label,
        "needs_clarification": needs_clarification,
        "overlap_points": overlap or ["Нужно сильнее сузить формулировки и явно зафиксировать общий результат."],
        "conflicts": conflicts,
        "short_reason": short_reason,
        "student_answers_count": student_answers_count,
        "teacher_answers_count": teacher_answers_count,
    }


def _heuristic_clarification_questions(student_payload: dict[str, Any], teacher_payload: dict[str, Any]) -> dict[
    str, Any]:
    return {
        "student_questions": [
            "Какой практический результат вы хотите получить по своей теме: отчет, прототип, исследование, модель, систему или методику?",
            "Какие данные, материалы или наработки у вас уже есть, и что из этого реально использовать в рамках задания?",
            "Какие границы темы для вас принципиальны, а чем вы готовы пожертвовать ради соответствия дисциплине?",
        ],
        "teacher_questions": [
            "Какие именно результаты по дисциплине обязательно должны быть проверяемыми и оценимыми?",
            "Какие формы результата допустимы: исследование, практическая работа, прототип, отчет, доклад, курсовой проект?",
            "Какие критерии приемки являются обязательными, а какие можно адаптировать под интерес студента?",
        ],
        "rationale": "Сформированы базовые вопросы для поиска пересечения между интересом студента и рамками дисциплины.",
    }


def _heuristic_agreed_spec(
        lab: dict[str, Any],
        session: dict[str, Any],
        student_payload: dict[str, Any],
        teacher_payload: dict[str, Any],
) -> dict[str, Any]:
    student_title = student_payload.get("title", "").strip()
    teacher_title = teacher_payload.get("title", "").strip()

    agreed_title = f"{student_title or 'Тема студента'} в рамках {teacher_title or 'дисциплины'}"
    agreed_description = (
        "Итоговое задание формируется на пересечении темы студента и темы преподавателя. "
        "Студент сохраняет предметный интерес, а преподаватель получает проверяемый результат, "
        "соответствующий дисциплине."
    )
    return {
        "work_type": lab.get("work_type") or WORK_TYPE_OTHER,
        "agreed_title": agreed_title,
        "agreed_description": agreed_description,
        "acceptance_criteria": {
            "must_have": [
                "Четко определенный объект и предмет работы.",
                "Описанный метод или способ решения задачи.",
                "Проверяемый результат, который можно оценить на защите.",
            ],
            "deliverables": [
                "Текст работы",
                "Материалы по теме",
                "Подготовка к защите",
            ],
            "evaluation_axes": [
                "соответствие дисциплине",
                "логика и обоснованность",
                "содержательность результата",
            ],
        },
        "compatibility_note": "Спецификация собрана эвристически.",
    }


def _heuristic_methodics(spec: dict[str, Any], context: str) -> dict[str, Any]:
    title = spec.get("agreed_title", "Методические рекомендации")
    desc = spec.get("agreed_description", "")
    criteria = spec.get("acceptance_criteria_json", {}) or spec.get("acceptance_criteria", {})
    must_have = "\n".join(
        [f"- {item}" for item in criteria.get("must_have", [])]) or "- Определить цель и ожидаемый результат."
    deliverables = "\n".join(
        [f"- {item}" for item in criteria.get("deliverables", [])]) or "- Подготовить результат работы."
    eval_axes = "\n".join([f"- {item}" for item in criteria.get("evaluation_axes", [])]) or "- соответствие теме"

    body = f"""# Методические рекомендации

## 1. Тема задания
{title}

## 2. Смысл работы
{desc}

## 3. Что необходимо сделать
{must_have}

## 4. Что должно быть представлено на проверку
{deliverables}

## 5. На что будет смотреть преподаватель
{eval_axes}

## 6. Как строить работу
1. Уточнить объект, предмет и границы задания.
2. Собрать и систематизировать материалы по теме.
3. Выполнить основную часть работы в выбранном формате.
4. Подготовить доказательства полученного результата.
5. Сформулировать выводы и подготовиться к защите.

## 7. Контекст, использованный при сборке методики
{_truncate(context, 3000)}
"""
    return {
        "title": f"Методика: {title}",
        "body_text": body.strip(),
        "checklist": [
            "Тема и цель не противоречат дисциплине.",
            "Есть проверяемый результат.",
            "Собраны материалы и аргументы для защиты.",
        ],
    }


def _heuristic_submission_analysis(spec: dict[str, Any], submission: dict[str, Any], context: str) -> dict[str, Any]:
    text = " ".join([
        submission.get("title", ""),
        submission.get("description", ""),
        _json_dumps(submission.get("analysis_json", {})),
        context,
    ]).lower()
    strengths = []
    risks = []
    if any(word in text for word in ["модель", "прототип", "система", "алгоритм", "анализ"]):
        strengths.append("В материалах просматривается содержательное ядро работы.")
    else:
        risks.append("По загруженным материалам пока неочевиден основной результат.")
    if any(word in text for word in ["эксперимент", "сравнение", "оценка", "тест"]):
        strengths.append("Есть признаки проверки результата или попытки его обосновать.")
    else:
        risks.append("Не хватает явных признаков проверки или оценки результата.")
    if not strengths:
        strengths.append("Есть базовый набор материалов для начала защиты.")

    return {
        "summary": "Предварительный анализ выполнен эвристически по тексту загруженных материалов.",
        "strengths": strengths,
        "risks": risks,
        "recommended_focus": [
            "четко сформулировать цель и итог работы",
            "показать, чем результат соответствует дисциплине",
            "подготовить ответы по методике и ограничениям",
        ],
    }


def _heuristic_question_pool(spec: dict[str, Any], analysis: dict[str, Any], pool_size: int = 8) -> dict[str, Any]:
    title = spec.get("agreed_title", "заданию")
    base = [
        f"В чем состоит основная цель вашей работы по теме «{title}»?",
        "Почему выбранная тема соответствует дисциплине и в чем здесь учебная ценность?",
        "Как вы определили границы задания и что сознательно оставили за пределами работы?",
        "Какие материалы, источники или данные вы использовали и почему именно их?",
        "Какой результат можно считать главным итогом вашей работы?",
        "Какие ограничения у полученного результата вы можете назвать сами?",
        "Чем ваш подход отличается от более простого или очевидного способа решения задачи?",
        "Как бы вы развивали работу дальше, если бы было больше времени?",
        "Какие критерии приемки из согласованного задания выполнены полностью, а какие частично?",
        "Что преподаватель может проверить в вашей работе объективно?",
    ]
    strengths = analysis.get("strengths", [])
    if strengths:
        base.append(f"Какие сильные стороны своей работы вы считаете главными: {', '.join(strengths[:2])}?")
    return {
        "questions": base[:pool_size],
        "strategy_note": "Пул сформирован эвристически и покрывает цель, метод, результат, ограничения и соответствие дисциплине.",
    }


def _heuristic_answer_evaluation(question: str, answer: str) -> dict[str, Any]:
    answer = (answer or "").strip()
    if not answer:
        return {
            "score": 0.0,
            "verdict": "no_answer",
            "strengths": [],
            "weaknesses": ["Ответ отсутствует."],
            "follow_up": "Ответьте предметно по сути вопроса и свяжите ответ с вашей работой.",
        }

    score = min(1.0, 0.2 + len(answer) / 800.0)
    verdict = "good" if score >= 0.7 else "partial" if score >= 0.45 else "weak"
    strengths = []
    weaknesses = []

    if len(answer) > 180:
        strengths.append("Ответ содержит развернутое пояснение.")
    else:
        weaknesses.append("Ответ получился слишком кратким.")

    if any(word in answer.lower() for word in ["цель", "метод", "результат", "критер", "данн", "анализ"]):
        strengths.append("Есть признаки предметного ответа.")
    else:
        weaknesses.append("Не хватает опоры на цель, метод или результат работы.")

    return {
        "score": round(score, 3),
        "verdict": verdict,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "follow_up": "Уточните ответ через цель, используемый подход и проверяемый результат.",
    }


def _heuristic_student_feedback(spec: dict[str, Any], qa_turns: list[dict[str, Any]], session: dict[str, Any]) -> dict[
    str, Any]:
    avg = 0.0
    if qa_turns:
        scores = [float((turn.get("evaluation_json") or {}).get("score", 0.0)) for turn in qa_turns]
        avg = sum(scores) / len(scores)
    rating = "хороший" if avg >= 0.7 else "удовлетворительный" if avg >= 0.45 else "слабый"
    return {
        "feedback_text": (
            f"Защита по теме «{spec.get('agreed_title', 'заданию')}» завершена. "
            f"Общий уровень ответов можно оценить как {rating}. "
            "Сильнее всего воспринимаются ответы, где вы прямо связываете цель, метод и результат. "
            "Слабее выглядят ответы, где не хватает конкретики или критериев проверки."
        ),
        "highlights": [
            "Связывайте ответы с согласованной темой задания.",
            "Показывайте, что именно было сделано и как это можно проверить.",
            "Не избегайте разговора об ограничениях работы.",
        ],
    }


def _heuristic_policy_items(feedback_texts: list[str]) -> dict[str, Any]:
    merged = " ".join(feedback_texts).lower()
    items = []
    if "критер" in merged:
        items.append({
            "kind": "evaluation",
            "title": "Явные критерии приемки",
            "body_text": "Перед защитой фиксируйте проверяемые критерии приемки и опирайтесь на них при оценивании.",
        })
    if "конкрет" in merged or "кратк" in merged:
        items.append({
            "kind": "defense",
            "title": "Требование конкретных ответов",
            "body_text": "Подталкивайте студента к предметным и доказательным ответам вместо общих формулировок.",
        })
    if "дисцип" in merged or "соответств" in merged:
        items.append({
            "kind": "alignment",
            "title": "Проверка соответствия дисциплине",
            "body_text": "На защите отдельно проверяйте, как работа связана с рамками дисциплины.",
        })

    if not items:
        items = [
            {
                "kind": "general",
                "title": "Фиксация ожиданий",
                "body_text": "Перед защитой полезно явно фиксировать ожидания к результату и качеству аргументации.",
            },
            {
                "kind": "general",
                "title": "Опора на доказательства",
                "body_text": "Оценивание должно опираться не только на итог, но и на объяснение способа его получения.",
            },
        ]
    return {"items": items}


class BaseAgent:
    def __init__(self, storage: Storage, llm_client: Any = None) -> None:
        self.storage = storage
        self.llm_client = llm_client

    def _ensure_lab(self, lab_id: str) -> dict[str, Any]:
        lab = self.storage.get_lab(lab_id)
        if not lab:
            raise ValueError(f"Задание не найдено: {lab_id}")
        return lab

    def _ensure_topic_session(self, topic_session_id: str) -> dict[str, Any]:
        session = self.storage.get_topic_session(topic_session_id)
        if not session:
            raise ValueError(f"Сессия согласования не найдена: {topic_session_id}")
        return session

    def _ensure_submission(self, submission_id: str) -> dict[str, Any]:
        submission = self.storage.get_submission(submission_id)
        if not submission:
            raise ValueError(f"Загрузка работы не найдена: {submission_id}")
        return submission

    def _ensure_defense_session(self, defense_session_id: str) -> dict[str, Any]:
        session = self.storage.get_defense_session(defense_session_id)
        if not session:
            raise ValueError(f"Сессия защиты не найдена: {defense_session_id}")
        return session


class IngestAgent(BaseAgent):
    def __init__(
            self,
            storage: Storage,
            llm_client: Any = None,
            *,
            upload_dir: str | Path = DEFAULT_UPLOAD_DIR,
    ) -> None:
        super().__init__(storage=storage, llm_client=llm_client)
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    def save_uploaded_file(
            self,
            *,
            lab_id: str,
            uploaded_file: Any,
            owner_role: str,
            stage: str,
    ) -> dict[str, Any]:
        data, original_name, mime_type = _material_bytes_and_name(uploaded_file)
        safe_name = _safe_filename(original_name)
        digest = hashlib.sha256(data).hexdigest()[:12]
        target_dir = self.upload_dir / lab_id / owner_role / stage
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / f"{digest}_{safe_name}"
        target_path.write_bytes(data)

        return {
            "filename": safe_name,
            "mime_type": mime_type,
            "file_path": str(target_path),
            "size_bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }

    def ingest_materials(
            self,
            *,
            lab_id: str,
            uploaded_files: Iterable[Any],
            owner_role: str,
            stage: str = MATERIAL_STAGE_GENERAL,
            title_prefix: str = "",
            extra_meta: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        self._ensure_lab(lab_id)
        uploaded_files = list(uploaded_files or [])
        if not uploaded_files:
            return []

        saved_records = []
        for file_obj in uploaded_files:
            saved = self.save_uploaded_file(
                lab_id=lab_id,
                uploaded_file=file_obj,
                owner_role=owner_role,
                stage=stage,
            )
            saved_records.append({"saved": saved, "file_obj": file_obj})

        parsed_docs = parse_uploaded_files(
            [item["file_obj"] for item in saved_records],
            owner=owner_role,
            stage=stage,
        )

        created: list[dict[str, Any]] = []
        saved_by_name = {item["saved"]["filename"]: item["saved"] for item in saved_records}

        for doc in parsed_docs:
            base_saved = saved_by_name.get(doc.filename)
            meta = {
                "parser_name": getattr(doc, "parser_name", ""),
                "archive_path": getattr(doc, "archive_path", None),
                "warnings": getattr(doc, "warnings", []),
                "is_supported": getattr(doc, "is_supported", True),
                "source_name": getattr(doc, "source_name", None),
            }
            if extra_meta:
                meta.update(extra_meta)
            if base_saved:
                meta.setdefault("sha256", base_saved.get("sha256"))
                meta.setdefault("size_bytes", base_saved.get("size_bytes"))

            created.append(
                self.storage.add_material(
                    lab_id=lab_id,
                    filename=getattr(doc, "display_name", None) or getattr(doc, "filename", "uploaded.bin"),
                    owner_role=owner_role,
                    stage=stage,
                    title=f"{title_prefix}{getattr(doc, 'filename', 'Материал')}".strip(),
                    mime_type=getattr(doc, "mime_type", "") or (base_saved.get("mime_type", "") if base_saved else ""),
                    file_path=base_saved.get("file_path", "") if base_saved else "",
                    extracted_text=getattr(doc, "text", ""),
                    meta=meta,
                )
            )

        if not created:
            for item in saved_records:
                saved = item["saved"]
                meta = dict(extra_meta or {})
                meta.update({"warnings": ["Файл сохранен, но текст не извлечен."]})
                created.append(
                    self.storage.add_material(
                        lab_id=lab_id,
                        filename=saved["filename"],
                        owner_role=owner_role,
                        stage=stage,
                        title=f"{title_prefix}{saved['filename']}".strip(),
                        mime_type=saved["mime_type"],
                        file_path=saved["file_path"],
                        extracted_text="",
                        meta=meta,
                    )
                )

        return created


class TopicAlignmentAgent(BaseAgent):
    def __init__(
            self,
            storage: Storage,
            llm_client: Any = None,
            *,
            relation_threshold: float = 0.55,
            max_rounds: int = 3,
    ) -> None:
        super().__init__(storage=storage, llm_client=llm_client)
        self.relation_threshold = relation_threshold
        self.max_rounds = max_rounds

    def get_or_create_session(self, lab_id: str) -> dict[str, Any]:
        self._ensure_lab(lab_id)
        session = self.storage.get_latest_topic_session_for_lab(lab_id)
        if session:
            return session
        return self.storage.create_topic_session(lab_id=lab_id, status=STATUS_DRAFT)

    def set_student_topic(
            self,
            *,
            topic_session_id: str,
            title: str,
            description: str = "",
            context: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        self._ensure_topic_session(topic_session_id)
        return self.storage.upsert_topic_input(
            topic_session_id=topic_session_id,
            side=SIDE_STUDENT,
            title=title,
            description=description,
            context=context,
        )

    def set_teacher_topic(
            self,
            *,
            topic_session_id: str,
            title: str,
            description: str = "",
            context: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        self._ensure_topic_session(topic_session_id)
        return self.storage.upsert_topic_input(
            topic_session_id=topic_session_id,
            side=SIDE_TEACHER,
            title=title,
            description=description,
            context=context,
        )

    def build_alignment_context(self, topic_session_id: str) -> dict[str, Any]:
        session = self._ensure_topic_session(topic_session_id)
        lab = self._ensure_lab(session["lab_id"])
        student_input = self.storage.get_topic_input(topic_session_id, SIDE_STUDENT) or {
            "title": "",
            "description": "",
            "context_json": {},
        }
        teacher_input = self.storage.get_topic_input(topic_session_id, SIDE_TEACHER) or {
            "title": "",
            "description": "",
            "context_json": {},
        }

        materials = self.storage.list_materials(session["lab_id"], stage=MATERIAL_STAGE_TOPIC_ALIGNMENT)
        student_materials = [m for m in materials if m["owner_role"] == MATERIAL_ROLE_STUDENT]
        teacher_materials = [m for m in materials if m["owner_role"] == MATERIAL_ROLE_TEACHER]

        student_context = build_context_bundle(
            [_material_to_doc_stub(m) for m in student_materials],
            max_chars_per_doc=6000,
            max_total_chars=24000,
        )
        teacher_context = build_context_bundle(
            [_material_to_doc_stub(m) for m in teacher_materials],
            max_chars_per_doc=6000,
            max_total_chars=24000,
        )

        turns = self.storage.list_topic_turns(topic_session_id)

        return {
            "lab": lab,
            "topic_session": session,
            "student_input": student_input,
            "teacher_input": teacher_input,
            "student_materials": student_materials,
            "teacher_materials": teacher_materials,
            "student_context_bundle": student_context,
            "teacher_context_bundle": teacher_context,
            "turns": turns,
        }

    def assess_relation(self, topic_session_id: str) -> dict[str, Any]:
        ctx = self.build_alignment_context(topic_session_id)
        student_input = ctx["student_input"]
        teacher_input = ctx["teacher_input"]
        session = ctx["topic_session"]

        fallback = _heuristic_relation(
            student_input,
            teacher_input,
            ctx["student_materials"],
            ctx["teacher_materials"],
            ctx["turns"],
        )

        system_prompt = (
            "Ты анализируешь две независимые темы: студенческую и преподавательскую. "
            "Нужно оценить, насколько их можно связать в одно учебное задание. "
            "Верни JSON с полями: relation_score, relation_label, needs_clarification, overlap_points, conflicts, short_reason."
        )
        user_prompt = f"""
Тема студента:
Название: {student_input.get("title", "")}
Описание: {student_input.get("description", "")}
Контекст: {_json_dumps(student_input.get("context_json", {}))}

Материалы студента:
{ctx["student_context_bundle"] or "[нет материалов]"}

Тема преподавателя:
Название: {teacher_input.get("title", "")}
Описание: {teacher_input.get("description", "")}
Контекст: {_json_dumps(teacher_input.get("context_json", {}))}

Материалы преподавателя:
{ctx["teacher_context_bundle"] or "[нет материалов]"}

История уточнений:
{_json_dumps(ctx["turns"])}
"""

        result = _call_llm_json(
            self.llm_client,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            fallback=fallback,
            schema_name="topic_relation_assessment",
        )

        student_answers_count = _count_alignment_answers(ctx["turns"], SIDE_STUDENT)
        teacher_answers_count = _count_alignment_answers(ctx["turns"], SIDE_TEACHER)

        score = float(result.get("relation_score", fallback["relation_score"]))
        label = str(result.get("relation_label", fallback["relation_label"]))
        needs_clarification = bool(result.get("needs_clarification", label in {"weak", "none"}))
        assessment_mode = str(result.get("assessment_mode", "llm"))

        # Главная страховка:
        # пока нет хотя бы по одному ответу от обеих сторон, не фиксируем тему автоматически.
        if student_answers_count == 0 or teacher_answers_count == 0:
            needs_clarification = True
            result["needs_clarification"] = True
            result["forced_first_round"] = True
            result["student_answers_count"] = student_answers_count
            result["teacher_answers_count"] = teacher_answers_count
            if label in {"partial", "direct"}:
                result["short_reason"] = (
                    "Темы уже близки, но система принудительно запускает один раунд уточнения, "
                    "чтобы зафиксировать ожидания студента и преподавателя."
                )

        effective_threshold = self.relation_threshold
        if assessment_mode == "heuristic":
            effective_threshold = min(self.relation_threshold, 0.18)

        if score >= effective_threshold and label in {"direct", "partial"} and not needs_clarification:
            status = STATUS_ALIGNED
        else:
            status = STATUS_NEEDS_CLARIFICATION

        updated = self.storage.update_topic_session(
            topic_session_id,
            status=status,
            round_no=int(session.get("round_no", 0)),
            relation_score=score,
            relation_label=label,
            summary_text=str(result.get("short_reason", "")),
            llm_assessment=result,
        )
        return {
            "topic_session": updated,
            "assessment": result,
        }

    def generate_clarification_questions(self, topic_session_id: str) -> dict[str, Any]:
        ctx = self.build_alignment_context(topic_session_id)
        student_input = ctx["student_input"]
        teacher_input = ctx["teacher_input"]
        assessment = (ctx["topic_session"].get("llm_assessment_json") or {}) if ctx["topic_session"] else {}

        fallback = _heuristic_clarification_questions(student_input, teacher_input)

        system_prompt = (
            "Ты помогаешь согласовать тему задания между студентом и преподавателем. "
            "Сформируй отдельные уточняющие вопросы студенту и преподавателю. "
            "Вопросы должны быть краткими, предметными и направленными на поиск пересечения. "
            "Верни JSON с полями: student_questions, teacher_questions, rationale."
        )
        user_prompt = f"""
Оценка связи:
{_json_dumps(assessment)}

Тема студента:
{_json_dumps(student_input)}

Тема преподавателя:
{_json_dumps(teacher_input)}
"""

        result = _call_llm_json(
            self.llm_client,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            fallback=fallback,
            schema_name="clarification_questions",
        )

        existing_questions = {
            (turn.get("side"), (turn.get("question_text") or "").strip())
            for turn in ctx["turns"]
            if turn.get("turn_kind") == "question" and (turn.get("question_text") or "").strip()
        }

        new_count = 0

        for question in _as_list(result.get("student_questions")):
            q = str(question).strip()
            if not q or (SIDE_STUDENT, q) in existing_questions:
                continue
            self.storage.add_topic_turn(
                topic_session_id=topic_session_id,
                side=SIDE_STUDENT,
                turn_kind="question",
                question_text=q,
                extra={"source": "llm"},
            )
            new_count += 1

        for question in _as_list(result.get("teacher_questions")):
            q = str(question).strip()
            if not q or (SIDE_TEACHER, q) in existing_questions:
                continue
            self.storage.add_topic_turn(
                topic_session_id=topic_session_id,
                side=SIDE_TEACHER,
                turn_kind="question",
                question_text=q,
                extra={"source": "llm"},
            )
            new_count += 1

        current = self._ensure_topic_session(topic_session_id)
        next_round = int(current.get("round_no", 0))
        if new_count > 0:
            next_round += 1

        self.storage.update_topic_session(
            topic_session_id,
            status=STATUS_NEEDS_CLARIFICATION,
            round_no=next_round,
        )

        return result

    def add_clarification_answers(
            self,
            *,
            topic_session_id: str,
            side: str,
            answers: str | list[str],
    ) -> list[dict[str, Any]]:
        self._ensure_topic_session(topic_session_id)
        answers_list = [answers] if isinstance(answers, str) else list(answers or [])
        created = []
        for answer in answers_list:
            if not str(answer).strip():
                continue
            created.append(
                self.storage.add_topic_turn(
                    topic_session_id=topic_session_id,
                    side=side,
                    turn_kind="answer",
                    answer_text=str(answer),
                    extra={"source": "user"},
                )
            )
        return created

    def run_alignment_cycle(self, topic_session_id: str) -> dict[str, Any]:
        assessment = self.assess_relation(topic_session_id)
        session = assessment["topic_session"]
        if not session:
            raise RuntimeError("Не удалось обновить сессию согласования.")

        if session["status"] == STATUS_ALIGNED:
            spec = self.build_agreed_spec(topic_session_id)
            finalized = self.finalize_alignment(topic_session_id, spec_override=spec)
            return {
                "status": "aligned",
                "assessment": assessment["assessment"],
                "spec": finalized["agreed_spec"],
                "lab": finalized["lab"],
            }

        if int(session.get("round_no", 0)) >= self.max_rounds:
            self.storage.update_topic_session(
                topic_session_id,
                status=STATUS_REJECTED,
                summary_text="Превышено число раундов уточнения без уверенного согласования темы.",
            )
            return {
                "status": "rejected",
                "assessment": assessment["assessment"],
            }

        questions = self.generate_clarification_questions(topic_session_id)
        return {
            "status": "needs_clarification",
            "assessment": assessment["assessment"],
            "questions": questions,
        }

    def build_agreed_spec(self, topic_session_id: str) -> dict[str, Any]:
        ctx = self.build_alignment_context(topic_session_id)
        lab = ctx["lab"]
        session = ctx["topic_session"]
        student_input = ctx["student_input"]
        teacher_input = ctx["teacher_input"]

        fallback = _heuristic_agreed_spec(lab, session, student_input, teacher_input)

        system_prompt = (
            "Ты формируешь итоговую согласованную спецификацию учебного задания. "
            "Нужно объединить тему студента и тему преподавателя так, чтобы итог удовлетворял обе стороны. "
            "Верни JSON с полями: work_type, agreed_title, agreed_description, acceptance_criteria."
        )
        user_prompt = f"""
Данные по заданию:
{_json_dumps(lab)}

Оценка связи тем:
{_json_dumps(session.get("llm_assessment_json", {}))}

Тема студента:
{_json_dumps(student_input)}

Тема преподавателя:
{_json_dumps(teacher_input)}

История уточнений:
{_json_dumps(ctx["turns"])}
"""

        result = _call_llm_json(
            self.llm_client,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            fallback=fallback,
            schema_name="agreed_assignment_spec",
        )
        result.setdefault("work_type", lab.get("work_type") or WORK_TYPE_OTHER)
        result.setdefault("agreed_title", fallback["agreed_title"])
        result.setdefault("agreed_description", fallback["agreed_description"])
        result.setdefault("acceptance_criteria", fallback["acceptance_criteria"])
        return result

    def finalize_alignment(
            self,
            topic_session_id: str,
            *,
            spec_override: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        ctx = self.build_alignment_context(topic_session_id)
        spec = spec_override or self.build_agreed_spec(topic_session_id)

        finalized = self.storage.finalize_assignment_from_agreed_spec(
            lab_id=ctx["lab"]["lab_id"],
            topic_session_id=topic_session_id,
            work_type=str(spec.get("work_type") or ctx["lab"].get("work_type") or WORK_TYPE_OTHER),
            agreed_title=str(spec.get("agreed_title", "")).strip() or ctx["lab"]["title"],
            agreed_description=str(spec.get("agreed_description", "")).strip() or ctx["lab"].get("description", ""),
            acceptance_criteria=spec.get("acceptance_criteria", {}),
            generated_from={
                "assessment": ctx["topic_session"].get("llm_assessment_json", {}),
                "student_input": ctx["student_input"],
                "teacher_input": ctx["teacher_input"],
            },
        )
        return finalized


class MethodicsAgent(BaseAgent):
    def build_methodics_context(self, lab_id: str) -> dict[str, Any]:
        lab = self._ensure_lab(lab_id)
        spec = self.storage.get_agreed_spec_by_lab(lab_id)
        topic_session = self.storage.get_latest_topic_session_for_lab(lab_id)
        topic_materials = self.storage.list_materials(lab_id, stage=MATERIAL_STAGE_TOPIC_ALIGNMENT)
        methodics_materials = self.storage.list_materials(lab_id, stage=MATERIAL_STAGE_METHODICS)
        context_bundle = build_context_bundle(
            [_material_to_doc_stub(m) for m in topic_materials + methodics_materials],
            max_chars_per_doc=7000,
            max_total_chars=40000,
        )
        return {
            "lab": lab,
            "spec": spec,
            "topic_session": topic_session,
            "topic_materials": topic_materials,
            "methodics_materials": methodics_materials,
            "context_bundle": context_bundle,
        }

    def generate_methodics(self, lab_id: str, *, save_as_material: bool = True) -> dict[str, Any]:
        ctx = self.build_methodics_context(lab_id)
        spec = ctx["spec"]
        if not spec:
            raise ValueError("Нельзя генерировать методику без согласованной спецификации задания.")

        fallback = _heuristic_methodics(spec, ctx["context_bundle"])

        system_prompt = (
            "Ты создаешь методические рекомендации по уже согласованному учебному заданию. "
            "Верни JSON с полями: title, body_text, checklist."
        )
        user_prompt = f"""
Задание:
{_json_dumps(ctx["lab"])}

Согласованная спецификация:
{_json_dumps(spec)}

Контекст:
{ctx["context_bundle"]}
"""

        result = _call_llm_json(
            self.llm_client,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            fallback=fallback,
            schema_name="methodics_artifact",
        )
        result.setdefault("title", fallback["title"])
        result.setdefault("body_text", fallback["body_text"])
        result.setdefault("checklist", fallback["checklist"])

        if save_as_material:
            self.storage.add_material(
                lab_id=lab_id,
                filename=DEFAULT_METHODICS_FILENAME,
                owner_role=MATERIAL_ROLE_SYSTEM,
                stage=MATERIAL_STAGE_METHODICS,
                title=result["title"],
                mime_type="text/markdown",
                file_path="",
                extracted_text=result["body_text"],
                meta={"checklist": result.get("checklist", []), "generated": True},
            )

        self.storage.update_lab(lab_id, config={"methodics_generated": True})
        return result

    def calibrate_policy(self, lab_id: str, *, persist: bool = True) -> dict[str, Any]:
        lab = self._ensure_lab(lab_id)
        spec = self.storage.get_agreed_spec_by_lab(lab_id)
        if not spec:
            raise ValueError("Нельзя калибровать политику без согласованной спецификации.")

        criteria = spec.get("acceptance_criteria_json", {})
        items = [
            {
                "kind": "alignment",
                "title": "Соответствие согласованной теме",
                "body_text": "На защите проверяется, насколько итог работы соответствует зафиксированной согласованной теме задания.",
            },
            {
                "kind": "evaluation",
                "title": "Проверяемый результат",
                "body_text": "Оценивание должно опираться на результат, который можно предъявить, описать и аргументировать.",
            },
            {
                "kind": "defense",
                "title": "Ответы по цели, методу и результату",
                "body_text": "На защите студент должен уметь связать цель работы, выбранный подход и полученный результат.",
            },
        ]

        for axis in criteria.get("evaluation_axes", []):
            items.append({
                "kind": "evaluation",
                "title": f"Критерий: {axis}",
                "body_text": f"При ревью и защите отдельно учитывается аспект «{axis}».",
            })

        if persist:
            for item in items:
                self.storage.add_policy_item(
                    lab_id=lab["lab_id"],
                    kind=item["kind"],
                    title=item["title"],
                    body_text=item["body_text"],
                    source="methodics",
                    meta={"generated": True},
                )

        self.storage.update_lab(lab_id, config={"policy_calibrated": True})
        return {"items": items}


class DefenseAgent(BaseAgent):
    def __init__(self, storage: Storage, llm_client: Any = None, *, max_questions: int = 6) -> None:
        super().__init__(storage=storage, llm_client=llm_client)
        self.max_questions = max_questions

    def build_submission_context(self, lab_id: str, submission_id: str) -> dict[str, Any]:
        lab = self._ensure_lab(lab_id)
        submission = self._ensure_submission(submission_id)
        spec = self.storage.get_agreed_spec_by_lab(lab_id)
        methodics_materials = self.storage.list_materials(lab_id, stage=MATERIAL_STAGE_METHODICS)
        submission_materials = [
            m for m in self.storage.list_materials(lab_id, stage=MATERIAL_STAGE_SUBMISSION)
            if m["owner_role"] == MATERIAL_ROLE_STUDENT
        ]
        context_bundle = build_context_bundle(
            [_material_to_doc_stub(m) for m in methodics_materials + submission_materials],
            max_chars_per_doc=6000,
            max_total_chars=45000,
        )
        return {
            "lab": lab,
            "submission": submission,
            "spec": spec,
            "methodics_materials": methodics_materials,
            "submission_materials": submission_materials,
            "context_bundle": context_bundle,
        }

    def analyze_submission(self, lab_id: str, submission_id: str) -> dict[str, Any]:
        ctx = self.build_submission_context(lab_id, submission_id)
        if not ctx["spec"]:
            raise ValueError("Нельзя анализировать работу без согласованной спецификации.")

        fallback = _heuristic_submission_analysis(ctx["spec"], ctx["submission"], ctx["context_bundle"])

        system_prompt = (
            "Ты выполняешь предварительный анализ загруженной студентом работы перед защитой. "
            "Верни JSON с полями: summary, strengths, risks, recommended_focus."
        )
        user_prompt = f"""
Согласованное задание:
{_json_dumps(ctx["spec"])}

Сведения о работе студента:
{_json_dumps(ctx["submission"])}

Контекст материалов:
{ctx["context_bundle"]}
"""

        result = _call_llm_json(
            self.llm_client,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            fallback=fallback,
            schema_name="submission_analysis",
        )

        updated = self.storage.update_submission(
            submission_id,
            analysis=result,
        )
        return {
            "submission": updated,
            "analysis": result,
        }

    def build_question_pool(
            self,
            *,
            lab_id: str,
            submission_id: str,
            pool_size: int = 8,
    ) -> dict[str, Any]:
        ctx = self.build_submission_context(lab_id, submission_id)
        analysis = ctx["submission"].get("analysis_json", {})
        if not analysis:
            analysis = self.analyze_submission(lab_id, submission_id)["analysis"]

        fallback = _heuristic_question_pool(ctx["spec"] or {}, analysis, pool_size=pool_size)

        system_prompt = (
            "Ты готовишь пул вопросов для защиты по уже согласованному учебному заданию. "
            "Вопросы должны проверять понимание цели, метода, результата, ограничений и соответствия дисциплине. "
            "Верни JSON с полями: questions, strategy_note."
        )
        user_prompt = f"""
Согласованная спецификация:
{_json_dumps(ctx["spec"])}

Анализ работы:
{_json_dumps(analysis)}

Контекст материалов:
{ctx["context_bundle"]}
"""

        result = _call_llm_json(
            self.llm_client,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            fallback=fallback,
            schema_name="question_pool",
        )
        questions = [str(q).strip() for q in _as_list(result.get("questions")) if str(q).strip()]
        if not questions:
            questions = fallback["questions"]

        return {
            "questions": questions[:pool_size],
            "strategy_note": result.get("strategy_note", fallback["strategy_note"]),
        }

    def start_defense(self, *, lab_id: str, submission_id: str, pool_size: int = 8) -> dict[str, Any]:
        pool = self.build_question_pool(lab_id=lab_id, submission_id=submission_id, pool_size=pool_size)
        session = self.storage.create_defense_session(
            lab_id=lab_id,
            submission_id=submission_id,
            status=STATUS_DRAFT,
            plan={
                "question_pool": pool["questions"],
                "strategy_note": pool["strategy_note"],
                "current_index": 0,
                "pending_question": None,
                "max_questions": min(self.max_questions, len(pool["questions"])),
            },
        )
        return session

    def next_question(self, defense_session_id: str) -> dict[str, Any]:
        session = self._ensure_defense_session(defense_session_id)
        plan = session.get("plan_json", {}) or {}
        pool = plan.get("question_pool", [])
        current_index = int(plan.get("current_index", 0))
        pending_question = plan.get("pending_question")

        if pending_question:
            return {
                "question": pending_question,
                "index": current_index,
                "already_pending": True,
            }

        max_questions = int(plan.get("max_questions", self.max_questions))
        if current_index >= min(max_questions, len(pool)):
            finished = self.finish_defense(defense_session_id)
            return {
                "question": None,
                "finished": True,
                "summary": finished,
            }

        question = pool[current_index]
        plan["pending_question"] = question
        self.storage.update_defense_session(defense_session_id, status=STATUS_NEEDS_CLARIFICATION, plan=plan)

        return {
            "question": question,
            "index": current_index + 1,
            "already_pending": False,
        }

    def submit_answer(self, defense_session_id: str, answer_text: str) -> dict[str, Any]:
        session = self._ensure_defense_session(defense_session_id)
        plan = session.get("plan_json", {}) or {}
        question = plan.get("pending_question")
        if not question:
            raise ValueError("Нет активного вопроса для ответа.")

        fallback = _heuristic_answer_evaluation(question, answer_text)

        system_prompt = (
            "Ты оцениваешь ответ студента на вопрос защиты. "
            "Верни JSON с полями: score, verdict, strengths, weaknesses, follow_up."
        )
        user_prompt = f"""
Вопрос:
{question}

Ответ студента:
{answer_text}
"""

        result = _call_llm_json(
            self.llm_client,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            fallback=fallback,
            schema_name="answer_evaluation",
        )

        self.storage.add_qa_turn(
            defense_session_id=defense_session_id,
            question_text=question,
            answer_text=answer_text,
            evaluation=result,
        )

        current_index = int(plan.get("current_index", 0))
        plan["current_index"] = current_index + 1
        plan["pending_question"] = None

        self.storage.update_defense_session(
            defense_session_id,
            status=STATUS_DRAFT,
            plan=plan,
        )

        return result

    def finish_defense(self, defense_session_id: str) -> dict[str, Any]:
        session = self._ensure_defense_session(defense_session_id)
        qa_turns = self.storage.list_qa_turns(defense_session_id)
        scores = [float((item.get("evaluation_json") or {}).get("score", 0.0)) for item in qa_turns]
        avg_score = round(sum(scores) / len(scores), 3) if scores else 0.0

        strengths = []
        weaknesses = []
        for turn in qa_turns:
            evaluation = turn.get("evaluation_json", {}) or {}
            strengths.extend(_as_list(evaluation.get("strengths")))
            weaknesses.extend(_as_list(evaluation.get("weaknesses")))

        summary = {
            "asked_questions": len(qa_turns),
            "average_score": avg_score,
            "strong_points": list(dict.fromkeys(strengths))[:8],
            "weak_points": list(dict.fromkeys(weaknesses))[:8],
        }
        score = {
            "average": avg_score,
            "normalized_100": round(avg_score * 100, 1),
        }

        updated = self.storage.update_defense_session(
            defense_session_id,
            status=STATUS_FINALIZED,
            summary=summary,
            score=score,
        )
        return updated or session


class FeedbackAgent(BaseAgent):
    def generate_student_feedback(self, lab_id: str, defense_session_id: str) -> dict[str, Any]:
        lab = self._ensure_lab(lab_id)
        session = self._ensure_defense_session(defense_session_id)
        spec = self.storage.get_agreed_spec_by_lab(lab_id) or {}
        qa_turns = self.storage.list_qa_turns(defense_session_id)

        fallback = _heuristic_student_feedback(spec, qa_turns, session)

        system_prompt = (
            "Ты формируешь итоговую обратную связь студенту по результатам защиты. "
            "Верни JSON с полями: feedback_text, highlights."
        )
        user_prompt = f"""
Задание:
{_json_dumps(lab)}

Согласованная спецификация:
{_json_dumps(spec)}

Итоги защиты:
{_json_dumps(session)}

Ход защиты:
{_json_dumps(qa_turns)}
"""

        result = _call_llm_json(
            self.llm_client,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            fallback=fallback,
            schema_name="student_feedback",
        )

        created = self.storage.add_student_feedback(
            lab_id=lab_id,
            defense_session_id=defense_session_id,
            feedback_text=str(result.get("feedback_text", fallback["feedback_text"])),
            extra={"highlights": result.get("highlights", fallback["highlights"]), "generated": True},
        )
        return created

    def register_teacher_feedback(
            self,
            *,
            lab_id: str,
            feedback_text: str,
            defense_session_id: Optional[str] = None,
            extra: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        self._ensure_lab(lab_id)
        return self.storage.add_teacher_feedback(
            lab_id=lab_id,
            defense_session_id=defense_session_id,
            feedback_text=feedback_text,
            extra=extra,
        )

    def update_policy_memory(self, lab_id: str) -> dict[str, Any]:
        self._ensure_lab(lab_id)
        teacher_feedback_items = self.storage.list_teacher_feedback(lab_id)
        feedback_texts = [item.get("feedback_text", "") for item in teacher_feedback_items if item.get("feedback_text")]

        fallback = _heuristic_policy_items(feedback_texts)

        system_prompt = (
            "Ты обновляешь policy memory проекта на основе замечаний преподавателя. "
            "Верни JSON с полем items, где каждый элемент содержит kind, title, body_text."
        )
        user_prompt = f"""
Замечания преподавателя:
{_json_dumps(teacher_feedback_items)}
"""

        result = _call_llm_json(
            self.llm_client,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            fallback=fallback,
            schema_name="policy_update",
        )

        created = []
        for item in _as_list(result.get("items")):
            if not isinstance(item, dict):
                continue
            title = str(item.get("title", "")).strip()
            body_text = str(item.get("body_text", "")).strip()
            if not title or not body_text:
                continue
            created.append(
                self.storage.add_policy_item(
                    lab_id=lab_id,
                    kind=str(item.get("kind", "general")),
                    title=title,
                    body_text=body_text,
                    source="teacher_feedback",
                    meta={"generated": True},
                )
            )

        return {
            "items": created,
            "source_feedback_count": len(teacher_feedback_items),
        }


def _material_to_doc_stub(material: dict[str, Any]) -> Any:
    class _DocStub:
        def __init__(self, material: dict[str, Any]) -> None:
            self.filename = material.get("filename", "material.txt")
            self.display_name = material.get("filename", "material.txt")
            self.text = material.get("extracted_text", "")
            self.owner = material.get("owner_role", "unknown")
            self.stage = material.get("stage", "general")
            self.parser_name = (material.get("meta_json") or {}).get("parser_name", "storage")
            self.mime_type = material.get("mime_type")
            self.source_name = material.get("file_path")
            self.archive_path = (material.get("meta_json") or {}).get("archive_path")
            self.warnings = (material.get("meta_json") or {}).get("warnings", [])
            self.is_supported = True
            self.sha256 = (material.get("meta_json") or {}).get("sha256", "")
            self.size_bytes = int((material.get("meta_json") or {}).get("size_bytes", 0))

        def to_context_block(self, max_chars: Optional[int] = None) -> str:
            text = self.text or ""
            if max_chars is not None and len(text) > max_chars:
                text = text[:max_chars].rstrip() + "\n...[обрезано]"
            return (
                f"[ИСТОЧНИК]\n"
                f"Файл: {self.display_name}\n"
                f"Владелец: {self.owner}\n"
                f"Этап: {self.stage}\n\n"
                f"{text or '[пустой текст]'}"
            )

    return _DocStub(material)


__all__ = [
    "IngestAgent",
    "TopicAlignmentAgent",
    "MethodicsAgent",
    "DefenseAgent",
    "FeedbackAgent",
]
