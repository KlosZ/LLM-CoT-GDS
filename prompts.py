"""
Централизованные промпты + JSON-схемы для системы автоматизации защиты работ.

Здесь:
- системные инструкции (tone/guardrails)
- шаблоны user-подсказок для разных стадий: подготовка (преподаватель), защита (студент), ревью (преподаватель)
- JSON Schema для структурированных ответов (под LLM response_format json_schema)
- additionalProperties=False
- русский язык и учебный контекст
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

JsonDict = Dict[str, Any]

# Base system prompts


SYSTEM_BASE_RU = """\
Ты — ассистент для автоматизации подготовки и защиты лабораторных/практических работ.
Контекст: система работает в двух ролях (Преподаватель/Студент), помогает:
1) генерировать методические указания и банк вопросов по материалам лабораторной работы;
2) анализировать работу студента на соответствие заданию/структуре;
3) проводить диалог защиты: задавать вопросы по работе и оценивать ответы;
4) собирать лог защиты и данные обратной связи.

ОБЯЗАТЕЛЬНЫЕ ПРАВИЛА:
- Опирайся на предоставленные материалы и текст работы студента. Если чего-то нет — прямо помечай как "не найдено в материалах".
- Не выдумывай факты о содержимом файлов. Если информации не хватает — задавай уточняющий вопрос (или фиксируй пробел).
- Вопросы должны быть по теме конкретной работы и проверять понимание.
- Избегай токсичности и "ловушек ради ловушек". Сложность должна быть честной.
- Пиши по-русски, кратко и структурированно.
"""

SYSTEM_STRICT_JSON = """\
Ты возвращаешь результат СТРОГО как один JSON-объект.
Запрещено: любой текст вне JSON, markdown, пояснения, лишние ключи.
"""

SYSTEM_GENERATION_EVALUATOR_RU = """\
Ты — независимый оценщик качества генерации в системе подготовки и защиты учебных работ.
Твоя задача — оценивать не студента и не саму учебную работу, а ответ, который сгенерировала тестируемая модель.

Оцени результат по пяти критериям целыми числами от 1 до 5:
1) relevance — релевантность входному контексту и части сценария;
2) completeness — полнота относительно задачи генерации;
3) clarity — ясность, структурность и понятность формулировок;
4) usefulness — практическая полезность для пользователя сценария;
5) correctness — корректность, отсутствие противоречий и необоснованных утверждений.

Шкала:
1 — результат почти непригоден;
2 — много существенных проблем;
3 — приемлемо, но есть заметные недостатки;
4 — хороший результат с небольшими недочетами;
5 — качественный результат без существенных замечаний.

Верни только JSON по заданной схеме. В comment кратко объясни главную причину оценки на русском языке.
"""

# Shared JSON schema building blocks


DIFFICULTY_ENUM = ["easy", "medium", "hard"]

QUESTION_SCHEMA: JsonDict = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "question_id": {"type": "string"},
        "difficulty": {"type": "string", "enum": DIFFICULTY_ENUM},
        "topic": {"type": "string"},
        "question": {"type": "string"},
        "expected_answer_outline": {
            "type": "array",
            "items": {"type": "string"},
        },
        "evaluation_criteria": {
            "type": "array",
            "items": {"type": "string"},
        },
        "followups": {
            "type": "array",
            "items": {"type": "string"},
        },
        "sources": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Ссылки на источники из материалов: material_id/название файла/раздел.",
        },
        "tags": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "question_id",
        "difficulty",
        "topic",
        "question",
        "expected_answer_outline",
        "evaluation_criteria",
        "followups",
        "sources",
        "tags",
    ],
}

RUBRIC_CRITERION_SCHEMA: JsonDict = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "criterion": {"type": "string"},
        "weight": {"type": "number", "minimum": 0, "maximum": 1},
        "description": {"type": "string"},
        "levels": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Уровни/градации (например: отлично/хорошо/уд./неуд.) или чек-лист.",
        },
    },
    "required": ["criterion", "weight", "description", "levels"],
}

# Schemas


SCHEMA_METHODICS_AND_BANK: JsonDict = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "lab_title": {"type": "string"},
        "methodics": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "goal": {"type": "string"},
                "tasks": {"type": "array", "items": {"type": "string"}},
                "prerequisites": {"type": "array", "items": {"type": "string"}},
                "inputs_outputs": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "inputs": {"type": "array", "items": {"type": "string"}},
                        "outputs": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["inputs", "outputs"],
                },
                "step_by_step": {"type": "array", "items": {"type": "string"}},
                "report_structure": {"type": "array", "items": {"type": "string"}},
                "format_requirements": {"type": "array", "items": {"type": "string"}},
                "common_mistakes": {"type": "array", "items": {"type": "string"}},
                "submission_checklist": {"type": "array", "items": {"type": "string"}},
                "academic_integrity": {"type": "array", "items": {"type": "string"}},
            },
            "required": [
                "goal",
                "tasks",
                "prerequisites",
                "inputs_outputs",
                "step_by_step",
                "report_structure",
                "format_requirements",
                "common_mistakes",
                "submission_checklist",
                "academic_integrity",
            ],
        },
        "rubric": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "scale": {"type": "string", "description": "Например: 0-100 или зачёт/незачёт."},
                "criteria": {"type": "array", "items": RUBRIC_CRITERION_SCHEMA},
                "notes": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["scale", "criteria", "notes"],
        },
        "question_bank": {
            "type": "array",
            "items": QUESTION_SCHEMA,
            "minItems": 6,
        },
        "coverage_map": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "topic": {"type": "string"},
                    "materials_covered": {"type": "array", "items": {"type": "string"}},
                    "questions": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["topic", "materials_covered", "questions"],
            },
            "description": "Покрытие тем: какие вопросы опираются на какие материалы.",
        },
        "quality_warnings": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["lab_title", "methodics", "rubric", "question_bank", "coverage_map", "quality_warnings"],
}

SCHEMA_SUBMISSION_ANALYSIS: JsonDict = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "overall_status": {"type": "string", "enum": ["ok", "warning", "fail"]},
        "summary": {"type": "string"},
        "compliance": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "structure": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "missing_sections": {"type": "array", "items": {"type": "string"}},
                        "extra_sections": {"type": "array", "items": {"type": "string"}},
                        "notes": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["missing_sections", "extra_sections", "notes"],
                },
                "task_coverage": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "covered_tasks": {"type": "array", "items": {"type": "string"}},
                        "uncovered_tasks": {"type": "array", "items": {"type": "string"}},
                        "notes": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["covered_tasks", "uncovered_tasks", "notes"],
                },
                "format": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "issues": {"type": "array", "items": {"type": "string"}},
                        "notes": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["issues", "notes"],
                },
            },
            "required": ["structure", "task_coverage", "format"],
        },
        "risk_flags": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Потенциальные риски: шаблонность, отсутствие выводов, несоответствие теме и т.п.",
        },
        "suggested_questions": {
            "type": "array",
            "items": QUESTION_SCHEMA,
            "description": "Предварительный набор вопросов именно по этой работе (не общий банк).",
        },
        "grading_hint": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "estimated_level": {"type": "string", "enum": ["high", "medium", "low", "unknown"]},
                "reasons": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["estimated_level", "reasons"],
        },
    },
    "required": [
        "overall_status",
        "summary",
        "compliance",
        "risk_flags",
        "suggested_questions",
        "grading_hint",
    ],
}

SCHEMA_NEXT_QUESTION_PICK: JsonDict = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "selected_question": QUESTION_SCHEMA,
        "reason": {"type": "string"},
        "should_ask_followup_after_answer": {"type": "boolean"},
    },
    "required": ["selected_question", "reason", "should_ask_followup_after_answer"],
}

SCHEMA_ANSWER_EVAL: JsonDict = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "score": {"type": "number", "minimum": 0, "maximum": 1},
        "label": {"type": "string", "enum": ["pass", "partial", "fail"]},
        "brief_feedback": {"type": "string"},
        "missing_points": {"type": "array", "items": {"type": "string"}},
        "major_errors": {"type": "array", "items": {"type": "string"}},
        "followup_question": {"type": ["string", "null"]},
        "rubric_alignment": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Какие критерии рубрики затронуты/как повлияло.",
        },
    },
    "required": [
        "score",
        "label",
        "brief_feedback",
        "missing_points",
        "major_errors",
        "followup_question",
        "rubric_alignment",
    ],
}

SCHEMA_TEACHER_CALIBRATION_BATCH: JsonDict = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "batch_id": {"type": "string"},
        "questions": {"type": "array", "items": QUESTION_SCHEMA, "minItems": 3},
        "notes_for_teacher": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["batch_id", "questions", "notes_for_teacher"],
}

SCHEMA_POLICY_UPDATE_SUGGESTION: JsonDict = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "good_items_to_add": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "kind": {"type": "string", "enum": ["good_question", "note"]},
                    "content": {"type": "object"},
                    "reason_tags": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["kind", "content", "reason_tags"],
            },
        },
        "bad_items_to_add": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "kind": {"type": "string", "enum": ["bad_question", "note"]},
                    "content": {"type": "object"},
                    "reason_tags": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["kind", "content", "reason_tags"],
            },
        },
        "policy_notes": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["good_items_to_add", "bad_items_to_add", "policy_notes"],
}

SCHEMA_GENERATION_EVALUATION: JsonDict = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "relevance": {
            "type": "integer",
            "minimum": 1,
            "maximum": 5,
            "description": "Релевантность результата входному контексту и части сценария.",
        },
        "completeness": {
            "type": "integer",
            "minimum": 1,
            "maximum": 5,
            "description": "Полнота результата относительно задачи генерации.",
        },
        "clarity": {
            "type": "integer",
            "minimum": 1,
            "maximum": 5,
            "description": "Ясность, структурность и понятность формулировок.",
        },
        "usefulness": {
            "type": "integer",
            "minimum": 1,
            "maximum": 5,
            "description": "Практическая полезность результата для пользователя сценария.",
        },
        "correctness": {
            "type": "integer",
            "minimum": 1,
            "maximum": 5,
            "description": "Корректность, отсутствие противоречий и необоснованных утверждений.",
        },
        "comment": {
            "type": "string",
            "description": "Краткое объяснение оценки на русском языке.",
        },
    },
    "required": [
        "relevance",
        "completeness",
        "clarity",
        "usefulness",
        "correctness",
        "comment",
    ],
}


# Prompt builders


def _json_pretty(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def _json_pretty_limited(obj: Any, *, max_chars: int = 12000) -> str:
    """
    Форматирует объект для промпта и ограничивает размер, чтобы оценщик
    не получал чрезмерно длинный контекст.
    """
    if isinstance(obj, str):
        text = obj.strip()
    else:
        text = _json_pretty(obj)
    if len(text) > max_chars:
        return text[:max_chars].rstrip() + "\n...[фрагмент сокращен]"
    return text


def build_generation_evaluation_prompt(
        *,
        scenario_part: str,
        input_context: Any,
        generated_output: Any,
        method_name: Optional[str] = None,
        extra_instruction: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Формирует prompt для сторонней LLM-оценки качества генерации.

    Функция используется в новом контуре оценки генерации. Модель-оценщик
    выставляет пять целых оценок от 1 до 5, а дальнейшая нормализация и
    объединение с эвристиками выполняются в evaluation.py.
    """
    system = SYSTEM_GENERATION_EVALUATOR_RU + "\n\n" + SYSTEM_STRICT_JSON

    method_block = f"\nМетод / этап: {method_name}" if method_name else ""
    extra_block = (
        f"\n\nДополнительные указания к оценке:\n{extra_instruction.strip()}"
        if extra_instruction
        else ""
    )

    user = f"""\
Часть сценария: {scenario_part}{method_block}

Входной контекст, на основе которого тестируемая модель должна была выполнить генерацию:
{_json_pretty_limited(input_context)}

Результат генерации тестируемой модели:
{_json_pretty_limited(generated_output)}
{extra_block}

ЗАДАЧА:
Оцени именно качество результата генерации для указанной части сценария.
Не оценивай студента, преподавателя или исходную учебную работу.
Не требуй сведений, которых невозможно вывести из входного контекста.

Критерии оценки:
- relevance: связан ли результат с входными данными и нужным этапом сценария;
- completeness: хватает ли результата для выполнения задачи этапа;
- clarity: понятны ли формулировки и структура;
- usefulness: поможет ли результат пользователю системы;
- correctness: нет ли противоречий, выдуманных фактов и необоснованных утверждений.

Верни JSON по схеме generation_evaluation.
"""
    return system, user


def build_methodics_and_bank_prompt(
        *,
        lab_title: str,
        lab_config: JsonDict,
        materials_digest: List[JsonDict],
) -> Tuple[str, str]:
    """
    Подготовка (Преподаватель): сгенерировать методичку, рубрику и общий банк вопросов.

    materials_digest: список объектов (например из storage/materials + извлеченный текст/сводка):
      [{"material_id":"mat_..","filename":"...","kind":"theory","text":"... (можно сокращённо)"}]
    """
    system = SYSTEM_BASE_RU + "\n\n" + SYSTEM_STRICT_JSON

    user = f"""\
Роль: Преподаватель. Стадия: подготовка лабораторной.

Название лабораторной: {lab_title}

Параметры (lab_config):
{_json_pretty(lab_config)}

Материалы лабораторной (digest):
- Каждый элемент содержит material_id, filename, kind и текст (возможно укороченный).
{_json_pretty(materials_digest)}

ЗАДАЧА:
1) Сгенерируй комплект методических указаний под эту лабораторную.
2) Сгенерируй рубрику оценивания (scale + критерии с весами, сумма весов = 1).
3) Сгенерируй общий банк вопросов для защиты:
   - сложности easy/medium/hard
   - вопросы должны быть привязаны к материалам (sources)
   - ожидаемый ответ — в виде тезисного outline (не эссе)
   - добавь followups (1-2) на случай частичного ответа
4) Дай coverage_map: темы → какие материалы покрыты → какие question_id
5) Если в материалах есть пробелы/неопределённости, запиши их в quality_warnings.

ОГРАНИЧЕНИЕ ПО РАЗМЕРУ:
- Сгенерируй ровно 12 вопросов в question_bank:
  4 easy, 5 medium, 3 hard (итого 12).
- Пиши кратко: expected_answer_outline 3–6 пунктов, evaluation_criteria 2–4 пункта, followups 0–2.

ТРЕБОВАНИЯ К КАЧЕСТВУ:
- Вопросы не должны быть "вообще про LLM", если это не часть материалов. Всё привязывай к конкретной лабораторной.
- Структура отчёта должна согласоваться с lab_config.required_sections (можно расширить, но не игнорировать).
- Не придумывай того, чего нет: если не хватает данных, укажи предупреждение.
"""
    return system, user


def build_teacher_calibration_batch_prompt(
        *,
        lab_title: str,
        lab_config: JsonDict,
        question_bank: List[JsonDict],
        round_index: int,
        top_n: int = 10,
) -> Tuple[str, str]:
    """
    Калибровка (Преподаватель): показать top-N вопросов, которые система будет задавать.

    Здесь LLM выбирает top-N из банка и добавляет заметки для преподавателя.
    """
    system = SYSTEM_BASE_RU + "\n\n" + SYSTEM_STRICT_JSON

    user = f"""\
Роль: Преподаватель. Стадия: калибровка вопросов.

Лабораторная: {lab_title}
Раунд калибровки: {round_index}
Параметры (lab_config):
{_json_pretty(lab_config)}

Банк вопросов (question_bank):
{_json_pretty(question_bank)}

ЗАДАЧА:
- Выбери TOP-{top_n} вопросов, которые наиболее релевантны и хорошо сбалансированы по сложности.
- Убедись, что вопросы покрывают разные темы и опираются на материалы.
- Для каждого вопроса проверь корректность и формулировку.
- Добавь notes_for_teacher: на что смотреть (сложность, двусмысленность, оффтоп, корректность ожидаемого ответа).

Верни JSON по схеме.
"""
    return system, user


def build_submission_analysis_prompt(
        *,
        lab_title: str,
        lab_config: JsonDict,
        rubric: JsonDict,
        materials_digest: List[JsonDict],
        submission_excerpt: str,
) -> Tuple[str, str]:
    """
    Стадия защиты: анализ работы студента на соответствие заданию/структуре/формату.

    submission_excerpt: лучше передавать не весь текст, а:
      - оглавление/заголовки
      - введение/цель/задачи
      - выводы
      - фрагменты ключевых разделов
    """
    system = SYSTEM_BASE_RU + "\n\n" + SYSTEM_STRICT_JSON

    user = f"""\
Роль: Система. Стадия: анализ работы студента перед защитой.

Лабораторная: {lab_title}

Параметры (lab_config):
{_json_pretty(lab_config)}

Рубрика:
{_json_pretty(rubric)}

Материалы лабораторной (digest):
{_json_pretty(materials_digest)}

Текст работы студента (excerpt):
{submission_excerpt}

ЗАДАЧА:
1) Оцени соответствие структуры: наличие обязательных разделов (lab_config.required_sections).
2) Оцени покрытие задач лабораторной (по методичке/материалам): что явно сделано, что нет.
3) Оцени формат/качество представления (без придирок к мелочам, только важное).
4) Сформируй risk_flags: существенные риски (нет выводов, подмена темы, отсутствие результатов, и т.п.).
5) Сформируй suggested_questions: вопросы именно по этой работе (опирайся на найденные слабые места и ключевые элементы).

Правила:
- Не выдумывай, если раздела не видно — считай "не найдено".
- Если excerpt слишком короткий, укажи это в summary/notes и будь аккуратен.
"""
    return system, user


def build_next_question_pick_prompt(
        *,
        lab_title: str,
        lab_config: JsonDict,
        question_plan: JsonDict,
        remaining_question_bank: List[JsonDict],
        asked_turns: List[JsonDict],
        submission_analysis: JsonDict,
        policy_good_examples: List[JsonDict],
        policy_bad_examples: List[JsonDict],
) -> Tuple[str, str]:
    """
    Во время защиты: выбрать следующий вопрос из оставшегося пула.

    asked_turns: список уже заданных (turns) с вопросом и кратким статусом ответа (если есть)
    question_plan: например {"easy":3,"medium":2,"hard":1} — сколько нужно всего
    remaining_question_bank: кандидаты
    """
    system = SYSTEM_BASE_RU + "\n\n" + SYSTEM_STRICT_JSON

    user = f"""\
Роль: Система. Стадия: проведение защиты. Нужно выбрать следующий вопрос.

Лабораторная: {lab_title}
Параметры (lab_config):
{_json_pretty(lab_config)}

План по сложности (сколько вопросов нужно в итоге):
{_json_pretty(question_plan)}

Уже заданные вопросы и статус (asked_turns):
{_json_pretty(asked_turns)}

Анализ работы студента (submission_analysis):
{_json_pretty(submission_analysis)}

Оставшиеся кандидаты вопросов (remaining_question_bank):
{_json_pretty(remaining_question_bank)}

Примеры предпочтений преподавателя (policy memory):
GOOD examples (делай похоже по стилю/уровню/привязке):
{_json_pretty(policy_good_examples)}

BAD examples (так НЕ делай; избегай причин/паттернов):
{_json_pretty(policy_bad_examples)}

ЗАДАЧА:
- Выбери один следующий вопрос (selected_question), чтобы:
  1) соблюдать план сложности (не тратить hard слишком рано, если не надо);
  2) закрывать пробелы из submission_analysis;
  3) избегать повторов тем;
  4) соответствовать предпочтениям преподавателя.
- Укажи короткую причину (reason).
- Укажи, стоит ли планировать followup после ответа (should_ask_followup_after_answer):
  true, если вопрос проверяет критичный пробел/есть высокий риск непонимания.
"""
    return system, user


def build_answer_evaluation_prompt(
        *,
        lab_title: str,
        rubric: JsonDict,
        question_obj: JsonDict,
        student_answer: str,
        materials_digest: Optional[List[JsonDict]] = None,
        strictness: float = 0.7,
) -> Tuple[str, str]:
    """
    Оценка ответа студента на вопрос в ходе защиты.

    strictness 0..1: чем выше, тем строже оценивание неполных ответов.
    """
    system = SYSTEM_BASE_RU + "\n\n" + SYSTEM_STRICT_JSON

    md = _json_pretty(materials_digest) if materials_digest is not None else "[]"

    user = f"""\
Роль: Система. Стадия: оценка ответа студента на вопрос защиты.

Лабораторная: {lab_title}
Строгость (0..1): {strictness}

Рубрика:
{_json_pretty(rubric)}

Вопрос (question_obj):
{_json_pretty(question_obj)}

Ответ студента (student_answer):
{student_answer}

Материалы лабораторной (digest, опционально):
{md}

ЗАДАЧА:
- Оцени ответ по сути вопроса:
  * score: 0..1
  * label: pass/partial/fail
  * brief_feedback: 1-3 предложения
  * missing_points: что не упомянуто из expected_answer_outline
  * major_errors: грубые ошибки/подмена смысла
  * followup_question: либо null, либо один уточняющий вопрос (короткий)
  * rubric_alignment: какие критерии рубрики затронуты (строки)

Правила:
- Не оценивай "красоту речи", оцени понимание.
- Если ответ частичный — укажи конкретно, чего не хватает.
- Если ответ неверен — коротко объясни, в чём ключевая ошибка.
"""
    return system, user


def build_policy_update_suggestion_prompt(
        *,
        lab_title: str,
        teacher_feedback_items: List[JsonDict],
        turns: List[JsonDict],
) -> Tuple[str, str]:
    """
    После ревью преподавателя: предложить, какие элементы добавить в policy memory.
    teacher_feedback_items: список записей (good/bad + tags + comment)
    turns: turns с вопросами/ответами, чтобы извлечь контент.
    """
    system = SYSTEM_BASE_RU + "\n\n" + SYSTEM_STRICT_JSON

    user = f"""\
Роль: Система. Стадия: обновление policy memory по обратной связи преподавателя.

Лабораторная: {lab_title}

Teacher feedback items:
{_json_pretty(teacher_feedback_items)}

Ход защиты (turns):
{_json_pretty(turns)}

ЗАДАЧА:
- На основе teacher_feedback_items предложи:
  1) good_items_to_add: примеры удачных вопросов/заметки (kind=good_question или note)
  2) bad_items_to_add: примеры неудачных вопросов/заметки (kind=bad_question или note)
- Для каждого item:
  - content: минимально достаточная структура (например, question + why_good/why_bad + difficulty + topic)
  - reason_tags: 1..4 тега (например: off_topic, ambiguous, too_easy, too_hard, good_depth, good_alignment)
- policy_notes: общие выводы для будущих генераций (например: "больше вопросов по разделу X").

Не выдумывай: используй только то, что есть в turns и teacher feedback.
"""
    return system, user


# Small utilities for embedding policy examples into prompts (optional)


def policy_examples_to_fewshot_text(
        good_examples: List[JsonDict],
        bad_examples: List[JsonDict],
        *,
        max_chars: int = 6000,
) -> str:
    """
    Превращает policy items в компактный текст (если вам удобнее few-shot в plain text,
    а не в JSON внутри промпта).
    """
    parts: List[str] = []

    def _clip(s: str) -> str:
        return s if len(s) <= max_chars else s[:max_chars] + "…"

    if good_examples:
        parts.append("GOOD (делай похоже):")
        for ex in good_examples:
            content = ex.get("content", ex)
            parts.append(_clip(json.dumps(content, ensure_ascii=False)))
    if bad_examples:
        parts.append("BAD (так не делай):")
        for ex in bad_examples:
            content = ex.get("content", ex)
            parts.append(_clip(json.dumps(content, ensure_ascii=False)))

    txt = "\n".join(parts)
    if len(txt) > max_chars:
        txt = txt[:max_chars] + "…"
    return txt


# Expose schemas in one place


SCHEMAS: JsonDict = {
    "methodics_and_bank": SCHEMA_METHODICS_AND_BANK,
    "submission_analysis": SCHEMA_SUBMISSION_ANALYSIS,
    "next_question_pick": SCHEMA_NEXT_QUESTION_PICK,
    "answer_eval": SCHEMA_ANSWER_EVAL,
    "teacher_calibration_batch": SCHEMA_TEACHER_CALIBRATION_BATCH,
    "policy_update_suggestion": SCHEMA_POLICY_UPDATE_SUGGESTION,
    "generation_evaluation": SCHEMA_GENERATION_EVALUATION,
}
