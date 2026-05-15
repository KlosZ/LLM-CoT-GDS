from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any

CSV_COLUMNS = [
    "case_code",
    "title",
    "description",
    "scenario_part",
    "method_name",
    "input_json",
    "generated_output_json",
    "expected_notes",
    "tags",
    "is_active",
]

DEFAULT_EVALUATION_CASES_CSV = Path("data/evaluation_cases_default.csv")


def _decode_csv_content(content: str | bytes) -> str:
    """
    Приводит CSV-содержимое к строке UTF-8.

    Streamlit file_uploader возвращает bytes, а встроенный демонстрационный
    файл читается как str. Функция унифицирует оба варианта и убирает BOM,
    который часто появляется в CSV, открытых или сохраненных через Excel.
    """
    if isinstance(content, bytes):
        return content.decode("utf-8-sig")
    return str(content).lstrip("\ufeff")


def parse_bool(value: Any, default: bool = True) -> bool:
    """
    Преобразует текстовое значение из CSV в bool.

    Поддерживаются русские и английские варианты: true/false, да/нет, 1/0.
    Пустая ячейка возвращает default.
    """
    text = str(value or "").strip().lower()
    if not text:
        return default
    if text in {"1", "true", "yes", "y", "да", "д", "истина", "активен", "active"}:
        return True
    if text in {"0", "false", "no", "n", "нет", "н", "ложь", "неактивен", "inactive"}:
        return False
    return default


def parse_tags(value: Any) -> list[str]:
    """
    Разбирает теги из CSV.

    Разрешены разделители ',' и ';'. Повторы удаляются с сохранением порядка.
    """
    text = str(value or "").strip()
    if not text:
        return []

    raw_items: list[str] = []
    for chunk in text.replace(";", ",").split(","):
        tag = chunk.strip()
        if tag:
            raw_items.append(tag)

    seen: set[str] = set()
    result: list[str] = []
    for tag in raw_items:
        key = tag.lower()
        if key not in seen:
            seen.add(key)
            result.append(tag)
    return result


def parse_json_cell(value: Any, *, row_no: int, column_name: str, errors: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Разбирает JSON-ячейку CSV.

    Для input_json и generated_output_json ожидается JSON-объект. Если ячейка
    пустая, возвращается пустой словарь. Если JSON некорректен, ошибка
    добавляется в список errors, а строка дальше не импортируется.
    """
    text = str(value or "").strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        errors.append(
            {
                "row_no": row_no,
                "column": column_name,
                "error": f"Некорректный JSON: {exc.msg} на позиции {exc.pos}",
            }
        )
        return {}

    if not isinstance(parsed, dict):
        errors.append(
            {
                "row_no": row_no,
                "column": column_name,
                "error": "Ожидался JSON-объект, например {\"key\": \"value\"}.",
            }
        )
        return {}
    return parsed


def parse_evaluation_cases_csv(content: str | bytes) -> dict[str, Any]:
    """
    Читает CSV с тест-кейсами оценки генерации.

    Обязательные колонки:
    - title;
    - scenario_part;
    - method_name;
    - input_json;
    - generated_output_json.

    Возвращает структуру:
    {
        "cases": [...],
        "errors": [...]
    }

    Ошибочные строки не попадают в cases, но отображаются в errors, чтобы
    пользователь мог исправить CSV без падения всего импорта.
    """
    text = _decode_csv_content(content)
    reader = csv.DictReader(io.StringIO(text))

    errors: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []

    if not reader.fieldnames:
        return {"cases": [], "errors": [{"row_no": 0, "error": "CSV-файл не содержит заголовков."}]}

    normalized_fieldnames = {name.strip() for name in reader.fieldnames if name}
    required = {"title", "scenario_part", "method_name", "input_json", "generated_output_json"}
    missing = sorted(required - normalized_fieldnames)
    if missing:
        return {
            "cases": [],
            "errors": [
                {
                    "row_no": 0,
                    "error": "В CSV отсутствуют обязательные колонки: " + ", ".join(missing),
                }
            ],
        }

    for row_index, raw_row in enumerate(reader, start=2):
        row = {str(key or "").strip(): value for key, value in raw_row.items()}
        row_errors_before = len(errors)

        title = str(row.get("title") or "").strip()
        scenario_part = str(row.get("scenario_part") or "").strip()
        method_name = str(row.get("method_name") or "").strip()

        if not title:
            errors.append({"row_no": row_index, "column": "title", "error": "Название кейса не заполнено."})
        if not scenario_part:
            errors.append({"row_no": row_index, "column": "scenario_part", "error": "Часть сценария не заполнена."})
        if not method_name:
            errors.append({"row_no": row_index, "column": "method_name", "error": "Метод / этап не заполнен."})

        input_json = parse_json_cell(row.get("input_json"), row_no=row_index, column_name="input_json", errors=errors)
        output_json = parse_json_cell(
            row.get("generated_output_json"),
            row_no=row_index,
            column_name="generated_output_json",
            errors=errors,
        )

        if len(errors) > row_errors_before:
            continue

        cases.append(
            {
                "case_code": str(row.get("case_code") or "").strip(),
                "title": title,
                "description": str(row.get("description") or "").strip(),
                "scenario_part": scenario_part,
                "method_name": method_name,
                "input_json": input_json,
                "generated_output_json": output_json,
                "expected_notes": str(row.get("expected_notes") or "").strip(),
                "tags": parse_tags(row.get("tags")),
                "is_active": parse_bool(row.get("is_active"), default=True),
            }
        )

    return {"cases": cases, "errors": errors}


def load_default_evaluation_cases_csv(path: str | Path = DEFAULT_EVALUATION_CASES_CSV) -> str:
    """
    Загружает встроенный демонстрационный CSV-набор тест-кейсов.
    """
    csv_path = Path(path)
    if not csv_path.exists():
        csv_path = Path(__file__).resolve().parent / path
    return csv_path.read_text(encoding="utf-8-sig")


def cases_to_csv_bytes(cases: list[dict[str, Any]]) -> bytes:
    """
    Сериализирует список кейсов в CSV-байты UTF-8 с BOM для удобного открытия в Excel.
    """
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=CSV_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    for item in cases:
        row = dict(item)
        row["input_json"] = json.dumps(row.get("input_json") or {}, ensure_ascii=False)
        row["generated_output_json"] = json.dumps(row.get("generated_output_json") or {}, ensure_ascii=False)
        tags = row.get("tags") or []
        if isinstance(tags, list):
            row["tags"] = ", ".join(str(tag) for tag in tags)
        row["is_active"] = "true" if parse_bool(row.get("is_active"), default=True) else "false"
        writer.writerow(row)
    return ("\ufeff" + buffer.getvalue()).encode("utf-8")
