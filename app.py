from __future__ import annotations

import csv
import io
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional, Tuple

import streamlit as st
from streamlit import runtime
from streamlit.web import cli as stcli

from core import ProjectCore, create_core
from storage import (
    SIDE_STUDENT,
    SIDE_TEACHER,
    STATUS_ALIGNED,
    STATUS_FINALIZED,
    STATUS_NEEDS_CLARIFICATION,
    STATUS_REJECTED,
    WORK_TYPE_COURSEWORK,
    WORK_TYPE_LAB,
    WORK_TYPE_OTHER,
    WORK_TYPE_PRACTICE,
    WORK_TYPE_REPORT,
    WORK_TYPE_RESEARCH,
)

runtime.exists()

try:
    from dotenv import load_dotenv  # type: ignore

    dotenv_override = (
                              os.getenv("LLM_DOTENV_OVERRIDE")
                              or os.getenv("DOTENV_OVERRIDE")
                              or "1"
                      ).strip().lower() not in {"0", "false", "no", "off"}
    load_dotenv(override=dotenv_override)
except Exception:
    pass

APP_TITLE = "Система формирования и защиты учебных заданий"
APP_CAPTION = (
    "Новая версия: сначала согласование темы между студентом и преподавателем, "
    "затем методичка, публикация, защита и ревью."
)

WORK_TYPE_LABELS = {
    WORK_TYPE_LAB: "Лабораторная работа",
    WORK_TYPE_PRACTICE: "Практическая работа",
    WORK_TYPE_RESEARCH: "НИР / исследовательская работа",
    WORK_TYPE_COURSEWORK: "Курсовая работа",
    WORK_TYPE_REPORT: "Доклад / отчет",
    WORK_TYPE_OTHER: "Другое",
}


def try_build_llm_client() -> Tuple[Any, Optional[str]]:
    try:
        import llm  # type: ignore
    except Exception as exc:
        return None, f"Не удалось импортировать llm.py: {type(exc).__name__}: {exc}"

    factory_names = [
        "create_llm_client",
        "build_llm_client",
        "get_llm_client",
    ]

    last_error: Optional[str] = None

    for name in factory_names:
        factory = getattr(llm, name, None)
        if callable(factory):
            try:
                client = factory()
                return client, None
            except TypeError:
                try:
                    client = factory(os.environ)
                    return client, None
                except Exception as exc:
                    last_error = f"{name}(os.environ): {type(exc).__name__}: {exc}"
            except Exception as exc:
                last_error = f"{name}(): {type(exc).__name__}: {exc}"

    client_cls = getattr(llm, "LLMClient", None)
    if client_cls:
        try:
            client = client_cls()
            return client, None
        except Exception as exc:
            last_error = f"LLMClient(): {type(exc).__name__}: {exc}"

    return None, last_error or "В llm.py не найден подходящий фабричный метод или класс LLMClient."


@st.cache_resource(show_spinner=False)
def get_cached_llm_client() -> Tuple[Any, Optional[str]]:
    return try_build_llm_client()


@st.cache_resource(show_spinner=False)
def get_core() -> ProjectCore:
    db_path = os.getenv("APP_DB_PATH", "data/app.sqlite3")
    upload_dir = os.getenv("APP_UPLOAD_DIR", "uploads")
    llm_client, _llm_error = get_cached_llm_client()

    return create_core(
        db_path=db_path,
        llm_client=llm_client,
        upload_dir=upload_dir,
        relation_threshold=float(os.getenv("RELATION_THRESHOLD", "0.55")),
        max_alignment_rounds=int(os.getenv("MAX_ALIGNMENT_ROUNDS", "3")),
        max_defense_questions=int(os.getenv("MAX_DEFENSE_QUESTIONS", "6")),
    )


def rerun() -> None:
    st.rerun()


def json_pretty(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def parse_json_text(value: str, *, default: Optional[Any] = None) -> Any:
    """
    Аккуратно разбирает JSON из текстового поля Streamlit.
    Пустая строка возвращает default.
    """
    text = (value or "").strip()
    if not text:
        return default
    return json.loads(text)


def report_rows_to_csv_bytes(rows: list[dict[str, Any]]) -> bytes:
    """
    Формирует CSV для скачивания итоговой таблицы оценки генерации.
    """
    if not rows:
        return b""

    preferred = [
        "run_id",
        "case_id",
        "case_title",
        "scenario_part",
        "method_name",
        "llm_relevance",
        "llm_completeness",
        "llm_clarity",
        "llm_usefulness",
        "llm_correctness",
        "llm_score",
        "heuristic_score",
        "final_score",
        "comment",
    ]
    extra = sorted({key for row in rows for key in row.keys()} - set(preferred))
    fieldnames = [key for key in preferred if any(key in row for row in rows)] + extra

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return buffer.getvalue().encode("utf-8-sig")


def scenario_label(value: str) -> str:
    mapping = {
        "topic_final": "Итоговая тема",
        "topic_clarification_questions": "Уточняющие вопросы",
        "topic_alignment_process": "Процесс согласования темы",
        "topic_alignment_round": "Раунд согласования темы",
        "defense_questions": "Вопросы для защиты",
    }
    return mapping.get(value, value or "—")


def work_type_label(value: str) -> str:
    return WORK_TYPE_LABELS.get(value, value)


def lab_option_label(lab: dict[str, Any]) -> str:
    return f"{lab['title']} · {work_type_label(lab['work_type'])} · {lab['status']}"


def status_badge(status: str) -> str:
    mapping = {
        STATUS_NEEDS_CLARIFICATION: "🟡 Нужно уточнение",
        STATUS_ALIGNED: "🟢 Согласовано",
        STATUS_FINALIZED: "✅ Зафиксировано",
        STATUS_REJECTED: "🔴 Отклонено",
    }
    return mapping.get(status, f"⚪ {status}")


def ensure_state_defaults() -> None:
    defaults = {
        "selected_lab_id": None,
        "selected_submission_id": None,
        "selected_defense_session_id": None,
        "selected_evaluation_run_id": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def pick_selected_lab(labs: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    selected_lab_id = st.session_state.get("selected_lab_id")
    if not labs:
        st.session_state["selected_lab_id"] = None
        return None

    if not selected_lab_id or all(item["lab_id"] != selected_lab_id for item in labs):
        st.session_state["selected_lab_id"] = labs[0]["lab_id"]
        selected_lab_id = labs[0]["lab_id"]

    return next((item for item in labs if item["lab_id"] == selected_lab_id), None)


def group_topic_turns_by_side(turns: list[dict[str, Any]]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {
        SIDE_STUDENT: {"question": [], "answer": []},
        SIDE_TEACHER: {"question": [], "answer": []},
    }
    for turn in turns:
        side = turn.get("side", "")
        kind = turn.get("turn_kind", "")
        if side in grouped and kind in grouped[side]:
            grouped[side][kind].append(turn)
    return grouped


def render_material_cards(materials: list[dict[str, Any]], title: str) -> None:
    st.markdown(f"**{title}**")
    if not materials:
        st.info("Пока нет материалов.")
        return

    for item in materials:
        with st.container(border=True):
            st.markdown(f"**{item.get('filename', 'Файл')}**")
            st.caption(
                f"Этап: {item.get('stage', '')} · "
                f"Роль: {item.get('owner_role', '')} · "
                f"Создано: {item.get('created_at', '')}"
            )
            text = (item.get("extracted_text") or "").strip()
            if text:
                st.text_area(
                    "Извлеченный текст",
                    value=text[:4000],
                    height=160,
                    key=f"text_{item['material_id']}",
                )
            meta = item.get("meta_json") or {}
            warnings = meta.get("warnings") or []
            if warnings:
                st.warning("\n".join(str(x) for x in warnings))


def render_policy_items(items: list[dict[str, Any]]) -> None:
    if not items:
        st.info("Policy memory пока пустая.")
        return

    for item in items:
        with st.container(border=True):
            st.markdown(f"**{item.get('title', 'Без названия')}**")
            st.caption(
                f"Тип: {item.get('kind', '')} · "
                f"Источник: {item.get('source', '')} · "
                f"Обновлено: {item.get('updated_at', '')}"
            )
            st.write(item.get("body_text", ""))


def render_feedback_list(items: list[dict[str, Any]], title: str) -> None:
    st.markdown(f"**{title}**")
    if not items:
        st.info("Пока нет записей.")
        return

    for item in items:
        with st.container(border=True):
            st.caption(item.get("created_at", ""))
            st.write(item.get("feedback_text", ""))
            extra = item.get("extra_json") or {}
            if extra:
                with st.expander("Дополнительно"):
                    st.code(json_pretty(extra), language="json")


def render_qa_turns(turns: list[dict[str, Any]]) -> None:
    if not turns:
        st.info("В этой защите пока нет вопросов и ответов.")
        return

    for idx, turn in enumerate(turns, start=1):
        with st.container(border=True):
            st.markdown(f"**Вопрос {idx}**")
            st.write(turn.get("question_text", ""))
            st.markdown("**Ответ**")
            st.write(turn.get("answer_text", ""))
            evaluation = turn.get("evaluation_json") or {}
            if evaluation:
                st.markdown("**Оценка ответа**")
                st.code(json_pretty(evaluation), language="json")


def render_llm_status() -> None:
    llm_client, llm_error = get_cached_llm_client()

    st.sidebar.divider()
    st.sidebar.subheader("LLM")

    if st.sidebar.button("Сбросить кэш ядра / LLM"):
        st.cache_resource.clear()
        rerun()

    if llm_client is not None:
        st.sidebar.success("LLM-клиент инициализирован.")
    else:
        st.sidebar.error("LLM-клиент не инициализирован.")
        if llm_error:
            with st.sidebar.expander("Причина"):
                st.code(llm_error)

    try:
        import llm  # type: ignore

        summary = llm.get_safe_config_summary() if hasattr(llm, "get_safe_config_summary") else {}
        if summary:
            with st.sidebar.expander("Диагностика .env"):
                st.code(json_pretty(summary), language="json")
    except Exception:
        pass

    if st.sidebar.button("Проверить LLM запросом"):
        if llm_client is None:
            st.sidebar.error("Сначала нужно исправить инициализацию LLM.")
            return
        try:
            answer = llm_client.generate_text(
                system_prompt="Ты тестовый помощник. Ответь ровно одним словом.",
                user_prompt="Ответь: OK",
            )
            st.sidebar.success(f"Ответ LLM: {answer}")
        except Exception as exc:
            st.sidebar.error(f"{type(exc).__name__}: {exc}")


def render_sidebar(core: ProjectCore) -> Optional[dict[str, Any]]:
    st.sidebar.title("Навигация")
    labs = core.storage.list_labs()

    if st.sidebar.button("Обновить список"):
        rerun()

    if labs:
        selected_lab = pick_selected_lab(labs)
        options = {lab_option_label(item): item["lab_id"] for item in labs}
        current_label = next(
            (label for label, lab_id in options.items() if lab_id == st.session_state["selected_lab_id"]),
            list(options.keys())[0],
        )

        selected_label = st.sidebar.selectbox(
            "Выберите задание",
            options=list(options.keys()),
            index=list(options.keys()).index(current_label),
        )
        st.session_state["selected_lab_id"] = options[selected_label]
        selected_lab = next(item for item in labs if item["lab_id"] == st.session_state["selected_lab_id"])
    else:
        selected_lab = None
        st.sidebar.info("Заданий пока нет.")

    render_llm_status()

    st.sidebar.divider()
    st.sidebar.subheader("Создать новое задание")

    with st.sidebar.form("create_assignment_form", clear_on_submit=False):
        teacher_title = st.text_input("Тема преподавателя / основа задания")
        teacher_description = st.text_area("Описание", height=120)
        work_type_ui = st.selectbox("Тип задания", options=list(WORK_TYPE_LABELS.keys()), format_func=work_type_label)
        discipline = st.text_input("Дисциплина / рамка курса")
        create_clicked = st.form_submit_button("Создать")

    if create_clicked:
        if not teacher_title.strip():
            st.sidebar.error("Нужно задать хотя бы тему преподавателя.")
        else:
            created = core.create_assignment(
                teacher_title=teacher_title.strip(),
                teacher_description=teacher_description.strip(),
                work_type=work_type_ui,
                config={"topic_alignment_enabled": True},
                seed_teacher_topic=True,
                teacher_topic_context={"discipline": discipline.strip()},
            )
            st.session_state["selected_lab_id"] = created["lab"]["lab_id"]
            st.sidebar.success("Задание создано.")
            rerun()

    st.sidebar.divider()
    st.sidebar.caption(
        f"База: {os.getenv('APP_DB_PATH', 'data/app.sqlite3')}\n\n"
        f"Загрузки: {os.getenv('APP_UPLOAD_DIR', 'uploads')}"
    )

    return selected_lab


def render_overview(core: ProjectCore, dashboard: dict[str, Any]) -> None:
    lab = dashboard["lab"]
    topic_session = dashboard["topic_session"]
    agreed_spec = dashboard["agreed_spec"]

    st.subheader("Обзор задания")

    col1, col2, col3 = st.columns(3)
    col1.metric("Тип задания", work_type_label(lab["work_type"]))
    col2.metric("Статус", lab["status"])
    col3.metric("Согласование", status_badge(topic_session["status"]) if topic_session else "нет")

    with st.container(border=True):
        st.markdown(f"**Название**: {lab['title']}")
        st.markdown(f"**Описание**: {lab.get('description', '') or '—'}")
        st.caption(f"Создано: {lab.get('created_at', '')} · Обновлено: {lab.get('updated_at', '')}")

    if topic_session:
        with st.container(border=True):
            st.markdown("**Текущее состояние согласования**")
            st.write(f"Статус: {status_badge(topic_session.get('status', ''))}")
            st.write(f"Раундов уточнения: {topic_session.get('round_no', 0)}")
            st.write(f"Оценка связи: {topic_session.get('relation_score')}")
            st.write(f"Метка связи: {topic_session.get('relation_label') or '—'}")
            if topic_session.get("summary_text"):
                st.info(topic_session["summary_text"])

    if agreed_spec:
        with st.container(border=True):
            st.markdown("**Зафиксированная спецификация**")
            st.markdown(f"**Тема**: {agreed_spec.get('agreed_title', '')}")
            st.write(agreed_spec.get("agreed_description", ""))
            criteria = agreed_spec.get("acceptance_criteria_json") or {}
            if criteria:
                with st.expander("Критерии приемки"):
                    st.code(json_pretty(criteria), language="json")

    with st.expander("Конфигурация задания"):
        st.code(json_pretty(lab.get("config_json") or {}), language="json")


def render_alignment_tab(core: ProjectCore, dashboard: dict[str, Any]) -> None:
    lab = dashboard["lab"]
    topic_session = dashboard["topic_session"]

    if not topic_session:
        st.error("Для задания не найдена сессия согласования.")
        return

    st.subheader("Согласование темы")
    st.info(
        "Сначала сохраните ответы студента и преподавателя. После этого один раз нажмите «Запустить цикл согласования».")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### Ввод студента")
        current_student = core.storage.get_topic_input(topic_session["topic_session_id"], SIDE_STUDENT) or {}
        with st.form("student_topic_form"):
            student_title = st.text_input("Тема студента", value=current_student.get("title", ""))
            student_description = st.text_area("Описание темы студента",
                                               value=current_student.get("description", ""),
                                               height=160)
            student_notes = st.text_area(
                "Контекст студента",
                value=(current_student.get("context_json") or {}).get("notes", ""),
                height=100,
                help="Например: практика, НИР, уже есть данные, нужен прототип и т.д.",
            )
            submit_student = st.form_submit_button("Сохранить тему студента")

        if submit_student:
            if not student_title.strip():
                st.error("Нужно ввести тему студента.")
            else:
                core.submit_student_topic(
                    topic_session_id=topic_session["topic_session_id"],
                    title=student_title.strip(),
                    description=student_description.strip(),
                    context={"notes": student_notes.strip()},
                )
                st.success("Тема студента сохранена.")
                rerun()

        student_files = st.file_uploader(
            "Материалы студента",
            accept_multiple_files=True,
            key=f"student_topic_files_{lab['lab_id']}",
        )
        if st.button("Загрузить материалы студента", key=f"upload_student_topic_{lab['lab_id']}"):
            if not student_files:
                st.warning("Файлы не выбраны.")
            else:
                core.upload_student_topic_materials(lab_id=lab["lab_id"], uploaded_files=student_files)
                st.success("Материалы студента загружены.")
                rerun()

    with col_right:
        st.markdown("### Ввод преподавателя")
        current_teacher = core.storage.get_topic_input(topic_session["topic_session_id"], SIDE_TEACHER) or {}
        with st.form("teacher_topic_form"):
            teacher_title = st.text_input("Тема преподавателя", value=current_teacher.get("title", ""))
            teacher_description = st.text_area("Описание темы преподавателя",
                                               value=current_teacher.get("description", ""),
                                               height=160)
            teacher_notes = st.text_area(
                "Контекст преподавателя",
                value=(current_teacher.get("context_json") or {}).get("notes", ""),
                height=100,
                help="Например: рамки дисциплины, обязательные критерии, допустимый формат результата.",
            )
            submit_teacher = st.form_submit_button("Сохранить тему преподавателя")

        if submit_teacher:
            if not teacher_title.strip():
                st.error("Нужно ввести тему преподавателя.")
            else:
                core.submit_teacher_topic(
                    topic_session_id=topic_session["topic_session_id"],
                    title=teacher_title.strip(),
                    description=teacher_description.strip(),
                    context={"notes": teacher_notes.strip()},
                )
                st.success("Тема преподавателя сохранена.")
                rerun()

        teacher_files = st.file_uploader(
            "Материалы преподавателя",
            accept_multiple_files=True,
            key=f"teacher_topic_files_{lab['lab_id']}",
        )
        if st.button("Загрузить материалы преподавателя", key=f"upload_teacher_topic_{lab['lab_id']}"):
            if not teacher_files:
                st.warning("Файлы не выбраны.")
            else:
                core.upload_teacher_topic_materials(lab_id=lab["lab_id"], uploaded_files=teacher_files)
                st.success("Материалы преподавателя загружены.")
                rerun()

    st.divider()

    col_a, col_b, col_c = st.columns([1, 1, 1])
    with col_a:
        if st.button("Запустить цикл согласования", type="primary"):
            result = core.run_alignment_cycle(topic_session_id=topic_session["topic_session_id"])
            st.session_state["last_alignment_result"] = result
            st.success(f"Цикл завершен: {result['status']}")
            rerun()

    with col_b:
        if st.button("Зафиксировать тему вручную"):
            try:
                core.finalize_alignment_manually(topic_session_id=topic_session["topic_session_id"])
                st.success("Согласованная тема зафиксирована.")
                rerun()
            except Exception as exc:
                st.error(str(exc))

    with col_c:
        if st.button("Обновить снимок"):
            rerun()

    current_session = core.get_topic_session(topic_session["topic_session_id"])
    with st.container(border=True):
        st.markdown("**Состояние согласования**")
        st.write(f"Статус: {status_badge(current_session.get('status', ''))}")
        st.write(f"Оценка связи: {current_session.get('relation_score')}")
        st.write(f"Метка связи: {current_session.get('relation_label') or '—'}")
        st.write(f"Раунд: {current_session.get('round_no', 0)}")
        if current_session.get("summary_text"):
            st.info(current_session["summary_text"])

        assessment = current_session.get("llm_assessment_json") or {}
        if assessment:
            with st.expander("Подробности оценки связи"):
                st.code(json_pretty(assessment), language="json")

    turns = core.storage.list_topic_turns(topic_session["topic_session_id"])
    grouped = group_topic_turns_by_side(turns)

    col_s, col_t = st.columns(2)

    with col_s:
        st.markdown("### Уточнения для студента")
        questions = grouped[SIDE_STUDENT]["question"]
        answers = grouped[SIDE_STUDENT]["answer"]

        if questions:
            st.markdown("**Вопросы**")
            for item in questions:
                st.write(f"- {item.get('question_text', '')}")
        else:
            st.info("Пока нет вопросов студенту.")

        if answers:
            with st.expander("Уже данные ответы"):
                for item in answers:
                    st.write(f"- {item.get('answer_text', '')}")

        with st.form("student_alignment_answers_form"):
            student_answers_text = st.text_area(
                "Ответы студента",
                height=180,
                help="Можно ввести один большой ответ или несколько ответов по строкам.",
            )
            student_new_files = st.file_uploader(
                "Дополнительные материалы студента",
                accept_multiple_files=True,
                key=f"student_alignment_answer_files_{lab['lab_id']}",
            )
            save_student_answers = st.form_submit_button("Сохранить ответы студента")

        if save_student_answers:
            answers_list = [line.strip() for line in student_answers_text.splitlines() if line.strip()]
            if not answers_list and not student_new_files:
                st.warning("Нужно ввести хотя бы один ответ или приложить файлы.")
            else:
                core.submit_alignment_answers(
                    topic_session_id=topic_session["topic_session_id"],
                    side=SIDE_STUDENT,
                    answers=answers_list or "",
                    uploaded_files=student_new_files,
                    rerun=False,
                )
                st.success("Ответы студента сохранены. Теперь нажмите «Запустить цикл согласования».")
                rerun()

    with col_t:
        st.markdown("### Уточнения для преподавателя")
        questions = grouped[SIDE_TEACHER]["question"]
        answers = grouped[SIDE_TEACHER]["answer"]

        if questions:
            st.markdown("**Вопросы**")
            for item in questions:
                st.write(f"- {item.get('question_text', '')}")
        else:
            st.info("Пока нет вопросов преподавателю.")

        if answers:
            with st.expander("Уже данные ответы"):
                for item in answers:
                    st.write(f"- {item.get('answer_text', '')}")

        with st.form("teacher_alignment_answers_form"):
            teacher_answers_text = st.text_area(
                "Ответы преподавателя",
                height=180,
                help="Можно ввести один большой ответ или несколько ответов по строкам.",
            )
            teacher_new_files = st.file_uploader(
                "Дополнительные материалы преподавателя",
                accept_multiple_files=True,
                key=f"teacher_alignment_answer_files_{lab['lab_id']}",
            )
            save_teacher_answers = st.form_submit_button("Сохранить ответы преподавателя")

        if save_teacher_answers:
            answers_list = [line.strip() for line in teacher_answers_text.splitlines() if line.strip()]
            if not answers_list and not teacher_new_files:
                st.warning("Нужно ввести хотя бы один ответ или приложить файлы.")
            else:
                core.submit_alignment_answers(
                    topic_session_id=topic_session["topic_session_id"],
                    side=SIDE_TEACHER,
                    answers=answers_list or "",
                    uploaded_files=teacher_new_files,
                    rerun=False,
                )
                st.success("Ответы преподавателя сохранены. Теперь нажмите «Запустить цикл согласования».")
                rerun()

    st.divider()
    render_material_cards(dashboard["topic_materials_student"], "Материалы студента по теме")
    st.divider()
    render_material_cards(dashboard["topic_materials_teacher"], "Материалы преподавателя по теме")


def render_methodics_tab(core: ProjectCore, dashboard: dict[str, Any]) -> None:
    lab = dashboard["lab"]
    agreed_spec = dashboard["agreed_spec"]

    st.subheader("Методичка, калибровка и публикация")

    if not agreed_spec:
        st.warning("Сначала нужно согласовать и зафиксировать тему задания.")
        return

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Сгенерировать методичку", type="primary"):
            try:
                core.generate_methodics(lab["lab_id"], save_as_material=True)
                st.success("Методичка создана.")
                rerun()
            except Exception as exc:
                st.error(str(exc))

    with col2:
        if st.button("Калибровать policy memory"):
            try:
                core.calibrate_policy(lab["lab_id"], persist=True)
                st.success("Policy memory обновлена.")
                rerun()
            except Exception as exc:
                st.error(str(exc))

    with col3:
        if st.button("Опубликовать задание"):
            try:
                core.publish_assignment(lab["lab_id"])
                st.success("Задание опубликовано.")
                rerun()
            except Exception as exc:
                st.error(str(exc))

    st.divider()
    render_material_cards(dashboard["methodics_materials"], "Сгенерированные методические материалы")

    st.divider()
    st.markdown("**Зафиксированная спецификация задания**")
    st.write(f"Тема: {agreed_spec.get('agreed_title', '')}")
    st.write(agreed_spec.get("agreed_description", ""))

    with st.expander("Критерии приемки"):
        st.code(json_pretty(agreed_spec.get("acceptance_criteria_json") or {}), language="json")


def render_submission_tab(core: ProjectCore, dashboard: dict[str, Any]) -> None:
    lab = dashboard["lab"]

    st.subheader("Работа студента")

    with st.form("create_submission_form"):
        student_name = st.text_input("Имя студента")
        submission_title = st.text_input("Название загружаемой работы")
        submission_description = st.text_area("Описание работы", height=150)
        submission_files = st.file_uploader(
            "Файлы работы студента",
            accept_multiple_files=True,
            key=f"submission_files_{lab['lab_id']}",
        )
        create_submission_btn = st.form_submit_button("Создать загрузку работы")

    if create_submission_btn:
        try:
            created = core.create_submission(
                lab_id=lab["lab_id"],
                student_name=student_name.strip(),
                title=submission_title.strip(),
                description=submission_description.strip(),
                uploaded_files=submission_files,
                auto_analyze=True,
            )
            st.session_state["selected_submission_id"] = created["submission"]["submission_id"]
            st.success("Работа студента загружена.")
            rerun()
        except Exception as exc:
            st.error(str(exc))

    submissions = dashboard["submissions"]
    if not submissions:
        st.info("Загрузок работ пока нет.")
        return

    submission_options = {
        f"{item.get('title') or 'Без названия'} · {item.get('student_name') or 'Студент'} · {item.get('created_at', '')}":
            item["submission_id"]
        for item in submissions
    }

    current_submission_id = st.session_state.get("selected_submission_id")
    if not current_submission_id or current_submission_id not in submission_options.values():
        current_submission_id = submissions[0]["submission_id"]
        st.session_state["selected_submission_id"] = current_submission_id

    selected_submission_label = next(
        label for label, sid in submission_options.items() if sid == st.session_state["selected_submission_id"]
    )
    selected_label = st.selectbox(
        "Выберите загрузку работы",
        options=list(submission_options.keys()),
        index=list(submission_options.keys()).index(selected_submission_label),
    )
    st.session_state["selected_submission_id"] = submission_options[selected_label]
    submission = core.storage.get_submission(st.session_state["selected_submission_id"])

    if not submission:
        st.warning("Не удалось загрузить выбранную работу.")
        return

    with st.container(border=True):
        st.markdown(f"**Название**: {submission.get('title', '') or '—'}")
        st.markdown(f"**Студент**: {submission.get('student_name', '') or '—'}")
        st.write(submission.get("description", "") or "Описание отсутствует.")
        st.caption(f"Создано: {submission.get('created_at', '')}")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Повторно проанализировать работу"):
            try:
                core.analyze_submission(lab_id=lab["lab_id"], submission_id=submission["submission_id"])
                st.success("Анализ обновлен.")
                rerun()
            except Exception as exc:
                st.error(str(exc))

    with col2:
        if st.button("Подготовить и начать защиту"):
            try:
                defense_session = core.start_defense(
                    lab_id=lab["lab_id"],
                    submission_id=submission["submission_id"],
                    pool_size=8,
                    auto_analyze_if_needed=True,
                )
                st.session_state["selected_defense_session_id"] = defense_session["defense_session_id"]
                st.success("Сессия защиты создана.")
                rerun()
            except Exception as exc:
                st.error(str(exc))

    analysis = submission.get("analysis_json") or {}
    if analysis:
        with st.expander("Анализ работы", expanded=True):
            st.code(json_pretty(analysis), language="json")

    render_material_cards(dashboard["submission_materials"], "Материалы этапа submission")


def render_defense_tab(core: ProjectCore, dashboard: dict[str, Any]) -> None:
    defense_sessions = dashboard["defense_sessions"]

    st.subheader("Защита")

    if not defense_sessions:
        st.info("Сессий защиты пока нет. Сначала создайте загрузку работы и запустите защиту.")
        return

    options = {
        f"{item.get('defense_session_id')} · {item.get('created_at', '')} · {item.get('status', '')}": item[
            "defense_session_id"]
        for item in defense_sessions
    }

    current_id = st.session_state.get("selected_defense_session_id")
    if not current_id or current_id not in options.values():
        current_id = defense_sessions[0]["defense_session_id"]
        st.session_state["selected_defense_session_id"] = current_id

    current_label = next(
        label for label, did in options.items() if did == st.session_state["selected_defense_session_id"])
    selected_label = st.selectbox(
        "Выберите сессию защиты",
        options=list(options.keys()),
        index=list(options.keys()).index(current_label),
    )
    st.session_state["selected_defense_session_id"] = options[selected_label]
    defense_session = core.storage.get_defense_session(st.session_state["selected_defense_session_id"])

    if not defense_session:
        st.warning("Не удалось открыть выбранную сессию защиты.")
        return

    with st.container(border=True):
        st.write(f"Статус: {defense_session.get('status', '')}")
        st.write(f"Создано: {defense_session.get('created_at', '')}")
        st.write(f"Обновлено: {defense_session.get('updated_at', '')}")

    summary = defense_session.get("summary_json") or {}
    score = defense_session.get("score_json") or {}

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Получить следующий вопрос", type="primary"):
            try:
                payload = core.next_defense_question(defense_session["defense_session_id"])
                st.session_state["last_question_payload"] = payload
                rerun()
            except Exception as exc:
                st.error(str(exc))

    with col2:
        if st.button("Завершить защиту"):
            try:
                core.finalize_defense(defense_session["defense_session_id"])
                st.success("Защита завершена.")
                rerun()
            except Exception as exc:
                st.error(str(exc))

    with col3:
        if st.button("Обновить данные защиты"):
            rerun()

    latest_session = core.storage.get_defense_session(defense_session["defense_session_id"]) or defense_session
    latest_plan = latest_session.get("plan_json") or {}
    pending_question = latest_plan.get("pending_question")

    if pending_question:
        st.markdown("### Текущий вопрос")
        st.info(pending_question)

        with st.form("submit_defense_answer_form"):
            answer_text = st.text_area("Ответ студента", height=220)
            submit_answer_btn = st.form_submit_button("Сохранить ответ")

        if submit_answer_btn:
            try:
                result = core.submit_defense_answer(latest_session["defense_session_id"], answer_text=answer_text)
                st.session_state["last_answer_evaluation"] = result
                st.success("Ответ сохранен и оценен.")
                rerun()
            except Exception as exc:
                st.error(str(exc))
    else:
        st.info("Сейчас активного вопроса нет. Можно запросить следующий вопрос.")

    if summary:
        with st.expander("Итоги защиты", expanded=True):
            st.code(json_pretty(summary), language="json")

    if score:
        with st.expander("Оценка", expanded=True):
            st.code(json_pretty(score), language="json")

    turns = core.storage.list_qa_turns(latest_session["defense_session_id"])
    st.markdown("### Ход защиты")
    render_qa_turns(turns)


def render_review_tab(core: ProjectCore, dashboard: dict[str, Any]) -> None:
    lab = dashboard["lab"]
    defense_sessions = dashboard["defense_sessions"]

    st.subheader("Ревью и policy memory")

    if defense_sessions:
        options = {
            f"{item.get('defense_session_id')} · {item.get('created_at', '')}": item["defense_session_id"]
            for item in defense_sessions
        }
        selected_label = st.selectbox(
            "Сессия защиты для ревью",
            options=list(options.keys()),
            index=0,
        )
        selected_defense_id = options[selected_label]
    else:
        selected_defense_id = None
        st.info("Сессий защиты пока нет. Обратную связь преподавателя можно сохранить и без них.")

    col1, col2 = st.columns(2)

    with col1:
        if selected_defense_id and st.button("Сгенерировать feedback студенту"):
            try:
                core.generate_student_feedback(lab_id=lab["lab_id"], defense_session_id=selected_defense_id)
                st.success("Обратная связь студенту сохранена.")
                rerun()
            except Exception as exc:
                st.error(str(exc))

    with col2:
        if st.button("Обновить policy memory из отзывов"):
            try:
                core.update_policy_from_teacher_feedback(lab["lab_id"])
                st.success("Policy memory обновлена.")
                rerun()
            except Exception as exc:
                st.error(str(exc))

    with st.form("teacher_feedback_form"):
        teacher_feedback_text = st.text_area(
            "Замечания преподавателя",
            height=220,
            help="Здесь можно зафиксировать замечания к качеству защиты, критериям, формулировкам, проверке результатов и т.д.",
        )
        update_policy_now = st.checkbox("Сразу обновить policy memory")
        save_teacher_feedback = st.form_submit_button("Сохранить замечания преподавателя")

    if save_teacher_feedback:
        if not teacher_feedback_text.strip():
            st.warning("Введите текст замечаний.")
        else:
            try:
                core.register_teacher_feedback(
                    lab_id=lab["lab_id"],
                    feedback_text=teacher_feedback_text.strip(),
                    defense_session_id=selected_defense_id,
                    extra={},
                    update_policy=update_policy_now,
                )
                st.success("Замечания преподавателя сохранены.")
                rerun()
            except Exception as exc:
                st.error(str(exc))

    st.divider()
    render_feedback_list(dashboard["student_feedback"], "Обратная связь студенту")
    st.divider()
    render_feedback_list(dashboard["teacher_feedback"], "Замечания преподавателя")
    st.divider()
    render_policy_items(dashboard["policy_items"])


def render_evaluation_tab(core: ProjectCore, dashboard: dict[str, Any]) -> None:
    """
    Служебная страница оценки качества генерации.

    Страница не является частью пользовательского сценария защиты. Она нужна для
    разработчика/проверяющего: создать сохраненные кейсы, прогнать их через
    LLM-оценщик и формальные эвристики, а затем получить итоговую таблицу.
    """
    lab = dashboard["lab"]
    lab_id = lab["lab_id"]

    st.subheader("Оценка генерации")
    st.info(
        "Здесь проверяется качество генерации выбранной модели. Итоговая оценка считается "
        "по формуле final_score = k * llm_score + (1 - k) * heuristic_score. "
        "По умолчанию k = 0.7. Оценки LLM 1–5 нормализуются в диапазон 0.0–1.0, "
        "а формальная оценка собирается из отдельных эвристик."
    )

    llm_client, llm_error = get_cached_llm_client()

    st.markdown("### 1. Создание кейсов из текущего состояния задания")
    st.caption(
        "Кейс сохраняет входной контекст и уже полученный результат генерации. "
        "После этого кейс можно многократно прогонять через оценку."
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("Кейс итоговой темы"):
            try:
                case = core.create_topic_final_evaluation_case_from_lab(lab_id=lab_id)
                st.success(f"Кейс создан: {case['case_id']}")
                rerun()
            except Exception as exc:
                st.error(str(exc))

    with col2:
        if st.button("Кейс уточняющих вопросов"):
            try:
                case = core.create_clarification_questions_evaluation_case_from_session(lab_id=lab_id)
                st.success(f"Кейс создан: {case['case_id']}")
                rerun()
            except Exception as exc:
                st.error(str(exc))

    with col3:
        if st.button("Кейс процесса согласования"):
            try:
                case = core.create_topic_process_evaluation_case_from_session(lab_id=lab_id)
                st.success(f"Кейс создан: {case['case_id']}")
                rerun()
            except Exception as exc:
                st.error(str(exc))

    with col4:
        defense_sessions = dashboard.get("defense_sessions") or []
        if defense_sessions:
            defense_options = {
                f"{item.get('created_at', '')} · {item.get('status', '')} · {item.get('defense_session_id')}": item[
                    "defense_session_id"]
                for item in defense_sessions
            }
            selected_defense_label = st.selectbox(
                "Сессия защиты",
                options=list(defense_options.keys()),
                key=f"evaluation_defense_session_{lab_id}",
            )
            selected_defense_id = defense_options[selected_defense_label]
            if st.button("Кейс вопросов защиты"):
                try:
                    case = core.create_defense_questions_evaluation_case_from_session(
                        defense_session_id=selected_defense_id,
                    )
                    st.success(f"Кейс создан: {case['case_id']}")
                    rerun()
                except Exception as exc:
                    st.error(str(exc))
        else:
            st.caption("Нет сессий защиты для формирования кейса.")

    with st.expander("Создать кейс вручную"):
        st.caption(
            "Ручной кейс удобен, если заказчик передал отдельные примеры: входные данные, "
            "результат генерации и пометки к ожидаемому поведению."
        )
        with st.form("manual_evaluation_case_form"):
            manual_title = st.text_input("Название кейса")
            manual_description = st.text_area("Описание / назначение", height=90)
            manual_scenario = st.selectbox(
                "Часть сценария",
                options=[
                    "topic_final",
                    "topic_clarification_questions",
                    "topic_alignment_process",
                    "defense_questions",
                ],
                format_func=scenario_label,
            )
            manual_method = st.text_input("Метод / этап", value="build_agreed_spec")
            manual_input = st.text_area(
                "input_json",
                value=json_pretty({
                    "student_topic": "",
                    "teacher_topic": "",
                }),
                height=180,
            )
            manual_output = st.text_area(
                "generated_output_json",
                value=json_pretty({"generated_topic": ""}),
                height=160,
            )
            manual_notes = st.text_area("expected_notes", height=80)
            manual_tags = st.text_input("Теги через запятую", value="manual")
            manual_active = st.checkbox("Активный кейс", value=True)
            manual_create = st.form_submit_button("Сохранить ручной кейс")

        if manual_create:
            try:
                input_json = parse_json_text(manual_input, default={})
                output_json = parse_json_text(manual_output, default={})
                tags = [item.strip() for item in manual_tags.split(",") if item.strip()]
                case = core.create_evaluation_case(
                    lab_id=lab_id,
                    scenario_part=manual_scenario,
                    method_name=manual_method.strip() or manual_scenario,
                    title=manual_title.strip() or "Ручной кейс оценки генерации",
                    description=manual_description.strip(),
                    input_json=input_json,
                    generated_output_json=output_json,
                    expected_notes=manual_notes.strip(),
                    tags=tags,
                    is_active=manual_active,
                )
                st.success(f"Кейс создан: {case['case_id']}")
                rerun()
            except Exception as exc:
                st.error(f"Не удалось создать кейс: {type(exc).__name__}: {exc}")

    with st.expander("Импорт кейсов из CSV / демонстрационный набор"):
        st.caption(
            "CSV нужен для массовой загрузки тест-кейсов: можно подготовить 20–25 строк в таблице, "
            "импортировать их в базу, затем запустить оценку и скачать итоговый отчет CSV. "
            "Встроенный файл data/evaluation_cases_default.csv содержит 20 демонстрационных кейсов: "
            "15 по согласованию темы и 5 по вопросам защиты."
        )

        default_csv_path = Path("data/evaluation_cases_default.csv")
        if default_csv_path.exists():
            default_csv_bytes = default_csv_path.read_bytes()
            st.download_button(
                "Скачать демонстрационный CSV с 20 кейсами",
                data=default_csv_bytes,
                file_name="evaluation_cases_default.csv",
                mime="text/csv",
            )
        else:
            st.warning("Файл data/evaluation_cases_default.csv не найден. Проверьте, что он добавлен в проект.")

        uploaded_cases_csv = st.file_uploader(
            "Загрузить CSV с тест-кейсами",
            type=["csv"],
            key=f"evaluation_cases_csv_upload_{lab_id}",
            help=(
                "Ожидаемые колонки: case_code, title, description, scenario_part, method_name, "
                "input_json, generated_output_json, expected_notes, tags, is_active."
            ),
        )

        col_import_1, col_import_2 = st.columns(2)
        with col_import_1:
            attach_import_to_lab = st.checkbox(
                "Привязать импортированные кейсы к текущему заданию",
                value=True,
                key=f"evaluation_attach_import_to_lab_{lab_id}",
                help="Если выключить, кейсы будут общими и появятся при выборе области проверки «Все задания».",
            )
        with col_import_2:
            update_existing_cases = st.checkbox(
                "Обновлять существующие кейсы с тем же названием",
                value=False,
                key=f"evaluation_update_existing_cases_{lab_id}",
                help="Дубликат ищется по title + scenario_part + method_name.",
            )

        col_seed, col_upload = st.columns(2)
        with col_seed:
            if st.button("Загрузить встроенные 20 кейсов в базу"):
                try:
                    summary = core.import_default_evaluation_cases_csv(
                        lab_id=lab_id,
                        attach_to_lab=attach_import_to_lab,
                        update_existing=update_existing_cases,
                    )
                    st.success(
                        "Импорт завершен: "
                        f"добавлено {summary['imported_count']}, "
                        f"обновлено {summary['updated_count']}, "
                        f"пропущено {summary['skipped_count']}, "
                        f"ошибок {summary['error_count']}."
                    )
                    if summary.get("errors"):
                        st.json(summary["errors"])
                    rerun()
                except Exception as exc:
                    st.error(f"Не удалось загрузить встроенный CSV: {type(exc).__name__}: {exc}")

        with col_upload:
            if st.button("Импортировать загруженный CSV"):
                if uploaded_cases_csv is None:
                    st.warning("Сначала выберите CSV-файл с тест-кейсами.")
                else:
                    try:
                        summary = core.import_evaluation_cases_from_csv(
                            uploaded_cases_csv.getvalue(),
                            lab_id=lab_id,
                            attach_to_lab=attach_import_to_lab,
                            update_existing=update_existing_cases,
                        )
                        st.success(
                            "Импорт завершен: "
                            f"добавлено {summary['imported_count']}, "
                            f"обновлено {summary['updated_count']}, "
                            f"пропущено {summary['skipped_count']}, "
                            f"ошибок {summary['error_count']}."
                        )
                        if summary.get("errors"):
                            st.json(summary["errors"])
                        rerun()
                    except Exception as exc:
                        st.error(f"Не удалось импортировать CSV: {type(exc).__name__}: {exc}")

        st.markdown(
            "После импорта оставьте фильтр «Все» или выберите нужную часть сценария, "
            "нажмите «Запустить оценку по выбранным кейсам», а затем скачайте итоговую таблицу CSV."
        )

    st.divider()
    st.markdown("### 2. Сохраненные кейсы")

    scope = st.radio(
        "Область проверки",
        options=["Текущее задание", "Все задания"],
        horizontal=True,
        key="evaluation_scope",
    )
    lab_filter = lab_id if scope == "Текущее задание" else None

    scenario_options = {
        "Все": None,
        "Итоговая тема": "topic_final",
        "Уточняющие вопросы": "topic_clarification_questions",
        "Процесс согласования темы": "topic_alignment_process",
        "Вопросы для защиты": "defense_questions",
    }
    scenario_selected = st.selectbox("Фильтр по части сценария", options=list(scenario_options.keys()))
    scenario_filter = scenario_options[scenario_selected]
    active_only = st.checkbox("Только активные кейсы", value=True)

    cases = core.list_evaluation_cases(
        lab_id=lab_filter,
        scenario_part=scenario_filter,
        active_only=active_only,
        limit=500,
    )

    case_rows = [
        {
            "case_id": item.get("case_id"),
            "title": item.get("title"),
            "scenario_part": scenario_label(item.get("scenario_part", "")),
            "method_name": item.get("method_name"),
            "active": item.get("is_active"),
            "updated_at": item.get("updated_at"),
        }
        for item in cases
    ]

    if case_rows:
        st.dataframe(case_rows, use_container_width=True, hide_index=True)
    else:
        st.info("По выбранным фильтрам кейсов пока нет.")

    if cases:
        with st.expander("Просмотр / управление одним кейсом"):
            case_options = {
                f"{item.get('title') or 'Без названия'} · {item.get('case_id')}": item["case_id"]
                for item in cases
            }
            selected_case_label = st.selectbox("Кейс", options=list(case_options.keys()))
            selected_case_id = case_options[selected_case_label]
            selected_case = next(item for item in cases if item["case_id"] == selected_case_id)

            col_case_1, col_case_2, col_case_3 = st.columns(3)
            with col_case_1:
                if st.button("Включить / выключить кейс"):
                    try:
                        core.update_evaluation_case(
                            selected_case_id,
                            is_active=not bool(selected_case.get("is_active")),
                        )
                        st.success("Статус кейса обновлен.")
                        rerun()
                    except Exception as exc:
                        st.error(str(exc))
            with col_case_2:
                if st.button("Прогнать только этот кейс"):
                    try:
                        use_judge = bool(st.session_state.get("evaluation_use_llm_judge", llm_client is not None))
                        k_value = float(st.session_state.get("evaluation_k", 0.7))
                        result = core.run_evaluation_case(
                            selected_case_id,
                            k=k_value,
                            use_llm_judge=use_judge,
                            save_result=True,
                        )
                        st.session_state["selected_evaluation_run_id"] = result["run_id"]
                        st.success("Кейс оценен.")
                        rerun()
                    except Exception as exc:
                        st.error(str(exc))
            with col_case_3:
                if st.button("Удалить кейс", type="secondary"):
                    try:
                        core.delete_evaluation_case(selected_case_id)
                        st.success("Кейс удален.")
                        rerun()
                    except Exception as exc:
                        st.error(str(exc))

            st.markdown("**input_json**")
            st.code(json_pretty(selected_case.get("input_json") or {}), language="json")
            st.markdown("**generated_output_json**")
            st.code(json_pretty(selected_case.get("generated_output_json") or {}), language="json")
            if selected_case.get("expected_notes"):
                st.markdown("**expected_notes**")
                st.write(selected_case.get("expected_notes"))

    st.divider()
    st.markdown("### 3. Запуск оценки")

    col_run_1, col_run_2, col_run_3 = st.columns([1, 1, 2])
    with col_run_1:
        k = st.slider(
            "k — доверие к LLM-оценке",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.05,
            key="evaluation_k",
            help="final_score = k * llm_score + (1 - k) * heuristic_score",
        )
    with col_run_2:
        use_llm_judge = st.checkbox(
            "Использовать LLM-оценщик",
            value=llm_client is not None,
            key="evaluation_use_llm_judge",
        )
        if use_llm_judge and llm_client is None:
            st.warning("LLM-клиент не инициализирован. Оценщик не сможет выполниться.")
            if llm_error:
                with st.expander("Причина ошибки LLM"):
                    st.code(llm_error)
    with col_run_3:
        run_title = st.text_input("Название запуска", value="Оценка генерации")

    if st.button("Запустить оценку по выбранным кейсам", type="primary"):
        try:
            if not cases:
                st.warning("Сначала создайте хотя бы один кейс или измените фильтры.")
            else:
                result = core.run_evaluation_suite(
                    lab_id=lab_filter,
                    scenario_part=scenario_filter,
                    active_only=active_only,
                    k=k,
                    use_llm_judge=use_llm_judge,
                    title=run_title.strip() or "Оценка генерации",
                    limit=500,
                )
                st.session_state["selected_evaluation_run_id"] = result["run"]["run_id"]
                if result.get("errors"):
                    st.warning(f"Оценка завершена с ошибками: {len(result['errors'])}")
                else:
                    st.success("Оценка завершена.")
                rerun()
        except Exception as exc:
            st.error(f"Не удалось запустить оценку: {type(exc).__name__}: {exc}")

    st.divider()
    st.markdown("### 4. Итоговая таблица")

    runs = core.list_evaluation_runs(limit=50)
    if not runs:
        st.info("Запусков оценки пока нет.")
        return

    run_options = {
        f"{item.get('created_at', '')} · {item.get('title') or 'Оценка'} · {item.get('status', '')} · {item.get('run_id')}":
            item["run_id"]
        for item in runs
    }

    current_run_id = st.session_state.get("selected_evaluation_run_id")
    if not current_run_id or current_run_id not in run_options.values():
        current_run_id = runs[0]["run_id"]
        st.session_state["selected_evaluation_run_id"] = current_run_id

    current_run_label = next(
        label for label, run_id in run_options.items() if run_id == st.session_state["selected_evaluation_run_id"]
    )
    selected_run_label = st.selectbox(
        "Запуск оценки",
        options=list(run_options.keys()),
        index=list(run_options.keys()).index(current_run_label),
    )
    st.session_state["selected_evaluation_run_id"] = run_options[selected_run_label]

    try:
        report = core.get_evaluation_run_report(st.session_state["selected_evaluation_run_id"])
    except Exception as exc:
        st.error(str(exc))
        return

    run = report["run"]
    report_rows = report.get("report_rows") or []
    results = report.get("results") or []

    col_metric_1, col_metric_2, col_metric_3, col_metric_4 = st.columns(4)
    col_metric_1.metric("Статус", run.get("status", ""))
    col_metric_2.metric("k", run.get("k", 0.7))
    col_metric_3.metric("Кейсов", len(report_rows))
    final_scores = [row.get("final_score") for row in report_rows if row.get("final_score") is not None]
    avg_score = sum(float(x) for x in final_scores) / len(final_scores) if final_scores else 0.0
    col_metric_4.metric("Средний final_score", round(avg_score, 4))

    if report_rows:
        st.dataframe(report_rows, use_container_width=True, hide_index=True)

        col_download_1, col_download_2 = st.columns(2)
        with col_download_1:
            st.download_button(
                "Скачать таблицу CSV",
                data=report_rows_to_csv_bytes(report_rows),
                file_name=f"evaluation_report_{run['run_id']}.csv",
                mime="text/csv",
            )
        with col_download_2:
            st.download_button(
                "Скачать подробности JSON",
                data=json_pretty(report).encode("utf-8"),
                file_name=f"evaluation_report_{run['run_id']}.json",
                mime="application/json",
            )

        with st.expander("Подробные результаты и эвристики"):
            for idx, item in enumerate(results, start=1):
                st.markdown(f"**Результат {idx}: {item.get('case_id') or 'без case_id'}**")
                st.caption(
                    f"Сценарий: {scenario_label(item.get('scenario_part', ''))} · "
                    f"Метод: {item.get('method_name', '')} · "
                    f"final_score: {item.get('final_score')}"
                )
                result_json = item.get("result_json") or {}
                st.code(json_pretty(result_json), language="json")
    else:
        st.info("В выбранном запуске нет результатов.")


def main() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="🎓",
        layout="wide",
    )
    ensure_state_defaults()

    core = get_core()

    st.title(APP_TITLE)
    st.caption(APP_CAPTION)

    llm_client, llm_error = get_cached_llm_client()
    if llm_client is None:
        st.warning(
            "LLM-клиент не инициализирован. Система работает в эвристическом режиме. "
            "Проверьте .env, ключ ProxyAPI и нажмите «Сбросить кэш ядра / LLM» в боковой панели."
        )
        if llm_error:
            with st.expander("Ошибка инициализации LLM"):
                st.code(llm_error)
    else:
        st.success("LLM-клиент инициализирован. Эвристический режим будет использоваться только как fallback.")

    selected_lab = render_sidebar(core)

    if not selected_lab:
        st.info(
            "Сначала создайте новое задание через боковую панель. "
            "После этого можно будет согласовать тему, подготовить методичку, "
            "загрузить работу студента и провести защиту."
        )
        return

    dashboard = core.get_lab_dashboard(selected_lab["lab_id"])

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
        [
            "Обзор",
            "Согласование темы",
            "Методичка и публикация",
            "Работа студента",
            "Защита",
            "Ревью и policy",
            "Оценка генерации",
        ]
    )

    with tab1:
        render_overview(core, dashboard)

    with tab2:
        render_alignment_tab(core, dashboard)

    with tab3:
        render_methodics_tab(core, dashboard)

    with tab4:
        render_submission_tab(core, dashboard)

    with tab5:
        render_defense_tab(core, dashboard)

    with tab6:
        render_review_tab(core, dashboard)

    with tab7:
        render_evaluation_tab(core, dashboard)


if __name__ == "__main__":
    # Если runtime существует
    if runtime.exists():
        # Вызываем функцию main()
        main()
    # Если runtime не существует
    else:
        # Устанавливаем аргументы командной строки
        sys.argv = ["streamlit", "run", sys.argv[0]]
        # Выходим из программы с помощью функции main() из модуля stcli
        sys.exit(stcli.main())
