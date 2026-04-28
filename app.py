from __future__ import annotations

import json
import os
from typing import Any, Optional

import streamlit as st
from dotenv import load_dotenv  # type: ignore

load_dotenv()

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


def try_build_llm_client() -> Any:
    try:
        import llm  # type: ignore
    except Exception:
        return None

    factory_names = [
        "create_llm_client",
        "build_llm_client",
        "get_llm_client",
    ]

    for name in factory_names:
        factory = getattr(llm, name, None)
        if callable(factory):
            try:
                return factory()
            except TypeError:
                try:
                    return factory(os.environ)
                except Exception:
                    pass
            except Exception:
                pass

    client_cls = getattr(llm, "LLMClient", None)
    if client_cls:
        try:
            return client_cls()
        except Exception:
            pass

    return None


@st.cache_resource(show_spinner=False)
def get_core() -> ProjectCore:
    db_path = os.getenv("APP_DB_PATH", "data/app.sqlite3")
    upload_dir = os.getenv("APP_UPLOAD_DIR", "uploads")
    llm_client = try_build_llm_client()

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
            student_description = st.text_area("Описание темы студента", value=current_student.get("description", ""),
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
                                               value=current_teacher.get("description", ""), height=160)
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

    if core.llm_client is None:
        st.warning(
            "LLM-клиент не инициализирован. Система работает в эвристическом режиме. "
            "Это допустимо для теста, но логика согласования будет проще."
        )

    selected_lab = render_sidebar(core)

    if not selected_lab:
        st.info(
            "Сначала создайте новое задание через боковую панель. "
            "После этого можно будет согласовать тему, подготовить методичку, "
            "загрузить работу студента и провести защиту."
        )
        return

    dashboard = core.get_lab_dashboard(selected_lab["lab_id"])

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        [
            "Обзор",
            "Согласование темы",
            "Методичка и публикация",
            "Работа студента",
            "Защита",
            "Ревью и policy",
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


if __name__ == "__main__":
    main()
