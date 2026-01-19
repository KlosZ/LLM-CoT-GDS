import os
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

import core
from storage import Storage

JsonDict = Dict[str, Any]


# Minimal .env loader (no python-dotenv dependency)

def load_env_file(env_path: str = ".env") -> None:
    p = Path(env_path)
    if not p.exists() or not p.is_file():
        return

    for raw_line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


# Streamlit helpers

@st.cache_resource
def get_services() -> core.AppServices:
    load_env_file(".env")
    storage = Storage()
    return core.create_services(storage)


def _pretty_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def _render_question(q: JsonDict) -> None:
    st.markdown(f"**Тема:** {q.get('topic', '')}  \n**Сложность:** `{q.get('difficulty', '')}`")
    st.markdown(f"**Вопрос:** {q.get('question', '')}")

    with st.expander("Ожидаемый ответ (outline)"):
        outline = q.get("expected_answer_outline") or []
        if isinstance(outline, list) and outline:
            st.markdown("\n".join([f"- {x}" for x in outline]))
        else:
            st.write("—")

    with st.expander("Критерии оценивания"):
        crit = q.get("evaluation_criteria") or []
        if isinstance(crit, list) and crit:
            st.markdown("\n".join([f"- {x}" for x in crit]))
        else:
            st.write("—")

    sources = q.get("sources") or []
    if sources:
        with st.expander("Источники (materials refs)"):
            st.markdown("\n".join([f"- {s}" for s in sources]))

    tags = q.get("tags") or []
    if tags:
        st.caption("tags: " + ", ".join(map(str, tags)))


def _list_sessions_for_lab(storage: Storage, lab_id: str) -> List[JsonDict]:
    # No public method in storage.py for sessions list; for app-level UI this is OK.
    with storage._connect() as conn:  # pylint: disable=protected-access
        rows = conn.execute(
            "SELECT * FROM defense_sessions WHERE lab_id=? ORDER BY started_at DESC;",
            (lab_id,),
        ).fetchall()
    out: List[JsonDict] = []
    for r in rows:
        out.append(
            {
                "session_id": r["session_id"],
                "submission_id": r["submission_id"],
                "student_label": r["student_label"],
                "status": r["status"],
                "started_at": r["started_at"],
                "finished_at": r["finished_at"],
                "policy_version": r["policy_version"],
            }
        )
    return out


def _lab_selector(svc: core.AppServices) -> Optional[str]:
    labs = svc.storage.list_labs()

    st.sidebar.subheader("Лабораторные")
    if labs:
        options = [f"{lab.title}  ({lab.lab_id})" for lab in labs]
        idx = 0
        if st.session_state.get("lab_id"):
            for i, lab in enumerate(labs):
                if lab.lab_id == st.session_state["lab_id"]:
                    idx = i
                    break
        sel = st.sidebar.selectbox("Выберите лабораторную", options, index=idx)
        lab_id = labs[options.index(sel)].lab_id
        st.session_state["lab_id"] = lab_id
    else:
        st.sidebar.info("Пока нет лабораторных. Создайте первую.")
        st.session_state["lab_id"] = None

    st.sidebar.divider()
    st.sidebar.markdown("**Создать новую**")
    new_title = st.sidebar.text_input("Название", value="", placeholder="Например: Защита ЛР по LLM")
    if st.sidebar.button("Создать", use_container_width=True, disabled=not new_title.strip()):
        lab = svc.storage.create_lab(new_title.strip())
        st.session_state["lab_id"] = lab.lab_id
        st.rerun()

    return st.session_state.get("lab_id")


def _teacher_materials_ui(svc: core.AppServices, lab_id: str) -> None:
    st.subheader("Материалы лабораторной")

    col1, col2 = st.columns([1, 1])
    with col1:
        kind = st.selectbox(
            "Тип материала (kind)",
            ["goal", "task", "theory", "assignment", "example", "report_structure", "other"],
            index=2,
        )
    with col2:
        st.caption(
            "Совет: загружайте отдельными файлами цель/задачи/теорию/структуру отчёта — так лучше покрытие вопросов.")

    up = st.file_uploader(
        "Загрузить материал (pdf/docx/txt/md/ipynb/zip)",
        type=["pdf", "docx", "txt", "md", "ipynb", "zip", "json", "csv", "tsv"],
        accept_multiple_files=False,
    )

    if st.button("Добавить материал", disabled=(up is None), use_container_width=True):
        try:
            data = up.getvalue()
            res = core.teacher_upload_material(
                svc,
                lab_id=lab_id,
                kind=kind,
                filename=up.name,
                data=data,
                mime=getattr(up, "type", None),
            )
            st.success(f"Материал добавлен: {res['material_id']} (извлечено символов: {res['extracted_chars']})")
            st.rerun()
        except Exception as e:
            st.error(f"Ошибка добавления материала: {e}")

    st.divider()

    mats = svc.storage.list_materials(lab_id)
    if not mats:
        st.info("Материалы не загружены.")
        return

    for m in mats:
        with st.expander(f"{m.kind} — {m.filename}  ({m.material_id})", expanded=False):
            cols = st.columns([1, 1, 1])
            cols[0].write(f"**MIME:** {m.mime}")
            cols[1].write(f"**Создано:** {m.created_at}")
            cols[2].write(f"**Файл:** {m.original_path}")

            preview = Path(m.extracted_text_path).read_text(encoding="utf-8", errors="replace") if Path(
                m.extracted_text_path).exists() else ""
            if preview.strip():
                st.text_area("Извлечённый текст (preview)", value=preview[:6000], height=250)
            else:
                st.warning("Извлечённый текст пустой (возможно, pdf скан/картинки — нужен OCR, его тут нет).")

            if st.button(f"Удалить материал {m.material_id}", type="secondary"):
                try:
                    svc.storage.delete_material(m.material_id, delete_files=True)
                    st.success("Удалено.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Ошибка удаления: {e}")


def _teacher_generate_ui(svc: core.AppServices, lab_id: str) -> None:
    st.subheader("Генерация методички и банка вопросов")

    if st.button("Сгенерировать (методичка + рубрика + банк)", use_container_width=True):
        try:
            out = core.teacher_generate_methodics(svc, lab_id=lab_id)
            st.success("Готово. Результат сохранён в lab.config['generated'].")
            st.session_state["last_generated"] = out
        except Exception as e:
            st.error(f"Ошибка генерации: {e}")

    gen = svc.storage.get_lab(lab_id).config.get("generated", {})
    if not gen:
        st.info("Пока ничего не сгенерировано.")
        return

    st.divider()
    st.markdown("### Методические указания")
    methodics = gen.get("methodics")
    if methodics:
        st.text_area("methodics (JSON)", value=_pretty_json(methodics), height=300)
    else:
        st.warning("methodics отсутствует.")

    st.markdown("### Рубрика")
    rubric = gen.get("rubric")
    if rubric:
        st.text_area("rubric (JSON)", value=_pretty_json(rubric), height=240)
    else:
        st.warning("rubric отсутствует.")

    st.markdown("### Банк вопросов")
    bank = gen.get("question_bank") or []
    st.write(f"Всего вопросов: **{len(bank)}**")
    if bank:
        with st.expander("Показать банк (первые 30)", expanded=False):
            for q in bank[:30]:
                st.markdown("---")
                _render_question(q)

    warnings = gen.get("quality_warnings") or []
    if warnings:
        st.warning("Предупреждения по качеству материалов:")
        st.markdown("\n".join([f"- {w}" for w in warnings]))


def _teacher_calibration_ui(svc: core.AppServices, lab_id: str) -> None:
    st.subheader("Калибровка вопросов под преподавателя (policy memory)")

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        round_index = st.number_input("Раунд", min_value=1, value=int(st.session_state.get("cal_round", 1)), step=1)
    with col2:
        top_n = st.number_input("TOP-N", min_value=3, value=int(st.session_state.get("cal_topn", 10)), step=1)
    with col3:
        if st.button("Получить batch", use_container_width=True):
            try:
                batch = core.teacher_get_calibration_batch(svc, lab_id=lab_id, round_index=int(round_index),
                                                           top_n=int(top_n))
                st.session_state["cal_batch"] = batch
                st.session_state["cal_round"] = int(round_index)
                st.session_state["cal_topn"] = int(top_n)
                st.success("Batch получен.")
            except Exception as e:
                st.error(f"Ошибка получения batch: {e}")

    batch = st.session_state.get("cal_batch")
    if not batch:
        st.info("Нажмите «Получить batch», чтобы увидеть top-N вопросов.")
        return

    questions = batch.get("questions") or []
    st.write(f"Batch: **{batch.get('batch_id', '')}** | вопросов: **{len(questions)}**")

    notes = batch.get("notes_for_teacher") or []
    if notes:
        with st.expander("Заметки для преподавателя"):
            st.markdown("\n".join([f"- {n}" for n in notes]))

    st.divider()
    st.markdown("### Разметка вопросов (good/bad + причины)")

    reason_tags = [
        "good_alignment",
        "good_depth",
        "clear_wording",
        "ambiguous",
        "off_topic",
        "too_easy",
        "too_hard",
        "wrong_expected_answer",
        "repeats",
    ]

    labeled: List[JsonDict] = []
    for i, q in enumerate(questions):
        st.markdown("---")
        st.markdown(f"#### Вопрос {i + 1}")
        _render_question(q)

        c1, c2 = st.columns([1, 2])
        with c1:
            label = st.radio(
                f"Оценка (Q{i})",
                ["skip", "good", "bad"],
                index=0,
                horizontal=True,
                key=f"cal_label_{i}",
            )
        with c2:
            tags = st.multiselect(f"Теги причин (Q{i})", reason_tags, default=[], key=f"cal_tags_{i}")

        comment = st.text_input(f"Комментарий (Q{i})", value="", key=f"cal_comment_{i}")
        if label != "skip":
            labeled.append({"question": q, "label": label, "reason_tags": tags, "comment": comment})

    st.divider()
    col_a, col_b = st.columns([1, 1])
    with col_a:
        if st.button("Сохранить разметку в policy memory", use_container_width=True, disabled=(len(labeled) == 0)):
            try:
                stats = core.teacher_apply_calibration_labels(svc, lab_id=lab_id, labeled_questions=labeled)
                st.success(f"Сохранено: {stats}")
            except Exception as e:
                st.error(f"Ошибка сохранения policy: {e}")

    with col_b:
        if st.button("Опубликовать версию (publish)", use_container_width=True):
            try:
                v = core.teacher_publish_lab(svc, lab_id=lab_id)
                st.success(f"Опубликована версия: {v}")
            except Exception as e:
                st.error(f"Ошибка publish: {e}")

    st.caption("Publish увеличивает published_version. Защиты будут запоминать, какая версия политики использовалась.")


def _teacher_sessions_ui(svc: core.AppServices, lab_id: str) -> None:
    st.subheader("Сессии защит и ревью")

    sessions = _list_sessions_for_lab(svc.storage, lab_id)
    if not sessions:
        st.info("Пока нет сессий защиты по этой лабораторной.")
        return

    opts = [f"{s['started_at']} | {s['student_label']} | {s['status']} | {s['session_id']}" for s in sessions]
    sel = st.selectbox("Выберите сессию", opts, index=0)
    sess_id = sessions[opts.index(sel)]["session_id"]

    sess = svc.storage.get_defense_session(sess_id)
    st.write(f"**session_id:** `{sess.session_id}`  \n**submission_id:** `{sess.submission_id}`  \n"
             f"**status:** `{sess.status}`  \n**policy_version:** `{sess.policy_version}`")

    turns = svc.storage.list_turns(sess_id)
    if not turns:
        st.warning("В сессии нет вопросов.")
    else:
        st.markdown("### Ход защиты")
        for t in turns:
            q = t.get("question") or {}
            ev = t.get("system_eval") or {}
            st.markdown("---")
            st.markdown(f"**Turn {t.get('idx')}** — `{t.get('turn_id')}`")
            _render_question(q)
            st.markdown("**Ответ студента:**")
            st.write(t.get("answer_text") or "—")

            if ev:
                with st.expander("Оценка системы (system_eval)"):
                    st.json(ev)

            # Teacher feedback inline
            st.markdown("**Оценка преподавателя:**")
            c1, c2, c3 = st.columns([1, 1, 2])

            with c1:
                q_label = st.selectbox(
                    f"Вопрос системы (turn {t['idx']})",
                    ["skip", "good", "bad"],
                    index=0,
                    key=f"tf_q_label_{t['turn_id']}",
                )
            with c2:
                a_label = st.selectbox(
                    f"Ответ студента (turn {t['idx']})",
                    ["skip", "good", "bad"],
                    index=0,
                    key=f"tf_a_label_{t['turn_id']}",
                )
            with c3:
                reason = st.multiselect(
                    f"Теги причин (turn {t['idx']})",
                    ["off_topic", "ambiguous", "too_easy", "too_hard", "good_depth", "good_alignment", "wrong_eval",
                     "other"],
                    default=[],
                    key=f"tf_tags_{t['turn_id']}",
                )
            comment = st.text_input(f"Комментарий преподавателя (turn {t['idx']})", value="",
                                    key=f"tf_comment_{t['turn_id']}")

            col_x, col_y = st.columns([1, 1])
            with col_x:
                if st.button(f"Сохранить фидбек по TURN {t['idx']}", key=f"save_tfb_{t['turn_id']}"):
                    try:
                        if q_label != "skip":
                            core.teacher_rate_question(
                                svc,
                                session_id=sess_id,
                                turn_id=t["turn_id"],
                                label=q_label,
                                reason_tags=reason,
                                comment=comment,
                            )
                        if a_label != "skip":
                            core.teacher_rate_answer(
                                svc,
                                session_id=sess_id,
                                turn_id=t["turn_id"],
                                label=a_label,
                                reason_tags=reason,
                                comment=comment,
                            )
                        st.success("Сохранено.")
                    except Exception as e:
                        st.error(f"Ошибка сохранения teacher feedback: {e}")
            with col_y:
                if st.button(f"Экспорт лога JSON (сессия)", key=f"export_{t['turn_id']}"):
                    try:
                        path = core.export_defense_log_json(svc, session_id=sess_id)
                        st.success(f"Экспортировано: {path}")
                    except Exception as e:
                        st.error(f"Ошибка экспорта: {e}")

    st.divider()
    st.markdown("### Итоговое ревью и обновление policy memory")

    teacher_summary = st.text_area(
        "Итоговый комментарий/оценка (teacher_summary JSON или текст)",
        value="",
        height=120,
        key="teacher_summary_text",
    )
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Сохранить teacher_summary", use_container_width=True):
            try:
                # allow plain text or json
                ts: JsonDict
                txt = teacher_summary.strip()
                if not txt:
                    ts = {}
                else:
                    try:
                        ts = json.loads(txt)
                        if not isinstance(ts, dict):
                            ts = {"text": txt}
                    except Exception:
                        ts = {"text": txt}
                core.teacher_finalize_session_review(svc, session_id=sess_id, teacher_summary=ts)
                st.success("Сохранено.")
            except Exception as e:
                st.error(f"Ошибка teacher_summary: {e}")

    with col2:
        if st.button("Предложить и применить обновление policy", use_container_width=True):
            try:
                out = core.teacher_update_policy_from_session(svc, lab_id=lab_id, session_id=sess_id, apply=True)
                st.success(f"Policy обновлена: {out.get('stats')}")
                with st.expander("Suggestion (JSON)"):
                    st.json(out.get("suggestion") or {})
            except Exception as e:
                st.error(f"Ошибка обновления policy: {e}")


def _student_ui(svc: core.AppServices, lab_id: str) -> None:
    st.subheader("Студент: загрузка и защита")

    # --- Upload submission
    st.markdown("### 1) Загрузить работу")
    student_label = st.text_input("ФИО / метка студента", value=st.session_state.get("student_label", "student"))
    st.session_state["student_label"] = student_label

    up = st.file_uploader(
        "Файл работы (pdf/docx/txt/md/ipynb/zip)",
        type=["pdf", "docx", "txt", "md", "ipynb", "zip", "json", "csv", "tsv"],
        accept_multiple_files=False,
        key="student_submission_uploader",
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Сохранить работу", use_container_width=True, disabled=(up is None)):
            try:
                res = core.student_upload_submission(
                    svc,
                    lab_id=lab_id,
                    filename=up.name,
                    data=up.getvalue(),
                    mime=getattr(up, "type", None),
                    student_id=student_label.strip() or None,
                )
                st.session_state["submission_id"] = res["submission_id"]
                st.success(f"Сохранено: {res['submission_id']} (извлечено символов: {res['extracted_chars']})")
            except Exception as e:
                st.error(f"Ошибка загрузки: {e}")
    with col2:
        st.caption(f"Текущая submission_id: `{st.session_state.get('submission_id', '—')}`")

    # --- Start defense
    st.markdown("### 2) Старт защиты")
    sub_id = st.session_state.get("submission_id")
    if st.button("Начать защиту (анализ + подготовка вопросов)", use_container_width=True, disabled=not sub_id):
        try:
            out = core.student_start_defense(
                svc,
                lab_id=lab_id,
                submission_id=sub_id,
                student_label=student_label.strip() or "student",
                max_pool=30,
            )
            st.session_state["session_id"] = out["session_id"]
            st.session_state["active_turn_id"] = None
            st.session_state["last_analysis"] = out.get("submission_analysis")
            st.success(f"Сессия начата: {out['session_id']} | пул вопросов: {out['question_pool_size']}")
        except Exception as e:
            st.error(f"Ошибка старта: {e}")

    sess_id = st.session_state.get("session_id")
    if not sess_id:
        st.info("Чтобы перейти к защите, загрузите работу и нажмите «Начать защиту».")
        return

    # show analysis
    analysis = st.session_state.get("last_analysis")
    if analysis:
        with st.expander("Авто-анализ работы (submission_analysis)", expanded=False):
            st.json(analysis)

    # --- Defense interaction
    st.markdown("### 3) Прохождение защиты (вопросы → ответы)")

    try:
        sess = svc.storage.get_defense_session(sess_id)
        st.write(f"**session_id:** `{sess.session_id}`  \n**status:** `{sess.status}`")
    except Exception as e:
        st.error(f"Не удалось загрузить сессию: {e}")
        return

    if sess.status != "in_progress":
        st.warning("Сессия уже завершена. Можно экспортировать лог и оставить feedback.")
    else:
        col_a, col_b = st.columns([1, 1])
        with col_a:
            if st.button("Получить следующий вопрос", use_container_width=True):
                try:
                    out = core.student_get_next_question(svc, session_id=sess_id)
                    if out.get("turn_id") is None:
                        st.session_state["active_turn_id"] = None
                        st.info("План вопросов выполнен. Сессия завершена автоматически.")
                    else:
                        st.session_state["active_turn_id"] = out["turn_id"]
                        st.session_state["last_question_out"] = out
                except Exception as e:
                    st.error(f"Ошибка выбора вопроса: {e}")

        with col_b:
            if st.button("Завершить защиту сейчас", use_container_width=True, type="secondary"):
                try:
                    summary = core.student_finish_defense(svc, session_id=sess_id)
                    st.success("Сессия завершена.")
                    with st.expander("Итог (system_summary)"):
                        st.json(summary)
                except Exception as e:
                    st.error(f"Ошибка завершения: {e}")

        # Show current question
        active_turn_id = st.session_state.get("active_turn_id")
        if active_turn_id:
            turns = svc.storage.list_turns(sess_id)
            turn = next((t for t in turns if t.get("turn_id") == active_turn_id), None)
            if not turn:
                st.warning("Активный turn не найден.")
            else:
                q = turn.get("question") or {}
                st.markdown("---")
                st.markdown("#### Текущий вопрос")
                _render_question(q)

                with st.form(key="answer_form", clear_on_submit=True):
                    ans = st.text_area("Ваш ответ", height=140, key="student_answer_text")
                    submitted = st.form_submit_button("Отправить ответ", use_container_width=True)

                if submitted:
                    if not ans.strip():
                        st.warning("Введите ответ.")
                    else:
                        try:
                            out = core.student_submit_answer(
                                svc,
                                session_id=sess_id,
                                turn_id=active_turn_id,
                                answer_text=ans,
                                allow_followup=True,
                            )
                            st.session_state["last_eval"] = out.get("system_eval")

                            if out.get("followup_turn_id"):
                                st.session_state["active_turn_id"] = out["followup_turn_id"]
                                st.info("Создан follow-up вопрос. Ответьте на него.")
                            else:
                                st.session_state["active_turn_id"] = None
                                st.success("Ответ принят. Можно запросить следующий вопрос.")

                            st.rerun()
                        except Exception as e:
                            st.error(f"Ошибка отправки ответа: {e}")

        last_eval = st.session_state.get("last_eval")
        if last_eval:
            with st.expander("Последняя оценка системы (answer_eval)"):
                st.json(last_eval)

    st.divider()
    st.markdown("### 4) Экспорт лога и обратная связь")

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Экспортировать лог защиты (JSON)", use_container_width=True):
            try:
                path = core.export_defense_log_json(svc, session_id=sess_id)
                st.success(f"Экспортировано: {path}")
            except Exception as e:
                st.error(f"Ошибка экспорта: {e}")

    with col2:
        rating = st.slider("Оценка процедуры (1–5)", min_value=1, max_value=5, value=4)
        comment = st.text_input("Комментарий", value="")
        tags = st.multiselect("Теги", ["понятно", "сложно", "слишком_легко", "по_делу", "оффтоп", "долго", "быстро"],
                              default=[])
        if st.button("Отправить feedback", use_container_width=True):
            try:
                fid = core.student_add_feedback(svc, session_id=sess_id, rating=rating, comment=comment, tags=tags)
                st.success(f"Feedback сохранён: {fid}")
            except Exception as e:
                st.error(f"Ошибка feedback: {e}")


# Main app

def main() -> None:
    st.set_page_config(page_title="LLM Defense Automation", layout="wide")

    svc = get_services()

    st.title("Система автоматизации защиты лабораторных/практических работ (LLM)")

    # Sidebar role selector
    role = st.sidebar.radio("Роль", ["Преподаватель", "Студент"], index=0)
    lab_id = _lab_selector(svc)

    if not lab_id:
        st.info("Создайте лабораторную в левом меню.")
        return

    lab = svc.storage.get_lab(lab_id)
    st.caption(
        f"Текущая лабораторная: **{lab.title}**  | lab_id: `{lab.lab_id}`" +
        "| published_version: `{lab.published_version}`"
    )

    if role == "Преподаватель":
        tabs = st.tabs(["Материалы", "Генерация", "Калибровка", "Сессии и ревью"])
        with tabs[0]:
            _teacher_materials_ui(svc, lab_id)
        with tabs[1]:
            _teacher_generate_ui(svc, lab_id)
        with tabs[2]:
            _teacher_calibration_ui(svc, lab_id)
        with tabs[3]:
            _teacher_sessions_ui(svc, lab_id)
    else:
        _student_ui(svc, lab_id)


if __name__ == "__main__":
    main()
