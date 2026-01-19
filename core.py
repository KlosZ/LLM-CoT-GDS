"""
Высокоуровневые сценарии (use-cases) для Streamlit-приложения:

Роль "Преподаватель":
- загрузка материалов
- генерация методички/рубрики/банка вопросов
- калибровка (top-N) + фиксация предпочтений (policy memory)
- просмотр логов защит, ревью, обновление policy по обратной связи

Роль "Студент":
- загрузка работы
- авто-анализ
- старт сессии защиты
- цикл 'вопрос -> ответ -> оценка' (+ опциональные follow-up)
- отправка лога преподавателю (в рамках проекта: экспорт JSON)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from storage import Storage
from agents import (
    IngestAgent,
    MethodicsAgent,
    DefenseAgent,
    FeedbackAgent,
    build_asked_turns_summary,
    remove_selected_question,
)

JsonDict = Dict[str, Any]


# Internal helpers for session system_summary


def _safe_json(obj: Any) -> Any:
    """Best-effort: ensure json-serializable."""
    try:
        json.dumps(obj, ensure_ascii=False)
        return obj
    except Exception:
        return {"_non_serializable": str(obj)}


def _get_session_system_summary(storage: Storage, session_id: str) -> JsonDict:
    sess = storage.get_defense_session(session_id)
    return dict(sess.system_summary or {})


def _set_session_system_summary(storage: Storage, session_id: str, summary: JsonDict) -> None:
    sess = storage.get_defense_session(session_id)
    merged = dict(sess.system_summary or {})
    merged.update(_safe_json(summary) or {})
    # storage.finish_defense_session overwrites status; we need update-only:
    with storage._tx() as conn:  # pylint: disable=protected-access
        conn.execute(
            "UPDATE defense_sessions SET system_summary_json=? WHERE session_id=?;",
            (json.dumps(merged, ensure_ascii=False), session_id),
        )


def _count_turns_by_difficulty(turns: List[JsonDict]) -> JsonDict:
    c = {"easy": 0, "medium": 0, "hard": 0, "followup": 0}
    for t in turns:
        q = (t.get("question") or {})
        d = q.get("difficulty")
        if d in ("easy", "medium", "hard"):
            c[d] += 1
        if "followup" in (q.get("tags") or []):
            c["followup"] += 1
    return c


def _plan_done(plan: JsonDict, counts: JsonDict) -> bool:
    for k in ("easy", "medium", "hard"):
        if int(counts.get(k, 0)) < int(plan.get(k, 0)):
            return False
    return True


def _make_followup_question_obj(
        *,
        base_question: JsonDict,
        followup_text: str,
        missing_points: List[str],
        idx: int,
) -> JsonDict:
    """
    Заворачиваем follow-up строку в объект, совместимый с QUESTION_SCHEMA.
    (Без генерации новой схемы через LLM - быстро и детерминированно.)
    """
    base_id = base_question.get("question_id") or "q"
    base_topic = base_question.get("topic") or "followup"
    base_sources = base_question.get("sources") or []
    base_diff = base_question.get("difficulty") or "medium"
    base_tags = list(base_question.get("tags") or [])

    # expected outline для follow-up - это то, чего не хватило
    outline = missing_points[:] if missing_points else ["Дать уточнение по сути вопроса."]

    q_obj = {
        "question_id": f"{base_id}__followup_{idx}",
        "difficulty": base_diff,
        "topic": f"{base_topic} (follow-up)",
        "question": followup_text.strip(),
        "expected_answer_outline": outline,
        "evaluation_criteria": [
            "Уточнение отвечает на поставленный follow-up вопрос",
            "Сохраняется связь с исходной работой/материалами",
        ],
        "followups": [],
        "sources": base_sources,
        "tags": sorted(list(set(base_tags + ["followup"]))),
    }
    return q_obj


def _normalize_reason_tags(tags: Optional[List[str]]) -> List[str]:
    if not tags:
        return []
    out = []
    for t in tags:
        s = str(t).strip().lower()
        s = re.sub(r"[^a-z0-9_]+", "_", s)
        if s:
            out.append(s)
    # dedupe keep order
    seen = set()
    res = []
    for t in out:
        if t in seen:
            continue
        seen.add(t)
        res.append(t)
    return res[:6]


# High-level facade


@dataclass(frozen=True)
class AppServices:
    storage: Storage
    ingest: IngestAgent
    methodics: MethodicsAgent
    defense: DefenseAgent
    feedback: FeedbackAgent


def create_services(storage: Optional[Storage] = None) -> AppServices:
    st = storage or Storage()
    return AppServices(
        storage=st,
        ingest=IngestAgent(st),
        methodics=MethodicsAgent(st),
        defense=DefenseAgent(st),
        feedback=FeedbackAgent(st),
    )


# Teacher flows


def teacher_upload_material(
        svc: AppServices,
        *,
        lab_id: str,
        kind: str,
        filename: str,
        data: bytes,
        mime: Optional[str] = None,
) -> JsonDict:
    """
    Преподаватель загружает материал (цель/задачи/теория/пример/структура отчёта).
    """
    return svc.ingest.ingest_teacher_material(
        lab_id=lab_id,
        kind=kind,
        filename=filename,
        data=data,
        mime=mime,
        parse=True,
    )


def teacher_generate_methodics(
        svc: AppServices,
        *,
        lab_id: str,
) -> JsonDict:
    """
    Генерация методички+рубрики+банка вопросов по загруженным материалам.
    """
    return svc.methodics.generate_methodics_and_bank(lab_id)


def teacher_get_calibration_batch(
        svc: AppServices,
        *,
        lab_id: str,
        round_index: int,
        top_n: int = 10,
) -> JsonDict:
    """
    Получить top-N вопросов для калибровки преподавателем.
    """
    return svc.methodics.make_calibration_batch(lab_id, round_index=round_index, top_n=top_n)


def teacher_apply_calibration_labels(
        svc: AppServices,
        *,
        lab_id: str,
        labeled_questions: List[JsonDict],
) -> Dict[str, int]:
    """
    Преподаватель после калибровки помечает вопросы как good/bad + теги/комментарии.
    Мы сохраняем это в policy_items (teacher preference memory).

    Формат labeled_questions (рекомендуемый):
      [{
        "question": {QUESTION_SCHEMA...},
        "label": "good"|"bad",
        "reason_tags": ["off_topic","too_easy",...],
        "comment": "..."
      }, ...]

    Возвращает статистику добавлений.
    """
    added_good = 0
    added_bad = 0
    added_notes = 0

    for item in labeled_questions:
        q = item.get("question") or {}
        label = str(item.get("label") or "").strip().lower()
        tags = _normalize_reason_tags(item.get("reason_tags") or [])
        comment = (item.get("comment") or "").strip()

        if label not in ("good", "bad"):
            continue

        if label == "good":
            kind = "good_question"
            added_good += 1
        else:
            kind = "bad_question"
            added_bad += 1

        content = {
            "question_id": q.get("question_id"),
            "difficulty": q.get("difficulty"),
            "topic": q.get("topic"),
            "question": q.get("question"),
            "expected_answer_outline": q.get("expected_answer_outline"),
            "sources": q.get("sources"),
            "why": comment,
        }
        svc.storage.add_policy_item(lab_id, kind=kind, content=content, reason_tags=tags)

        # optional: also store notes separately if comment is substantial
        if comment and len(comment) >= 10:
            svc.storage.add_policy_item(
                lab_id,
                kind="note",
                content={"note": comment, "about_question_id": q.get("question_id")},
                reason_tags=tags,
            )
            added_notes += 1

    return {"added_good": added_good, "added_bad": added_bad, "added_notes": added_notes}


def teacher_publish_lab(
        svc: AppServices,
        *,
        lab_id: str,
) -> int:
    """
    Заморозить/опубликовать текущую "версию" (policy_version), которую будут использовать защиты.
    """
    return svc.storage.publish_lab_version(lab_id)


# Student flows


def student_upload_submission(
        svc: AppServices,
        *,
        lab_id: str,
        filename: str,
        data: bytes,
        mime: Optional[str] = None,
        student_id: Optional[str] = None,
) -> JsonDict:
    """
    Студент загружает работу. Сохраняем submission + извлечённый текст.
    """
    return svc.ingest.ingest_student_submission(
        lab_id=lab_id,
        filename=filename,
        data=data,
        mime=mime,
        student_id=student_id,
        parse=True,
    )


def student_start_defense(
        svc: AppServices,
        *,
        lab_id: str,
        submission_id: str,
        student_label: str = "student",
        max_pool: int = 30,
) -> Dict[str, Any]:
    """
    Старт сессии защиты:
    - создаём defense_session
    - анализируем submission
    - формируем пул вопросов + план по сложности
    - сохраняем всё в system_summary (чтобы UI мог продолжать без потери состояния)

    Возвращает:
      {
        "session_id": ...,
        "submission_analysis": ...,
        "question_plan": ...,
        "question_pool_size": ...,
        "policy_version": ...,
      }
    """
    lab = svc.storage.get_lab(lab_id)
    sess = svc.storage.create_defense_session(
        lab_id=lab_id,
        submission_id=submission_id,
        student_label=student_label,
        policy_version=lab.published_version,
    )

    analysis = svc.defense.analyze_submission(lab_id, submission_id)
    pool = svc.defense.compose_question_pool(lab_id, analysis, prefer_suggested=True, max_total=max_pool)

    lab_cfg = lab.config or {}
    plan = (lab_cfg.get("question_plan") or {"easy": 3, "medium": 2, "hard": 1})
    # normalize plan ints
    plan = {k: int(plan.get(k, 0)) for k in ("easy", "medium", "hard")}

    _set_session_system_summary(svc.storage, sess.session_id, {
        "submission_analysis": analysis,
        "question_plan": plan,
        "remaining_question_bank": pool,
        "policy_version_used": sess.policy_version,
        "started_by": "student_start_defense",
    })

    return {
        "session_id": sess.session_id,
        "submission_analysis": analysis,
        "question_plan": plan,
        "question_pool_size": len(pool),
        "policy_version": sess.policy_version,
    }


def student_get_next_question(
        svc: AppServices,
        *,
        session_id: str,
) -> Dict[str, Any]:
    """
    Выбирает и добавляет следующий вопрос в qa_turns:
    - учитывает plan, уже заданные вопросы, пробелы анализа, policy memory
    - обновляет remaining_question_bank в system_summary

    Возвращает:
      {
        "turn_id": ...,
        "question": {...},
        "selection_reason": "...",
        "should_followup": bool,
        "progress": {"asked": {...}, "plan": {...}, "done": bool},
      }
    """
    sess = svc.storage.get_defense_session(session_id)
    if sess.status != "in_progress":
        raise RuntimeError(f"Session is not active: status={sess.status}")

    state = _get_session_system_summary(svc.storage, session_id)
    analysis = state.get("submission_analysis") or {}
    plan = state.get("question_plan") or {"easy": 3, "medium": 2, "hard": 1}
    remaining = state.get("remaining_question_bank") or []

    turns = svc.storage.list_turns(session_id)
    asked_counts = _count_turns_by_difficulty(turns)

    # if plan done -> finish
    if _plan_done(plan, asked_counts):
        svc.storage.finish_defense_session(session_id, system_summary={"auto_finish": True, "reason": "plan_done"})
        return {
            "turn_id": None,
            "question": None,
            "selection_reason": "План вопросов выполнен.",
            "should_followup": False,
            "progress": {"asked": asked_counts, "plan": plan, "done": True},
        }

    asked_turns_summary = build_asked_turns_summary(svc.storage, session_id, max_items=20)

    pick = svc.defense.pick_next_question(
        lab_id=sess.lab_id,
        question_plan=plan,
        remaining_question_bank=remaining,
        asked_turns=asked_turns_summary,
        submission_analysis=analysis,
    )

    selected = (pick.get("selected_question") or {})
    reason = pick.get("reason") or ""
    should_followup = bool(pick.get("should_ask_followup_after_answer"))

    if not selected or not (selected.get("question") or "").strip():
        # If LLM failed, fallback: take first remaining
        if not remaining:
            raise RuntimeError("No questions remaining to ask.")
        selected = remaining[0]
        reason = reason or "Fallback: первый вопрос из оставшегося пула."
        should_followup = False

    # append question turn
    turn_id = svc.storage.append_question(session_id, selected)

    # remove from remaining and update state
    remaining2 = remove_selected_question(remaining, selected)
    _set_session_system_summary(svc.storage, session_id, {
        "remaining_question_bank": remaining2,
        "last_pick_reason": reason,
        "last_pick_should_followup": should_followup,
    })

    # progress snapshot
    turns2 = svc.storage.list_turns(session_id)
    asked_counts2 = _count_turns_by_difficulty(turns2)
    done = _plan_done(plan, asked_counts2)

    return {
        "turn_id": turn_id,
        "question": selected,
        "selection_reason": reason,
        "should_followup": should_followup,
        "progress": {"asked": asked_counts2, "plan": plan, "done": done},
    }


def student_submit_answer(
        svc: AppServices,
        *,
        session_id: str,
        turn_id: str,
        answer_text: str,
        allow_followup: bool = True,
) -> Dict[str, Any]:
    """
    Студент отвечает на вопрос:
    - оцениваем ответ через LLM
    - сохраняем в qa_turns (answer_text + system_eval_json)
    - опционально добавляем follow-up вопрос как новый turn (если модель предложила)
    """
    sess = svc.storage.get_defense_session(session_id)
    if sess.status != "in_progress":
        raise RuntimeError(f"Session is not active: status={sess.status}")

    # get question from turn
    turns = svc.storage.list_turns(session_id)
    turn = next((t for t in turns if t.get("turn_id") == turn_id), None)
    if not turn:
        raise KeyError(f"Turn not found: {turn_id}")

    q_obj = turn.get("question") or {}
    eval_json = svc.defense.evaluate_answer(
        lab_id=sess.lab_id,
        question_obj=q_obj,
        student_answer=answer_text,
        strictness=None,
        include_materials_digest=False,
    )

    svc.storage.submit_answer(
        turn_id=turn_id,
        answer_text=answer_text,
        answer_json={},
        system_eval=eval_json,
    )

    # follow-up handling
    followup_turn_id: Optional[str] = None
    followup_question_obj: Optional[JsonDict] = None

    if allow_followup:
        state = _get_session_system_summary(svc.storage, session_id)
        lab_cfg = svc.storage.get_lab(sess.lab_id).config or {}
        max_followups = int(lab_cfg.get("max_followups_per_question", 1))

        followup_text = eval_json.get("followup_question")
        missing_points = eval_json.get("missing_points") or []
        if isinstance(followup_text, str) and followup_text.strip():
            # count already created followups for this base question_id
            base_qid = q_obj.get("question_id") or ""
            existing_followups = 0
            for t in svc.storage.list_turns(session_id):
                qq = t.get("question") or {}
                qid = qq.get("question_id") or ""
                if base_qid and qid.startswith(base_qid + "__followup_"):
                    existing_followups += 1

            if existing_followups < max_followups:
                followup_question_obj = _make_followup_question_obj(
                    base_question=q_obj,
                    followup_text=followup_text,
                    missing_points=missing_points if isinstance(missing_points, list) else [],
                    idx=existing_followups + 1,
                )
                followup_turn_id = svc.storage.append_question(session_id, followup_question_obj)

                # optional: remember we created followup
                _set_session_system_summary(svc.storage, session_id, {
                    "last_followup_created": True,
                    "last_followup_for_turn_id": turn_id,
                })

    return {
        "turn_id": turn_id,
        "system_eval": eval_json,
        "followup_turn_id": followup_turn_id,
        "followup_question": followup_question_obj,
    }


def student_finish_defense(
        svc: AppServices,
        *,
        session_id: str,
) -> JsonDict:
    """
    Явное завершение защиты (если студент/интерфейс решает закончить).
    """
    sess = svc.storage.get_defense_session(session_id)
    turns = svc.storage.list_turns(session_id)
    state = _get_session_system_summary(svc.storage, session_id)
    plan = state.get("question_plan") or {}

    counts = _count_turns_by_difficulty(turns)
    done = _plan_done(plan, counts) if plan else False

    summary = {
        "manual_finish": True,
        "plan_done": done,
        "asked_counts": counts,
    }
    svc.storage.finish_defense_session(session_id, system_summary=summary)
    return summary


def student_add_feedback(
        svc: AppServices,
        *,
        session_id: str,
        rating: Optional[int] = None,
        comment: str = "",
        tags: Optional[List[str]] = None,
) -> str:
    """
    Обратная связь студента по процедуре защиты.
    """
    return svc.storage.add_student_feedback(session_id, rating=rating, comment=comment, tags=tags or [])


# Teacher review flows


def teacher_rate_question(
        svc: AppServices,
        *,
        session_id: str,
        turn_id: str,
        label: str,  # good/bad
        score: Optional[float] = None,
        reason_tags: Optional[List[str]] = None,
        comment: str = "",
) -> str:
    """
    Преподаватель оценивает качество вопроса системы.
    """
    label = str(label).strip().lower()
    if label not in ("good", "bad"):
        raise ValueError("label must be 'good' or 'bad'")
    return svc.storage.add_teacher_feedback(
        session_id=session_id,
        target="question",
        label=label,
        turn_id=turn_id,
        score=score,
        reason_tags=_normalize_reason_tags(reason_tags),
        comment=comment,
    )


def teacher_rate_answer(
        svc: AppServices,
        *,
        session_id: str,
        turn_id: str,
        label: str,  # good/bad (answer quality)
        score: Optional[float] = None,
        reason_tags: Optional[List[str]] = None,
        comment: str = "",
) -> str:
    """
    Преподаватель оценивает ответ студента (или корректность авто-оценки системы).
    """
    label = str(label).strip().lower()
    if label not in ("good", "bad"):
        raise ValueError("label must be 'good' or 'bad'")
    return svc.storage.add_teacher_feedback(
        session_id=session_id,
        target="answer",
        label=label,
        turn_id=turn_id,
        score=score,
        reason_tags=_normalize_reason_tags(reason_tags),
        comment=comment,
    )


def teacher_finalize_session_review(
        svc: AppServices,
        *,
        session_id: str,
        teacher_summary: JsonDict,
) -> None:
    """
    Преподаватель добавляет итоговый комментарий/оценку к сессии (teacher_summary).
    """
    svc.storage.set_teacher_session_summary(session_id, teacher_summary=teacher_summary)


def teacher_update_policy_from_session(
        svc: AppServices,
        *,
        lab_id: str,
        session_id: str,
        apply: bool = True,
) -> Dict[str, Any]:
    """
    На основании teacher_feedback по сессии - предложить и (опционально) применить policy updates.
    """
    suggestion = svc.feedback.suggest_policy_updates(lab_id, session_id)
    stats = {"added_good": 0, "added_bad": 0, "added_notes": 0}
    if apply:
        stats = svc.feedback.apply_policy_updates(lab_id, suggestion)
    return {"suggestion": suggestion, "stats": stats}


# Logs / exports


def export_defense_log_json(
        svc: AppServices,
        *,
        session_id: str,
        filename: Optional[str] = None,
) -> str:
    """
    Экспорт полного лога защиты в JSON файл. Возвращает путь.
    """
    p = svc.storage.export_defense_log_json(session_id, filename=filename)
    return str(p)


def get_defense_log(
        svc: AppServices,
        *,
        session_id: str,
) -> JsonDict:
    """
    Получить полный лог защиты (dict), без экспорта на диск.
    """
    return svc.storage.build_defense_log(session_id)
