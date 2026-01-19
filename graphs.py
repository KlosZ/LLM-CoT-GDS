"""
LangGraph-оркестрация (мультиагентный сценарий) поверх наших use-case функций. Здесь мы собираем 2 графа:
1) teacher_graph: подготовка/калибровка (генерация методички+банка, выдача top-N для калибровки)
2) student_graph: защита (start -> next_question -> submit_answer -> finish)
Если langgraph не установлен/несовместим, будет понятная ошибка. 

Идея:
- UI (Streamlit) управляет "action" в состоянии графа
- Graph.invoke(state) возвращает обновлённый state (результаты шагов)

Пример использования:

    from storage import Storage
    from core import create_services
    from graphs import build_teacher_graph, build_student_graph
    
    svc = create_services(Storage())
    tg = build_teacher_graph(svc)
    sg = build_student_graph(svc)
    
    # Teacher generate
    out = tg.invoke({"action":"generate", "lab_id": lab_id})
    
    # Student start
    out = sg.invoke({"action":"start", "lab_id": lab_id, "submission_id": sub_id, "student_label":"Иванов"})
"""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional, TypedDict, cast

import core

JsonDict = Dict[str, Any]


# LangGraph import (compatibility layer)


def _import_langgraph():
    """
    LangGraph versions differ a bit; we try common imports.
    Returns (StateGraph, END).
    """
    try:
        from langgraph.graph import StateGraph, END  # type: ignore
        return StateGraph, END
    except Exception:
        # Some versions expose END as a string constant or in different module
        try:
            from langgraph.graph import StateGraph  # type: ignore
            END = "END"
            return StateGraph, END
        except Exception as e:
            raise RuntimeError(
                "LangGraph is required for graphs.py. Install it:\n"
                "  pip install langgraph\n"
                "If installed, your version may be incompatible with this scaffold."
            ) from e


StateGraph, END = _import_langgraph()


# Teacher graph state


class TeacherState(TypedDict, total=False):
    action: Literal["generate", "calibrate"]
    lab_id: str
    round_index: int
    top_n: int
    result: JsonDict
    error: str


# Student graph state


class StudentState(TypedDict, total=False):
    action: Literal["start", "next_question", "submit_answer", "finish"]
    lab_id: str
    submission_id: str
    session_id: str
    student_label: str
    turn_id: str
    answer_text: str
    allow_followup: bool
    result: JsonDict
    error: str


# Teacher graph nodes


def _teacher_generate_node(state: TeacherState, svc: core.AppServices) -> TeacherState:
    lab_id = state["lab_id"]
    generated = core.teacher_generate_methodics(svc, lab_id=lab_id)
    return cast(TeacherState, {"result": {"generated": generated}})


def _teacher_calibrate_node(state: TeacherState, svc: core.AppServices) -> TeacherState:
    lab_id = state["lab_id"]
    round_index = int(state.get("round_index", 1))
    top_n = int(state.get("top_n", 10))
    batch = core.teacher_get_calibration_batch(svc, lab_id=lab_id, round_index=round_index, top_n=top_n)
    return cast(TeacherState, {"result": {"calibration_batch": batch}})


def _teacher_supervisor(state: TeacherState) -> str:
    return str(state.get("action") or "generate")


def _teacher_error_handler(state: TeacherState) -> TeacherState:
    # просто заглушка: если кто-то направит сюда, завершить.
    return cast(TeacherState, {"error": state.get("error", "unknown_error")})


# Student graph nodes


def _student_start_node(state: StudentState, svc: core.AppServices) -> StudentState:
    lab_id = state["lab_id"]
    submission_id = state["submission_id"]
    student_label = state.get("student_label", "student")

    out = core.student_start_defense(
        svc,
        lab_id=lab_id,
        submission_id=submission_id,
        student_label=student_label,
        max_pool=30,
    )
    # сохраним session_id в state, чтобы следующие action могли его использовать
    return cast(StudentState, {
        "session_id": out["session_id"],
        "result": {"start": out},
    })


def _student_next_question_node(state: StudentState, svc: core.AppServices) -> StudentState:
    session_id = state["session_id"]
    out = core.student_get_next_question(svc, session_id=session_id)
    # out может вернуть turn_id=None если план выполнен
    upd: StudentState = {"result": {"next_question": out}}
    if out.get("turn_id"):
        upd["turn_id"] = out["turn_id"]
    return upd


def _student_submit_answer_node(state: StudentState, svc: core.AppServices) -> StudentState:
    session_id = state["session_id"]
    turn_id = state["turn_id"]
    answer_text = state.get("answer_text", "")
    allow_followup = bool(state.get("allow_followup", True))

    out = core.student_submit_answer(
        svc,
        session_id=session_id,
        turn_id=turn_id,
        answer_text=answer_text,
        allow_followup=allow_followup,
    )

    # Если создали follow-up turn, то можно сразу перекинуть UI на него,
    # или оставить как есть (решит UI). Здесь мы сохраняем followup_turn_id.
    upd: StudentState = {"result": {"submit_answer": out}}
    if out.get("followup_turn_id"):
        upd["turn_id"] = out["followup_turn_id"]
        # Чтобы UI знал, что сейчас активен follow-up, можно проставить флаг
        upd["result"]["submit_answer"]["active_turn_is_followup"] = True
    return upd


def _student_finish_node(state: StudentState, svc: core.AppServices) -> StudentState:
    session_id = state["session_id"]
    summary = core.student_finish_defense(svc, session_id=session_id)
    # Экспорт лога JSON можно вызывать отдельным действием из UI,
    # но для удобства можно сделать тут (закомментировано):
    # path = core.export_defense_log_json(svc, session_id=session_id)
    return cast(StudentState, {"result": {"finish": summary}})


def _student_supervisor(state: StudentState) -> str:
    return str(state.get("action") or "start")


def _student_error_handler(state: StudentState) -> StudentState:
    return cast(StudentState, {"error": state.get("error", "unknown_error")})


# Build graphs


def build_teacher_graph(svc: core.AppServices):
    """
    Компилирует граф преподавателя.

    Ожидаемый входной state:
      {"action":"generate","lab_id": "..."} или {"action":"calibrate","lab_id":"...","round_index":1,"top_n":10}
    """
    g = StateGraph(TeacherState)

    g.add_node("supervisor", _teacher_supervisor)
    g.add_node("generate", lambda s: _teacher_generate_node(cast(TeacherState, s), svc))
    g.add_node("calibrate", lambda s: _teacher_calibrate_node(cast(TeacherState, s), svc))
    g.add_node("error", _teacher_error_handler)

    g.set_entry_point("supervisor")

    g.add_conditional_edges(
        "supervisor",
        _teacher_supervisor,
        {
            "generate": "generate",
            "calibrate": "calibrate",
        },
    )

    g.add_edge("generate", END)
    g.add_edge("calibrate", END)
    g.add_edge("error", END)

    return g.compile()


def build_student_graph(svc: core.AppServices):
    """
    Компилирует граф студента.

    Ожидаемый входной state по action:
    - start:
        {"action":"start","lab_id":"...","submission_id":"...","student_label":"..."}
    - next_question:
        {"action":"next_question","session_id":"..."}
    - submit_answer:
        {"action":"submit_answer","session_id":"...","turn_id":"...","answer_text":"...","allow_followup":true}
    - finish:
        {"action":"finish","session_id":"..."}
    """
    g = StateGraph(StudentState)

    g.add_node("supervisor", _student_supervisor)
    g.add_node("start", lambda s: _student_start_node(cast(StudentState, s), svc))
    g.add_node("next_question", lambda s: _student_next_question_node(cast(StudentState, s), svc))
    g.add_node("submit_answer", lambda s: _student_submit_answer_node(cast(StudentState, s), svc))
    g.add_node("finish", lambda s: _student_finish_node(cast(StudentState, s), svc))
    g.add_node("error", _student_error_handler)

    g.set_entry_point("supervisor")

    g.add_conditional_edges(
        "supervisor",
        _student_supervisor,
        {
            "start": "start",
            "next_question": "next_question",
            "submit_answer": "submit_answer",
            "finish": "finish",
        },
    )

    g.add_edge("start", END)
    g.add_edge("next_question", END)
    g.add_edge("submit_answer", END)
    g.add_edge("finish", END)
    g.add_edge("error", END)

    return g.compile()


# Convenience wrappers


def invoke_teacher(svc: core.AppServices, state: TeacherState) -> TeacherState:
    """One-shot helper."""
    g = build_teacher_graph(svc)
    return cast(TeacherState, g.invoke(state))


def invoke_student(svc: core.AppServices, state: StudentState) -> StudentState:
    """One-shot helper."""
    g = build_student_graph(svc)
    return cast(StudentState, g.invoke(state))
