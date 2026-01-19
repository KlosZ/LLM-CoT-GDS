"""
Реализация узких (в смысле ответственности) агентов для системы 
автоматизации защиты лабораторных/практических работ.

- IngestAgent: загрузка материалов преподавателя/работы студента + извлечение текста
- MethodicsAgent: генерация методички/рубрики/банка вопросов + калибровка топ-N
- DefenseAgent: анализ работы студента, подбор следующего вопроса, оценка ответа
- FeedbackAgent: обработка обратной связи преподавателя -> обновление policy memory

"Дообучение" реализуется как policy memory: хорошие/плохие 
примеры и заметки преподавателя, которые подмешиваются в промпт.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from storage import Storage
from parsers import extract_text_from_bytes, split_to_chunks, ExtractResult
from llm import call_llm
import prompts

JsonDict = Dict[str, Any]


# Small utilities


def _read_text_file(path: str, *, max_chars: int = 8000) -> str:
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    text = text.strip()
    if max_chars and len(text) > max_chars:
        return text[:max_chars] + "…"
    return text


def _clip_text(text: str, *, max_chars: int = 8000) -> str:
    t = (text or "").strip()
    if max_chars and len(t) > max_chars:
        return t[:max_chars] + "…"
    return t


def _dedupe_questions(qs: List[JsonDict]) -> List[JsonDict]:
    seen = set()
    out: List[JsonDict] = []
    for q in qs:
        key = (q.get("question", "") or "").strip().lower()
        key = re.sub(r"\s+", " ", key)
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(q)
    return out


def _count_by_difficulty(questions: List[JsonDict]) -> JsonDict:
    c = {"easy": 0, "medium": 0, "hard": 0}
    for q in questions:
        d = q.get("difficulty")
        if d in c:
            c[d] += 1
    return c


def _get_generated(storage: Storage, lab_id: str) -> JsonDict:
    lab = storage.get_lab(lab_id)
    cfg = lab.config or {}
    return cfg.get("generated", {}) if isinstance(cfg, dict) else {}


def _set_generated(storage: Storage, lab_id: str, generated: JsonDict) -> None:
    lab = storage.get_lab(lab_id)
    cfg = dict(lab.config or {})
    cfg["generated"] = generated
    storage.update_lab(lab_id, config=cfg)


def _get_lab_config(storage: Storage, lab_id: str) -> JsonDict:
    lab = storage.get_lab(lab_id)
    return dict(lab.config or {})


def _make_materials_digest(storage: Storage, lab_id: str, *, per_item_chars: int = 3500) -> List[JsonDict]:
    mats = storage.list_materials(lab_id)
    digest: List[JsonDict] = []
    for m in mats:
        text = _read_text_file(m.extracted_text_path, max_chars=per_item_chars)
        digest.append({
            "material_id": m.material_id,
            "filename": m.filename,
            "kind": m.kind,
            "text": text,
        })
    return digest


def _make_submission_excerpt(storage: Storage, submission_id: str, *, max_chars: int = 9000) -> str:
    sub = storage.get_submission(submission_id)
    full = _read_text_file(sub.extracted_text_path, max_chars=0)

    if not full.strip():
        return ""

    # Heuristic excerpt: headings + beginning + end
    lines = [ln.rstrip() for ln in full.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    headings: List[str] = []
    for ln in lines[:800]:
        s = ln.strip()
        if not s:
            continue
        if len(s) > 120:
            continue
        if re.match(r"^(глава|раздел|введение|заключение|содержание|цель|задач|теори|результат|вывод)", s.lower()):
            headings.append(s)
        elif re.match(r"^\d+(\.\d+)*\s+\S+", s):
            headings.append(s)

        if len(headings) >= 30:
            break

    head_block = "\n".join(headings[:30]).strip()
    start = full[: min(4500, len(full))].strip()
    tail = full[-min(2500, len(full)):].strip()

    excerpt_parts = []
    if head_block:
        excerpt_parts.append("=== Похоже на заголовки/оглавление (извлечено эвристикой) ===\n" + head_block)
    excerpt_parts.append("=== Начало работы ===\n" + start)
    if tail and tail not in start:
        excerpt_parts.append("=== Конец работы ===\n" + tail)

    excerpt = "\n\n".join(excerpt_parts).strip()
    return _clip_text(excerpt, max_chars=max_chars)


# Ingest Agent


class IngestAgent:
    def __init__(self, storage: Storage):
        self.storage = storage

    def ingest_teacher_material(
            self,
            lab_id: str,
            *,
            kind: str,
            filename: str,
            data: bytes,
            mime: Optional[str] = None,
            parse: bool = True,
            parse_meta: Optional[JsonDict] = None,
    ) -> JsonDict:
        """
        Сохраняет материал преподавателя в БД/файлы и (опционально) извлекает текст.
        Возвращает dict с material_id и краткой сводкой.
        """
        extracted_text = ""
        meta: JsonDict = dict(parse_meta or {})
        meta["parser"] = None

        if parse:
            res = extract_text_from_bytes(filename, data, mime=mime)
            extracted_text = res.text
            meta.update(res.meta)
            meta["parser"] = "extract_text_from_bytes"

        mat = self.storage.add_material(
            lab_id=lab_id,
            kind=kind,
            filename=filename,
            data=data,
            mime=mime or (meta.get("mime") or "application/octet-stream"),
            extracted_text=extracted_text,
            meta=meta,
        )

        return {
            "material_id": mat.material_id,
            "filename": mat.filename,
            "kind": mat.kind,
            "mime": mat.mime,
            "extracted_chars": len(extracted_text or ""),
        }

    def ingest_student_submission(
            self,
            lab_id: str,
            *,
            filename: str,
            data: bytes,
            mime: Optional[str] = None,
            student_id: Optional[str] = None,
            parse: bool = True,
            parse_meta: Optional[JsonDict] = None,
    ) -> JsonDict:
        """
        Сохраняет работу студента. Возвращает dict с submission_id и метаданными.
        """
        extracted_text = ""
        meta: JsonDict = dict(parse_meta or {})
        meta["parser"] = None

        if parse:
            res = extract_text_from_bytes(filename, data, mime=mime)
            extracted_text = res.text
            meta.update(res.meta)
            meta["parser"] = "extract_text_from_bytes"

        sub = self.storage.add_submission(
            lab_id=lab_id,
            filename=filename,
            data=data,
            student_id=student_id,
            mime=mime or (meta.get("mime") or "application/octet-stream"),
            extracted_text=extracted_text,
            meta=meta,
        )

        return {
            "submission_id": sub.submission_id,
            "filename": sub.filename,
            "mime": sub.mime,
            "student_id": sub.student_id,
            "extracted_chars": len(extracted_text or ""),
        }


# Methodics Agent


class MethodicsAgent:
    def __init__(self, storage: Storage):
        self.storage = storage

    def generate_methodics_and_bank(self, lab_id: str) -> JsonDict:
        """
        Генерация методички + рубрики + банка вопросов по материалам преподавателя.
        Сохраняет результат в lab.config["generated"].
        """
        lab = self.storage.get_lab(lab_id)
        lab_cfg = _get_lab_config(self.storage, lab_id)
        mats_digest = _make_materials_digest(self.storage, lab_id)

        system, user = prompts.build_methodics_and_bank_prompt(
            lab_title=lab.title,
            lab_config=lab_cfg,
            materials_digest=mats_digest,
        )

        resp = call_llm(
            user_prompt=user,
            system_prompt=system,
            json_schema=prompts.SCHEMA_METHODICS_AND_BANK,
            schema_name="methodics_and_bank",
            strict_json=True,
        )

        generated = dict(resp.json or {})
        generated["generated_at"] = generated.get("generated_at") or None  # optional
        # Normalize: enforce existence keys
        generated.setdefault("lab_title", lab.title)

        # Save to lab config
        cur = _get_generated(self.storage, lab_id)
        cur.update({
            "methodics": generated.get("methodics"),
            "rubric": generated.get("rubric"),
            "question_bank": generated.get("question_bank"),
            "coverage_map": generated.get("coverage_map"),
            "quality_warnings": generated.get("quality_warnings"),
            "lab_title": generated.get("lab_title", lab.title),
            "generated_at_utc": cur.get("generated_at_utc") or None,
        })
        _set_generated(self.storage, lab_id, cur)

        return generated

    def make_calibration_batch(self, lab_id: str, *, round_index: int, top_n: int = 10) -> JsonDict:
        """
        Выдать топ-N вопросов для калибровки преподавателем (по уже сгенерированному банку).
        """
        lab = self.storage.get_lab(lab_id)
        lab_cfg = _get_lab_config(self.storage, lab_id)
        gen = _get_generated(self.storage, lab_id)
        bank = gen.get("question_bank") or []
        if not bank:
            raise RuntimeError("question_bank is empty. Generate methodics/bank first.")

        system, user = prompts.build_teacher_calibration_batch_prompt(
            lab_title=lab.title,
            lab_config=lab_cfg,
            question_bank=bank,
            round_index=round_index,
            top_n=top_n,
        )

        resp = call_llm(
            user_prompt=user,
            system_prompt=system,
            json_schema=prompts.SCHEMA_TEACHER_CALIBRATION_BATCH,
            schema_name="teacher_calibration_batch",
            strict_json=True,
        )
        return resp.json or {}


# Defense Agent


class DefenseAgent:
    def __init__(self, storage: Storage):
        self.storage = storage

    def analyze_submission(self, lab_id: str, submission_id: str) -> JsonDict:
        """
        Анализ работы студента на соответствие заданию/структуре.
        Требует, чтобы в generated уже была rubric (желательно), иначе возьмём заглушку.
        """
        lab = self.storage.get_lab(lab_id)
        lab_cfg = _get_lab_config(self.storage, lab_id)

        gen = _get_generated(self.storage, lab_id)
        rubric = gen.get("rubric") or {"scale": "draft", "criteria": [], "notes": ["Rubric not generated yet."]}

        mats_digest = _make_materials_digest(self.storage, lab_id)
        excerpt = _make_submission_excerpt(self.storage, submission_id)

        system, user = prompts.build_submission_analysis_prompt(
            lab_title=lab.title,
            lab_config=lab_cfg,
            rubric=rubric,
            materials_digest=mats_digest,
            submission_excerpt=excerpt,
        )

        resp = call_llm(
            user_prompt=user,
            system_prompt=system,
            json_schema=prompts.SCHEMA_SUBMISSION_ANALYSIS,
            schema_name="submission_analysis",
            strict_json=True,
        )
        return resp.json or {}

    def compose_question_pool(
            self,
            lab_id: str,
            submission_analysis: JsonDict,
            *,
            prefer_suggested: bool = True,
            max_total: int = 30,
    ) -> List[JsonDict]:
        """
        Собирает пул вопросов для защиты:
        - suggested_questions из анализа работы
        - + общий question_bank из generated
        Дедупликация по тексту вопроса.
        """
        gen = _get_generated(self.storage, lab_id)
        bank = gen.get("question_bank") or []
        suggested = submission_analysis.get("suggested_questions") or []

        pool: List[JsonDict] = []
        if prefer_suggested:
            pool.extend(suggested)
            pool.extend(bank)
        else:
            pool.extend(bank)
            pool.extend(suggested)

        pool = _dedupe_questions(pool)
        if max_total and len(pool) > max_total:
            pool = pool[:max_total]
        return pool

    def pick_next_question(
            self,
            lab_id: str,
            *,
            question_plan: JsonDict,
            remaining_question_bank: List[JsonDict],
            asked_turns: List[JsonDict],
            submission_analysis: JsonDict,
    ) -> JsonDict:
        """
        Выбирает следующий вопрос (через LLM) с учётом:
        - плана по сложности
        - уже заданных вопросов
        - предпочтений преподавателя (policy memory)
        """
        lab = self.storage.get_lab(lab_id)
        lab_cfg = _get_lab_config(self.storage, lab_id)

        good, bad = self.storage.get_policy_examples(lab_id)

        # Safety: do not send too big candidates
        candidates = remaining_question_bank[:40]

        system, user = prompts.build_next_question_pick_prompt(
            lab_title=lab.title,
            lab_config=lab_cfg,
            question_plan=question_plan,
            remaining_question_bank=candidates,
            asked_turns=asked_turns,
            submission_analysis=submission_analysis,
            policy_good_examples=good,
            policy_bad_examples=bad,
        )

        resp = call_llm(
            user_prompt=user,
            system_prompt=system,
            json_schema=prompts.SCHEMA_NEXT_QUESTION_PICK,
            schema_name="next_question_pick",
            strict_json=True,
        )
        return resp.json or {}

    def evaluate_answer(
            self,
            lab_id: str,
            *,
            question_obj: JsonDict,
            student_answer: str,
            strictness: Optional[float] = None,
            include_materials_digest: bool = False,
    ) -> JsonDict:
        """
        Оценивает ответ студента на один вопрос (через LLM).
        """
        lab = self.storage.get_lab(lab_id)
        gen = _get_generated(self.storage, lab_id)
        rubric = gen.get("rubric") or {"scale": "draft", "criteria": [], "notes": []}

        lab_cfg = _get_lab_config(self.storage, lab_id)
        st = strictness
        if st is None:
            st = float(lab_cfg.get("strictness", 0.7))

        mats_digest = _make_materials_digest(self.storage, lab_id,
                                             per_item_chars=2400) if include_materials_digest else None

        system, user = prompts.build_answer_evaluation_prompt(
            lab_title=lab.title,
            rubric=rubric,
            question_obj=question_obj,
            student_answer=student_answer,
            materials_digest=mats_digest,
            strictness=float(st),
        )

        resp = call_llm(
            user_prompt=user,
            system_prompt=system,
            json_schema=prompts.SCHEMA_ANSWER_EVAL,
            schema_name="answer_eval",
            strict_json=True,
        )
        return resp.json or {}


# Feedback Agent


class FeedbackAgent:
    def __init__(self, storage: Storage):
        self.storage = storage

    def list_teacher_feedback(self, session_id: str) -> List[JsonDict]:
        """
        Вытаскивает teacher_feedback по session_id.
        В storage.py нет отдельного метода, поэтому читаем напрямую (это ок для учебного проекта).
        """
        with self.storage._connect() as conn:  # pylint: disable=protected-access
            rows = conn.execute(
                "SELECT * FROM teacher_feedback WHERE session_id=? ORDER BY created_at ASC;",
                (session_id,),
            ).fetchall()

        out: List[JsonDict] = []
        for r in rows:
            out.append({
                "feedback_id": r["feedback_id"],
                "session_id": r["session_id"],
                "target": r["target"],
                "turn_id": r["turn_id"],
                "label": r["label"],
                "score": r["score"],
                "reason_tags_json": r["reason_tags_json"],
                "comment": r["comment"],
                "created_at": r["created_at"],
            })
        return out

    def suggest_policy_updates(self, lab_id: str, session_id: str) -> JsonDict:
        """
        На основе teacher_feedback + turns предлагает, какие policy items добавить.
        """
        lab = self.storage.get_lab(lab_id)

        turns = self.storage.list_turns(session_id)
        tfb_raw = self.list_teacher_feedback(session_id)

        # normalize teacher feedback items for prompt
        teacher_feedback_items: List[JsonDict] = []
        for r in tfb_raw:
            # parse reason_tags_json safely (may not exist if inserted old)
            reason_tags = []
            try:
                import json as _json
                reason_tags = _json.loads(r.get("reason_tags_json") or "[]")
            except Exception:
                reason_tags = []
            teacher_feedback_items.append({
                "feedback_id": r["feedback_id"],
                "target": r["target"],
                "turn_id": r["turn_id"],
                "label": r["label"],
                "score": r["score"],
                "reason_tags": reason_tags,
                "comment": r["comment"],
                "created_at": r["created_at"],
            })

        system, user = prompts.build_policy_update_suggestion_prompt(
            lab_title=lab.title,
            teacher_feedback_items=teacher_feedback_items,
            turns=turns,
        )

        resp = call_llm(
            user_prompt=user,
            system_prompt=system,
            json_schema=prompts.SCHEMA_POLICY_UPDATE_SUGGESTION,
            schema_name="policy_update_suggestion",
            strict_json=True,
        )
        return resp.json or {}

    def apply_policy_updates(self, lab_id: str, suggestion: JsonDict) -> Dict[str, int]:
        """
        Применяет suggestion (good/bad items) в policy_items таблицу.
        Возвращает статистику добавлений.
        """
        added_good = 0
        added_bad = 0
        added_notes = 0

        for item in suggestion.get("good_items_to_add", []) or []:
            kind = item.get("kind", "note")
            content = item.get("content") or {}
            tags = item.get("reason_tags") or []
            self.storage.add_policy_item(lab_id, kind=kind, content=content, reason_tags=tags)
            if kind == "good_question":
                added_good += 1
            else:
                added_notes += 1

        for item in suggestion.get("bad_items_to_add", []) or []:
            kind = item.get("kind", "note")
            content = item.get("content") or {}
            tags = item.get("reason_tags") or []
            self.storage.add_policy_item(lab_id, kind=kind, content=content, reason_tags=tags)
            if kind == "bad_question":
                added_bad += 1
            else:
                added_notes += 1

        return {"added_good": added_good, "added_bad": added_bad, "added_notes": added_notes}


# Optional: convenience orchestration helpers (still "agent-side")


def build_asked_turns_summary(storage: Storage, session_id: str, *, max_items: int = 20) -> List[JsonDict]:
    """
    Для промпта выбора следующего вопроса: компактное описание уже заданных вопросов.
    """
    turns = storage.list_turns(session_id)
    out: List[JsonDict] = []
    for t in turns[-max_items:]:
        q = t.get("question") or {}
        se = t.get("system_eval") or {}
        out.append({
            "question_id": q.get("question_id"),
            "difficulty": q.get("difficulty"),
            "topic": q.get("topic"),
            "question": _clip_text(q.get("question", ""), max_chars=220),
            "answer_label": se.get("label"),
            "answer_score": se.get("score"),
        })
    return out


def remove_selected_question(remaining: List[JsonDict], selected: JsonDict) -> List[JsonDict]:
    """
    Удаляет выбранный вопрос из remaining_question_bank (по question_id или тексту).
    """
    sid = (selected or {}).get("question_id")
    stxt = ((selected or {}).get("question") or "").strip().lower()
    stxt = re.sub(r"\s+", " ", stxt)

    out: List[JsonDict] = []
    for q in remaining:
        if sid and q.get("question_id") == sid:
            continue
        qtxt = (q.get("question") or "").strip().lower()
        qtxt = re.sub(r"\s+", " ", qtxt)
        if stxt and qtxt == stxt:
            continue
        out.append(q)
    return out
