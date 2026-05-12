"""
SQLite + файловое хранилище для Streamlit-приложения (роль Преподаватель/Студент).

Цели:
- хранить "лабы" (lab_id) + их конфиг
- хранить материалы преподавателя и извлечённый текст
- хранить студенческие работы (submissions) и извлечённый текст
- хранить сессии защиты (defense_sessions) + ход Q/A (qa_turns)
- хранить обратную связь студента и преподавателя
- хранить "политику" / предпочтения преподавателя (policy_items) для псевдо-дообучения (RAG/few-shot)
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

SCHEMA_VERSION = 3

STATUS_DRAFT = "draft"
STATUS_NEEDS_CLARIFICATION = "needs_clarification"
STATUS_ALIGNED = "aligned"
STATUS_REJECTED = "rejected"
STATUS_FINALIZED = "finalized"

SIDE_STUDENT = "student"
SIDE_TEACHER = "teacher"
SIDE_SYSTEM = "system"

MATERIAL_ROLE_STUDENT = "student"
MATERIAL_ROLE_TEACHER = "teacher"
MATERIAL_ROLE_SYSTEM = "system"

MATERIAL_STAGE_TOPIC_ALIGNMENT = "topic_alignment"
MATERIAL_STAGE_METHODICS = "methodics"
MATERIAL_STAGE_SUBMISSION = "submission"
MATERIAL_STAGE_GENERAL = "general"

WORK_TYPE_LAB = "lab"
WORK_TYPE_PRACTICE = "practice"
WORK_TYPE_RESEARCH = "research"
WORK_TYPE_COURSEWORK = "coursework"
WORK_TYPE_REPORT = "report"
WORK_TYPE_OTHER = "other"

EVALUATION_SCENARIO_TOPIC_FINAL = "topic_final"
EVALUATION_SCENARIO_TOPIC_QUESTIONS = "topic_clarification_questions"
EVALUATION_SCENARIO_TOPIC_PROCESS = "topic_alignment_process"
EVALUATION_SCENARIO_DEFENSE_QUESTIONS = "defense_questions"

EVALUATION_STATUS_CREATED = "created"
EVALUATION_STATUS_RUNNING = "running"
EVALUATION_STATUS_FINISHED = "finished"
EVALUATION_STATUS_FAILED = "failed"


# Helpers


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def dumps_json(value: Any) -> str:
    if value is None:
        return "{}"
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def loads_json(value: Optional[str], default: Any = None) -> Any:
    if value is None or value == "":
        return {} if default is None else default
    try:
        return json.loads(value)
    except Exception:
        return {} if default is None else default


def row_to_dict(row: sqlite3.Row | None) -> Optional[dict[str, Any]]:
    if row is None:
        return None
    return {key: row[key] for key in row.keys()}


def rows_to_dicts(rows: Iterable[sqlite3.Row]) -> list[dict[str, Any]]:
    return [row_to_dict(row) for row in rows if row is not None]  # type: ignore[arg-type]


# Main storage


class Storage:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # Schema / migration

    def init_db(self) -> None:
        with self.connect() as conn:
            current_version = self._get_user_version(conn)

            if current_version == 0:
                self._create_schema_v2(conn)
                self._set_user_version(conn, SCHEMA_VERSION)
                return

            if current_version < SCHEMA_VERSION:
                self._migrate_to_v2_best_effort(conn)
                self._set_user_version(conn, SCHEMA_VERSION)
                return

            # На случай, если версия уже равна 2, но каких-то таблиц нет
            self._create_schema_v2(conn)

    def _get_user_version(self, conn: sqlite3.Connection) -> int:
        row = conn.execute("PRAGMA user_version;").fetchone()
        return int(row[0]) if row else 0

    def _set_user_version(self, conn: sqlite3.Connection, version: int) -> None:
        conn.execute(f"PRAGMA user_version = {int(version)};")

    def _create_schema_v2(self, conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS labs (
                lab_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                description TEXT DEFAULT '',
                work_type TEXT NOT NULL DEFAULT 'other',
                status TEXT NOT NULL DEFAULT 'draft',
                config_json TEXT NOT NULL DEFAULT '{}',
                agreed_spec_id TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_labs_status ON labs(status);
            CREATE INDEX IF NOT EXISTS idx_labs_work_type ON labs(work_type);

            CREATE TABLE IF NOT EXISTS materials (
                material_id TEXT PRIMARY KEY,
                lab_id TEXT NOT NULL,
                owner_role TEXT NOT NULL DEFAULT 'teacher',
                stage TEXT NOT NULL DEFAULT 'general',
                title TEXT DEFAULT '',
                filename TEXT NOT NULL,
                mime_type TEXT DEFAULT '',
                file_path TEXT DEFAULT '',
                extracted_text TEXT DEFAULT '',
                meta_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                FOREIGN KEY (lab_id) REFERENCES labs(lab_id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_materials_lab_id ON materials(lab_id);
            CREATE INDEX IF NOT EXISTS idx_materials_stage ON materials(stage);
            CREATE INDEX IF NOT EXISTS idx_materials_owner_role ON materials(owner_role);

            CREATE TABLE IF NOT EXISTS topic_sessions (
                topic_session_id TEXT PRIMARY KEY,
                lab_id TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'draft',
                round_no INTEGER NOT NULL DEFAULT 0,
                relation_score REAL,
                relation_label TEXT DEFAULT '',
                summary_text TEXT DEFAULT '',
                llm_assessment_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (lab_id) REFERENCES labs(lab_id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_topic_sessions_lab_id ON topic_sessions(lab_id);
            CREATE INDEX IF NOT EXISTS idx_topic_sessions_status ON topic_sessions(status);

            CREATE TABLE IF NOT EXISTS topic_inputs (
                topic_input_id TEXT PRIMARY KEY,
                topic_session_id TEXT NOT NULL,
                side TEXT NOT NULL,
                title TEXT NOT NULL,
                description TEXT DEFAULT '',
                context_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(topic_session_id, side),
                FOREIGN KEY (topic_session_id) REFERENCES topic_sessions(topic_session_id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_topic_inputs_session_id ON topic_inputs(topic_session_id);
            CREATE INDEX IF NOT EXISTS idx_topic_inputs_side ON topic_inputs(side);

            CREATE TABLE IF NOT EXISTS topic_turns (
                topic_turn_id TEXT PRIMARY KEY,
                topic_session_id TEXT NOT NULL,
                side TEXT NOT NULL,
                turn_kind TEXT NOT NULL,
                question_text TEXT DEFAULT '',
                answer_text TEXT DEFAULT '',
                extra_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                FOREIGN KEY (topic_session_id) REFERENCES topic_sessions(topic_session_id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_topic_turns_session_id ON topic_turns(topic_session_id);
            CREATE INDEX IF NOT EXISTS idx_topic_turns_side ON topic_turns(side);

            CREATE TABLE IF NOT EXISTS agreed_specs (
                spec_id TEXT PRIMARY KEY,
                lab_id TEXT NOT NULL,
                topic_session_id TEXT,
                work_type TEXT NOT NULL DEFAULT 'other',
                agreed_title TEXT NOT NULL,
                agreed_description TEXT DEFAULT '',
                acceptance_criteria_json TEXT NOT NULL DEFAULT '{}',
                generated_from_json TEXT NOT NULL DEFAULT '{}',
                status TEXT NOT NULL DEFAULT 'finalized',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (lab_id) REFERENCES labs(lab_id) ON DELETE CASCADE,
                FOREIGN KEY (topic_session_id) REFERENCES topic_sessions(topic_session_id) ON DELETE SET NULL
            );

            CREATE INDEX IF NOT EXISTS idx_agreed_specs_lab_id ON agreed_specs(lab_id);

            CREATE TABLE IF NOT EXISTS submissions (
                submission_id TEXT PRIMARY KEY,
                lab_id TEXT NOT NULL,
                student_name TEXT DEFAULT '',
                title TEXT DEFAULT '',
                description TEXT DEFAULT '',
                file_bundle_json TEXT NOT NULL DEFAULT '{}',
                analysis_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (lab_id) REFERENCES labs(lab_id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_submissions_lab_id ON submissions(lab_id);

            CREATE TABLE IF NOT EXISTS defense_sessions (
                defense_session_id TEXT PRIMARY KEY,
                lab_id TEXT NOT NULL,
                submission_id TEXT,
                status TEXT NOT NULL DEFAULT 'draft',
                plan_json TEXT NOT NULL DEFAULT '{}',
                summary_json TEXT NOT NULL DEFAULT '{}',
                score_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (lab_id) REFERENCES labs(lab_id) ON DELETE CASCADE,
                FOREIGN KEY (submission_id) REFERENCES submissions(submission_id) ON DELETE SET NULL
            );

            CREATE INDEX IF NOT EXISTS idx_defense_sessions_lab_id ON defense_sessions(lab_id);
            CREATE INDEX IF NOT EXISTS idx_defense_sessions_submission_id ON defense_sessions(submission_id);

            CREATE TABLE IF NOT EXISTS qa_turns (
                qa_turn_id TEXT PRIMARY KEY,
                defense_session_id TEXT NOT NULL,
                question_text TEXT NOT NULL,
                answer_text TEXT DEFAULT '',
                evaluation_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                FOREIGN KEY (defense_session_id) REFERENCES defense_sessions(defense_session_id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_qa_turns_defense_session_id ON qa_turns(defense_session_id);

            CREATE TABLE IF NOT EXISTS teacher_feedback (
                teacher_feedback_id TEXT PRIMARY KEY,
                lab_id TEXT NOT NULL,
                defense_session_id TEXT,
                feedback_text TEXT NOT NULL,
                extra_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                FOREIGN KEY (lab_id) REFERENCES labs(lab_id) ON DELETE CASCADE,
                FOREIGN KEY (defense_session_id) REFERENCES defense_sessions(defense_session_id) ON DELETE SET NULL
            );

            CREATE INDEX IF NOT EXISTS idx_teacher_feedback_lab_id ON teacher_feedback(lab_id);

            CREATE TABLE IF NOT EXISTS student_feedback (
                student_feedback_id TEXT PRIMARY KEY,
                lab_id TEXT NOT NULL,
                defense_session_id TEXT,
                feedback_text TEXT NOT NULL,
                extra_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                FOREIGN KEY (lab_id) REFERENCES labs(lab_id) ON DELETE CASCADE,
                FOREIGN KEY (defense_session_id) REFERENCES defense_sessions(defense_session_id) ON DELETE SET NULL
            );

            CREATE INDEX IF NOT EXISTS idx_student_feedback_lab_id ON student_feedback(lab_id);

            CREATE TABLE IF NOT EXISTS policy_items (
                policy_item_id TEXT PRIMARY KEY,
                lab_id TEXT,
                kind TEXT NOT NULL DEFAULT 'general',
                title TEXT NOT NULL,
                body_text TEXT NOT NULL,
                score REAL,
                source TEXT NOT NULL DEFAULT 'teacher_feedback',
                meta_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (lab_id) REFERENCES labs(lab_id) ON DELETE SET NULL
            );

            CREATE INDEX IF NOT EXISTS idx_policy_items_lab_id ON policy_items(lab_id);
            CREATE INDEX IF NOT EXISTS idx_policy_items_kind ON policy_items(kind);

            CREATE TABLE IF NOT EXISTS evaluation_cases (
                case_id TEXT PRIMARY KEY,
                lab_id TEXT,
                scenario_part TEXT NOT NULL,
                method_name TEXT NOT NULL,
                title TEXT DEFAULT '',
                description TEXT DEFAULT '',
                input_json TEXT NOT NULL DEFAULT '{}',
                generated_output_json TEXT NOT NULL DEFAULT '{}',
                expected_notes TEXT DEFAULT '',
                tags_json TEXT NOT NULL DEFAULT '[]',
                is_active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (lab_id) REFERENCES labs(lab_id) ON DELETE SET NULL
            );

            CREATE INDEX IF NOT EXISTS idx_evaluation_cases_lab_id ON evaluation_cases(lab_id);
            CREATE INDEX IF NOT EXISTS idx_evaluation_cases_scenario_part ON evaluation_cases(scenario_part);
            CREATE INDEX IF NOT EXISTS idx_evaluation_cases_method_name ON evaluation_cases(method_name);
            CREATE INDEX IF NOT EXISTS idx_evaluation_cases_is_active ON evaluation_cases(is_active);

            CREATE TABLE IF NOT EXISTS evaluation_runs (
                run_id TEXT PRIMARY KEY,
                title TEXT DEFAULT '',
                model_name TEXT DEFAULT '',
                judge_model_name TEXT DEFAULT '',
                k REAL NOT NULL DEFAULT 0.7,
                status TEXT NOT NULL DEFAULT 'created',
                meta_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                finished_at TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_evaluation_runs_status ON evaluation_runs(status);
            CREATE INDEX IF NOT EXISTS idx_evaluation_runs_created_at ON evaluation_runs(created_at);

            CREATE TABLE IF NOT EXISTS evaluation_results (
                result_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                case_id TEXT,
                scenario_part TEXT NOT NULL,
                method_name TEXT NOT NULL,
                llm_scores_json TEXT NOT NULL DEFAULT '{}',
                llm_score REAL,
                heuristic_metrics_json TEXT NOT NULL DEFAULT '[]',
                heuristic_score REAL,
                final_score REAL,
                result_json TEXT NOT NULL DEFAULT '{}',
                comment TEXT DEFAULT '',
                created_at TEXT NOT NULL,
                FOREIGN KEY (run_id) REFERENCES evaluation_runs(run_id) ON DELETE CASCADE,
                FOREIGN KEY (case_id) REFERENCES evaluation_cases(case_id) ON DELETE SET NULL
            );

            CREATE INDEX IF NOT EXISTS idx_evaluation_results_run_id ON evaluation_results(run_id);
            CREATE INDEX IF NOT EXISTS idx_evaluation_results_case_id ON evaluation_results(case_id);
            CREATE INDEX IF NOT EXISTS idx_evaluation_results_scenario_part ON evaluation_results(scenario_part);
            """
        )

    def _migrate_to_v2_best_effort(self, conn: sqlite3.Connection) -> None:
        """
        Мягкая миграция без предположений о полной старой схеме.
        Если таблиц/колонок нет — создаем.
        Если таблицы есть, но не хватает колонок — добавляем.
        """
        self._create_schema_v2(conn)

        self._ensure_column(conn, "labs", "work_type", "TEXT NOT NULL DEFAULT 'other'")
        self._ensure_column(conn, "labs", "status", "TEXT NOT NULL DEFAULT 'draft'")
        self._ensure_column(conn, "labs", "config_json", "TEXT NOT NULL DEFAULT '{}'")
        self._ensure_column(conn, "labs", "agreed_spec_id", "TEXT")
        self._ensure_column(conn, "labs", "updated_at", "TEXT")

        self._ensure_column(conn, "materials", "owner_role", "TEXT NOT NULL DEFAULT 'teacher'")
        self._ensure_column(conn, "materials", "stage", "TEXT NOT NULL DEFAULT 'general'")
        self._ensure_column(conn, "materials", "meta_json", "TEXT NOT NULL DEFAULT '{}'")
        self._ensure_column(conn, "materials", "extracted_text", "TEXT DEFAULT ''")

        # Подтягиваем updated_at там, где нужно
        self._ensure_column(conn, "submissions", "updated_at", "TEXT")
        self._ensure_column(conn, "defense_sessions", "updated_at", "TEXT")

        # Если updated_at пустой — выставим created_at
        self._backfill_updated_at(conn, "labs")
        self._backfill_updated_at(conn, "submissions")
        self._backfill_updated_at(conn, "defense_sessions")

    def _ensure_column(self, conn: sqlite3.Connection, table: str, column: str, definition: str) -> None:
        columns = self._get_table_columns(conn, table)
        if column not in columns:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition};")

    def _get_table_columns(self, conn: sqlite3.Connection, table: str) -> set[str]:
        rows = conn.execute(f"PRAGMA table_info({table});").fetchall()
        return {row["name"] for row in rows}

    def _backfill_updated_at(self, conn: sqlite3.Connection, table: str) -> None:
        columns = self._get_table_columns(conn, table)
        if "created_at" in columns and "updated_at" in columns:
            conn.execute(
                f"""
                UPDATE {table}
                SET updated_at = COALESCE(updated_at, created_at, ?)
                WHERE updated_at IS NULL OR updated_at = '';
                """,
                (utc_now(),),
            )

    # Labs / assignments

    def create_lab(
            self,
            *,
            title: str,
            description: str = "",
            work_type: str = WORK_TYPE_OTHER,
            status: str = STATUS_DRAFT,
            config: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        lab_id = new_id("lab")
        now = utc_now()

        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO labs (
                    lab_id, title, description, work_type, status, config_json, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (lab_id, title, description, work_type, status, dumps_json(config), now, now),
            )
            row = conn.execute("SELECT * FROM labs WHERE lab_id = ?;", (lab_id,)).fetchone()
            return self._decode_lab(row_to_dict(row))

    def list_labs(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM labs ORDER BY created_at DESC, title ASC;"
            ).fetchall()
            return [self._decode_lab(item) for item in rows_to_dicts(rows)]

    def get_lab(self, lab_id: str) -> Optional[dict[str, Any]]:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM labs WHERE lab_id = ?;", (lab_id,)).fetchone()
            return self._decode_lab(row_to_dict(row))

    def update_lab(
            self,
            lab_id: str,
            *,
            title: Optional[str] = None,
            description: Optional[str] = None,
            work_type: Optional[str] = None,
            status: Optional[str] = None,
            config: Optional[dict[str, Any]] = None,
            agreed_spec_id: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        current = self.get_lab(lab_id)
        if not current:
            return None

        merged_config = current["config_json"]
        if config is not None:
            merged = dict(merged_config)
            merged.update(config)
            merged_config = merged

        now = utc_now()

        with self.connect() as conn:
            conn.execute(
                """
                UPDATE labs
                SET
                    title = ?,
                    description = ?,
                    work_type = ?,
                    status = ?,
                    config_json = ?,
                    agreed_spec_id = ?,
                    updated_at = ?
                WHERE lab_id = ?;
                """,
                (
                    title if title is not None else current["title"],
                    description if description is not None else current["description"],
                    work_type if work_type is not None else current["work_type"],
                    status if status is not None else current["status"],
                    dumps_json(merged_config),
                    agreed_spec_id if agreed_spec_id is not None else current["agreed_spec_id"],
                    now,
                    lab_id,
                ),
            )
            row = conn.execute("SELECT * FROM labs WHERE lab_id = ?;", (lab_id,)).fetchone()
            return self._decode_lab(row_to_dict(row))

    def delete_lab(self, lab_id: str) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM labs WHERE lab_id = ?;", (lab_id,))

    # Materials

    def add_material(
            self,
            *,
            lab_id: str,
            filename: str,
            owner_role: str = MATERIAL_ROLE_TEACHER,
            stage: str = MATERIAL_STAGE_GENERAL,
            title: str = "",
            mime_type: str = "",
            file_path: str = "",
            extracted_text: str = "",
            meta: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        material_id = new_id("mat")
        now = utc_now()

        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO materials (
                    material_id, lab_id, owner_role, stage, title, filename, mime_type,
                    file_path, extracted_text, meta_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    material_id,
                    lab_id,
                    owner_role,
                    stage,
                    title,
                    filename,
                    mime_type,
                    file_path,
                    extracted_text,
                    dumps_json(meta),
                    now,
                ),
            )
            row = conn.execute("SELECT * FROM materials WHERE material_id = ?;", (material_id,)).fetchone()
            return self._decode_material(row_to_dict(row))

    def list_materials(
            self,
            lab_id: str,
            *,
            owner_role: Optional[str] = None,
            stage: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        where = ["lab_id = ?"]
        params: list[Any] = [lab_id]

        if owner_role:
            where.append("owner_role = ?")
            params.append(owner_role)

        if stage:
            where.append("stage = ?")
            params.append(stage)

        sql = f"""
        SELECT * FROM materials
        WHERE {" AND ".join(where)}
        ORDER BY created_at ASC, filename ASC;
        """

        with self.connect() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()
            return [self._decode_material(item) for item in rows_to_dicts(rows)]

    def get_material(self, material_id: str) -> Optional[dict[str, Any]]:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM materials WHERE material_id = ?;", (material_id,)).fetchone()
            return self._decode_material(row_to_dict(row))

    def delete_material(self, material_id: str) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM materials WHERE material_id = ?;", (material_id,))

    # Topic alignment

    def create_topic_session(
            self,
            *,
            lab_id: str,
            status: str = STATUS_DRAFT,
    ) -> dict[str, Any]:
        topic_session_id = new_id("topic")
        now = utc_now()

        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO topic_sessions (
                    topic_session_id, lab_id, status, round_no, relation_score, relation_label,
                    summary_text, llm_assessment_json, created_at, updated_at
                )
                VALUES (?, ?, ?, 0, NULL, '', '', '{}', ?, ?);
                """,
                (topic_session_id, lab_id, status, now, now),
            )
            row = conn.execute(
                "SELECT * FROM topic_sessions WHERE topic_session_id = ?;",
                (topic_session_id,),
            ).fetchone()
            return self._decode_topic_session(row_to_dict(row))

    def get_topic_session(self, topic_session_id: str) -> Optional[dict[str, Any]]:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM topic_sessions WHERE topic_session_id = ?;",
                (topic_session_id,),
            ).fetchone()
            return self._decode_topic_session(row_to_dict(row))

    def get_latest_topic_session_for_lab(self, lab_id: str) -> Optional[dict[str, Any]]:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM topic_sessions
                WHERE lab_id = ?
                ORDER BY created_at DESC
                LIMIT 1;
                """,
                (lab_id,),
            ).fetchone()
            return self._decode_topic_session(row_to_dict(row))

    def update_topic_session(
            self,
            topic_session_id: str,
            *,
            status: Optional[str] = None,
            round_no: Optional[int] = None,
            relation_score: Optional[float] = None,
            relation_label: Optional[str] = None,
            summary_text: Optional[str] = None,
            llm_assessment: Optional[dict[str, Any]] = None,
    ) -> Optional[dict[str, Any]]:
        current = self.get_topic_session(topic_session_id)
        if not current:
            return None

        now = utc_now()

        merged_assessment = current["llm_assessment_json"]
        if llm_assessment is not None:
            merged = dict(merged_assessment)
            merged.update(llm_assessment)
            merged_assessment = merged

        with self.connect() as conn:
            conn.execute(
                """
                UPDATE topic_sessions
                SET
                    status = ?,
                    round_no = ?,
                    relation_score = ?,
                    relation_label = ?,
                    summary_text = ?,
                    llm_assessment_json = ?,
                    updated_at = ?
                WHERE topic_session_id = ?;
                """,
                (
                    status if status is not None else current["status"],
                    round_no if round_no is not None else current["round_no"],
                    relation_score if relation_score is not None else current["relation_score"],
                    relation_label if relation_label is not None else current["relation_label"],
                    summary_text if summary_text is not None else current["summary_text"],
                    dumps_json(merged_assessment),
                    now,
                    topic_session_id,
                ),
            )
            row = conn.execute(
                "SELECT * FROM topic_sessions WHERE topic_session_id = ?;",
                (topic_session_id,),
            ).fetchone()
            return self._decode_topic_session(row_to_dict(row))

    def upsert_topic_input(
            self,
            *,
            topic_session_id: str,
            side: str,
            title: str,
            description: str = "",
            context: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        now = utc_now()

        with self.connect() as conn:
            existing = conn.execute(
                """
                SELECT * FROM topic_inputs
                WHERE topic_session_id = ? AND side = ?;
                """,
                (topic_session_id, side),
            ).fetchone()

            if existing:
                topic_input_id = existing["topic_input_id"]
                current_context = loads_json(existing["context_json"], default={})
                merged = dict(current_context)
                if context:
                    merged.update(context)

                conn.execute(
                    """
                    UPDATE topic_inputs
                    SET title = ?, description = ?, context_json = ?, updated_at = ?
                    WHERE topic_input_id = ?;
                    """,
                    (title, description, dumps_json(merged), now, topic_input_id),
                )
            else:
                topic_input_id = new_id("tinput")
                conn.execute(
                    """
                    INSERT INTO topic_inputs (
                        topic_input_id, topic_session_id, side, title, description,
                        context_json, created_at, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        topic_input_id,
                        topic_session_id,
                        side,
                        title,
                        description,
                        dumps_json(context),
                        now,
                        now,
                    ),
                )

            row = conn.execute(
                "SELECT * FROM topic_inputs WHERE topic_input_id = ?;",
                (topic_input_id,),
            ).fetchone()
            return self._decode_topic_input(row_to_dict(row))

    def list_topic_inputs(self, topic_session_id: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM topic_inputs
                WHERE topic_session_id = ?
                ORDER BY created_at ASC;
                """,
                (topic_session_id,),
            ).fetchall()
            return [self._decode_topic_input(item) for item in rows_to_dicts(rows)]

    def get_topic_input(self, topic_session_id: str, side: str) -> Optional[dict[str, Any]]:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM topic_inputs
                WHERE topic_session_id = ? AND side = ?;
                """,
                (topic_session_id, side),
            ).fetchone()
            return self._decode_topic_input(row_to_dict(row))

    def add_topic_turn(
            self,
            *,
            topic_session_id: str,
            side: str,
            turn_kind: str,
            question_text: str = "",
            answer_text: str = "",
            extra: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        topic_turn_id = new_id("tturn")
        now = utc_now()

        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO topic_turns (
                    topic_turn_id, topic_session_id, side, turn_kind,
                    question_text, answer_text, extra_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    topic_turn_id,
                    topic_session_id,
                    side,
                    turn_kind,
                    question_text,
                    answer_text,
                    dumps_json(extra),
                    now,
                ),
            )
            row = conn.execute(
                "SELECT * FROM topic_turns WHERE topic_turn_id = ?;",
                (topic_turn_id,),
            ).fetchone()
            return self._decode_topic_turn(row_to_dict(row))

    def list_topic_turns(self, topic_session_id: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM topic_turns
                WHERE topic_session_id = ?
                ORDER BY created_at ASC;
                """,
                (topic_session_id,),
            ).fetchall()
            return [self._decode_topic_turn(item) for item in rows_to_dicts(rows)]

    # Agreed specification

    def save_agreed_spec(
            self,
            *,
            lab_id: str,
            topic_session_id: Optional[str],
            work_type: str,
            agreed_title: str,
            agreed_description: str = "",
            acceptance_criteria: Optional[dict[str, Any]] = None,
            generated_from: Optional[dict[str, Any]] = None,
            status: str = STATUS_FINALIZED,
    ) -> dict[str, Any]:
        current = self.get_agreed_spec_by_lab(lab_id)
        now = utc_now()

        with self.connect() as conn:
            if current:
                spec_id = current["spec_id"]
                conn.execute(
                    """
                    UPDATE agreed_specs
                    SET
                        topic_session_id = ?,
                        work_type = ?,
                        agreed_title = ?,
                        agreed_description = ?,
                        acceptance_criteria_json = ?,
                        generated_from_json = ?,
                        status = ?,
                        updated_at = ?
                    WHERE spec_id = ?;
                    """,
                    (
                        topic_session_id,
                        work_type,
                        agreed_title,
                        agreed_description,
                        dumps_json(acceptance_criteria),
                        dumps_json(generated_from),
                        status,
                        now,
                        spec_id,
                    ),
                )
            else:
                spec_id = new_id("spec")
                conn.execute(
                    """
                    INSERT INTO agreed_specs (
                        spec_id, lab_id, topic_session_id, work_type, agreed_title,
                        agreed_description, acceptance_criteria_json, generated_from_json,
                        status, created_at, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        spec_id,
                        lab_id,
                        topic_session_id,
                        work_type,
                        agreed_title,
                        agreed_description,
                        dumps_json(acceptance_criteria),
                        dumps_json(generated_from),
                        status,
                        now,
                        now,
                    ),
                )

            conn.execute(
                """
                UPDATE labs
                SET agreed_spec_id = ?, work_type = ?, title = ?, description = ?, status = ?, updated_at = ?
                WHERE lab_id = ?;
                """,
                (
                    spec_id,
                    work_type,
                    agreed_title,
                    agreed_description,
                    STATUS_FINALIZED,
                    now,
                    lab_id,
                ),
            )

            row = conn.execute("SELECT * FROM agreed_specs WHERE spec_id = ?;", (spec_id,)).fetchone()
            return self._decode_agreed_spec(row_to_dict(row))

    def get_agreed_spec(self, spec_id: str) -> Optional[dict[str, Any]]:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM agreed_specs WHERE spec_id = ?;", (spec_id,)).fetchone()
            return self._decode_agreed_spec(row_to_dict(row))

    def get_agreed_spec_by_lab(self, lab_id: str) -> Optional[dict[str, Any]]:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM agreed_specs
                WHERE lab_id = ?
                ORDER BY created_at DESC
                LIMIT 1;
                """,
                (lab_id,),
            ).fetchone()
            return self._decode_agreed_spec(row_to_dict(row))

    # Submissions

    def create_submission(
            self,
            *,
            lab_id: str,
            student_name: str = "",
            title: str = "",
            description: str = "",
            file_bundle: Optional[dict[str, Any]] = None,
            analysis: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        submission_id = new_id("sub")
        now = utc_now()

        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO submissions (
                    submission_id, lab_id, student_name, title, description,
                    file_bundle_json, analysis_json, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    submission_id,
                    lab_id,
                    student_name,
                    title,
                    description,
                    dumps_json(file_bundle),
                    dumps_json(analysis),
                    now,
                    now,
                ),
            )
            row = conn.execute("SELECT * FROM submissions WHERE submission_id = ?;", (submission_id,)).fetchone()
            return self._decode_submission(row_to_dict(row))

    def update_submission(
            self,
            submission_id: str,
            *,
            student_name: Optional[str] = None,
            title: Optional[str] = None,
            description: Optional[str] = None,
            file_bundle: Optional[dict[str, Any]] = None,
            analysis: Optional[dict[str, Any]] = None,
    ) -> Optional[dict[str, Any]]:
        current = self.get_submission(submission_id)
        if not current:
            return None

        merged_file_bundle = current["file_bundle_json"]
        if file_bundle is not None:
            merged = dict(merged_file_bundle)
            merged.update(file_bundle)
            merged_file_bundle = merged

        merged_analysis = current["analysis_json"]
        if analysis is not None:
            merged = dict(merged_analysis)
            merged.update(analysis)
            merged_analysis = merged

        now = utc_now()

        with self.connect() as conn:
            conn.execute(
                """
                UPDATE submissions
                SET
                    student_name = ?,
                    title = ?,
                    description = ?,
                    file_bundle_json = ?,
                    analysis_json = ?,
                    updated_at = ?
                WHERE submission_id = ?;
                """,
                (
                    student_name if student_name is not None else current["student_name"],
                    title if title is not None else current["title"],
                    description if description is not None else current["description"],
                    dumps_json(merged_file_bundle),
                    dumps_json(merged_analysis),
                    now,
                    submission_id,
                ),
            )
            row = conn.execute("SELECT * FROM submissions WHERE submission_id = ?;", (submission_id,)).fetchone()
            return self._decode_submission(row_to_dict(row))

    def get_submission(self, submission_id: str) -> Optional[dict[str, Any]]:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM submissions WHERE submission_id = ?;", (submission_id,)).fetchone()
            return self._decode_submission(row_to_dict(row))

    def list_submissions(self, lab_id: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM submissions
                WHERE lab_id = ?
                ORDER BY created_at DESC;
                """,
                (lab_id,),
            ).fetchall()
            return [self._decode_submission(item) for item in rows_to_dicts(rows)]

    # Defense

    def create_defense_session(
            self,
            *,
            lab_id: str,
            submission_id: Optional[str] = None,
            status: str = STATUS_DRAFT,
            plan: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        defense_session_id = new_id("def")
        now = utc_now()

        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO defense_sessions (
                    defense_session_id, lab_id, submission_id, status,
                    plan_json, summary_json, score_json, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, '{}', '{}', ?, ?);
                """,
                (
                    defense_session_id,
                    lab_id,
                    submission_id,
                    status,
                    dumps_json(plan),
                    now,
                    now,
                ),
            )
            row = conn.execute(
                "SELECT * FROM defense_sessions WHERE defense_session_id = ?;",
                (defense_session_id,),
            ).fetchone()
            return self._decode_defense_session(row_to_dict(row))

    def get_defense_session(self, defense_session_id: str) -> Optional[dict[str, Any]]:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM defense_sessions WHERE defense_session_id = ?;",
                (defense_session_id,),
            ).fetchone()
            return self._decode_defense_session(row_to_dict(row))

    def list_defense_sessions(self, lab_id: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM defense_sessions
                WHERE lab_id = ?
                ORDER BY created_at DESC;
                """,
                (lab_id,),
            ).fetchall()
            return [self._decode_defense_session(item) for item in rows_to_dicts(rows)]

    def update_defense_session(
            self,
            defense_session_id: str,
            *,
            status: Optional[str] = None,
            plan: Optional[dict[str, Any]] = None,
            summary: Optional[dict[str, Any]] = None,
            score: Optional[dict[str, Any]] = None,
    ) -> Optional[dict[str, Any]]:
        current = self.get_defense_session(defense_session_id)
        if not current:
            return None

        merged_plan = current["plan_json"]
        if plan is not None:
            merged = dict(merged_plan)
            merged.update(plan)
            merged_plan = merged

        merged_summary = current["summary_json"]
        if summary is not None:
            merged = dict(merged_summary)
            merged.update(summary)
            merged_summary = merged

        merged_score = current["score_json"]
        if score is not None:
            merged = dict(merged_score)
            merged.update(score)
            merged_score = merged

        now = utc_now()

        with self.connect() as conn:
            conn.execute(
                """
                UPDATE defense_sessions
                SET
                    status = ?,
                    plan_json = ?,
                    summary_json = ?,
                    score_json = ?,
                    updated_at = ?
                WHERE defense_session_id = ?;
                """,
                (
                    status if status is not None else current["status"],
                    dumps_json(merged_plan),
                    dumps_json(merged_summary),
                    dumps_json(merged_score),
                    now,
                    defense_session_id,
                ),
            )
            row = conn.execute(
                "SELECT * FROM defense_sessions WHERE defense_session_id = ?;",
                (defense_session_id,),
            ).fetchone()
            return self._decode_defense_session(row_to_dict(row))

    def add_qa_turn(
            self,
            *,
            defense_session_id: str,
            question_text: str,
            answer_text: str = "",
            evaluation: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        qa_turn_id = new_id("qa")
        now = utc_now()

        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO qa_turns (
                    qa_turn_id, defense_session_id, question_text,
                    answer_text, evaluation_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?);
                """,
                (
                    qa_turn_id,
                    defense_session_id,
                    question_text,
                    answer_text,
                    dumps_json(evaluation),
                    now,
                ),
            )
            row = conn.execute("SELECT * FROM qa_turns WHERE qa_turn_id = ?;", (qa_turn_id,)).fetchone()
            return self._decode_qa_turn(row_to_dict(row))

    def list_qa_turns(self, defense_session_id: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM qa_turns
                WHERE defense_session_id = ?
                ORDER BY created_at ASC;
                """,
                (defense_session_id,),
            ).fetchall()
            return [self._decode_qa_turn(item) for item in rows_to_dicts(rows)]

    # Feedback

    def add_teacher_feedback(
            self,
            *,
            lab_id: str,
            feedback_text: str,
            defense_session_id: Optional[str] = None,
            extra: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        teacher_feedback_id = new_id("tfb")
        now = utc_now()

        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO teacher_feedback (
                    teacher_feedback_id, lab_id, defense_session_id,
                    feedback_text, extra_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?);
                """,
                (
                    teacher_feedback_id,
                    lab_id,
                    defense_session_id,
                    feedback_text,
                    dumps_json(extra),
                    now,
                ),
            )
            row = conn.execute(
                "SELECT * FROM teacher_feedback WHERE teacher_feedback_id = ?;",
                (teacher_feedback_id,),
            ).fetchone()
            return self._decode_feedback(row_to_dict(row))

    def list_teacher_feedback(self, lab_id: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM teacher_feedback
                WHERE lab_id = ?
                ORDER BY created_at DESC;
                """,
                (lab_id,),
            ).fetchall()
            return [self._decode_feedback(item) for item in rows_to_dicts(rows)]

    def add_student_feedback(
            self,
            *,
            lab_id: str,
            feedback_text: str,
            defense_session_id: Optional[str] = None,
            extra: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        student_feedback_id = new_id("sfb")
        now = utc_now()

        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO student_feedback (
                    student_feedback_id, lab_id, defense_session_id,
                    feedback_text, extra_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?);
                """,
                (
                    student_feedback_id,
                    lab_id,
                    defense_session_id,
                    feedback_text,
                    dumps_json(extra),
                    now,
                ),
            )
            row = conn.execute(
                "SELECT * FROM student_feedback WHERE student_feedback_id = ?;",
                (student_feedback_id,),
            ).fetchone()
            return self._decode_feedback(row_to_dict(row))

    def list_student_feedback(self, lab_id: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM student_feedback
                WHERE lab_id = ?
                ORDER BY created_at DESC;
                """,
                (lab_id,),
            ).fetchall()
            return [self._decode_feedback(item) for item in rows_to_dicts(rows)]

    # Policy memory

    def add_policy_item(
            self,
            *,
            title: str,
            body_text: str,
            kind: str = "general",
            lab_id: Optional[str] = None,
            score: Optional[float] = None,
            source: str = "teacher_feedback",
            meta: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        policy_item_id = new_id("pol")
        now = utc_now()

        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO policy_items (
                    policy_item_id, lab_id, kind, title, body_text,
                    score, source, meta_json, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    policy_item_id,
                    lab_id,
                    kind,
                    title,
                    body_text,
                    score,
                    source,
                    dumps_json(meta),
                    now,
                    now,
                ),
            )
            row = conn.execute(
                "SELECT * FROM policy_items WHERE policy_item_id = ?;",
                (policy_item_id,),
            ).fetchone()
            return self._decode_policy_item(row_to_dict(row))

    def list_policy_items(
            self,
            *,
            lab_id: Optional[str] = None,
            kind: Optional[str] = None,
            limit: int = 100,
    ) -> list[dict[str, Any]]:
        where: list[str] = []
        params: list[Any] = []

        if lab_id is not None:
            where.append("lab_id = ?")
            params.append(lab_id)

        if kind is not None:
            where.append("kind = ?")
            params.append(kind)

        where_sql = f"WHERE {' AND '.join(where)}" if where else ""

        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM policy_items
                {where_sql}
                ORDER BY updated_at DESC, created_at DESC
                LIMIT ?;
                """,
                (*params, limit),
            ).fetchall()
            return [self._decode_policy_item(item) for item in rows_to_dicts(rows)]

    # High-level helpers for new flow

    def create_assignment_with_topic_session(
            self,
            *,
            teacher_title: str,
            teacher_description: str = "",
            work_type: str = WORK_TYPE_OTHER,
            config: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Удобный helper:
        1) создает lab,
        2) сразу создает topic_session.
        """
        lab = self.create_lab(
            title=teacher_title or "Новое задание",
            description=teacher_description,
            work_type=work_type,
            status=STATUS_DRAFT,
            config=config,
        )
        topic_session = self.create_topic_session(lab_id=lab["lab_id"], status=STATUS_DRAFT)
        lab = self.update_lab(lab["lab_id"], config={"topic_session_id": topic_session["topic_session_id"]})
        return {
            "lab": lab,
            "topic_session": topic_session,
        }

    def finalize_assignment_from_agreed_spec(
            self,
            *,
            lab_id: str,
            topic_session_id: Optional[str],
            work_type: str,
            agreed_title: str,
            agreed_description: str,
            acceptance_criteria: Optional[dict[str, Any]] = None,
            generated_from: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        spec = self.save_agreed_spec(
            lab_id=lab_id,
            topic_session_id=topic_session_id,
            work_type=work_type,
            agreed_title=agreed_title,
            agreed_description=agreed_description,
            acceptance_criteria=acceptance_criteria,
            generated_from=generated_from,
            status=STATUS_FINALIZED,
        )

        if topic_session_id:
            self.update_topic_session(
                topic_session_id,
                status=STATUS_FINALIZED,
                summary_text="Согласованная тема зафиксирована.",
            )

        lab = self.update_lab(
            lab_id,
            title=agreed_title,
            description=agreed_description,
            work_type=work_type,
            status=STATUS_FINALIZED,
            agreed_spec_id=spec["spec_id"],
            config={"assignment_ready": True},
        )

        return {
            "lab": lab,
            "agreed_spec": spec,
        }

    # Evaluation cases / generation quality checks

    def create_evaluation_case(
            self,
            *,
            scenario_part: str,
            method_name: str,
            title: str = "",
            description: str = "",
            lab_id: Optional[str] = None,
            input_data: Optional[dict[str, Any]] = None,
            generated_output: Optional[dict[str, Any]] = None,
            input_json: Optional[dict[str, Any]] = None,
            generated_output_json: Optional[dict[str, Any]] = None,
            expected_notes: str = "",
            tags: Optional[list[str]] = None,
            is_active: bool = True,
    ) -> dict[str, Any]:
        """
        Создает сохраненный кейс оценки генерации.

        Это не unit-тест кода, а тестовый пример качества генерации:
        входной контекст + результат генерации + часть сценария, которую нужно оценить.
        """
        case_id = new_id("ecase")
        now = utc_now()
        input_payload = input_json if input_json is not None else input_data
        output_payload = generated_output_json if generated_output_json is not None else generated_output

        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO evaluation_cases (
                    case_id, lab_id, scenario_part, method_name, title, description,
                    input_json, generated_output_json, expected_notes, tags_json,
                    is_active, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    case_id,
                    lab_id,
                    scenario_part,
                    method_name,
                    title,
                    description,
                    dumps_json(input_payload),
                    dumps_json(output_payload),
                    expected_notes,
                    dumps_json(tags or []),
                    1 if is_active else 0,
                    now,
                    now,
                ),
            )
            row = conn.execute(
                "SELECT * FROM evaluation_cases WHERE case_id = ?;",
                (case_id,),
            ).fetchone()
            return self._decode_evaluation_case(row_to_dict(row))

    def update_evaluation_case(
            self,
            case_id: str,
            *,
            scenario_part: Optional[str] = None,
            method_name: Optional[str] = None,
            title: Optional[str] = None,
            description: Optional[str] = None,
            lab_id: Optional[str] = None,
            input_data: Optional[dict[str, Any]] = None,
            generated_output: Optional[dict[str, Any]] = None,
            input_json: Optional[dict[str, Any]] = None,
            generated_output_json: Optional[dict[str, Any]] = None,
            expected_notes: Optional[str] = None,
            tags: Optional[list[str]] = None,
            is_active: Optional[bool] = None,
    ) -> Optional[dict[str, Any]]:
        """
        Обновляет сохраненный кейс оценки генерации.
        """
        current = self.get_evaluation_case(case_id)
        if not current:
            return None

        input_payload = current["input_json"]
        if input_data is not None:
            input_payload = input_data
        if input_json is not None:
            input_payload = input_json

        output_payload = current["generated_output_json"]
        if generated_output is not None:
            output_payload = generated_output
        if generated_output_json is not None:
            output_payload = generated_output_json

        now = utc_now()

        with self.connect() as conn:
            conn.execute(
                """
                UPDATE evaluation_cases
                SET
                    lab_id = ?,
                    scenario_part = ?,
                    method_name = ?,
                    title = ?,
                    description = ?,
                    input_json = ?,
                    generated_output_json = ?,
                    expected_notes = ?,
                    tags_json = ?,
                    is_active = ?,
                    updated_at = ?
                WHERE case_id = ?;
                """,
                (
                    lab_id if lab_id is not None else current["lab_id"],
                    scenario_part if scenario_part is not None else current["scenario_part"],
                    method_name if method_name is not None else current["method_name"],
                    title if title is not None else current["title"],
                    description if description is not None else current["description"],
                    dumps_json(input_payload),
                    dumps_json(output_payload),
                    expected_notes if expected_notes is not None else current["expected_notes"],
                    dumps_json(tags if tags is not None else current["tags_json"]),
                    1 if (is_active if is_active is not None else current["is_active"]) else 0,
                    now,
                    case_id,
                ),
            )
            row = conn.execute(
                "SELECT * FROM evaluation_cases WHERE case_id = ?;",
                (case_id,),
            ).fetchone()
            return self._decode_evaluation_case(row_to_dict(row))

    def get_evaluation_case(self, case_id: str) -> Optional[dict[str, Any]]:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM evaluation_cases WHERE case_id = ?;",
                (case_id,),
            ).fetchone()
            return self._decode_evaluation_case(row_to_dict(row))

    def list_evaluation_cases(
            self,
            *,
            lab_id: Optional[str] = None,
            scenario_part: Optional[str] = None,
            method_name: Optional[str] = None,
            active_only: bool = False,
            limit: int = 200,
    ) -> list[dict[str, Any]]:
        where: list[str] = []
        params: list[Any] = []

        if lab_id is not None:
            where.append("lab_id = ?")
            params.append(lab_id)
        if scenario_part is not None:
            where.append("scenario_part = ?")
            params.append(scenario_part)
        if method_name is not None:
            where.append("method_name = ?")
            params.append(method_name)
        if active_only:
            where.append("is_active = 1")

        where_sql = f"WHERE {' AND '.join(where)}" if where else ""

        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM evaluation_cases
                {where_sql}
                ORDER BY updated_at DESC, created_at DESC
                LIMIT ?;
                """,
                (*params, limit),
            ).fetchall()
            return [self._decode_evaluation_case(item) for item in rows_to_dicts(rows)]

    def delete_evaluation_case(self, case_id: str) -> None:
        """
        Удаляет кейс. Результаты прошлых запусков сохраняются, но case_id в них станет NULL.
        """
        with self.connect() as conn:
            conn.execute("DELETE FROM evaluation_cases WHERE case_id = ?;", (case_id,))

    def create_evaluation_run(
            self,
            *,
            title: str = "",
            model_name: str = "",
            judge_model_name: str = "",
            k: float = 0.7,
            status: str = EVALUATION_STATUS_CREATED,
            meta: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Создает запуск оценки генерации.

        k — коэффициент доверия к LLM-оценщику в формуле:
        final_score = k * llm_score + (1 - k) * heuristic_score.
        """
        run_id = new_id("erun")
        now = utc_now()

        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO evaluation_runs (
                    run_id, title, model_name, judge_model_name, k,
                    status, meta_json, created_at, finished_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL);
                """,
                (
                    run_id,
                    title,
                    model_name,
                    judge_model_name,
                    float(k),
                    status,
                    dumps_json(meta),
                    now,
                ),
            )
            row = conn.execute(
                "SELECT * FROM evaluation_runs WHERE run_id = ?;",
                (run_id,),
            ).fetchone()
            return self._decode_evaluation_run(row_to_dict(row))

    def update_evaluation_run(
            self,
            run_id: str,
            *,
            title: Optional[str] = None,
            model_name: Optional[str] = None,
            judge_model_name: Optional[str] = None,
            k: Optional[float] = None,
            status: Optional[str] = None,
            meta: Optional[dict[str, Any]] = None,
            finished_at: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        current = self.get_evaluation_run(run_id)
        if not current:
            return None

        merged_meta = current["meta_json"]
        if meta is not None:
            merged = dict(merged_meta)
            merged.update(meta)
            merged_meta = merged

        resolved_status = status if status is not None else current["status"]
        resolved_finished_at = finished_at
        if resolved_finished_at is None:
            resolved_finished_at = current.get("finished_at")
        if status in {EVALUATION_STATUS_FINISHED, EVALUATION_STATUS_FAILED} and not resolved_finished_at:
            resolved_finished_at = utc_now()

        with self.connect() as conn:
            conn.execute(
                """
                UPDATE evaluation_runs
                SET
                    title = ?,
                    model_name = ?,
                    judge_model_name = ?,
                    k = ?,
                    status = ?,
                    meta_json = ?,
                    finished_at = ?
                WHERE run_id = ?;
                """,
                (
                    title if title is not None else current["title"],
                    model_name if model_name is not None else current["model_name"],
                    judge_model_name if judge_model_name is not None else current["judge_model_name"],
                    float(k) if k is not None else current["k"],
                    resolved_status,
                    dumps_json(merged_meta),
                    resolved_finished_at,
                    run_id,
                ),
            )
            row = conn.execute(
                "SELECT * FROM evaluation_runs WHERE run_id = ?;",
                (run_id,),
            ).fetchone()
            return self._decode_evaluation_run(row_to_dict(row))

    def finish_evaluation_run(
            self,
            run_id: str,
            *,
            status: str = EVALUATION_STATUS_FINISHED,
            meta: Optional[dict[str, Any]] = None,
    ) -> Optional[dict[str, Any]]:
        """
        Помечает запуск оценки завершенным или завершенным с ошибкой.
        """
        return self.update_evaluation_run(
            run_id,
            status=status,
            meta=meta,
            finished_at=utc_now(),
        )

    def get_evaluation_run(self, run_id: str) -> Optional[dict[str, Any]]:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM evaluation_runs WHERE run_id = ?;",
                (run_id,),
            ).fetchone()
            return self._decode_evaluation_run(row_to_dict(row))

    def list_evaluation_runs(
            self,
            *,
            status: Optional[str] = None,
            limit: int = 100,
    ) -> list[dict[str, Any]]:
        where = "WHERE status = ?" if status else ""
        params: tuple[Any, ...] = (status, limit) if status else (limit,)

        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM evaluation_runs
                {where}
                ORDER BY created_at DESC
                LIMIT ?;
                """,
                params,
            ).fetchall()
            return [self._decode_evaluation_run(item) for item in rows_to_dicts(rows)]

    def delete_evaluation_run(self, run_id: str) -> None:
        """
        Удаляет запуск оценки и связанные с ним результаты.
        """
        with self.connect() as conn:
            conn.execute("DELETE FROM evaluation_runs WHERE run_id = ?;", (run_id,))

    def save_evaluation_result(
            self,
            *,
            run_id: str,
            case_id: Optional[str],
            scenario_part: str,
            method_name: str,
            llm_scores: Optional[dict[str, Any]] = None,
            llm_score: Optional[float] = None,
            heuristic_metrics: Optional[list[dict[str, Any]]] = None,
            heuristic_score: Optional[float] = None,
            final_score: Optional[float] = None,
            result: Optional[dict[str, Any]] = None,
            comment: str = "",
    ) -> dict[str, Any]:
        """
        Сохраняет результат оценки одного кейса в рамках запуска.
        """
        result_id = new_id("eres")
        now = utc_now()

        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO evaluation_results (
                    result_id, run_id, case_id, scenario_part, method_name,
                    llm_scores_json, llm_score, heuristic_metrics_json,
                    heuristic_score, final_score, result_json, comment, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    result_id,
                    run_id,
                    case_id,
                    scenario_part,
                    method_name,
                    dumps_json(llm_scores),
                    llm_score,
                    dumps_json(heuristic_metrics or []),
                    heuristic_score,
                    final_score,
                    dumps_json(result),
                    comment,
                    now,
                ),
            )
            row = conn.execute(
                "SELECT * FROM evaluation_results WHERE result_id = ?;",
                (result_id,),
            ).fetchone()
            return self._decode_evaluation_result(row_to_dict(row))

    def save_evaluation_result_from_dict(
            self,
            *,
            run_id: str,
            case_id: Optional[str],
            result: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Удобный адаптер для результата из evaluation.py.
        """
        llm_block = result.get("llm_score") or {}
        return self.save_evaluation_result(
            run_id=run_id,
            case_id=case_id,
            scenario_part=str(result.get("scenario_part") or ""),
            method_name=str(result.get("method_name") or ""),
            llm_scores=llm_block.get("raw_scores") or llm_block,
            llm_score=llm_block.get("normalized_score"),
            heuristic_metrics=result.get("heuristic_metrics") or [],
            heuristic_score=result.get("heuristic_score"),
            final_score=result.get("final_score") or result.get("scenario_score"),
            result=result,
            comment=str(result.get("comment") or ""),
        )

    def get_evaluation_result(self, result_id: str) -> Optional[dict[str, Any]]:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM evaluation_results WHERE result_id = ?;",
                (result_id,),
            ).fetchone()
            return self._decode_evaluation_result(row_to_dict(row))

    def list_evaluation_results(
            self,
            *,
            run_id: Optional[str] = None,
            case_id: Optional[str] = None,
            scenario_part: Optional[str] = None,
            limit: int = 500,
    ) -> list[dict[str, Any]]:
        where: list[str] = []
        params: list[Any] = []

        if run_id is not None:
            where.append("run_id = ?")
            params.append(run_id)
        if case_id is not None:
            where.append("case_id = ?")
            params.append(case_id)
        if scenario_part is not None:
            where.append("scenario_part = ?")
            params.append(scenario_part)

        where_sql = f"WHERE {' AND '.join(where)}" if where else ""

        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM evaluation_results
                {where_sql}
                ORDER BY created_at DESC
                LIMIT ?;
                """,
                (*params, limit),
            ).fetchall()
            return [self._decode_evaluation_result(item) for item in rows_to_dicts(rows)]

    def build_evaluation_report_rows(self, *, run_id: str) -> list[dict[str, Any]]:
        """
        Возвращает плоские строки для таблицы в Streamlit/CSV.
        """
        results = self.list_evaluation_results(run_id=run_id, limit=10000)
        rows: list[dict[str, Any]] = []
        for item in results:
            case = self.get_evaluation_case(item["case_id"]) if item.get("case_id") else None
            llm_scores = item.get("llm_scores_json") or {}
            rows.append(
                {
                    "run_id": item.get("run_id"),
                    "case_id": item.get("case_id"),
                    "case_title": case.get("title", "") if case else "",
                    "scenario_part": item.get("scenario_part"),
                    "method_name": item.get("method_name"),
                    "llm_relevance": llm_scores.get("relevance"),
                    "llm_completeness": llm_scores.get("completeness"),
                    "llm_clarity": llm_scores.get("clarity"),
                    "llm_usefulness": llm_scores.get("usefulness"),
                    "llm_correctness": llm_scores.get("correctness"),
                    "llm_score": item.get("llm_score"),
                    "heuristic_score": item.get("heuristic_score"),
                    "final_score": item.get("final_score"),
                    "comment": item.get("comment", ""),
                    "heuristic_metrics_json": dumps_json(item.get("heuristic_metrics_json") or []),
                    "result_json": dumps_json(item.get("result_json") or {}),
                }
            )
        return rows

    # Decode helpers

    def _decode_evaluation_case(self, item: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
        if not item:
            return None
        item["input_json"] = loads_json(item.get("input_json"), default={})
        item["generated_output_json"] = loads_json(item.get("generated_output_json"), default={})
        item["tags_json"] = loads_json(item.get("tags_json"), default=[])
        item["is_active"] = bool(item.get("is_active"))
        return item

    def _decode_evaluation_run(self, item: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
        if not item:
            return None
        item["meta_json"] = loads_json(item.get("meta_json"), default={})
        return item

    def _decode_evaluation_result(self, item: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
        if not item:
            return None
        item["llm_scores_json"] = loads_json(item.get("llm_scores_json"), default={})
        item["heuristic_metrics_json"] = loads_json(item.get("heuristic_metrics_json"), default=[])
        item["result_json"] = loads_json(item.get("result_json"), default={})
        return item

    def _decode_lab(self, item: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
        if not item:
            return None
        item["config_json"] = loads_json(item.get("config_json"), default={})
        return item

    def _decode_material(self, item: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
        if not item:
            return None
        item["meta_json"] = loads_json(item.get("meta_json"), default={})
        return item

    def _decode_topic_session(self, item: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
        if not item:
            return None
        item["llm_assessment_json"] = loads_json(item.get("llm_assessment_json"), default={})
        return item

    def _decode_topic_input(self, item: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
        if not item:
            return None
        item["context_json"] = loads_json(item.get("context_json"), default={})
        return item

    def _decode_topic_turn(self, item: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
        if not item:
            return None
        item["extra_json"] = loads_json(item.get("extra_json"), default={})
        return item

    def _decode_agreed_spec(self, item: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
        if not item:
            return None
        item["acceptance_criteria_json"] = loads_json(item.get("acceptance_criteria_json"), default={})
        item["generated_from_json"] = loads_json(item.get("generated_from_json"), default={})
        return item

    def _decode_submission(self, item: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
        if not item:
            return None
        item["file_bundle_json"] = loads_json(item.get("file_bundle_json"), default={})
        item["analysis_json"] = loads_json(item.get("analysis_json"), default={})
        return item

    def _decode_defense_session(self, item: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
        if not item:
            return None
        item["plan_json"] = loads_json(item.get("plan_json"), default={})
        item["summary_json"] = loads_json(item.get("summary_json"), default={})
        item["score_json"] = loads_json(item.get("score_json"), default={})
        return item

    def _decode_qa_turn(self, item: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
        if not item:
            return None
        item["evaluation_json"] = loads_json(item.get("evaluation_json"), default={})
        return item

    def _decode_feedback(self, item: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
        if not item:
            return None
        item["extra_json"] = loads_json(item.get("extra_json"), default={})
        return item

    def _decode_policy_item(self, item: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
        if not item:
            return None
        item["meta_json"] = loads_json(item.get("meta_json"), default={})
        return item


# Factory


def create_storage(db_path: str | Path) -> Storage:
    storage = Storage(db_path)
    storage.init_db()
    return storage


# Manual smoke test


if __name__ == "__main__":
    db = create_storage("data/app.sqlite3")

    bundle = db.create_assignment_with_topic_session(
        teacher_title="Тестовое задание",
        teacher_description="Проверка нового контура согласования темы",
        work_type=WORK_TYPE_RESEARCH,
        config={"topic_alignment_enabled": True},
    )

    lab = bundle["lab"]
    topic_session = bundle["topic_session"]

    db.upsert_topic_input(
        topic_session_id=topic_session["topic_session_id"],
        side=SIDE_STUDENT,
        title="Анализ документов организации",
        description="Интересует задача обработки внутренних документов и отчетности",
        context={"origin": "practice"},
    )

    db.upsert_topic_input(
        topic_session_id=topic_session["topic_session_id"],
        side=SIDE_TEACHER,
        title="Методы обработки текстовых данных в рамках дисциплины",
        description="Нужно, чтобы тема проверялась в пределах дисциплины",
        context={"discipline": "ИИ в образовании"},
    )

    db.add_material(
        lab_id=lab["lab_id"],
        filename="teacher_notes.pdf",
        owner_role=MATERIAL_ROLE_TEACHER,
        stage=MATERIAL_STAGE_TOPIC_ALIGNMENT,
        title="Материалы преподавателя",
        file_path="uploads/teacher_notes.pdf",
    )

    db.add_topic_turn(
        topic_session_id=topic_session["topic_session_id"],
        side=SIDE_TEACHER,
        turn_kind="question",
        question_text="Какие именно типы документов студент хочет анализировать?",
    )

    db.update_topic_session(
        topic_session["topic_session_id"],
        status=STATUS_NEEDS_CLARIFICATION,
        round_no=1,
        relation_score=0.46,
        relation_label="weak",
        summary_text="Есть частичное пересечение, но нужны уточнения.",
        llm_assessment={
            "overlap_points": ["обработка документов", "анализ содержания"],
            "conflicts": ["не заданы границы результата"],
        },
    )

    result = db.finalize_assignment_from_agreed_spec(
        lab_id=lab["lab_id"],
        topic_session_id=topic_session["topic_session_id"],
        work_type=WORK_TYPE_RESEARCH,
        agreed_title="Исследование методов автоматизированного анализа учебных и организационных документов",
        agreed_description="Студент выполняет исследовательское задание в рамках дисциплины с опорой на обработку текстовых данных.",
        acceptance_criteria={
            "deliverables": ["описание предметной области", "прототип", "отчет"],
            "evaluation_axes": ["корректность", "обоснованность", "соответствие дисциплине"],
        },
        generated_from={"source": "llm_alignment"},
    )

    print("LAB:")
    print(result["lab"])
    print("\nSPEC:")
    print(result["agreed_spec"])
