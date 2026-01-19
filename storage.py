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
import os
import re
import shutil
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

JsonDict = Dict[str, Any]


# Helpers


def now_iso() -> str:
    """UTC ISO-8601 timestamp."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def new_id(prefix: str) -> str:
    """Generate stable, unique IDs like lab_xxx, mat_xxx, sub_xxx, etc."""
    return f"{prefix}_{uuid.uuid4().hex}"


def _safe_filename(name: str, max_len: int = 120) -> str:
    """Sanitize filenames for disk storage."""
    name = name.strip().replace("\\", "_").replace("/", "_")
    name = re.sub(r"[^A-Za-z0-9А-Яа-яЁё_.\-\s]+", "_", name)
    name = re.sub(r"\s+", " ", name).strip()
    if not name:
        name = "file"
    if len(name) > max_len:
        stem, ext = os.path.splitext(name)
        name = stem[: max_len - len(ext)] + ext
    return name


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=False)


def _json_loads(s: Optional[str]) -> Any:
    if not s:
        return None
    return json.loads(s)


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def _atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding=encoding)
    tmp.replace(path)


# Data classes (returns)


@dataclass(frozen=True)
class Lab:
    lab_id: str
    title: str
    status: str
    config: JsonDict
    created_at: str
    updated_at: str
    published_version: int


@dataclass(frozen=True)
class Material:
    material_id: str
    lab_id: str
    kind: str
    filename: str
    mime: str
    original_path: str
    extracted_text_path: str
    meta: JsonDict
    created_at: str


@dataclass(frozen=True)
class Submission:
    submission_id: str
    lab_id: str
    student_id: Optional[str]
    filename: str
    mime: str
    original_path: str
    extracted_text_path: str
    meta: JsonDict
    created_at: str


@dataclass(frozen=True)
class DefenseSession:
    session_id: str
    lab_id: str
    submission_id: str
    student_label: str
    status: str
    started_at: str
    finished_at: Optional[str]
    policy_version: int
    system_summary: JsonDict
    teacher_summary: JsonDict


# Storage main


class Storage:
    """
    Основная точка доступа к данным.

    По умолчанию:
      data_root = ./data
      db_path   = ./data/app.sqlite3

    Можно переопределить через env:
      APP_DATA_DIR
      APP_DB_PATH
    """

    SCHEMA_VERSION = 1

    def __init__(self, data_root: Optional[Union[str, Path]] = None, db_path: Optional[Union[str, Path]] = None):
        env_data_dir = os.getenv("APP_DATA_DIR")
        env_db_path = os.getenv("APP_DB_PATH")

        self.data_root = Path(data_root or env_data_dir or "./data").resolve()
        self.db_path = Path(db_path or env_db_path or (self.data_root / "app.sqlite3")).resolve()

        self.data_root.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # SQLite utilities

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30, isolation_level=None, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        return conn

    @contextmanager
    def _tx(self) -> Iterable[sqlite3.Connection]:
        conn = self._connect()
        try:
            conn.execute("BEGIN;")
            yield conn
            conn.execute("COMMIT;")
        except Exception:
            conn.execute("ROLLBACK;")
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._tx() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS schema_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
            """)
            row = conn.execute("SELECT value FROM schema_meta WHERE key='schema_version';").fetchone()
            if row is None:
                conn.execute(
                    "INSERT INTO schema_meta(key,value) VALUES('schema_version', ?);",
                    (str(self.SCHEMA_VERSION),)
                )
            else:
                ver = int(row["value"])
                if ver != self.SCHEMA_VERSION:
                    # Для учебного проекта: простая защита от несовместимости
                    raise RuntimeError(f"DB schema_version={ver} != expected {self.SCHEMA_VERSION}. Migration needed.")

            # Main tables
            conn.execute("""
                CREATE TABLE IF NOT EXISTS labs (
                    lab_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    status TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    published_version INTEGER NOT NULL DEFAULT 0
                );
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS materials (
                    material_id TEXT PRIMARY KEY,
                    lab_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    mime TEXT NOT NULL,
                    original_path TEXT NOT NULL,
                    extracted_text_path TEXT NOT NULL,
                    meta_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(lab_id) REFERENCES labs(lab_id) ON DELETE CASCADE
                );
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS submissions (
                    submission_id TEXT PRIMARY KEY,
                    lab_id TEXT NOT NULL,
                    student_id TEXT,
                    filename TEXT NOT NULL,
                    mime TEXT NOT NULL,
                    original_path TEXT NOT NULL,
                    extracted_text_path TEXT NOT NULL,
                    meta_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(lab_id) REFERENCES labs(lab_id) ON DELETE CASCADE
                );
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS defense_sessions (
                    session_id TEXT PRIMARY KEY,
                    lab_id TEXT NOT NULL,
                    submission_id TEXT NOT NULL,
                    student_label TEXT NOT NULL,
                    status TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    policy_version INTEGER NOT NULL DEFAULT 0,
                    system_summary_json TEXT NOT NULL DEFAULT '{}',
                    teacher_summary_json TEXT NOT NULL DEFAULT '{}',
                    FOREIGN KEY(lab_id) REFERENCES labs(lab_id) ON DELETE CASCADE,
                    FOREIGN KEY(submission_id) REFERENCES submissions(submission_id) ON DELETE CASCADE
                );
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS qa_turns (
                    turn_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    idx INTEGER NOT NULL,
                    question_json TEXT NOT NULL,
                    answer_text TEXT NOT NULL DEFAULT '',
                    answer_json TEXT NOT NULL DEFAULT '{}',
                    system_eval_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES defense_sessions(session_id) ON DELETE CASCADE
                );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_qa_turns_session_idx ON qa_turns(session_id, idx);")

            conn.execute("""
                CREATE TABLE IF NOT EXISTS student_feedback (
                    feedback_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    rating INTEGER,
                    comment TEXT NOT NULL DEFAULT '',
                    tags_json TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES defense_sessions(session_id) ON DELETE CASCADE
                );
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS teacher_feedback (
                    feedback_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    target TEXT NOT NULL, -- 'question' | 'answer' | 'session'
                    turn_id TEXT,         -- nullable for session-level feedback
                    label TEXT NOT NULL,  -- 'good' | 'bad'
                    score REAL,
                    reason_tags_json TEXT NOT NULL DEFAULT '[]',
                    comment TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES defense_sessions(session_id) ON DELETE CASCADE,
                    FOREIGN KEY(turn_id) REFERENCES qa_turns(turn_id) ON DELETE SET NULL
                );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_teacher_feedback_session ON teacher_feedback(session_id);")

            conn.execute("""
                CREATE TABLE IF NOT EXISTS policy_items (
                    item_id TEXT PRIMARY KEY,
                    lab_id TEXT NOT NULL,
                    kind TEXT NOT NULL,   -- 'good_question'|'bad_question'|'note' etc.
                    content_json TEXT NOT NULL,
                    reason_tags_json TEXT NOT NULL DEFAULT '[]',
                    source_feedback_id TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(lab_id) REFERENCES labs(lab_id) ON DELETE CASCADE,
                    FOREIGN KEY(source_feedback_id) REFERENCES teacher_feedback(feedback_id) ON DELETE SET NULL
                );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_policy_items_lab_kind ON policy_items(lab_id, kind);")

    # Paths

    def lab_dir(self, lab_id: str) -> Path:
        return self.data_root / "labs" / lab_id

    def _materials_dir(self, lab_id: str) -> Path:
        return self.lab_dir(lab_id) / "materials"

    def _submissions_dir(self, lab_id: str) -> Path:
        return self.lab_dir(lab_id) / "submissions"

    def _extracts_dir(self, lab_id: str) -> Path:
        return self.lab_dir(lab_id) / "extracts"

    def _exports_dir(self, lab_id: str) -> Path:
        return self.lab_dir(lab_id) / "exports"

    # Labs

    def create_lab(self, title: str, config: Optional[JsonDict] = None, status: str = "draft") -> Lab:
        lab_id = new_id("lab")
        created = now_iso()
        cfg = config or default_lab_config()
        with self._tx() as conn:
            conn.execute("""
                INSERT INTO labs(lab_id,title,status,config_json,created_at,updated_at,published_version)
                VALUES(?,?,?,?,?,?,0);
            """, (lab_id, title.strip(), status, _json_dumps(cfg), created, created))
        # ensure dirs exist
        self._materials_dir(lab_id).mkdir(parents=True, exist_ok=True)
        self._submissions_dir(lab_id).mkdir(parents=True, exist_ok=True)
        self._extracts_dir(lab_id).mkdir(parents=True, exist_ok=True)
        self._exports_dir(lab_id).mkdir(parents=True, exist_ok=True)
        return self.get_lab(lab_id)

    def list_labs(self) -> List[Lab]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM labs ORDER BY created_at DESC;").fetchall()
        return [self._row_to_lab(r) for r in rows]

    def get_lab(self, lab_id: str) -> Lab:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM labs WHERE lab_id=?;", (lab_id,)).fetchone()
        if row is None:
            raise KeyError(f"Lab not found: {lab_id}")
        return self._row_to_lab(row)

    def update_lab(self, lab_id: str, *, title: Optional[str] = None, status: Optional[str] = None,
                   config: Optional[JsonDict] = None) -> Lab:
        lab = self.get_lab(lab_id)
        new_title = title.strip() if title is not None else lab.title
        new_status = status if status is not None else lab.status
        new_cfg = config if config is not None else lab.config
        updated = now_iso()
        with self._tx() as conn:
            conn.execute("""
                UPDATE labs
                SET title=?, status=?, config_json=?, updated_at=?
                WHERE lab_id=?;
            """, (new_title, new_status, _json_dumps(new_cfg), updated, lab_id))
        return self.get_lab(lab_id)

    def publish_lab_version(self, lab_id: str) -> int:
        """Increment published_version (useful to freeze teacher policy/config). Returns new version."""
        with self._tx() as conn:
            row = conn.execute("SELECT published_version FROM labs WHERE lab_id=?;", (lab_id,)).fetchone()
            if row is None:
                raise KeyError(f"Lab not found: {lab_id}")
            new_ver = int(row["published_version"]) + 1
            conn.execute("""
                UPDATE labs SET published_version=?, updated_at=? WHERE lab_id=?;
            """, (new_ver, now_iso(), lab_id))
        return new_ver

    def delete_lab(self, lab_id: str, *, delete_files: bool = True) -> None:
        """Delete lab record (cascades). Optionally delete lab directory."""
        with self._tx() as conn:
            conn.execute("DELETE FROM labs WHERE lab_id=?;", (lab_id,))
        if delete_files:
            lab_dir = self.lab_dir(lab_id)
            if lab_dir.exists():
                shutil.rmtree(lab_dir, ignore_errors=True)

    # Materials (teacher uploads)

    def add_material(
            self,
            lab_id: str,
            kind: str,
            filename: str,
            data: bytes,
            *,
            mime: str = "application/octet-stream",
            extracted_text: str = "",
            meta: Optional[JsonDict] = None
    ) -> Material:
        """Store teacher material file + extracted text (if you already parsed it)."""
        _ = self.get_lab(lab_id)  # validate exists

        material_id = new_id("mat")
        created = now_iso()
        safe_name = _safe_filename(filename)
        orig_path = self._materials_dir(lab_id) / f"{material_id}_{safe_name}"
        text_path = self._extracts_dir(lab_id) / "materials" / f"{material_id}.txt"

        _atomic_write_bytes(orig_path, data)
        _atomic_write_text(text_path, extracted_text or "", encoding="utf-8")

        m = meta or {}
        with self._tx() as conn:
            conn.execute("""
                INSERT INTO materials(material_id,lab_id,kind,filename,mime,original_path,extracted_text_path,meta_json,created_at)
                VALUES(?,?,?,?,?,?,?,?,?);
            """, (
                material_id, lab_id, kind, safe_name, mime,
                str(orig_path), str(text_path), _json_dumps(m), created
            ))
        return self.get_material(material_id)

    def list_materials(self, lab_id: str) -> List[Material]:
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT * FROM materials WHERE lab_id=? ORDER BY created_at DESC;
            """, (lab_id,)).fetchall()
        return [self._row_to_material(r) for r in rows]

    def get_material(self, material_id: str) -> Material:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM materials WHERE material_id=?;", (material_id,)).fetchone()
        if row is None:
            raise KeyError(f"Material not found: {material_id}")
        return self._row_to_material(row)

    def delete_material(self, material_id: str, *, delete_files: bool = True) -> None:
        mat = self.get_material(material_id)
        with self._tx() as conn:
            conn.execute("DELETE FROM materials WHERE material_id=?;", (material_id,))
        if delete_files:
            for p in [Path(mat.original_path), Path(mat.extracted_text_path)]:
                try:
                    p.unlink(missing_ok=True)  # py3.8+ has missing_ok
                except TypeError:
                    if p.exists():
                        p.unlink()

    # Submissions (student uploads)

    def add_submission(
            self,
            lab_id: str,
            filename: str,
            data: bytes,
            *,
            student_id: Optional[str] = None,
            mime: str = "application/octet-stream",
            extracted_text: str = "",
            meta: Optional[JsonDict] = None
    ) -> Submission:
        """Store student submission file + extracted text."""
        _ = self.get_lab(lab_id)  # validate exists

        submission_id = new_id("sub")
        created = now_iso()
        safe_name = _safe_filename(filename)
        orig_path = self._submissions_dir(lab_id) / f"{submission_id}_{safe_name}"
        text_path = self._extracts_dir(lab_id) / "submissions" / f"{submission_id}.txt"

        _atomic_write_bytes(orig_path, data)
        _atomic_write_text(text_path, extracted_text or "", encoding="utf-8")

        m = meta or {}
        with self._tx() as conn:
            conn.execute("""
                INSERT INTO submissions(submission_id,lab_id,student_id,filename,mime,original_path,extracted_text_path,meta_json,created_at)
                VALUES(?,?,?,?,?,?,?,?,?);
            """, (
                submission_id, lab_id, student_id, safe_name, mime,
                str(orig_path), str(text_path), _json_dumps(m), created
            ))
        return self.get_submission(submission_id)

    def get_submission(self, submission_id: str) -> Submission:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM submissions WHERE submission_id=?;", (submission_id,)).fetchone()
        if row is None:
            raise KeyError(f"Submission not found: {submission_id}")
        return self._row_to_submission(row)

    def list_submissions(self, lab_id: str) -> List[Submission]:
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT * FROM submissions WHERE lab_id=? ORDER BY created_at DESC;
            """, (lab_id,)).fetchall()
        return [self._row_to_submission(r) for r in rows]

    # Defense sessions

    def create_defense_session(
            self,
            lab_id: str,
            submission_id: str,
            *,
            student_label: str = "student",
            policy_version: Optional[int] = None
    ) -> DefenseSession:
        """Start a defense session."""
        lab = self.get_lab(lab_id)
        _ = self.get_submission(submission_id)

        sess_id = new_id("sess")
        started = now_iso()
        ver = int(policy_version if policy_version is not None else lab.published_version)

        with self._tx() as conn:
            conn.execute("""
                INSERT INTO defense_sessions(session_id,lab_id,submission_id,student_label,status,started_at,policy_version,system_summary_json,teacher_summary_json)
                VALUES(?,?,?,?,? ,?,?,?,?);
            """, (sess_id, lab_id, submission_id, student_label, "in_progress", started, ver, "{}", "{}"))
        return self.get_defense_session(sess_id)

    def get_defense_session(self, session_id: str) -> DefenseSession:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM defense_sessions WHERE session_id=?;", (session_id,)).fetchone()
        if row is None:
            raise KeyError(f"Defense session not found: {session_id}")
        return self._row_to_session(row)

    def finish_defense_session(self, session_id: str, *, system_summary: Optional[JsonDict] = None) -> None:
        """Mark session finished and optionally store system summary JSON."""
        finished = now_iso()
        sys_sum = system_summary or {}
        with self._tx() as conn:
            conn.execute("""
                UPDATE defense_sessions
                SET status='finished', finished_at=?, system_summary_json=?
                WHERE session_id=?;
            """, (finished, _json_dumps(sys_sum), session_id))

    def set_teacher_session_summary(self, session_id: str, teacher_summary: JsonDict) -> None:
        with self._tx() as conn:
            conn.execute("""
                UPDATE defense_sessions SET teacher_summary_json=? WHERE session_id=?;
            """, (_json_dumps(teacher_summary), session_id))

    # Q/A turns

    def append_question(self, session_id: str, question: JsonDict) -> str:
        """Add a new question turn. Returns turn_id."""
        created = now_iso()
        turn_id = new_id("turn")
        idx = self._next_turn_idx(session_id)
        with self._tx() as conn:
            conn.execute("""
                INSERT INTO qa_turns(turn_id,session_id,idx,question_json,answer_text,answer_json,system_eval_json,created_at)
                VALUES(?,?,?,?,?,?,?,?);
            """, (turn_id, session_id, idx, _json_dumps(question), "", "{}", "{}", created))
        return turn_id

    def submit_answer(
            self,
            turn_id: str,
            answer_text: str,
            *,
            answer_json: Optional[JsonDict] = None,
            system_eval: Optional[JsonDict] = None
    ) -> None:
        """Attach student's answer + optional structured eval to an existing turn."""
        ans_j = answer_json or {}
        eval_j = system_eval or {}
        with self._tx() as conn:
            conn.execute("""
                UPDATE qa_turns
                SET answer_text=?, answer_json=?, system_eval_json=?
                WHERE turn_id=?;
            """, (answer_text or "", _json_dumps(ans_j), _json_dumps(eval_j), turn_id))

    def list_turns(self, session_id: str) -> List[JsonDict]:
        """Return all turns as dicts (ready for building log)."""
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT * FROM qa_turns WHERE session_id=? ORDER BY idx ASC;
            """, (session_id,)).fetchall()
        out: List[JsonDict] = []
        for r in rows:
            out.append({
                "turn_id": r["turn_id"],
                "idx": r["idx"],
                "question": _json_loads(r["question_json"]) or {},
                "answer_text": r["answer_text"] or "",
                "answer": _json_loads(r["answer_json"]) or {},
                "system_eval": _json_loads(r["system_eval_json"]) or {},
                "created_at": r["created_at"],
            })
        return out

    def _next_turn_idx(self, session_id: str) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COALESCE(MAX(idx), -1) AS m FROM qa_turns WHERE session_id=?;",
                               (session_id,)).fetchone()
        return int(row["m"]) + 1

    # Feedback

    def add_student_feedback(
            self,
            session_id: str,
            *,
            rating: Optional[int] = None,
            comment: str = "",
            tags: Optional[List[str]] = None
    ) -> str:
        fid = new_id("sfb")
        created = now_iso()
        with self._tx() as conn:
            conn.execute("""
                INSERT INTO student_feedback(feedback_id,session_id,rating,comment,tags_json,created_at)
                VALUES(?,?,?,?,?,?);
            """, (fid, session_id, rating, comment or "", _json_dumps(tags or []), created))
        return fid

    def add_teacher_feedback(
            self,
            session_id: str,
            *,
            target: str,  # 'question'|'answer'|'session'
            label: str,  # 'good'|'bad'
            turn_id: Optional[str] = None,
            score: Optional[float] = None,
            reason_tags: Optional[List[str]] = None,
            comment: str = ""
    ) -> str:
        fid = new_id("tfb")
        created = now_iso()
        with self._tx() as conn:
            conn.execute("""
                INSERT INTO teacher_feedback(feedback_id,session_id,target,turn_id,label,score,reason_tags_json,comment,created_at)
                VALUES(?,?,?,?,?,?,?,?,?);
            """, (
            fid, session_id, target, turn_id, label, score, _json_dumps(reason_tags or []), comment or "", created))
        return fid

    # Policy items (teacher preference memory)

    def add_policy_item(
            self,
            lab_id: str,
            *,
            kind: str,
            content: JsonDict,
            reason_tags: Optional[List[str]] = None,
            source_feedback_id: Optional[str] = None
    ) -> str:
        item_id = new_id("pol")
        created = now_iso()
        with self._tx() as conn:
            conn.execute("""
                INSERT INTO policy_items(item_id,lab_id,kind,content_json,reason_tags_json,source_feedback_id,created_at)
                VALUES(?,?,?,?,?,?,?);
            """, (
            item_id, lab_id, kind, _json_dumps(content), _json_dumps(reason_tags or []), source_feedback_id, created))
        return item_id

    def list_policy_items(self, lab_id: str, *, kind: Optional[str] = None, limit: int = 100) -> List[JsonDict]:
        with self._connect() as conn:
            if kind:
                rows = conn.execute("""
                    SELECT * FROM policy_items WHERE lab_id=? AND kind=? ORDER BY created_at DESC LIMIT ?;
                """, (lab_id, kind, limit)).fetchall()
            else:
                rows = conn.execute("""
                    SELECT * FROM policy_items WHERE lab_id=? ORDER BY created_at DESC LIMIT ?;
                """, (lab_id, limit)).fetchall()
        out: List[JsonDict] = []
        for r in rows:
            out.append({
                "item_id": r["item_id"],
                "lab_id": r["lab_id"],
                "kind": r["kind"],
                "content": _json_loads(r["content_json"]) or {},
                "reason_tags": _json_loads(r["reason_tags_json"]) or [],
                "source_feedback_id": r["source_feedback_id"],
                "created_at": r["created_at"],
            })
        return out

    def get_policy_examples(
            self,
            lab_id: str,
            *,
            good_kind: str = "good_question",
            bad_kind: str = "bad_question",
            good_limit: int = 8,
            bad_limit: int = 6
    ) -> Tuple[List[JsonDict], List[JsonDict]]:
        """
        Вернёт (good_examples, bad_examples) для few-shot/RAG.
        Здесь без embeddings: просто последние записи.
        """
        good = self.list_policy_items(lab_id, kind=good_kind, limit=good_limit)
        bad = self.list_policy_items(lab_id, kind=bad_kind, limit=bad_limit)
        return good, bad

    # Logs / export

    def build_defense_log(self, session_id: str) -> JsonDict:
        """Assemble full defense log (submission + turns + summaries + feedback references)."""
        sess = self.get_defense_session(session_id)
        sub = self.get_submission(sess.submission_id)
        turns = self.list_turns(session_id)

        with self._connect() as conn:
            sfb = conn.execute("SELECT * FROM student_feedback WHERE session_id=? ORDER BY created_at ASC;",
                               (session_id,)).fetchall()
            tfb = conn.execute("SELECT * FROM teacher_feedback WHERE session_id=? ORDER BY created_at ASC;",
                               (session_id,)).fetchall()

        student_feedback = [{
            "feedback_id": r["feedback_id"],
            "rating": r["rating"],
            "comment": r["comment"],
            "tags": _json_loads(r["tags_json"]) or [],
            "created_at": r["created_at"],
        } for r in sfb]

        teacher_feedback = [{
            "feedback_id": r["feedback_id"],
            "target": r["target"],
            "turn_id": r["turn_id"],
            "label": r["label"],
            "score": r["score"],
            "reason_tags": _json_loads(r["reason_tags_json"]) or [],
            "comment": r["comment"],
            "created_at": r["created_at"],
        } for r in tfb]

        return {
            "session": {
                "session_id": sess.session_id,
                "lab_id": sess.lab_id,
                "submission_id": sess.submission_id,
                "student_label": sess.student_label,
                "status": sess.status,
                "started_at": sess.started_at,
                "finished_at": sess.finished_at,
                "policy_version": sess.policy_version,
                "system_summary": sess.system_summary,
                "teacher_summary": sess.teacher_summary,
            },
            "submission": {
                "submission_id": sub.submission_id,
                "filename": sub.filename,
                "mime": sub.mime,
                "student_id": sub.student_id,
                "original_path": sub.original_path,
                "extracted_text_path": sub.extracted_text_path,
                "meta": sub.meta,
                "created_at": sub.created_at,
            },
            "turns": turns,
            "student_feedback": student_feedback,
            "teacher_feedback": teacher_feedback,
        }

    def export_defense_log_json(self, session_id: str, *, filename: Optional[str] = None) -> Path:
        """Export defense log to lab exports folder and return path."""
        sess = self.get_defense_session(session_id)
        log = self.build_defense_log(session_id)
        name = filename or f"defense_log_{session_id}.json"
        safe_name = _safe_filename(name)
        out_path = self._exports_dir(sess.lab_id) / safe_name
        _atomic_write_text(out_path, _json_dumps(log), encoding="utf-8")
        return out_path

    # Row mappers

    def _row_to_lab(self, r: sqlite3.Row) -> Lab:
        return Lab(
            lab_id=r["lab_id"],
            title=r["title"],
            status=r["status"],
            config=_json_loads(r["config_json"]) or {},
            created_at=r["created_at"],
            updated_at=r["updated_at"],
            published_version=int(r["published_version"]),
        )

    def _row_to_material(self, r: sqlite3.Row) -> Material:
        return Material(
            material_id=r["material_id"],
            lab_id=r["lab_id"],
            kind=r["kind"],
            filename=r["filename"],
            mime=r["mime"],
            original_path=r["original_path"],
            extracted_text_path=r["extracted_text_path"],
            meta=_json_loads(r["meta_json"]) or {},
            created_at=r["created_at"],
        )

    def _row_to_submission(self, r: sqlite3.Row) -> Submission:
        return Submission(
            submission_id=r["submission_id"],
            lab_id=r["lab_id"],
            student_id=r["student_id"],
            filename=r["filename"],
            mime=r["mime"],
            original_path=r["original_path"],
            extracted_text_path=r["extracted_text_path"],
            meta=_json_loads(r["meta_json"]) or {},
            created_at=r["created_at"],
        )

    def _row_to_session(self, r: sqlite3.Row) -> DefenseSession:
        return DefenseSession(
            session_id=r["session_id"],
            lab_id=r["lab_id"],
            submission_id=r["submission_id"],
            student_label=r["student_label"],
            status=r["status"],
            started_at=r["started_at"],
            finished_at=r["finished_at"],
            policy_version=int(r["policy_version"]),
            system_summary=_json_loads(r["system_summary_json"]) or {},
            teacher_summary=_json_loads(r["teacher_summary_json"]) or {},
        )


# Default config (editable in UI)


def default_lab_config() -> JsonDict:
    """
    Базовые параметры для формализации сценария:
    - числа вопросов по уровням сложности
    - циклы калибровки
    - требуемые разделы отчёта
    """
    return {
        "question_plan": {"easy": 3, "medium": 2, "hard": 1},
        "max_calibration_rounds": 5,
        "min_teacher_approval_rate": 0.8,
        "max_followups_per_question": 1,
        "required_sections": [
            "Цель и постановка задачи",
            "Теория",
            "Ход выполнения",
            "Результаты",
            "Выводы",
            "Список источников",
        ],
        "strictness": 0.7,
        "allowed_sources": "materials_only",  # materials_only | open_knowledge
        "grading_mode": "draft",  # draft | teacher_only
    }
