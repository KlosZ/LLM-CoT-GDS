from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional

from agents import (
    DefenseAgent,
    FeedbackAgent,
    IngestAgent,
    MethodicsAgent,
    TopicAlignmentAgent,
)
from storage import (
    MATERIAL_ROLE_STUDENT,
    MATERIAL_ROLE_TEACHER,
    MATERIAL_STAGE_METHODICS,
    MATERIAL_STAGE_SUBMISSION,
    MATERIAL_STAGE_TOPIC_ALIGNMENT,
    SIDE_STUDENT,
    SIDE_TEACHER,
    STATUS_FINALIZED,
    WORK_TYPE_OTHER,
    Storage,
    create_storage,
)


class ProjectCore:
    """
    Главный orchestration-слой проекта.

    Здесь нет низкоуровневой LLM-логики и нет SQL:
    - SQL живет в storage.py
    - взаимодействие с моделью живет в agents.py

    core.py отвечает за сценарии приложения:
    - создание задания
    - согласование темы
    - подготовка задания к публикации
    - прием работы студента
    - защита
    - обратная связь и policy memory
    """

    def __init__(
            self,
            storage: Storage,
            llm_client: Any = None,
            *,
            upload_dir: str | Path = "uploads",
            relation_threshold: float = 0.55,
            max_alignment_rounds: int = 3,
            max_defense_questions: int = 6,
    ) -> None:
        self.storage = storage
        self.llm_client = llm_client

        self.ingest = IngestAgent(
            storage=self.storage,
            llm_client=self.llm_client,
            upload_dir=upload_dir,
        )
        self.topic_alignment = TopicAlignmentAgent(
            storage=self.storage,
            llm_client=self.llm_client,
            relation_threshold=relation_threshold,
            max_rounds=max_alignment_rounds,
        )
        self.methodics = MethodicsAgent(
            storage=self.storage,
            llm_client=self.llm_client,
        )
        self.defense = DefenseAgent(
            storage=self.storage,
            llm_client=self.llm_client,
            max_questions=max_defense_questions,
        )
        self.feedback = FeedbackAgent(
            storage=self.storage,
            llm_client=self.llm_client,
        )

    # Factory / boot

    @classmethod
    def from_db_path(
            cls,
            db_path: str | Path,
            llm_client: Any = None,
            *,
            upload_dir: str | Path = "uploads",
            relation_threshold: float = 0.55,
            max_alignment_rounds: int = 3,
            max_defense_questions: int = 6,
    ) -> "ProjectCore":
        storage = create_storage(db_path)
        return cls(
            storage=storage,
            llm_client=llm_client,
            upload_dir=upload_dir,
            relation_threshold=relation_threshold,
            max_alignment_rounds=max_alignment_rounds,
            max_defense_questions=max_defense_questions,
        )

    # Basic getters

    def get_lab(self, lab_id: str) -> dict[str, Any]:
        lab = self.storage.get_lab(lab_id)
        if not lab:
            raise ValueError(f"Задание не найдено: {lab_id}")
        return lab

    def get_topic_session(self, topic_session_id: str) -> dict[str, Any]:
        session = self.storage.get_topic_session(topic_session_id)
        if not session:
            raise ValueError(f"Сессия согласования темы не найдена: {topic_session_id}")
        return session

    def get_latest_topic_session_for_lab(self, lab_id: str) -> Optional[dict[str, Any]]:
        self.get_lab(lab_id)
        return self.storage.get_latest_topic_session_for_lab(lab_id)

    def get_agreed_spec(self, lab_id: str) -> Optional[dict[str, Any]]:
        self.get_lab(lab_id)
        return self.storage.get_agreed_spec_by_lab(lab_id)

    # Assignment creation

    def create_assignment(
            self,
            *,
            teacher_title: str,
            teacher_description: str = "",
            work_type: str = WORK_TYPE_OTHER,
            config: Optional[dict[str, Any]] = None,
            seed_teacher_topic: bool = True,
            teacher_topic_context: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Создает новый контейнер задания и сессию согласования темы.
        По желанию сразу записывает ввод преподавателя как исходную тему.
        """
        bundle = self.storage.create_assignment_with_topic_session(
            teacher_title=teacher_title or "Новое задание",
            teacher_description=teacher_description,
            work_type=work_type,
            config=config or {"topic_alignment_enabled": True},
        )

        if seed_teacher_topic:
            self.topic_alignment.set_teacher_topic(
                topic_session_id=bundle["topic_session"]["topic_session_id"],
                title=teacher_title or "Тема преподавателя",
                description=teacher_description,
                context=teacher_topic_context or {},
            )

        return bundle

    def ensure_topic_session_for_lab(self, lab_id: str) -> dict[str, Any]:
        lab = self.get_lab(lab_id)
        topic_session_id = (lab.get("config_json") or {}).get("topic_session_id")
        if topic_session_id:
            session = self.storage.get_topic_session(topic_session_id)
            if session:
                return session
        return self.topic_alignment.get_or_create_session(lab_id)

    # Topic input

    def submit_student_topic(
            self,
            *,
            lab_id: Optional[str] = None,
            topic_session_id: Optional[str] = None,
            title: str,
            description: str = "",
            context: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        session = self._resolve_topic_session(lab_id=lab_id, topic_session_id=topic_session_id)
        return self.topic_alignment.set_student_topic(
            topic_session_id=session["topic_session_id"],
            title=title,
            description=description,
            context=context,
        )

    def submit_teacher_topic(
            self,
            *,
            lab_id: Optional[str] = None,
            topic_session_id: Optional[str] = None,
            title: str,
            description: str = "",
            context: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        session = self._resolve_topic_session(lab_id=lab_id, topic_session_id=topic_session_id)
        return self.topic_alignment.set_teacher_topic(
            topic_session_id=session["topic_session_id"],
            title=title,
            description=description,
            context=context,
        )

    # Topic materials

    def upload_student_topic_materials(
            self,
            *,
            lab_id: str,
            uploaded_files: Iterable[Any],
            extra_meta: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        self.get_lab(lab_id)
        return self.ingest.ingest_materials(
            lab_id=lab_id,
            uploaded_files=uploaded_files,
            owner_role=MATERIAL_ROLE_STUDENT,
            stage=MATERIAL_STAGE_TOPIC_ALIGNMENT,
            title_prefix="Материал студента: ",
            extra_meta=extra_meta,
        )

    def upload_teacher_topic_materials(
            self,
            *,
            lab_id: str,
            uploaded_files: Iterable[Any],
            extra_meta: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        self.get_lab(lab_id)
        return self.ingest.ingest_materials(
            lab_id=lab_id,
            uploaded_files=uploaded_files,
            owner_role=MATERIAL_ROLE_TEACHER,
            stage=MATERIAL_STAGE_TOPIC_ALIGNMENT,
            title_prefix="Материал преподавателя: ",
            extra_meta=extra_meta,
        )

    # Topic alignment cycle

    def get_alignment_snapshot(
            self,
            *,
            lab_id: Optional[str] = None,
            topic_session_id: Optional[str] = None,
    ) -> dict[str, Any]:
        session = self._resolve_topic_session(lab_id=lab_id, topic_session_id=topic_session_id)
        lab = self.get_lab(session["lab_id"])

        return {
            "lab": lab,
            "topic_session": session,
            "student_input": self.storage.get_topic_input(session["topic_session_id"], SIDE_STUDENT),
            "teacher_input": self.storage.get_topic_input(session["topic_session_id"], SIDE_TEACHER),
            "topic_turns": self.storage.list_topic_turns(session["topic_session_id"]),
            "student_materials": self.storage.list_materials(
                lab["lab_id"],
                owner_role=MATERIAL_ROLE_STUDENT,
                stage=MATERIAL_STAGE_TOPIC_ALIGNMENT,
            ),
            "teacher_materials": self.storage.list_materials(
                lab["lab_id"],
                owner_role=MATERIAL_ROLE_TEACHER,
                stage=MATERIAL_STAGE_TOPIC_ALIGNMENT,
            ),
            "agreed_spec": self.storage.get_agreed_spec_by_lab(lab["lab_id"]),
        }

    def run_alignment_cycle(
            self,
            *,
            lab_id: Optional[str] = None,
            topic_session_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Один полный цикл:
        - оценка связи
        - если надо, генерация уточняющих вопросов
        - если связь уже достаточна, формирование и фиксация спецификации
        """
        session = self._resolve_topic_session(lab_id=lab_id, topic_session_id=topic_session_id)
        result = self.topic_alignment.run_alignment_cycle(session["topic_session_id"])
        return {
            **result,
            "snapshot": self.get_alignment_snapshot(topic_session_id=session["topic_session_id"]),
        }

    def submit_alignment_answers(
            self,
            *,
            topic_session_id: str,
            side: str,
            answers: str | list[str],
            uploaded_files: Optional[Iterable[Any]] = None,
            rerun: bool = True,
    ) -> dict[str, Any]:
        """
        Сохраняет ответы стороны на уточняющие вопросы.
        По желанию сразу прикладывает новые материалы и повторно запускает цикл.
        """
        session = self.get_topic_session(topic_session_id)
        lab_id = session["lab_id"]

        created_answers = self.topic_alignment.add_clarification_answers(
            topic_session_id=topic_session_id,
            side=side,
            answers=answers,
        )

        uploaded_materials: list[dict[str, Any]] = []
        if uploaded_files:
            if side == SIDE_STUDENT:
                uploaded_materials = self.upload_student_topic_materials(
                    lab_id=lab_id,
                    uploaded_files=uploaded_files,
                    extra_meta={"source": "alignment_answer"},
                )
            elif side == SIDE_TEACHER:
                uploaded_materials = self.upload_teacher_topic_materials(
                    lab_id=lab_id,
                    uploaded_files=uploaded_files,
                    extra_meta={"source": "alignment_answer"},
                )

        if rerun:
            rerun_result = self.run_alignment_cycle(topic_session_id=topic_session_id)
        else:
            rerun_result = {"status": "saved_only"}

        return {
            "answers": created_answers,
            "materials": uploaded_materials,
            "result": rerun_result,
        }

    def finalize_alignment_manually(
            self,
            *,
            lab_id: Optional[str] = None,
            topic_session_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Ручная фиксация согласованной темы.
        Полезно, если LLM уже собрала контекст, но преподаватель хочет завершить этап вручную.
        """
        session = self._resolve_topic_session(lab_id=lab_id, topic_session_id=topic_session_id)
        result = self.topic_alignment.finalize_alignment(session["topic_session_id"])
        return {
            **result,
            "snapshot": self.get_alignment_snapshot(topic_session_id=session["topic_session_id"]),
        }

    # Methodics / calibration / publication

    def generate_methodics(self, lab_id: str, *, save_as_material: bool = True) -> dict[str, Any]:
        self._require_agreed_spec(lab_id)
        return self.methodics.generate_methodics(lab_id, save_as_material=save_as_material)

    def calibrate_policy(self, lab_id: str, *, persist: bool = True) -> dict[str, Any]:
        self._require_agreed_spec(lab_id)
        return self.methodics.calibrate_policy(lab_id, persist=persist)

    def prepare_assignment_after_alignment(
            self,
            lab_id: str,
            *,
            generate_methodics: bool = True,
            calibrate_policy: bool = True,
            publish: bool = False,
    ) -> dict[str, Any]:
        """
        Подготавливает задание после того, как тема уже согласована.
        """
        self._require_agreed_spec(lab_id)

        result: dict[str, Any] = {
            "lab": self.get_lab(lab_id),
            "methodics": None,
            "policy": None,
        }

        if generate_methodics:
            result["methodics"] = self.generate_methodics(lab_id)

        if calibrate_policy:
            result["policy"] = self.calibrate_policy(lab_id)

        if publish:
            result["lab"] = self.publish_assignment(lab_id)

        return result

    def publish_assignment(self, lab_id: str) -> dict[str, Any]:
        """
        Публикация не меняет смысл already finalized задания,
        а просто ставит флаг доступности для студента.
        """
        lab = self.get_lab(lab_id)
        if not self.storage.get_agreed_spec_by_lab(lab_id):
            raise ValueError("Нельзя публиковать задание без согласованной спецификации.")

        updated = self.storage.update_lab(
            lab_id,
            status=STATUS_FINALIZED,
            config={
                "published": True,
                "publication_state": "published",
            },
        )
        if not updated:
            raise RuntimeError("Не удалось опубликовать задание.")
        return updated

    # Submission

    def create_submission(
            self,
            *,
            lab_id: str,
            student_name: str = "",
            title: str = "",
            description: str = "",
            uploaded_files: Optional[Iterable[Any]] = None,
            auto_analyze: bool = True,
    ) -> dict[str, Any]:
        """
        Создает submission и, если переданы файлы, загружает их в stage=submission.
        """
        lab = self.get_lab(lab_id)
        if not lab:
            raise ValueError(f"Задание не найдено: {lab_id}")

        submission = self.storage.create_submission(
            lab_id=lab_id,
            student_name=student_name,
            title=title,
            description=description,
            file_bundle={},
            analysis={},
        )

        uploaded_materials: list[dict[str, Any]] = []
        if uploaded_files:
            uploaded_materials = self.ingest.ingest_materials(
                lab_id=lab_id,
                uploaded_files=uploaded_files,
                owner_role=MATERIAL_ROLE_STUDENT,
                stage=MATERIAL_STAGE_SUBMISSION,
                title_prefix="Работа студента: ",
                extra_meta={
                    "submission_id": submission["submission_id"],
                    "student_name": student_name,
                },
            )

            submission = self.storage.update_submission(
                submission["submission_id"],
                file_bundle={
                    "material_ids": [item["material_id"] for item in uploaded_materials],
                    "filenames": [item["filename"] for item in uploaded_materials],
                    "materials_count": len(uploaded_materials),
                },
            ) or submission

        analysis = None
        if auto_analyze:
            analysis_result = self.analyze_submission(
                lab_id=lab_id,
                submission_id=submission["submission_id"],
            )
            submission = analysis_result["submission"]
            analysis = analysis_result["analysis"]

        return {
            "submission": submission,
            "materials": uploaded_materials,
            "analysis": analysis,
        }

    def analyze_submission(self, *, lab_id: str, submission_id: str) -> dict[str, Any]:
        self._require_agreed_spec(lab_id)
        return self.defense.analyze_submission(lab_id, submission_id)

    # Defense

    def start_defense(
            self,
            *,
            lab_id: str,
            submission_id: str,
            pool_size: int = 8,
            auto_analyze_if_needed: bool = True,
    ) -> dict[str, Any]:
        self._require_agreed_spec(lab_id)

        submission = self.storage.get_submission(submission_id)
        if not submission:
            raise ValueError(f"Работа не найдена: {submission_id}")

        if auto_analyze_if_needed and not (submission.get("analysis_json") or {}):
            self.analyze_submission(lab_id=lab_id, submission_id=submission_id)

        return self.defense.start_defense(
            lab_id=lab_id,
            submission_id=submission_id,
            pool_size=pool_size,
        )

    def next_defense_question(self, defense_session_id: str) -> dict[str, Any]:
        return self.defense.next_question(defense_session_id)

    def submit_defense_answer(self, defense_session_id: str, answer_text: str) -> dict[str, Any]:
        return self.defense.submit_answer(defense_session_id, answer_text)

    def submit_answer_and_get_next(self, defense_session_id: str, answer_text: str) -> dict[str, Any]:
        evaluation = self.submit_defense_answer(defense_session_id, answer_text)
        next_question = self.next_defense_question(defense_session_id)
        return {
            "evaluation": evaluation,
            "next": next_question,
        }

    def finalize_defense(self, defense_session_id: str) -> dict[str, Any]:
        return self.defense.finish_defense(defense_session_id)

    # Feedback / policy loop

    def generate_student_feedback(self, *, lab_id: str, defense_session_id: str) -> dict[str, Any]:
        return self.feedback.generate_student_feedback(lab_id, defense_session_id)

    def register_teacher_feedback(
            self,
            *,
            lab_id: str,
            feedback_text: str,
            defense_session_id: Optional[str] = None,
            extra: Optional[dict[str, Any]] = None,
            update_policy: bool = False,
    ) -> dict[str, Any]:
        item = self.feedback.register_teacher_feedback(
            lab_id=lab_id,
            feedback_text=feedback_text,
            defense_session_id=defense_session_id,
            extra=extra,
        )

        policy_update = None
        if update_policy:
            policy_update = self.feedback.update_policy_memory(lab_id)

        return {
            "teacher_feedback": item,
            "policy_update": policy_update,
        }

    def update_policy_from_teacher_feedback(self, lab_id: str) -> dict[str, Any]:
        return self.feedback.update_policy_memory(lab_id)

    # Full snapshots for UI

    def get_lab_dashboard(self, lab_id: str) -> dict[str, Any]:
        """
        Удобная сборка всех данных по заданию для app.py.
        """
        lab = self.get_lab(lab_id)
        topic_session = self.storage.get_latest_topic_session_for_lab(lab_id)
        agreed_spec = self.storage.get_agreed_spec_by_lab(lab_id)

        return {
            "lab": lab,
            "topic_session": topic_session,
            "topic_inputs": self.storage.list_topic_inputs(topic_session["topic_session_id"]) if topic_session else [],
            "topic_turns": self.storage.list_topic_turns(topic_session["topic_session_id"]) if topic_session else [],
            "agreed_spec": agreed_spec,
            "topic_materials_student": self.storage.list_materials(
                lab_id,
                owner_role=MATERIAL_ROLE_STUDENT,
                stage=MATERIAL_STAGE_TOPIC_ALIGNMENT,
            ),
            "topic_materials_teacher": self.storage.list_materials(
                lab_id,
                owner_role=MATERIAL_ROLE_TEACHER,
                stage=MATERIAL_STAGE_TOPIC_ALIGNMENT,
            ),
            "methodics_materials": self.storage.list_materials(
                lab_id,
                stage=MATERIAL_STAGE_METHODICS,
            ),
            "submission_materials": self.storage.list_materials(
                lab_id,
                stage=MATERIAL_STAGE_SUBMISSION,
            ),
            "submissions": self.storage.list_submissions(lab_id),
            "defense_sessions": self.storage.list_defense_sessions(lab_id),
            "teacher_feedback": self.storage.list_teacher_feedback(lab_id),
            "student_feedback": self.storage.list_student_feedback(lab_id),
            "policy_items": self.storage.list_policy_items(lab_id=lab_id, limit=200),
        }

    # End-to-end helpers

    def run_alignment_to_completion_or_pause(
            self,
            *,
            lab_id: str,
    ) -> dict[str, Any]:
        """
        Обертка для app.py - в зависимости от стадии:
        — запускает очередной цикл согласования,
        — дает вопросы,
        — либо фиксирует задание,
        — либо переводит сессию в rejected.
        """
        session = self.ensure_topic_session_for_lab(lab_id)
        return self.run_alignment_cycle(topic_session_id=session["topic_session_id"])

    def finalize_and_prepare_assignment(
            self,
            *,
            lab_id: str,
            generate_methodics: bool = True,
            calibrate_policy: bool = True,
            publish: bool = True,
    ) -> dict[str, Any]:
        """
        Полезный быстрый сценарий:
        тема уже согласована -> генерим методичку -> калибруем policy -> публикуем.
        """
        self._require_agreed_spec(lab_id)
        return self.prepare_assignment_after_alignment(
            lab_id,
            generate_methodics=generate_methodics,
            calibrate_policy=calibrate_policy,
            publish=publish,
        )

    def run_full_defense_cycle(
            self,
            *,
            lab_id: str,
            submission_id: str,
            answers: list[str],
            pool_size: int = 8,
            generate_feedback: bool = True,
    ) -> dict[str, Any]:
        """
        Упрощенный сценарий для автотестов или демо:
        запускает защиту, прогоняет список ответов, завершает сессию.
        """
        defense_session = self.start_defense(
            lab_id=lab_id,
            submission_id=submission_id,
            pool_size=pool_size,
        )

        qa_log: list[dict[str, Any]] = []

        next_payload = self.next_defense_question(defense_session["defense_session_id"])
        answer_index = 0

        while not next_payload.get("finished"):
            question = next_payload.get("question")
            if question is None:
                break

            answer_text = answers[answer_index] if answer_index < len(answers) else ""
            step = self.submit_answer_and_get_next(
                defense_session["defense_session_id"],
                answer_text=answer_text,
            )
            qa_log.append(
                {
                    "question": question,
                    "answer": answer_text,
                    "evaluation": step["evaluation"],
                }
            )
            next_payload = step["next"]
            answer_index += 1

        final_session = self.finalize_defense(defense_session["defense_session_id"])

        feedback_item = None
        if generate_feedback:
            feedback_item = self.generate_student_feedback(
                lab_id=lab_id,
                defense_session_id=defense_session["defense_session_id"],
            )

        return {
            "defense_session": final_session,
            "qa_log": qa_log,
            "student_feedback": feedback_item,
        }

    # Internal helpers

    def _resolve_topic_session(
            self,
            *,
            lab_id: Optional[str],
            topic_session_id: Optional[str],
    ) -> dict[str, Any]:
        if topic_session_id:
            return self.get_topic_session(topic_session_id)
        if lab_id:
            return self.ensure_topic_session_for_lab(lab_id)
        raise ValueError("Нужно передать lab_id или topic_session_id.")

    def _require_agreed_spec(self, lab_id: str) -> dict[str, Any]:
        spec = self.storage.get_agreed_spec_by_lab(lab_id)
        if not spec:
            raise ValueError("Сначала нужно согласовать и зафиксировать тему задания.")
        return spec


# Thin functional wrappers


def create_core(
        db_path: str | Path,
        llm_client: Any = None,
        *,
        upload_dir: str | Path = "uploads",
        relation_threshold: float = 0.55,
        max_alignment_rounds: int = 3,
        max_defense_questions: int = 6,
) -> ProjectCore:
    return ProjectCore.from_db_path(
        db_path=db_path,
        llm_client=llm_client,
        upload_dir=upload_dir,
        relation_threshold=relation_threshold,
        max_alignment_rounds=max_alignment_rounds,
        max_defense_questions=max_defense_questions,
    )


# Manual smoke test


if __name__ == "__main__":
    core = create_core("data/app.sqlite3")

    created = core.create_assignment(
        teacher_title="Методы анализа текстовых данных",
        teacher_description="Нужно сформировать задание в рамках дисциплины.",
        work_type=WORK_TYPE_OTHER,
    )
    lab = created["lab"]
    topic_session = created["topic_session"]

    core.submit_student_topic(
        topic_session_id=topic_session["topic_session_id"],
        title="Анализ русскоязычных документов организации",
        description="Интересует задача обработки и классификации внутренних документов.",
        context={"origin": "practice"},
    )

    result = core.run_alignment_cycle(topic_session_id=topic_session["topic_session_id"])
    print("ALIGNMENT RESULT:")
    print(result["status"])

    if result["status"] == "needs_clarification":
        core.submit_alignment_answers(
            topic_session_id=topic_session["topic_session_id"],
            side=SIDE_STUDENT,
            answers="Я готов сузить тему до анализа текстовых документов и представить прототип с отчетом.",
            rerun=True,
        )

    dashboard = core.get_lab_dashboard(lab["lab_id"])
    print("\nDASHBOARD KEYS:")
    print(list(dashboard.keys()))
