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
    STATUS_ALIGNED,
    STATUS_DRAFT,
    STATUS_FINALIZED,
    STATUS_NEEDS_CLARIFICATION,
    STATUS_REJECTED,
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

    Важное исправление в этой версии:
    этап согласования больше не отклоняет тему только из-за достижения
    max_alignment_rounds, если последняя оценка связи уже уверенная.
    Это защитный слой над TopicAlignmentAgent на случай, если внутри агента
    проверка числа раундов выполняется раньше проверки relation_score.
    """

    STRONG_ALIGNMENT_LABELS = {
        "strongly_related",
        "related",
        "aligned",
        "same_topic",
        "highly_related",
        "very_related",
    }

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
        self.relation_threshold = float(relation_threshold)
        self.max_alignment_rounds = int(max_alignment_rounds)
        self.max_defense_questions = int(max_defense_questions)

        self.ingest = IngestAgent(
            storage=self.storage,
            llm_client=self.llm_client,
            upload_dir=upload_dir,
        )
        self.topic_alignment = TopicAlignmentAgent(
            storage=self.storage,
            llm_client=self.llm_client,
            relation_threshold=self.relation_threshold,
            max_rounds=self.max_alignment_rounds,
        )
        self.methodics = MethodicsAgent(
            storage=self.storage,
            llm_client=self.llm_client,
        )
        self.defense = DefenseAgent(
            storage=self.storage,
            llm_client=self.llm_client,
            max_questions=self.max_defense_questions,
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

        Исправление:
        если в сессии уже есть уверенная оценка связи, тема фиксируется
        до нового запуска агента. После запуска агента результат также
        проверяется повторно, чтобы восстановиться из ошибочного rejected.
        """
        session = self._resolve_topic_session(lab_id=lab_id, topic_session_id=topic_session_id)

        preflight = self._finalize_if_confident_alignment(session, source="preflight")
        if preflight is not None:
            return preflight

        result = self.topic_alignment.run_alignment_cycle(session["topic_session_id"])

        fresh_session = self.storage.get_topic_session(session["topic_session_id"]) or session
        postflight = self._finalize_if_confident_alignment(
            fresh_session,
            source="postflight",
            previous_result=result,
        )
        if postflight is not None:
            return postflight

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
        self._make_topic_session_finalizable(session["topic_session_id"])
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

    # Generation evaluation

    def create_evaluation_case(
            self,
            *,
            scenario_part: str,
            method_name: str,
            title: str = "",
            description: str = "",
            lab_id: Optional[str] = None,
            input_json: Optional[dict[str, Any]] = None,
            generated_output_json: Optional[dict[str, Any]] = None,
            expected_notes: str = "",
            tags: Optional[list[str]] = None,
            is_active: bool = True,
    ) -> dict[str, Any]:
        """
        Создает сохраненный кейс оценки генерации.

        Кейс хранит входной контекст, результат генерации и указание на часть
        сценария. Это не unit-тест кода, а пример для проверки качества ответа
        модели через LLM-оценщик и формальные эвристики.
        """
        if lab_id:
            self.get_lab(lab_id)

        return self.storage.create_evaluation_case(
            scenario_part=scenario_part,
            method_name=method_name,
            title=title,
            description=description,
            lab_id=lab_id,
            input_json=input_json or {},
            generated_output_json=generated_output_json or {},
            expected_notes=expected_notes,
            tags=tags or [],
            is_active=is_active,
        )

    def update_evaluation_case(
            self,
            case_id: str,
            **fields: Any,
    ) -> Optional[dict[str, Any]]:
        """
        Обновляет сохраненный кейс оценки генерации.
        """
        return self.storage.update_evaluation_case(case_id, **fields)

    def get_evaluation_case(self, case_id: str) -> dict[str, Any]:
        """
        Возвращает один кейс оценки генерации.
        """
        case = self.storage.get_evaluation_case(case_id)
        if not case:
            raise ValueError(f"Кейс оценки генерации не найден: {case_id}")
        return case

    def list_evaluation_cases(
            self,
            *,
            lab_id: Optional[str] = None,
            scenario_part: Optional[str] = None,
            method_name: Optional[str] = None,
            active_only: bool = False,
            limit: int = 200,
    ) -> list[dict[str, Any]]:
        """
        Возвращает сохраненные кейсы оценки генерации.
        """
        if lab_id:
            self.get_lab(lab_id)
        return self.storage.list_evaluation_cases(
            lab_id=lab_id,
            scenario_part=scenario_part,
            method_name=method_name,
            active_only=active_only,
            limit=limit,
        )

    def delete_evaluation_case(self, case_id: str) -> None:
        """
        Удаляет кейс оценки генерации.
        """
        self.storage.delete_evaluation_case(case_id)

    def create_topic_final_evaluation_case_from_lab(
            self,
            *,
            lab_id: str,
            title: str = "",
            description: str = "",
            tags: Optional[list[str]] = None,
            is_active: bool = True,
    ) -> dict[str, Any]:
        """
        Создает кейс оценки итоговой согласованной темы по текущему состоянию задания.

        Вход: тема студента, тема преподавателя, сессия согласования.
        Выход: сохраненная согласованная спецификация задания.
        """
        snapshot = self.get_alignment_snapshot(lab_id=lab_id)
        agreed_spec = snapshot.get("agreed_spec")
        if not agreed_spec:
            raise ValueError("Для кейса итоговой темы нужна уже зафиксированная согласованная спецификация.")

        return self.create_evaluation_case(
            lab_id=lab_id,
            scenario_part="topic_final",
            method_name="build_agreed_spec",
            title=title or "Оценка итоговой согласованной темы",
            description=description,
            input_json={
                "lab": snapshot.get("lab"),
                "topic_session": snapshot.get("topic_session"),
                "student_topic": snapshot.get("student_input"),
                "teacher_topic": snapshot.get("teacher_input"),
                "topic_turns": snapshot.get("topic_turns") or [],
            },
            generated_output_json={"generated_topic": agreed_spec},
            tags=tags or ["topic", "final"],
            is_active=is_active,
        )

    def create_clarification_questions_evaluation_case_from_session(
            self,
            *,
            lab_id: Optional[str] = None,
            topic_session_id: Optional[str] = None,
            title: str = "",
            description: str = "",
            tags: Optional[list[str]] = None,
            is_active: bool = True,
    ) -> dict[str, Any]:
        """
        Создает кейс оценки уточняющих вопросов по текущей сессии согласования.
        """
        snapshot = self.get_alignment_snapshot(lab_id=lab_id, topic_session_id=topic_session_id)
        session = snapshot["topic_session"]
        turns = snapshot.get("topic_turns") or []

        student_questions = [
            str(item.get("question_text") or "").strip()
            for item in turns
            if item.get("turn_kind") == "question" and item.get("side") == SIDE_STUDENT and str(
                item.get("question_text") or "").strip()
        ]
        teacher_questions = [
            str(item.get("question_text") or "").strip()
            for item in turns
            if item.get("turn_kind") == "question" and item.get("side") == SIDE_TEACHER and str(
                item.get("question_text") or "").strip()
        ]

        if not student_questions and not teacher_questions:
            raise ValueError("В сессии пока нет уточняющих вопросов для формирования кейса.")

        return self.create_evaluation_case(
            lab_id=session["lab_id"],
            scenario_part="topic_clarification_questions",
            method_name="generate_clarification_questions",
            title=title or "Оценка уточняющих вопросов",
            description=description,
            input_json={
                "lab": snapshot.get("lab"),
                "topic_session": session,
                "student_topic": snapshot.get("student_input"),
                "teacher_topic": snapshot.get("teacher_input"),
            },
            generated_output_json={
                "generated_questions": {
                    "student_questions": student_questions,
                    "teacher_questions": teacher_questions,
                }
            },
            tags=tags or ["topic", "questions"],
            is_active=is_active,
        )

    def create_topic_process_evaluation_case_from_session(
            self,
            *,
            lab_id: Optional[str] = None,
            topic_session_id: Optional[str] = None,
            title: str = "",
            description: str = "",
            tags: Optional[list[str]] = None,
            is_active: bool = True,
    ) -> dict[str, Any]:
        """
        Создает кейс оценки процесса согласования темы.

        Процесс собирается из сохраненных вопросов по сессии. Если сессия уже
        финализирована, finalized=True, а finalized_round берется из round_no.
        """
        snapshot = self.get_alignment_snapshot(lab_id=lab_id, topic_session_id=topic_session_id)
        session = snapshot["topic_session"]
        turns = snapshot.get("topic_turns") or []
        agreed_spec = snapshot.get("agreed_spec")

        questions = {
            "student_questions": [
                str(item.get("question_text") or "").strip()
                for item in turns
                if item.get("turn_kind") == "question" and item.get("side") == SIDE_STUDENT and str(
                    item.get("question_text") or "").strip()
            ],
            "teacher_questions": [
                str(item.get("question_text") or "").strip()
                for item in turns
                if item.get("turn_kind") == "question" and item.get("side") == SIDE_TEACHER and str(
                    item.get("question_text") or "").strip()
            ],
        }

        generated_topic = agreed_spec or {
            "agreed_title": snapshot["lab"].get("title", ""),
            "agreed_description": snapshot["lab"].get("description", ""),
        }

        rounds = [
            {
                "round_no": int(session.get("round_no") or 1),
                "student_topic": snapshot.get("student_input"),
                "teacher_topic": snapshot.get("teacher_input"),
                "generated_topic": generated_topic,
                "generated_questions": questions,
            }
        ]

        finalized = bool(agreed_spec or session.get("status") == STATUS_FINALIZED)
        finalized_round = int(session.get("round_no") or 1) if finalized else None

        return self.create_evaluation_case(
            lab_id=session["lab_id"],
            scenario_part="topic_alignment_process",
            method_name="run_alignment_process",
            title=title or "Оценка процесса согласования темы",
            description=description,
            input_json={
                "lab": snapshot.get("lab"),
                "student_topic": snapshot.get("student_input"),
                "teacher_topic": snapshot.get("teacher_input"),
                "rounds": rounds,
                "finalized": finalized,
                "finalized_round": finalized_round,
                "max_rounds": self.max_alignment_rounds,
            },
            generated_output_json={"rounds": rounds},
            tags=tags or ["topic", "process"],
            is_active=is_active,
        )

    def create_defense_questions_evaluation_case_from_session(
            self,
            *,
            defense_session_id: str,
            title: str = "",
            description: str = "",
            tags: Optional[list[str]] = None,
            is_active: bool = True,
    ) -> dict[str, Any]:
        """
        Создает кейс оценки вопросов для защиты по существующей defense-сессии.
        """
        session = self.storage.get_defense_session(defense_session_id)
        if not session:
            raise ValueError(f"Сессия защиты не найдена: {defense_session_id}")

        lab_id = session["lab_id"]
        lab = self.get_lab(lab_id)
        submission = self.storage.get_submission(session.get("submission_id")) if session.get(
            "submission_id") else None
        spec = self.storage.get_agreed_spec_by_lab(lab_id)
        qa_turns = self.storage.list_qa_turns(defense_session_id)
        plan = session.get("plan_json") or {}
        question_pool = plan.get("question_pool") or [item.get("question_text") for item in qa_turns if
                                                      item.get("question_text")]

        if not question_pool:
            raise ValueError("В сессии защиты пока нет сгенерированных вопросов.")

        return self.create_evaluation_case(
            lab_id=lab_id,
            scenario_part="defense_questions",
            method_name="build_question_pool",
            title=title or "Оценка вопросов для защиты",
            description=description,
            input_json={
                "lab": lab,
                "agreed_spec": spec,
                "submission": submission,
                "submission_analysis": (submission or {}).get("analysis_json", {}),
                "defense_session": session,
                "qa_turns": qa_turns,
            },
            generated_output_json={"generated_questions": question_pool},
            tags=tags or ["defense", "questions"],
            is_active=is_active,
        )

    def run_evaluation_case(
            self,
            case_id: str,
            *,
            run_id: Optional[str] = None,
            k: float = 0.7,
            use_llm_judge: bool = True,
            save_result: bool = True,
    ) -> dict[str, Any]:
        """
        Запускает оценку одного кейса.

        Сначала при необходимости вызывается сторонняя модель-оценщик, затем
        evaluation.py считает эвристики, нормализует LLM-оценку и объединяет
        обе части по формуле final_score = k * llm_score + (1 - k) * heuristic_score.
        """
        case = self.get_evaluation_case(case_id)
        own_run = False

        if run_id is None:
            run = self.storage.create_evaluation_run(
                title=f"Оценка кейса: {case.get('title') or case_id}",
                model_name=self._get_model_name(),
                judge_model_name=self._get_model_name(),
                k=k,
                status="running",
                meta={"mode": "single_case", "use_llm_judge": use_llm_judge},
            )
            run_id = run["run_id"]
            own_run = True

        result = self._evaluate_case_payload(case, k=k, use_llm_judge=use_llm_judge)

        saved_result = None
        if save_result:
            saved_result = self.storage.save_evaluation_result_from_dict(
                run_id=run_id,
                case_id=case_id,
                result=result,
            )

        if own_run:
            self.storage.finish_evaluation_run(
                run_id,
                status="finished",
                meta={"result_count": 1},
            )

        return {
            "run_id": run_id,
            "case": case,
            "result": result,
            "saved_result": saved_result,
        }

    def run_evaluation_suite(
            self,
            *,
            lab_id: Optional[str] = None,
            scenario_part: Optional[str] = None,
            method_name: Optional[str] = None,
            active_only: bool = True,
            k: float = 0.7,
            use_llm_judge: bool = True,
            title: str = "",
            limit: int = 200,
    ) -> dict[str, Any]:
        """
        Прогоняет набор сохраненных кейсов и сохраняет итоговую таблицу результатов.

        Возвращает запуск, список результатов и плоские строки отчета для Streamlit/CSV.
        Ошибка в одном кейсе не останавливает весь прогон: по такому кейсу сохраняется
        результат с final_score=0 и текстом ошибки.
        """
        if lab_id:
            self.get_lab(lab_id)

        cases = self.storage.list_evaluation_cases(
            lab_id=lab_id,
            scenario_part=scenario_part,
            method_name=method_name,
            active_only=active_only,
            limit=limit,
        )

        run = self.storage.create_evaluation_run(
            title=title or "Оценка генерации",
            model_name=self._get_model_name(),
            judge_model_name=self._get_model_name() if use_llm_judge else "",
            k=k,
            status="running",
            meta={
                "lab_id": lab_id,
                "scenario_part": scenario_part,
                "method_name": method_name,
                "active_only": active_only,
                "use_llm_judge": use_llm_judge,
                "case_count": len(cases),
            },
        )

        saved_results: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []

        for case in cases:
            case_id = case["case_id"]
            try:
                result = self._evaluate_case_payload(case, k=k, use_llm_judge=use_llm_judge)
                saved = self.storage.save_evaluation_result_from_dict(
                    run_id=run["run_id"],
                    case_id=case_id,
                    result=result,
                )
                saved_results.append(saved)
            except Exception as exc:
                error_result = self._build_failed_evaluation_result(case, exc, k=k)
                saved = self.storage.save_evaluation_result_from_dict(
                    run_id=run["run_id"],
                    case_id=case_id,
                    result=error_result,
                )
                saved_results.append(saved)
                errors.append({"case_id": case_id, "error": str(exc)})

        final_status = "failed" if errors else "finished"
        finished_run = self.storage.finish_evaluation_run(
            run["run_id"],
            status=final_status,
            meta={
                "result_count": len(saved_results),
                "error_count": len(errors),
                "errors": errors[:20],
            },
        )

        report_rows = self.storage.build_evaluation_report_rows(run_id=run["run_id"])

        return {
            "run": finished_run or run,
            "cases": cases,
            "results": saved_results,
            "report_rows": report_rows,
            "errors": errors,
        }

    def list_evaluation_runs(self, *, status: Optional[str] = None, limit: int = 100) -> list[dict[str, Any]]:
        """
        Возвращает историю запусков оценки генерации.
        """
        return self.storage.list_evaluation_runs(status=status, limit=limit)

    def get_evaluation_run_report(self, run_id: str) -> dict[str, Any]:
        """
        Возвращает запуск, подробные результаты и плоскую таблицу отчета.
        """
        run = self.storage.get_evaluation_run(run_id)
        if not run:
            raise ValueError(f"Запуск оценки не найден: {run_id}")
        return {
            "run": run,
            "results": self.storage.list_evaluation_results(run_id=run_id, limit=10000),
            "report_rows": self.storage.build_evaluation_report_rows(run_id=run_id),
        }

    def _evaluate_case_payload(
            self,
            case: dict[str, Any],
            *,
            k: float,
            use_llm_judge: bool,
    ) -> dict[str, Any]:
        prepared_case = self._prepare_case_for_evaluation(case, use_llm_judge=use_llm_judge)

        from evaluation import evaluate_case

        return evaluate_case(prepared_case, k=k)

    def _prepare_case_for_evaluation(self, case: dict[str, Any], *, use_llm_judge: bool) -> dict[str, Any]:
        """
        При необходимости добавляет к кейсу LLM-оценку смыслового качества.
        Эвристики при этом считаются отдельно в evaluation.py.
        """
        import copy

        prepared = copy.deepcopy(case)
        if not use_llm_judge:
            return prepared

        scenario_part = str(prepared.get("scenario_part") or "")
        method_name = str(prepared.get("method_name") or "")
        input_json = prepared.get("input_json") or {}
        output_json = prepared.get("generated_output_json") or {}

        if scenario_part == "topic_alignment_process":
            prepared["input_json"] = self._attach_llm_scores_to_process_case(
                input_json=input_json,
                output_json=output_json,
                method_name=method_name,
            )
            return prepared

        prepared["llm_scores_json"] = self._judge_generation_with_llm(
            scenario_part=scenario_part,
            method_name=method_name,
            input_context=input_json,
            generated_output=output_json,
        )
        return prepared

    def _attach_llm_scores_to_process_case(
            self,
            *,
            input_json: dict[str, Any],
            output_json: dict[str, Any],
            method_name: str,
    ) -> dict[str, Any]:
        """
        Для процесса согласования добавляет LLM-оценки к каждому раунду.
        Формула процесса остается в evaluation.py: сначала round_score, затем scenario_score.
        """
        import copy

        prepared_input = copy.deepcopy(input_json or {})
        rounds = prepared_input.get("rounds") or (output_json or {}).get("rounds") or []
        prepared_rounds: list[dict[str, Any]] = []

        for index, round_item in enumerate(rounds, start=1):
            item = copy.deepcopy(round_item or {})
            base_context = {
                "student_topic": item.get("student_topic") or prepared_input.get("student_topic"),
                "teacher_topic": item.get("teacher_topic") or prepared_input.get("teacher_topic"),
                "round_no": item.get("round_no", index),
                "previous_rounds_count": index - 1,
                "max_rounds": prepared_input.get("max_rounds", self.max_alignment_rounds),
            }

            generated_topic = item.get("generated_topic")
            if generated_topic:
                item["llm_scores_topic"] = self._judge_generation_with_llm(
                    scenario_part="topic_final",
                    method_name="build_agreed_spec",
                    input_context=base_context,
                    generated_output=generated_topic,
                    extra_instruction="Оцени формулировку темы внутри одного раунда согласования.",
                )

            generated_questions = item.get("generated_questions") or item.get("questions")
            if generated_questions:
                item["llm_scores_questions"] = self._judge_generation_with_llm(
                    scenario_part="topic_clarification_questions",
                    method_name="generate_clarification_questions",
                    input_context=base_context,
                    generated_output=generated_questions,
                    extra_instruction="Оцени уточняющие вопросы внутри одного раунда согласования.",
                )

            prepared_rounds.append(item)

        prepared_input["rounds"] = prepared_rounds
        return prepared_input

    def _judge_generation_with_llm(
            self,
            *,
            scenario_part: str,
            method_name: str,
            input_context: Any,
            generated_output: Any,
            extra_instruction: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Вызывает стороннюю модель-оценщик.

        Если переданный llm_client уже умеет evaluate_generation, используется он.
        Иначе используется функция evaluate_generation_with_llm из llm.py.
        """
        if self.llm_client is not None and hasattr(self.llm_client, "evaluate_generation"):
            return self.llm_client.evaluate_generation(
                scenario_part=scenario_part,
                method_name=method_name,
                input_context=input_context,
                generated_output=generated_output,
                extra_instruction=extra_instruction,
            )

        from llm import evaluate_generation_with_llm

        return evaluate_generation_with_llm(
            scenario_part=scenario_part,
            method_name=method_name,
            input_context=input_context,
            generated_output=generated_output,
            extra_instruction=extra_instruction,
            client=self.llm_client,
        )

    def _build_failed_evaluation_result(
            self,
            case: dict[str, Any],
            exc: Exception,
            *,
            k: float,
    ) -> dict[str, Any]:
        return {
            "scenario_part": str(case.get("scenario_part") or ""),
            "method_name": str(case.get("method_name") or ""),
            "llm_score": None,
            "heuristic_metrics": [
                {
                    "name": "evaluation_error",
                    "value": type(exc).__name__,
                    "score": 0.0,
                    "weight": 1.0,
                    "comment": str(exc),
                }
            ],
            "heuristic_score": 0.0,
            "final_score": 0.0,
            "k": k,
            "evaluation_mode": "failed",
            "comment": f"Ошибка при оценке кейса: {exc}",
            "extra": {"case_id": case.get("case_id"), "error_type": type(exc).__name__},
        }

    def _get_model_name(self) -> str:
        """
        Возвращает имя модели для метаданных evaluation_run.
        """
        if self.llm_client is not None:
            config = getattr(self.llm_client, "config", None)
            model = getattr(config, "model", None)
            if model:
                return str(model)
            model = getattr(self.llm_client, "model", None)
            if model:
                return str(model)

        import os

        return os.getenv("MODEL_NAME", "")

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

    def _finalize_if_confident_alignment(
            self,
            session: dict[str, Any],
            *,
            source: str,
            previous_result: Optional[dict[str, Any]] = None,
    ) -> Optional[dict[str, Any]]:
        topic_session_id = session["topic_session_id"]
        lab_id = session["lab_id"]

        if self.storage.get_agreed_spec_by_lab(lab_id):
            return {
                "status": STATUS_FINALIZED,
                "reason": "already_finalized",
                "snapshot": self.get_alignment_snapshot(topic_session_id=topic_session_id),
            }

        fresh_session = self.storage.get_topic_session(topic_session_id) or session
        assessment = self._extract_alignment_assessment(fresh_session, previous_result)

        if not self._is_confident_alignment(fresh_session, assessment):
            return None

        self._make_topic_session_finalizable(topic_session_id)
        result = self.topic_alignment.finalize_alignment(topic_session_id)

        return {
            **result,
            "status": result.get("status") or STATUS_FINALIZED,
            "reason": result.get("reason") or f"confident_alignment_{source}",
            "auto_fixed_rejected_round_limit": fresh_session.get("status") == STATUS_REJECTED,
            "snapshot": self.get_alignment_snapshot(topic_session_id=topic_session_id),
        }

    def _extract_alignment_assessment(
            self,
            session: dict[str, Any],
            previous_result: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        candidates: list[Any] = []

        if previous_result:
            candidates.extend(
                [
                    previous_result.get("assessment"),
                    previous_result.get("llm_assessment_json"),
                    previous_result.get("result"),
                ]
            )
            snapshot = previous_result.get("snapshot")
            if isinstance(snapshot, dict):
                snapshot_session = snapshot.get("topic_session")
                if isinstance(snapshot_session, dict):
                    candidates.append(snapshot_session.get("llm_assessment_json"))

        candidates.append(session.get("llm_assessment_json"))

        merged: dict[str, Any] = {}
        for item in candidates:
            if isinstance(item, dict):
                merged.update(item)

        if "relation_score" not in merged:
            merged["relation_score"] = session.get("relation_score")
        if "relation_label" not in merged:
            merged["relation_label"] = session.get("relation_label")
        if "summary_text" not in merged:
            merged["short_reason"] = session.get("summary_text")

        return merged

    def _is_confident_alignment(self, session: dict[str, Any], assessment: dict[str, Any]) -> bool:
        score = self._safe_float(assessment.get("relation_score", session.get("relation_score")))
        label = str(assessment.get("relation_label") or session.get("relation_label") or "").strip().lower()
        needs_clarification = assessment.get("needs_clarification")
        conflicts = assessment.get("conflicts") or []

        if not isinstance(conflicts, list):
            conflicts = [conflicts]

        score_is_enough = score is not None and score >= self.relation_threshold
        label_is_strong = label in self.STRONG_ALIGNMENT_LABELS
        no_conflicts = len(conflicts) == 0
        not_requesting_clarification = needs_clarification is False

        return no_conflicts and (not_requesting_clarification or label_is_strong) and (
                score_is_enough or label_is_strong)

    def _make_topic_session_finalizable(self, topic_session_id: str) -> None:
        """
        Снимает ошибочный rejected перед фиксацией темы.
        Метод написан осторожно: если в storage.py сигнатура update_topic_session
        немного отличается, приложение не падает на этом вспомогательном шаге.
        """
        session = self.storage.get_topic_session(topic_session_id)
        if not session:
            return

        if session.get("status") not in {STATUS_REJECTED, STATUS_NEEDS_CLARIFICATION, STATUS_DRAFT}:
            return

        assessment = self._extract_alignment_assessment(session)
        summary = (
                assessment.get("short_reason")
                or assessment.get("summary_text")
                or session.get("summary_text")
                or "Темы уверенно связаны, сессия подготовлена к фиксации."
        )

        self._try_update_topic_session(
            topic_session_id,
            status=STATUS_ALIGNED,
            summary_text=summary,
            relation_score=assessment.get("relation_score", session.get("relation_score")),
            relation_label=assessment.get("relation_label", session.get("relation_label")),
            llm_assessment=assessment,
        )

    def _try_update_topic_session(self, topic_session_id: str, **fields: Any) -> Optional[dict[str, Any]]:
        updater = getattr(self.storage, "update_topic_session", None)
        if not callable(updater):
            return None

        attempts = (
            lambda: updater(topic_session_id, **fields),
            lambda: updater(topic_session_id=topic_session_id, **fields),
            lambda: updater(topic_session_id, fields),
            lambda: updater(topic_session_id=topic_session_id, updates=fields),
        )

        for attempt in attempts:
            try:
                updated = attempt()
                if isinstance(updated, dict):
                    return updated
                if updated is not None:
                    return self.storage.get_topic_session(topic_session_id)
            except TypeError:
                continue
            except Exception:
                return None

        return None

    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        if value is None or value == "":
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None


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
