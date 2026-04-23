"""
Единый клиент для вызова LLM (OpenAI-compatible) из вашего приложения.

Особенности:
- Читает настройки из .env / окружения
- Работает в 2 режимах:
    1) Через python-библиотеку openai (если установлена)
    2) Через прямой HTTP (urllib из stdlib), если openai не установлена
- Поддерживает структурированный JSON-вывод:
    - если передан json_schema: пытается включить response_format json_schema (если API поддерживает)
    - при несовместимости автоматически откатывается на "json_object" и/или промпт-инструкцию
"""

from __future__ import annotations

import ast
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except Exception:
    pass

JsonDict = Dict[str, Any]


# Errors


class LLMError(RuntimeError):
    pass


class LLMAuthError(LLMError):
    pass


class LLMRateLimitError(LLMError):
    pass


class LLMHTTPError(LLMError):
    def __init__(self, status: int, body: str):
        super().__init__(f"LLM HTTP error {status}: {body[:500]}")
        self.status = status
        self.body = body


class LLMOutputParseError(LLMError):
    pass


# Data classes


@dataclass(frozen=True)
class LLMConfig:
    api_key: str
    base_url: str
    model: str
    temperature: float
    max_tokens: int
    timeout_seconds: int

    @staticmethod
    def from_env() -> "LLMConfig":
        api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
        if not api_key:
            raise LLMAuthError(
                "OPENAI_API_KEY is missing. Put it in your .env and load it, "
                "or set it as an environment variable."
            )

        base_url = (os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").strip().rstrip("/")
        model = (os.getenv("MODEL_NAME") or "gpt-4o-mini").strip()

        temperature = _env_float("LLM_TEMPERATURE", 0.2)
        max_tokens = _env_int("LLM_MAX_TOKENS", 1200)
        timeout = _env_int("LLM_TIMEOUT_SECONDS", 60)

        return LLMConfig(
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_seconds=timeout,
        )


@dataclass(frozen=True)
class LLMResponse:
    text: str
    json: Optional[JsonDict]
    raw: JsonDict
    model: str
    latency_ms: int


# Schema registry


def _schema_str() -> JsonDict:
    return {"type": "string"}


def _schema_num() -> JsonDict:
    return {"type": "number"}


def _schema_bool() -> JsonDict:
    return {"type": "boolean"}


def _schema_array(item_schema: Optional[JsonDict] = None) -> JsonDict:
    return {
        "type": "array",
        "items": item_schema or {"type": "string"},
    }


def _schema_obj(properties: JsonDict, required: Optional[List[str]] = None) -> JsonDict:
    return {
        "type": "object",
        "properties": properties,
        "required": required or list(properties.keys()),
        "additionalProperties": True,
    }


SCHEMA_REGISTRY: Dict[str, JsonDict] = {
    "topic_relation_assessment": _schema_obj(
        {
            "relation_score": _schema_num(),
            "relation_label": _schema_str(),
            "needs_clarification": _schema_bool(),
            "overlap_points": _schema_array(_schema_str()),
            "conflicts": _schema_array(_schema_str()),
            "short_reason": _schema_str(),
        }
    ),
    "clarification_questions": _schema_obj(
        {
            "student_questions": _schema_array(_schema_str()),
            "teacher_questions": _schema_array(_schema_str()),
            "rationale": _schema_str(),
        }
    ),
    "agreed_assignment_spec": _schema_obj(
        {
            "work_type": _schema_str(),
            "agreed_title": _schema_str(),
            "agreed_description": _schema_str(),
            "acceptance_criteria": _schema_obj(
                {
                    "must_have": _schema_array(_schema_str()),
                    "deliverables": _schema_array(_schema_str()),
                    "evaluation_axes": _schema_array(_schema_str()),
                }
            ),
        }
    ),
    "methodics_artifact": _schema_obj(
        {
            "title": _schema_str(),
            "body_text": _schema_str(),
            "checklist": _schema_array(_schema_str()),
        }
    ),
    "submission_analysis": _schema_obj(
        {
            "summary": _schema_str(),
            "strengths": _schema_array(_schema_str()),
            "risks": _schema_array(_schema_str()),
            "recommended_focus": _schema_array(_schema_str()),
        }
    ),
    "question_pool": _schema_obj(
        {
            "questions": _schema_array(_schema_str()),
            "strategy_note": _schema_str(),
        }
    ),
    "answer_evaluation": _schema_obj(
        {
            "score": _schema_num(),
            "verdict": _schema_str(),
            "strengths": _schema_array(_schema_str()),
            "weaknesses": _schema_array(_schema_str()),
            "follow_up": _schema_str(),
        }
    ),
    "student_feedback": _schema_obj(
        {
            "feedback_text": _schema_str(),
            "highlights": _schema_array(_schema_str()),
        }
    ),
    "policy_update": _schema_obj(
        {
            "items": {
                "type": "array",
                "items": _schema_obj(
                    {
                        "kind": _schema_str(),
                        "title": _schema_str(),
                        "body_text": _schema_str(),
                    }
                ),
            },
        }
    ),
}


def get_json_schema(schema_name: Optional[str]) -> Optional[JsonDict]:
    if not schema_name:
        return None
    return SCHEMA_REGISTRY.get(schema_name)


# Public high-level client


class LLMClient:
    def __init__(self, config: Optional[LLMConfig] = None) -> None:
        self.config = config or LLMConfig.from_env()

    def generate(
            self,
            *,
            user_prompt: Optional[str] = None,
            system_prompt: Optional[str] = None,
            messages: Optional[List[JsonDict]] = None,
            json_schema: Optional[JsonDict] = None,
            schema_name: Optional[str] = None,
            strict_json: bool = True,
    ) -> LLMResponse:
        resolved_schema = json_schema or get_json_schema(schema_name)
        return call_llm(
            user_prompt or "",
            system_prompt=system_prompt,
            messages=messages,
            json_schema=resolved_schema,
            schema_name=schema_name or "output",
            strict_json=strict_json,
            config=self.config,
        )

    def generate_text(
            self,
            *,
            user_prompt: Optional[str] = None,
            system_prompt: Optional[str] = None,
            messages: Optional[List[JsonDict]] = None,
    ) -> str:
        response = self.generate(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            messages=messages,
            json_schema=None,
            schema_name=None,
        )
        return response.text

    def generate_json(
            self,
            *,
            user_prompt: Optional[str] = None,
            system_prompt: Optional[str] = None,
            messages: Optional[List[JsonDict]] = None,
            json_schema: Optional[JsonDict] = None,
            schema_name: Optional[str] = None,
            strict_json: bool = True,
    ) -> JsonDict:
        response = self.generate(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            messages=messages,
            json_schema=json_schema,
            schema_name=schema_name,
            strict_json=strict_json,
        )
        if response.json is None:
            parsed = _parse_json_best_effort(response.text)
            if isinstance(parsed, dict):
                return parsed
            raise LLMOutputParseError("Structured JSON response was expected but not parsed.")
        return response.json

    def complete_json(self, **kwargs: Any) -> JsonDict:
        return self.generate_json(**kwargs)

    def invoke_json(self, **kwargs: Any) -> JsonDict:
        return self.generate_json(**kwargs)

    def chat_json(self, **kwargs: Any) -> JsonDict:
        return self.generate_json(**kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        schema_name = kwargs.get("schema_name")
        json_schema = kwargs.get("json_schema")
        if schema_name or json_schema:
            return self.generate_json(**kwargs)
        return self.generate_text(**kwargs)


def create_llm_client(config: Optional[LLMConfig] = None) -> LLMClient:
    return LLMClient(config=config)


def build_llm_client(config: Optional[LLMConfig] = None) -> LLMClient:
    return create_llm_client(config=config)


def get_llm_client(config: Optional[LLMConfig] = None) -> LLMClient:
    return create_llm_client(config=config)


def call_llm_json(
        user_prompt: str,
        *,
        system_prompt: Optional[str] = None,
        messages: Optional[List[JsonDict]] = None,
        json_schema: Optional[JsonDict] = None,
        schema_name: Optional[str] = None,
        strict_json: bool = True,
        config: Optional[LLMConfig] = None,
) -> JsonDict:
    schema = json_schema or get_json_schema(schema_name)
    response = call_llm(
        user_prompt=user_prompt,
        system_prompt=system_prompt,
        messages=messages,
        json_schema=schema,
        schema_name=schema_name or "output",
        strict_json=strict_json,
        config=config,
    )
    if response.json is None:
        raise LLMOutputParseError("JSON response was expected but was not parsed.")
    return response.json


# Public low-level API


def call_llm(
        user_prompt: str,
        *,
        system_prompt: Optional[str] = None,
        messages: Optional[List[JsonDict]] = None,
        json_schema: Optional[JsonDict] = None,
        schema_name: str = "output",
        strict_json: bool = True,
        config: Optional[LLMConfig] = None,
) -> LLMResponse:
    """
    Универсальный вызов LLM.

    Варианты использования:
    1) call_llm("Сделай X")
    2) call_llm(..., system_prompt="Ты строгий экзаменатор")
    3) call_llm(..., messages=[...])  # массив сообщений
    4) call_llm(..., json_schema={...})  # строгий JSON

    Возвращает LLMResponse:
      - text: финальный текст ответа
      - json: распарсенный JSON (если удалось/запрошено)
      - raw: "сырой" ответ API
    """
    cfg = config or LLMConfig.from_env()

    if messages is None:
        msgs = _build_messages(user_prompt, system_prompt=system_prompt)
    else:
        msgs = messages

    # Если требуется JSON, усилим системную/пользовательскую инструкцию
    if json_schema is not None:
        msgs = _inject_json_instructions(msgs, json_schema=json_schema, strict=strict_json)

    t0 = time.time()

    # 1) Пробуем использовать OpenAI-библиотеку (если установлена)
    try:
        resp_raw = _call_with_openai_lib(
            cfg,
            msgs,
            json_schema=json_schema,
            schema_name=schema_name,
            strict_json=strict_json,
        )
        latency_ms = int((time.time() - t0) * 1000)
        return _normalize_response(resp_raw, cfg.model, latency_ms, json_schema=json_schema)
    except ModuleNotFoundError:
        # OpenAI-библиотека не установлена -> используем HTTP-фоллбек
        pass
    except Exception as e:
        # Если библиотека есть, но упала из-за несовместимости response_format,
        # попробуем HTTP-фоллбеком (часто помогает с base_url прокси).
        # Важно: не "глотать" auth/rate-limit.
        if isinstance(e, (LLMAuthError, LLMRateLimitError, LLMHTTPError)):
            raise
        # иначе - фоллбек
        pass

    # 2) HTTP fallback
    resp_raw = _call_with_http(
        cfg,
        msgs,
        json_schema=json_schema,
        schema_name=schema_name,
        strict_json=strict_json,
    )
    latency_ms = int((time.time() - t0) * 1000)
    return _normalize_response(resp_raw, cfg.model, latency_ms, json_schema=json_schema)


# Message helpers


def _build_messages(user_prompt: str, *, system_prompt: Optional[str]) -> List[JsonDict]:
    msgs: List[JsonDict] = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": user_prompt})
    return msgs


def _inject_json_instructions(messages: List[JsonDict], *, json_schema: JsonDict, strict: bool) -> List[JsonDict]:
    """
    Усиливаем промпт, чтобы даже при отсутствии response_format модель вернула JSON.
    """
    schema_text = json.dumps(json_schema, ensure_ascii=False)
    instr = (
        "Верни результат СТРОГО как один JSON-объект без пояснений и без markdown-кодов.\n"
        f"JSON должен соответствовать схеме: {schema_text}\n"
    )
    if strict:
        instr += "Никаких лишних ключей. Никакого текста вне JSON.\n"

    # Добавим инструкцию в начало system, либо создадим system
    msgs = [dict(m) for m in messages]
    if msgs and msgs[0].get("role") == "system":
        msgs[0]["content"] = (msgs[0].get("content", "").rstrip() + "\n\n" + instr).strip()
    else:
        msgs.insert(0, {"role": "system", "content": instr})
    return msgs


# OpenAI python library caller


def _call_with_openai_lib(
        cfg: LLMConfig,
        messages: List[JsonDict],
        *,
        json_schema: Optional[JsonDict],
        schema_name: str,
        strict_json: bool,
) -> JsonDict:
    """
    Использует OpenAI-библиотеку, если она установлена.
    Поддерживает прокси-серверы base_url.
    """
    try:
        from openai import OpenAI  # type: ignore
    except ModuleNotFoundError as e:
        raise e

    client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)

    kwargs: JsonDict = {
        "model": cfg.model,
        "messages": messages,
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
    }

    # При необходимости используем структурированный вывод
    if json_schema is not None:
        # Более новый стиль OpenAI: response_format={"type":"json_schema","json_schema":{...}}
        # Некоторые прокси/старые серверы могут не поддерживать это -> ошибка -> фоллбек.
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "schema": json_schema,
                "strict": bool(strict_json),
            },
        }

    try:
        resp = client.chat.completions.create(**kwargs)
        raw = resp.model_dump() if hasattr(resp, "model_dump") else json.loads(resp.json())
        return raw
    except Exception as e:
        # Пробуем более менее строгий JSON, если формат схемы не поддерживается
        if json_schema is not None:
            try:
                kwargs2 = dict(kwargs)
                kwargs2["response_format"] = {"type": "json_object"}
                resp2 = client.chat.completions.create(**kwargs2)
                raw2 = resp2.model_dump() if hasattr(resp2, "model_dump") else json.loads(resp2.json())
                return raw2
            except Exception:
                pass
        _raise_mapped_openai_error(e)
        raise


def _raise_mapped_openai_error(e: Exception) -> None:
    msg = str(e).lower()
    if "api key" in msg or "authentication" in msg or "401" in msg:
        raise LLMAuthError(str(e))
    if "rate limit" in msg or "429" in msg:
        raise LLMRateLimitError(str(e))


# HTTP caller (urllib)


def _call_with_http(
        cfg: LLMConfig,
        messages: List[JsonDict],
        *,
        json_schema: Optional[JsonDict],
        schema_name: str,
        strict_json: bool
) -> JsonDict:
    """
    Прямой HTTP-вызов к {base_url}/chat/completions (совместим с OpenAI).
    """
    url = cfg.base_url.rstrip("/") + "/chat/completions"

    payload: JsonDict = {
        "model": cfg.model,
        "messages": messages,
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
    }

    if json_schema is not None:
        # Сначала пробуем через json_schema (если сервер его поддерживает)
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "schema": json_schema,
                "strict": bool(strict_json),
            },
        }

    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url=url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {cfg.api_key}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=cfg.timeout_seconds) as resp:
            resp_body = resp.read().decode("utf-8", errors="replace")
            return json.loads(resp_body)
    except urllib.error.HTTPError as he:
        resp_body = he.read().decode("utf-8", errors="replace") if hasattr(he, "read") else ""
        # json_schema не поддерживается -> пробуем через json_object
        if json_schema is not None and he.code in (400, 422):
            try:
                payload2 = dict(payload)
                payload2["response_format"] = {"type": "json_object"}
                body2 = json.dumps(payload2, ensure_ascii=False).encode("utf-8")
                req2 = urllib.request.Request(
                    url=url,
                    data=body2,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {cfg.api_key}",
                    },
                    method="POST",
                )
                with urllib.request.urlopen(req2, timeout=cfg.timeout_seconds) as resp2:
                    resp_body2 = resp2.read().decode("utf-8", errors="replace")
                    return json.loads(resp_body2)
            except Exception:
                pass

        if he.code == 401:
            raise LLMAuthError(resp_body or str(he))
        if he.code == 429:
            raise LLMRateLimitError(resp_body or str(he))
        raise LLMHTTPError(he.code, resp_body or str(he))
    except urllib.error.URLError as ue:
        raise LLMError(f"LLM URL error: {ue}") from ue


# Response normalization + JSON parsing


def _normalize_response(
        raw: JsonDict,
        model: str,
        latency_ms: int,
        *,
        json_schema: Optional[JsonDict]
) -> LLMResponse:
    """
    Преобразует ответ, совместимый с OpenAI, в формат LLMResponse.
    """
    text = _extract_text(raw)
    parsed: Optional[JsonDict] = None

    if json_schema is not None:
        # Пробуем прямой JSON-парсинг
        parsed = _parse_json_best_effort(text)
        if parsed is None:
            # Иногда модель возвращает JSON в виде нетипичного поля; пробуем его парсить
            parsed = _parse_json_from_raw(raw)
        if parsed is None:
            raise LLMOutputParseError(
                "Модель должна была возвращать JSON, но выходные данные не удалось обработать как JSON."
            )

    return LLMResponse(
        text=text,
        json=parsed,
        raw=raw,
        model=model,
        latency_ms=latency_ms,
    )


def _extract_text(raw: JsonDict) -> str:
    """
    OpenAI chat.completions style:
      raw["choices"][0]["message"]["content"]
    """
    try:
        choices = raw.get("choices") or []
        if not choices:
            return ""

        msg = choices[0].get("message") or {}
        content = msg.get("content")

        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        parts.append(str(item.get("text", "")))
                    elif "text" in item:
                        parts.append(str(item.get("text", "")))
                elif isinstance(item, str):
                    parts.append(item)
            return "\n".join(part.strip() for part in parts if str(part).strip()).strip()

        if content is None:
            content = msg.get("text", "")

        return str(content or "").strip()
    except Exception:
        return ""


def _parse_json_from_raw(raw: JsonDict) -> Optional[JsonDict]:
    """
    Некоторые поставщики размещают структурированные результаты в другом месте.
    Пробуем несколько известных шаблонов.
    """
    # 1) message.content как строка уже обработана
    # 2) message.tool_calls[].function.arguments (стиль вызова функций OpenAI)
    try:
        choices = raw.get("choices") or []
        if not choices:
            return None
        msg = choices[0].get("message") or {}
        tool_calls = msg.get("tool_calls") or []
        for tc in tool_calls:
            fn = tc.get("function") or {}
            args = fn.get("arguments")
            if isinstance(args, str):
                obj = _parse_json_best_effort(args)
                if isinstance(obj, dict):
                    return obj
            elif isinstance(args, dict):
                return args
    except Exception:
        pass

    # 3) Иногда возвращается {"output": {...}}
    out = raw.get("output")
    if isinstance(out, dict):
        return out

    return None


def _parse_json_best_effort(text: str) -> Optional[JsonDict]:
    """
    Анализ JSON-объекта из текста.
    - Принимает чистый JSON
    - Принимает JSON, заключенный в блоки Markdown
    - Принимает дополнительный текст: пытается извлечь первый блок {...}
    - Принимает словарь, подобный Python, через ast.literal_eval (одинарные кавычки, True/None)
    """
    if not text:
        return None

    t = text.strip()

    # Удаляем md-блоки, если есть
    t = re.sub(r"^\s*```(?:json)?\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*```\s*$", "", t)

    # 1) чистый JSON
    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    # 2) извлекаем первый {...} и пробуем JSON
    obj_str = _extract_first_json_object(t)
    if obj_str:
        try:
            obj = json.loads(obj_str)
            return obj if isinstance(obj, dict) else None
        except Exception:
            pass

    # 3) python-literal fallback (single quotes / True / None)
    try:
        obj = ast.literal_eval(t)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    if obj_str:
        try:
            obj = ast.literal_eval(obj_str)
            return obj if isinstance(obj, dict) else None
        except Exception:
            pass

    return None


def _extract_first_json_object(s: str) -> Optional[str]:
    """
    Извлекаем первый JSON-объект {...} с помощью подсчета фигурных скобок (обрабатывает вложенные скобки).
    """
    start = s.find("{")
    if start < 0:
        return None

    depth = 0
    in_str = False
    esc = False

    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
            continue

        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start:i + 1]

    return None


# Env helpers


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None or str(v).strip() == "":
        return default
    try:
        return int(str(v).strip())
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None or str(v).strip() == "":
        return default
    try:
        return float(str(v).strip().replace(",", "."))
    except ValueError:
        return default


__all__ = [
    "LLMError",
    "LLMAuthError",
    "LLMRateLimitError",
    "LLMHTTPError",
    "LLMOutputParseError",
    "LLMConfig",
    "LLMResponse",
    "LLMClient",
    "SCHEMA_REGISTRY",
    "get_json_schema",
    "call_llm",
    "call_llm_json",
    "create_llm_client",
    "build_llm_client",
    "get_llm_client",
]
