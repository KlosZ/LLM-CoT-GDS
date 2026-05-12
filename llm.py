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

    # Важно: load_dotenv() без override=True не перезаписывает уже существующие
    # переменные окружения Windows. Из-за этого приложение может брать старый
    # OPENAI_API_KEY и получать 401 Invalid API Key.
    dotenv_override = (
                              os.getenv("LLM_DOTENV_OVERRIDE")
                              or os.getenv("DOTENV_OVERRIDE")
                              or "1"
                      ).strip().lower() not in {"0", "false", "no", "off"}
    load_dotenv(override=dotenv_override)
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


# Env helpers


def _clean_env_value(value: Optional[str]) -> str:
    if value is None:
        return ""
    value = str(value).strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1].strip()
    return value


def _env_int(name: str, default: int) -> int:
    value = _clean_env_value(os.getenv(name))
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = _clean_env_value(os.getenv(name))
    if not value:
        return default
    try:
        return float(value.replace(",", "."))
    except ValueError:
        return default


def _mask_secret(value: str) -> str:
    value = _clean_env_value(value)
    if not value:
        return ""
    if len(value) <= 10:
        return value[:2] + "***"
    return value[:6] + "***" + value[-4:]


def _sanitize_schema_name(name: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(name or "output")).strip("_")
    return name[:64] or "output"


def get_safe_config_summary() -> JsonDict:
    """
    Безопасная диагностика для UI/консоли. Ключ маскируется.
    """
    openai_key = _clean_env_value(os.getenv("OPENAI_API_KEY"))
    proxy_key = _clean_env_value(os.getenv("PROXYAPI_API_KEY"))
    selected_key = openai_key or proxy_key

    return {
        "OPENAI_API_KEY_present": bool(openai_key),
        "PROXYAPI_API_KEY_present": bool(proxy_key),
        "selected_key_masked": _mask_secret(selected_key),
        "OPENAI_BASE_URL": _clean_env_value(os.getenv("OPENAI_BASE_URL")),
        "PROXYAPI_BASE_URL": _clean_env_value(os.getenv("PROXYAPI_BASE_URL")),
        "MODEL_NAME": _clean_env_value(os.getenv("MODEL_NAME")),
        "LLM_TEMPERATURE": _clean_env_value(os.getenv("LLM_TEMPERATURE")),
        "LLM_MAX_TOKENS": _clean_env_value(os.getenv("LLM_MAX_TOKENS")),
        "LLM_TIMEOUT_SECONDS": _clean_env_value(os.getenv("LLM_TIMEOUT_SECONDS")),
    }


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
        openai_key = _clean_env_value(os.getenv("OPENAI_API_KEY"))
        proxy_key = _clean_env_value(os.getenv("PROXYAPI_API_KEY"))
        api_key = openai_key or proxy_key

        if not api_key:
            raise LLMAuthError(
                "LLM API key is missing. Set OPENAI_API_KEY in .env. "
                "For ProxyAPI you may also use PROXYAPI_API_KEY."
            )

        base_url = (
                _clean_env_value(os.getenv("OPENAI_BASE_URL"))
                or _clean_env_value(os.getenv("PROXYAPI_BASE_URL"))
                or (
                    "https://api.proxyapi.ru/openai/v1"
                    if proxy_key and not openai_key
                    else "https://api.openai.com/v1"
                )
        ).rstrip("/")

        model = _clean_env_value(os.getenv("MODEL_NAME")) or "gpt-4o-mini"
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
    "generation_evaluation": _schema_obj(
        {
            "relevance": _schema_num(),
            "completeness": _schema_num(),
            "clarity": _schema_num(),
            "usefulness": _schema_num(),
            "correctness": _schema_num(),
            "comment": _schema_str(),
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

    def evaluate_generation(
            self,
            *,
            scenario_part: str,
            input_context: Any,
            generated_output: Any,
            method_name: Optional[str] = None,
            extra_instruction: Optional[str] = None,
    ) -> JsonDict:
        return evaluate_generation_with_llm(
            scenario_part=scenario_part,
            input_context=input_context,
            generated_output=generated_output,
            method_name=method_name,
            extra_instruction=extra_instruction,
            client=self,
        )

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
        msgs[0]["content"] = (str(msgs[0].get("content", "")).rstrip() + "\n\n" + instr).strip()
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
                "name": _sanitize_schema_name(schema_name),
                "schema": json_schema,
                "strict": bool(strict_json),
            },
        }

    try:
        resp = client.chat.completions.create(**kwargs)
        raw = resp.model_dump() if hasattr(resp, "model_dump") else json.loads(resp.json())
        return raw
    except Exception as first_exc:
        _raise_mapped_openai_error(first_exc)

        # Пробуем более менее строгий JSON, если формат схемы не поддерживается
        if json_schema is not None:
            try:
                kwargs2 = dict(kwargs)
                kwargs2["response_format"] = {"type": "json_object"}
                resp2 = client.chat.completions.create(**kwargs2)
                raw2 = resp2.model_dump() if hasattr(resp2, "model_dump") else json.loads(resp2.json())
                return raw2
            except Exception as second_exc:
                _raise_mapped_openai_error(second_exc)

                raise second_exc

        raise first_exc


def _raise_mapped_openai_error(e: Exception) -> None:
    msg = str(e).lower()
    status_code = getattr(e, "status_code", None)

    if status_code == 401 or "api key" in msg or "authentication" in msg or "401" in msg:
        raise LLMAuthError(str(e))
    if status_code == 429 or "rate limit" in msg or "429" in msg:
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
                "name": _sanitize_schema_name(schema_name),
                "schema": json_schema,
                "strict": bool(strict_json),
            },
        }

    return _post_json_with_fallback(
        url=url,
        api_key=cfg.api_key,
        payload=payload,
        timeout_seconds=cfg.timeout_seconds,
        allow_json_object_fallback=json_schema is not None,
    )


def _post_json_with_fallback(
        *,
        url: str,
        api_key: str,
        payload: JsonDict,
        timeout_seconds: int,
        allow_json_object_fallback: bool,
) -> JsonDict:
    try:
        return _post_json(url=url, api_key=api_key, payload=payload, timeout_seconds=timeout_seconds)
    except urllib.error.HTTPError as http_error:
        response_body = http_error.read().decode("utf-8", errors="replace") if hasattr(http_error, "read") else ""

        if allow_json_object_fallback and http_error.code in (400, 404, 422):
            try:
                payload2 = dict(payload)
                payload2["response_format"] = {"type": "json_object"}
                return _post_json(
                    url=url,
                    api_key=api_key,
                    payload=payload2,
                    timeout_seconds=timeout_seconds,
                )
            except urllib.error.HTTPError as http_error2:
                response_body2 = (
                    http_error2.read().decode("utf-8", errors="replace")
                    if hasattr(http_error2, "read")
                    else ""
                )
                _raise_mapped_http_error(http_error2.code, response_body2 or str(http_error2))
            except urllib.error.URLError as url_error2:
                raise LLMError(f"LLM URL error: {url_error2}") from url_error2

        _raise_mapped_http_error(http_error.code, response_body or str(http_error))
        raise
    except urllib.error.URLError as url_error:
        raise LLMError(f"LLM URL error: {url_error}") from url_error


def _post_json(
        *,
        url: str,
        api_key: str,
        payload: JsonDict,
        timeout_seconds: int,
) -> JsonDict:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url=url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
        resp_body = resp.read().decode("utf-8", errors="replace")
        return json.loads(resp_body)


def _raise_mapped_http_error(status_code: int, body: str) -> None:
    body_l = str(body or "").lower()
    if status_code == 401:
        raise LLMAuthError(body or "LLM authentication error")
    if status_code == 429:
        raise LLMRateLimitError(body or "LLM rate limit error")
    if "api key" in body_l or "authentication" in body_l:
        raise LLMAuthError(body)
    raise LLMHTTPError(status_code, body)


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
        if choices:
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


# Generation evaluation helper

GENERATION_EVALUATION_SYSTEM_PROMPT = """
Ты независимый оценщик качества генерации в системе подготовки и защиты учебных работ.
Твоя задача — оценить не студента и не саму учебную работу, а качество ответа, который сгенерировала тестируемая модель.

Поставь целые оценки от 1 до 5 по пяти критериям:
1. relevance — релевантность входному контексту и части сценария;
2. completeness — полнота ответа относительно задачи генерации;
3. clarity — ясность, структурность и понятность формулировок;
4. usefulness — практическая полезность результата для пользователя сценария;
5. correctness — корректность, отсутствие противоречий и необоснованных утверждений.

Шкала:
1 — результат почти непригоден;
2 — много существенных проблем;
3 — приемлемо, но есть заметные недостатки;
4 — хороший результат с небольшими недочетами;
5 — качественный результат без существенных замечаний.

Верни только JSON по заданной схеме. Все оценки должны быть целыми числами от 1 до 5.
В comment кратко объясни главную причину оценки на русском языке.
""".strip()


def _json_dumps_compact(value: Any, max_chars: int = 12000) -> str:
    try:
        if isinstance(value, str):
            text = value
        else:
            text = json.dumps(value, ensure_ascii=False, indent=2)
    except Exception:
        text = str(value)
    text = text.strip()
    if len(text) > max_chars:
        return text[:max_chars].rstrip() + "\n...[фрагмент сокращен]"
    return text


def build_generation_evaluation_prompt(
        *,
        scenario_part: str,
        input_context: Any,
        generated_output: Any,
        method_name: Optional[str] = None,
        extra_instruction: Optional[str] = None,
) -> str:
    """
    Формирует пользовательский prompt для LLM-оценщика.

    Важно: prompt описывает именно проверяемую генерацию. Формальные метрики
    считаются отдельно в evaluation.py, а здесь модель оценивает смысловое
    качество по пяти критериям.
    """
    method_block = f"\nМетод / этап: {method_name}" if method_name else ""
    extra_block = f"\n\nДополнительные указания к оценке:\n{extra_instruction.strip()}" if extra_instruction else ""

    return f"""
Часть сценария: {scenario_part}{method_block}

Входной контекст, на основе которого тестируемая модель должна была выполнить генерацию:
{_json_dumps_compact(input_context)}

Результат генерации тестируемой модели:
{_json_dumps_compact(generated_output)}
{extra_block}

Оцени результат генерации по пяти критериям. Не сравнивай его с идеальным ответом, которого нет во входных данных. Оцени, насколько результат подходит для указанной части сценария и насколько он полезен в системе подготовки/защиты учебной работы.
""".strip()


def _normalize_generation_evaluation_response(data: JsonDict) -> JsonDict:
    """
    Приводит ответ LLM-оценщика к стабильному формату.

    Даже при JSON-схеме совместимый провайдер может вернуть числа строками или
    дробные значения. Для отчета нужны целые оценки 1–5.
    """
    normalized: JsonDict = {}
    for key in ("relevance", "completeness", "clarity", "usefulness", "correctness"):
        value = data.get(key)
        try:
            value_int = int(round(float(value)))
        except Exception:
            value_int = 1
        normalized[key] = max(1, min(5, value_int))

    comment = data.get("comment") or data.get("reason") or data.get("summary") or ""
    normalized["comment"] = str(comment).strip()
    return normalized


def evaluate_generation_with_llm(
        *,
        scenario_part: str,
        input_context: Any,
        generated_output: Any,
        method_name: Optional[str] = None,
        extra_instruction: Optional[str] = None,
        client: Optional[LLMClient] = None,
        config: Optional[LLMConfig] = None,
) -> JsonDict:
    """
    Оценивает качество одной генерации сторонней моделью.

    Функция возвращает JSON со значениями 1–5:
    relevance, completeness, clarity, usefulness, correctness, comment.

    Эти значения затем передаются в evaluation.py, где средняя LLM-оценка
    нормализуется по формуле (raw_score - 1) / 4 и объединяется с heuristic_score.
    """
    judge = client or LLMClient(config=config)
    prompt = build_generation_evaluation_prompt(
        scenario_part=scenario_part,
        method_name=method_name,
        input_context=input_context,
        generated_output=generated_output,
        extra_instruction=extra_instruction,
    )

    response = judge.generate(
        system_prompt=GENERATION_EVALUATION_SYSTEM_PROMPT,
        user_prompt=prompt,
        schema_name="generation_evaluation",
        strict_json=True,
    )
    if response.json is None:
        raise LLMOutputParseError("Generation evaluation response was expected as JSON.")

    result = _normalize_generation_evaluation_response(response.json)
    result["judge_model"] = response.model
    result["judge_latency_ms"] = response.latency_ms
    return result


# Smoke test

def smoke_test() -> JsonDict:
    """
    Быстрая проверка из консоли:
        python -c "from llm import smoke_test; print(smoke_test())"
    """
    client = LLMClient()
    return client.generate_json(
        schema_name="topic_relation_assessment",
        system_prompt="Ты оцениваешь смысловую связь двух учебных тем. Верни только JSON.",
        user_prompt=(
            "Тема преподавателя: Классификация текстовых сообщений методами машинного обучения.\n"
            "Описание преподавателя: Нужно разработать учебный прототип, который принимает короткий текст "
            "на естественном языке, относит его к одному из заранее заданных классов и объясняет общий "
            "принцип работы алгоритма.\n\n"
            "Тема студента: Сортировка обращений граждан по подразделениям администрации.\n"
            "Описание студента: Я хочу сделать программу, которая читает короткие обращения граждан и "
            "автоматически предлагает, куда их направить: в отдел ЖКХ, транспорта, благоустройства или "
            "социальной поддержки."
        ),
    )


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
    "get_safe_config_summary",
    "call_llm",
    "call_llm_json",
    "create_llm_client",
    "build_llm_client",
    "get_llm_client",
    "GENERATION_EVALUATION_SYSTEM_PROMPT",
    "build_generation_evaluation_prompt",
    "evaluate_generation_with_llm",
    "smoke_test",
]
