"""
Единый клиент для вызова LLM (OpenAI-compatible) из вашего приложения.

Особенности:
- Читает настройки из .env / окружения:
    OPENAI_API_KEY        (обязательно)
    OPENAI_BASE_URL       (опционально, по умолчанию https://api.openai.com/v1)
    MODEL_NAME            (опционально)
    LLM_TEMPERATURE       (опционально)
    LLM_MAX_TOKENS        (опционально)
    LLM_TIMEOUT_SECONDS   (опционально)
- Работает в 2 режимах:
    1) Через python-библиотеку openai (если установлена)
    2) Через прямой HTTP (urllib из stdlib), если openai не установлена
- Поддерживает структурированный JSON-вывод:
    - если передан json_schema: пытается включить response_format json_schema (если API поддерживает)
    - при несовместимости автоматически откатывается на "json_object" и/или промпт-инструкцию
"""

from __future__ import annotations

import json
import ast
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

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
                "or set as an environment variable."
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


# Public API


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

    # 1) Try OpenAI python library (if installed)
    try:
        resp_raw = _call_with_openai_lib(
            cfg, msgs,
            json_schema=json_schema,
            schema_name=schema_name,
            strict_json=strict_json
        )
        latency_ms = int((time.time() - t0) * 1000)
        return _normalize_response(resp_raw, cfg.model, latency_ms, json_schema=json_schema)
    except ModuleNotFoundError:
        # openai library not installed -> fallback to HTTP
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
        cfg, msgs,
        json_schema=json_schema,
        schema_name=schema_name,
        strict_json=strict_json
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
        "Верни результат СТРОГО как один JSON-объект без пояснений, без markdown-кодов.\n"
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
        strict_json: bool
) -> JsonDict:
    """
    Uses openai python library if installed (OpenAI-compatible).
    Supports base_url proxies.
    """
    # OpenAI may exist under different import paths depending on version.
    # Primary: from openai import OpenAI
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

    # Try structured output if requested
    if json_schema is not None:
        # Newer OpenAI-style: response_format={"type":"json_schema","json_schema":{...}}
        # Some proxies/older servers may not support this -> will raise -> caller fallback.
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
        # Convert to plain dict-like
        raw = resp.model_dump() if hasattr(resp, "model_dump") else json.loads(resp.json())
        return raw
    except Exception as e:
        # Try softer JSON mode if schema format not supported
        if json_schema is not None:
            try:
                kwargs2 = dict(kwargs)
                kwargs2["response_format"] = {"type": "json_object"}
                resp2 = client.chat.completions.create(**kwargs2)
                raw2 = resp2.model_dump() if hasattr(resp2, "model_dump") else json.loads(resp2.json())
                return raw2
            except Exception:
                pass
        # Map common errors
        _raise_mapped_openai_error(e)
        raise


def _raise_mapped_openai_error(e: Exception) -> None:
    msg = str(e).lower()
    if "api key" in msg or "authentication" in msg or "401" in msg:
        raise LLMAuthError(str(e))
    if "rate limit" in msg or "429" in msg:
        raise LLMRateLimitError(str(e))
    # otherwise do nothing here


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
    Direct HTTP call to {base_url}/chat/completions (OpenAI-compatible).
    """
    url = cfg.base_url.rstrip("/") + "/chat/completions"

    payload: JsonDict = {
        "model": cfg.model,
        "messages": messages,
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
    }

    if json_schema is not None:
        # Try "json_schema" mode first (if server supports)
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
        # If json_schema not supported, retry with json_object
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


def _normalize_response(raw: JsonDict, model: str, latency_ms: int, *, json_schema: Optional[JsonDict]) -> LLMResponse:
    """
    Normalize OpenAI-compatible response into LLMResponse.
    """
    text = _extract_text(raw)
    parsed: Optional[JsonDict] = None

    if json_schema is not None:
        # Try direct JSON parse
        parsed = _parse_json_best_effort(text)
        if parsed is None:
            # Sometimes model returns JSON in a tool-like field; attempt to find it
            parsed = _parse_json_from_raw(raw)
        if parsed is None:
            raise LLMOutputParseError(
                "Model was asked to return JSON, but output could not be parsed as JSON."
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
        if content is None:
            # Some providers may use "text"
            content = msg.get("text", "")
        return str(content or "").strip()
    except Exception:
        return ""


def _parse_json_from_raw(raw: JsonDict) -> Optional[JsonDict]:
    """
    Some providers put structured outputs elsewhere. Attempt a few known patterns.
    """
    # 1) message.content as string already handled
    # 2) message.tool_calls[].function.arguments (OpenAI function calling style)
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

    # 3) Some servers return {"output": {...}}
    out = raw.get("output")
    if isinstance(out, dict):
        return out

    return None


def _parse_json_best_effort(text: str) -> Optional[JsonDict]:
    """
    Parse JSON object from text.
    - Accepts pure JSON
    - Accepts JSON wrapped in markdown fences
    - Accepts extra text: tries to extract first {...} block
    - Accepts python-like dict via ast.literal_eval (single quotes, True/None)
    """
    if not text:
        return None

    t = text.strip()

    # Remove markdown fences if present
    t = re.sub(r"^\s*```(?:json)?\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*```\s*$", "", t)

    # 1) direct json
    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    # 2) extract first {...} and try json
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
    Extract first {...} JSON object using brace counting (handles nested braces).
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
                return s[start: i + 1]

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
