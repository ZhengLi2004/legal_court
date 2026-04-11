"""Evaluator profile and OpenAI-compatible NLI labelers for Step 11."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import OpenAI

from benchmarks.experiments.eval.faithfulness_pipeline import (
    VALID_EVAL_LABELS,
    ClaimAssertion,
    EvidenceWindow,
    NliLabeler,
)

CLAIM2_NLI_PROMPT_VERSION = "step11_claim2_nli_v1"
CLAIM2_NLI_MAX_OUTPUT_TOKENS = 16

CLAIM2_NLI_SYSTEM_PROMPT = (
    "你是一名法律事实忠实度评测器。"
    "请判断给定证据窗口与给定断言之间的关系。"
    "只允许输出一个大写标签：E、N、C。"
    "E=证据足以支持断言；N=证据不足以支持也不足以否定；C=证据与断言矛盾。"
    "禁止输出解释、标点或其他文本。"
)

CLAIM2_NLI_USER_TEMPLATE = """请判断下列证据窗口与断言之间的关系，并只输出一个标签：E、N 或 C。

Assertion:
{assertion_text}

Evidence Section:
{section}

Evidence Window:
{window_text}
"""


def _claim2_chat_completion_kwargs(spec: "EvaluatorSpec") -> dict[str, Any]:
    """Return provider-specific chat completion kwargs for Claim 2 evaluators.

    The current OpenAI-compatible gateway exposes `Qwen3.5-35B-A3B` as a
    reasoning-style model that may emit only `reasoning_content` unless
    thinking is disabled via `chat_template_kwargs.enable_thinking=False`.
    Claim 2 requires a single E/N/C label in `message.content`, so we force
    no-thinking mode for this model family here.
    """

    if spec.backend == "openai_chat" and spec.model_name == "Qwen3.5-35B-A3B":
        return {
            "extra_body": {
                "chat_template_kwargs": {
                    "enable_thinking": False,
                }
            }
        }

    return {}


@dataclass(frozen=True)
class EvaluatorSpec:
    """One evaluator endpoint or local model used by the Claim 2 pipeline.

    Attributes:
        name: Stable evaluator name written into votes and summaries.
        model_name: Remote model id or local model display name.
        base_url: OpenAI-compatible base URL for chat-backed evaluators.
        api_key_env: Environment-variable name that stores the API key.
        endpoint_version: Version tag for caching and reproducibility.
        cohort: Evaluator cohort, such as `primary` or `supplementary`.
        backend: Backend type, either `openai_chat` or `local_nli`.
        local_model_path: Local model path used by `local_nli`.
        device: Preferred device string for local models.
        max_input_tokens: Maximum evaluator input budget recorded in snapshots.
    """

    name: str
    model_name: str
    base_url: str = ""
    api_key_env: str = ""
    endpoint_version: str = ""
    cohort: str = "primary"
    backend: str = "openai_chat"
    local_model_path: str = ""
    device: str = ""
    max_input_tokens: int = 8192

    def to_snapshot(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model_name": self.model_name,
            "base_url": self.base_url,
            "api_key_env": self.api_key_env,
            "endpoint_version": self.endpoint_version,
            "cohort": self.cohort,
            "backend": self.backend,
            "local_model_path": self.local_model_path,
            "device": self.device,
            "max_input_tokens": self.max_input_tokens,
            "api_key_fingerprint": _secret_fingerprint(os.getenv(self.api_key_env, "")),
        }


@dataclass(frozen=True)
class EvaluatorProfile:
    """Named collection of primary and supplementary Claim 2 evaluators.

    Attributes:
        profile_name: Human-readable profile identifier.
        primary_evaluators: Evaluators used for the main Claim 2 vote.
        supplementary_evaluators: Optional appendix-only evaluators.
    """

    profile_name: str
    primary_evaluators: tuple[EvaluatorSpec, ...]
    supplementary_evaluators: tuple[EvaluatorSpec, ...] = ()

    def to_snapshot(self) -> dict[str, Any]:
        return {
            "profile_name": self.profile_name,
            "primary_evaluators": [
                item.to_snapshot() for item in self.primary_evaluators
            ],
            "supplementary_evaluators": [
                item.to_snapshot() for item in self.supplementary_evaluators
            ],
        }


def _secret_fingerprint(value: str) -> dict[str, Any]:
    text = str(value or "")

    if not text:
        return {"set": False, "sha256_prefix": ""}

    return {
        "set": True,
        "sha256_prefix": hashlib.sha256(text.encode("utf-8")).hexdigest()[:12],
    }


def build_claim2_prompt_snapshot() -> dict[str, Any]:
    """Return the frozen prompt bundle used by Claim 2 NLI evaluators.

    Returns:
        A JSON-serializable prompt bundle used for evaluator snapshots and caches.
    """

    return {
        "prompt_version": CLAIM2_NLI_PROMPT_VERSION,
        "system_prompt": CLAIM2_NLI_SYSTEM_PROMPT,
        "user_template": CLAIM2_NLI_USER_TEMPLATE,
    }


def load_evaluator_profile(path: str | Path) -> EvaluatorProfile:
    """Load and validate one evaluator profile JSON file.

    Args:
        path: Profile JSON path.

    Returns:
        A validated evaluator profile object.

    Raises:
        ValueError: If the JSON payload is malformed or violates profile rules.
    """

    payload = json.loads(Path(path).read_text(encoding="utf-8"))

    if not isinstance(payload, dict):
        raise ValueError("Evaluator profile must be a JSON object.")

    profile_name = str(payload.get("profile_name", "") or "").strip()

    if not profile_name:
        raise ValueError("Evaluator profile must contain non-empty `profile_name`.")

    primary = _parse_evaluator_group(
        payload.get("primary_evaluators"), cohort="primary"
    )

    supplementary = _parse_evaluator_group(
        payload.get("supplementary_evaluators", []),
        cohort="supplementary",
        allow_empty=True,
    )

    if len(primary) < 2:
        raise ValueError(
            "Evaluator profile must contain at least 2 primary evaluators."
        )

    names = [item.name for item in (*primary, *supplementary)]

    if len(set(names)) != len(names):
        raise ValueError("Evaluator names must be unique across the whole profile.")

    return EvaluatorProfile(
        profile_name=profile_name,
        primary_evaluators=tuple(primary),
        supplementary_evaluators=tuple(supplementary),
    )


def _parse_evaluator_group(
    raw_group: Any,
    *,
    cohort: str,
    allow_empty: bool = False,
) -> list[EvaluatorSpec]:
    if raw_group is None:
        if allow_empty:
            return []

        raise ValueError(f"Evaluator profile missing `{cohort}_evaluators`.")

    if not isinstance(raw_group, list):
        raise ValueError(f"`{cohort}_evaluators` must be a list.")

    if not raw_group and not allow_empty:
        raise ValueError(f"`{cohort}_evaluators` must not be empty.")

    group: list[EvaluatorSpec] = []

    for index, row in enumerate(raw_group):
        if not isinstance(row, dict):
            raise ValueError(f"{cohort}[{index}] must be a JSON object.")

        backend = str(row.get("backend", "openai_chat") or "").strip().lower()

        if backend not in {"openai_chat", "local_nli"}:
            raise ValueError(
                f"{cohort}[{index}] has unsupported `backend`: {backend!r}."
            )

        base_url = str(row.get("base_url", "") or "").strip()
        base_url_env = str(row.get("base_url_env", "") or "").strip()

        if not base_url and base_url_env:
            base_url = str(os.getenv(base_url_env, "") or "").strip()

        group.append(
            EvaluatorSpec(
                name=_require_str(row, "name", cohort=cohort, index=index),
                model_name=_require_str(row, "model_name", cohort=cohort, index=index),
                base_url=(base_url if backend == "openai_chat" else ""),
                api_key_env=(
                    _require_str(row, "api_key_env", cohort=cohort, index=index)
                    if backend == "openai_chat"
                    else ""
                ),
                endpoint_version=(
                    _require_str(row, "endpoint_version", cohort=cohort, index=index)
                    if backend == "openai_chat"
                    else str(row.get("endpoint_version", "") or "").strip()
                    or "transformers-local-v1"
                ),
                cohort=cohort,
                backend=backend,
                local_model_path=(
                    _require_str(row, "local_model_path", cohort=cohort, index=index)
                    if backend == "local_nli"
                    else ""
                ),
                device=(
                    str(row.get("device", "") or "").strip()
                    if backend == "local_nli"
                    else ""
                ),
                max_input_tokens=_require_positive_int(
                    row,
                    "max_input_tokens",
                    cohort=cohort,
                    index=index,
                    default=8192,
                ),
            )
        )

    return group


def _require_str(
    row: dict[str, Any],
    field_name: str,
    *,
    cohort: str,
    index: int,
) -> str:
    value = str(row.get(field_name, "") or "").strip()

    if not value:
        raise ValueError(f"{cohort}[{index}] missing non-empty `{field_name}`.")

    return value


def _require_positive_int(
    row: dict[str, Any],
    field_name: str,
    *,
    cohort: str,
    index: int,
    default: int,
) -> int:
    raw = row.get(field_name, default)

    try:
        value = int(raw)

    except Exception as exc:
        raise ValueError(
            f"{cohort}[{index}] has invalid `{field_name}`: {raw!r}."
        ) from exc

    if value <= 0:
        raise ValueError(f"{cohort}[{index}] `{field_name}` must be > 0.")

    return value


def probe_evaluator_profile(
    profile: EvaluatorProfile,
) -> list[dict[str, Any]]:
    """Probe every evaluator in a profile and return health-check metadata.

    Args:
        profile: Evaluator profile whose members should be probed.

    Returns:
        One health-check row per evaluator.
    """

    rows: list[dict[str, Any]] = []

    for spec in (*profile.primary_evaluators, *profile.supplementary_evaluators):
        rows.append(probe_evaluator(spec))

    return rows


def probe_evaluator(spec: EvaluatorSpec) -> dict[str, Any]:
    """Probe one evaluator endpoint or local model without scoring any case.

    Args:
        spec: Evaluator specification to probe.

    Returns:
        A health-check payload describing latency, response excerpt, and errors.
    """

    if spec.backend == "local_nli":
        return _probe_local_nli_evaluator(spec)

    started_at = time.perf_counter()
    api_key = os.getenv(spec.api_key_env, "")

    try:
        if not api_key:
            raise ValueError(
                f"Missing API key environment variable for evaluator {spec.name}: {spec.api_key_env}"
            )

        client = OpenAI(base_url=spec.base_url, api_key=api_key)
        response = client.chat.completions.create(
            model=spec.model_name,
            messages=[
                {"role": "system", "content": CLAIM2_NLI_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": "Assertion:\n测试断言\n\nEvidence Section:\n测试\n\nEvidence Window:\n测试证据\n\n只输出 E",
                },
            ],
            max_tokens=CLAIM2_NLI_MAX_OUTPUT_TOKENS,
            temperature=0.0,
            **_claim2_chat_completion_kwargs(spec),
        )

        content = str(response.choices[0].message.content or "").strip()
        normalized = _normalize_raw_label(content)

        if normalized != "E":
            raise ValueError(f"Probe returned unexpected label: {content!r}")

        return {
            "name": spec.name,
            "cohort": spec.cohort,
            "ok": True,
            "model_name": spec.model_name,
            "base_url": spec.base_url,
            "endpoint_version": spec.endpoint_version,
            "api_key_env": spec.api_key_env,
            "max_input_tokens": spec.max_input_tokens,
            "api_key_fingerprint": _secret_fingerprint(api_key),
            "latency_ms": round((time.perf_counter() - started_at) * 1000.0, 3),
            "response_excerpt": content[:80],
            "error_type": "",
            "error_detail": "",
        }

    except Exception as exc:
        return {
            "name": spec.name,
            "cohort": spec.cohort,
            "ok": False,
            "model_name": spec.model_name,
            "base_url": spec.base_url,
            "endpoint_version": spec.endpoint_version,
            "api_key_env": spec.api_key_env,
            "max_input_tokens": spec.max_input_tokens,
            "api_key_fingerprint": _secret_fingerprint(api_key),
            "latency_ms": round((time.perf_counter() - started_at) * 1000.0, 3),
            "response_excerpt": "",
            "error_type": type(exc).__name__,
            "error_detail": str(exc),
        }


def _probe_local_nli_evaluator(spec: EvaluatorSpec) -> dict[str, Any]:
    started_at = time.perf_counter()

    try:
        label, raw_label = _predict_local_nli_label(
            spec,
            premise="法院认为，被告未按约支付货款。",
            hypothesis="法院支持了支付货款的诉请。",
        )

        if label not in VALID_EVAL_LABELS:
            raise ValueError(f"Probe returned unexpected label: {raw_label!r}")

        return {
            "name": spec.name,
            "cohort": spec.cohort,
            "ok": True,
            "backend": spec.backend,
            "model_name": spec.model_name,
            "base_url": "",
            "local_model_path": spec.local_model_path,
            "device": _resolve_local_nli_device(spec),
            "endpoint_version": spec.endpoint_version,
            "max_input_tokens": spec.max_input_tokens,
            "effective_max_input_tokens": _resolve_local_nli_max_length(spec),
            "api_key_env": "",
            "api_key_fingerprint": _secret_fingerprint(""),
            "latency_ms": round((time.perf_counter() - started_at) * 1000.0, 3),
            "response_excerpt": raw_label,
            "error_type": "",
            "error_detail": "",
        }

    except Exception as exc:
        return {
            "name": spec.name,
            "cohort": spec.cohort,
            "ok": False,
            "backend": spec.backend,
            "model_name": spec.model_name,
            "base_url": "",
            "local_model_path": spec.local_model_path,
            "device": _resolve_local_nli_device(spec),
            "endpoint_version": spec.endpoint_version,
            "max_input_tokens": spec.max_input_tokens,
            "effective_max_input_tokens": _resolve_local_nli_max_length(spec),
            "api_key_env": "",
            "api_key_fingerprint": _secret_fingerprint(""),
            "latency_ms": round((time.perf_counter() - started_at) * 1000.0, 3),
            "response_excerpt": "",
            "error_type": type(exc).__name__,
            "error_detail": str(exc),
        }


class OpenAINliLabeler(NliLabeler):
    """OpenAI-compatible evaluator with disk-backed label cache.

    Args:
        spec: Evaluator specification describing the remote endpoint.
        cache_dir: Directory used to persist label-cache entries.
        max_attempts: Maximum number of contract-preserving retries.
    """

    def __init__(
        self,
        spec: EvaluatorSpec,
        *,
        cache_dir: str | Path,
        max_attempts: int = 3,
    ) -> None:
        api_key = os.getenv(spec.api_key_env, "")

        if not api_key:
            raise ValueError(
                f"Missing API key environment variable for evaluator {spec.name}: {spec.api_key_env}"
            )

        self.name = spec.name
        self.spec = spec
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_attempts = int(max_attempts)
        self._client = OpenAI(base_url=spec.base_url, api_key=api_key)

    def label(self, assertion: ClaimAssertion, window: EvidenceWindow) -> str:
        cache_path = self._cache_path(assertion, window)
        cached = _try_load_cached_label(cache_path)

        if cached is not None:
            return cached

        messages = [
            {"role": "system", "content": CLAIM2_NLI_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _build_chat_evaluator_user_content(
                    spec=self.spec,
                    assertion_text=assertion.assertion_text,
                    section=window.section,
                    window_text=window.text,
                ),
            },
        ]

        last_error: Exception | None = None

        for attempt in range(1, self.max_attempts + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self.spec.model_name,
                    messages=messages,
                    max_tokens=CLAIM2_NLI_MAX_OUTPUT_TOKENS,
                    temperature=0.0,
                    **_claim2_chat_completion_kwargs(self.spec),
                )

                content = str(response.choices[0].message.content or "").strip()
                label = _normalize_raw_label(content)

                _write_cached_label(
                    cache_path,
                    {
                        "assertion_id": assertion.assertion_id,
                        "case_uid": assertion.case_uid,
                        "claim_id": assertion.claim_id,
                        "window_id": window.window_id,
                        "evaluator_name": self.name,
                        "model_name": self.spec.model_name,
                        "endpoint_version": self.spec.endpoint_version,
                        "max_input_tokens": self.spec.max_input_tokens,
                        "prompt_version": CLAIM2_NLI_PROMPT_VERSION,
                        "label": label,
                        "raw_response": content,
                        "attempt": attempt,
                    },
                )

                return label

            except Exception as exc:
                last_error = exc

                if attempt < self.max_attempts:
                    time.sleep(0.2 * attempt)

        raise ValueError(
            f"Evaluator {self.name} failed to emit a valid E/N/C label after {self.max_attempts} attempts: {last_error}"
        )

    def _cache_path(self, assertion: ClaimAssertion, window: EvidenceWindow) -> Path:
        payload = {
            "assertion_id": assertion.assertion_id,
            "case_uid": assertion.case_uid,
            "claim_id": assertion.claim_id,
            "window_id": window.window_id,
            "evaluator_name": self.name,
            "model_name": self.spec.model_name,
            "endpoint_version": self.spec.endpoint_version,
            "max_input_tokens": self.spec.max_input_tokens,
            "prompt_version": CLAIM2_NLI_PROMPT_VERSION,
        }

        digest = hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()

        return self.cache_dir / f"{digest}.json"


class LocalNliLabeler(NliLabeler):
    """Local transformers-based NLI evaluator with disk-backed cache.

    Args:
        spec: Evaluator specification describing the local model.
        cache_dir: Directory used to persist label-cache entries.
    """

    def __init__(
        self,
        spec: EvaluatorSpec,
        *,
        cache_dir: str | Path,
    ) -> None:
        if spec.backend != "local_nli":
            raise ValueError(
                f"LocalNliLabeler requires `local_nli` backend, got {spec.backend!r}"
            )

        self.name = spec.name
        self.spec = spec
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def label(self, assertion: ClaimAssertion, window: EvidenceWindow) -> str:
        cache_path = self._cache_path(assertion, window)
        cached = _try_load_cached_label(cache_path)

        if cached is not None:
            return cached

        label, raw_label = _predict_local_nli_label(
            self.spec,
            premise=window.text,
            hypothesis=assertion.assertion_text,
        )

        _write_cached_label(
            cache_path,
            {
                "assertion_id": assertion.assertion_id,
                "case_uid": assertion.case_uid,
                "claim_id": assertion.claim_id,
                "window_id": window.window_id,
                "evaluator_name": self.name,
                "backend": self.spec.backend,
                "model_name": self.spec.model_name,
                "local_model_path": self.spec.local_model_path,
                "device": _resolve_local_nli_device(self.spec),
                "endpoint_version": self.spec.endpoint_version,
                "max_input_tokens": self.spec.max_input_tokens,
                "effective_max_input_tokens": _resolve_local_nli_max_length(self.spec),
                "prompt_version": CLAIM2_NLI_PROMPT_VERSION,
                "label": label,
                "raw_response": raw_label,
            },
        )

        return label

    def _cache_path(self, assertion: ClaimAssertion, window: EvidenceWindow) -> Path:
        payload = {
            "assertion_id": assertion.assertion_id,
            "case_uid": assertion.case_uid,
            "claim_id": assertion.claim_id,
            "window_id": window.window_id,
            "evaluator_name": self.name,
            "backend": self.spec.backend,
            "model_name": self.spec.model_name,
            "local_model_path": self.spec.local_model_path,
            "device": _resolve_local_nli_device(self.spec),
            "endpoint_version": self.spec.endpoint_version,
            "max_input_tokens": self.spec.max_input_tokens,
            "effective_max_input_tokens": _resolve_local_nli_max_length(self.spec),
            "prompt_version": CLAIM2_NLI_PROMPT_VERSION,
        }

        digest = hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()

        return self.cache_dir / f"{digest}.json"


def build_primary_labelers(
    profile: EvaluatorProfile,
    *,
    cache_dir: str | Path,
) -> list[NliLabeler]:
    """Instantiate cached labelers for the primary evaluator cohort.

    Args:
        profile: Evaluator profile containing primary evaluator specs.
        cache_dir: Shared cache directory for emitted votes.

    Returns:
        Instantiated labelers in profile order.
    """

    return [
        _build_labeler(spec, cache_dir=cache_dir) for spec in profile.primary_evaluators
    ]


def build_supplementary_labelers(
    profile: EvaluatorProfile,
    *,
    cache_dir: str | Path,
) -> list[NliLabeler]:
    """Instantiate cached labelers for the supplementary evaluator cohort.

    Args:
        profile: Evaluator profile containing supplementary evaluator specs.
        cache_dir: Shared cache directory for emitted votes.

    Returns:
        Instantiated labelers in profile order.
    """

    return [
        _build_labeler(spec, cache_dir=cache_dir)
        for spec in profile.supplementary_evaluators
    ]


def build_labelers_for_specs(
    specs: list[EvaluatorSpec] | tuple[EvaluatorSpec, ...],
    *,
    cache_dir: str | Path,
) -> list[NliLabeler]:
    """Instantiate cached labelers for an explicit evaluator spec list.

    Args:
        specs: Explicit evaluator specs to instantiate.
        cache_dir: Shared cache directory for emitted votes.

    Returns:
        Instantiated labelers in the requested order.
    """

    return [_build_labeler(spec, cache_dir=cache_dir) for spec in specs]


def _build_labeler(spec: EvaluatorSpec, *, cache_dir: str | Path) -> NliLabeler:
    if spec.backend == "local_nli":
        return LocalNliLabeler(spec, cache_dir=cache_dir)

    return OpenAINliLabeler(spec, cache_dir=cache_dir)


def _normalize_raw_label(content: str) -> str:
    text = str(content or "").strip().upper()
    compact = "".join(ch for ch in text if ch.isalpha())

    if compact in VALID_EVAL_LABELS:
        return compact

    matches = re.findall(r"(?:^|[^A-Z])(E|N|C)(?:$|[^A-Z])", text)

    if matches:
        unique = sorted(set(matches))

        if len(unique) == 1:
            return unique[0]

    raise ValueError(f"Unsupported evaluator label output: {content!r}")


def _normalize_local_nli_label(content: str) -> str:
    text = str(content or "").strip().lower().replace("_", " ").replace("-", " ")

    if "entail" in text:
        return "E"

    if "neutral" in text:
        return "N"

    if "contrad" in text:
        return "C"

    raise ValueError(f"Unsupported local NLI label output: {content!r}")


def _build_chat_evaluator_user_content(
    *,
    spec: EvaluatorSpec,
    assertion_text: str,
    section: str,
    window_text: str,
) -> str:
    budget_chars = int(spec.max_input_tokens) * 2

    prefix = CLAIM2_NLI_USER_TEMPLATE.format(
        assertion_text=assertion_text,
        section=section,
        window_text="",
    )

    remaining = max(0, budget_chars - len(CLAIM2_NLI_SYSTEM_PROMPT) - len(prefix))
    trimmed_window_text = str(window_text or "")[:remaining] if remaining else ""

    return CLAIM2_NLI_USER_TEMPLATE.format(
        assertion_text=assertion_text,
        section=section,
        window_text=trimmed_window_text,
    )


def _try_load_cached_label(path: Path) -> str | None:
    if not path.exists():
        return None

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        label = _normalize_raw_label(str(payload.get("label", "") or ""))

    except Exception:
        return None

    return label


def _write_cached_label(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


_LOCAL_NLI_RUNTIME_CACHE: dict[tuple[str, str], tuple[Any, Any, str]] = {}


def _resolve_local_nli_device(spec: EvaluatorSpec) -> str:
    if spec.device:
        return spec.device

    try:
        import torch

        if torch.cuda.is_available():
            return "cuda:0"

    except Exception:
        pass

    return "cpu"


def _resolve_local_nli_max_length(spec: EvaluatorSpec) -> int:
    runtime = _LOCAL_NLI_RUNTIME_CACHE.get(
        (spec.local_model_path, _resolve_local_nli_device(spec))
    )

    tokenizer = runtime[0] if runtime is not None else None
    model = runtime[1] if runtime is not None else None
    return _compute_local_nli_max_length(spec, tokenizer=tokenizer, model=model)


def _compute_local_nli_max_length(
    spec: EvaluatorSpec,
    *,
    tokenizer: Any | None,
    model: Any | None,
) -> int:
    candidates = [int(spec.max_input_tokens)]
    tokenizer_limit = int(getattr(tokenizer, "model_max_length", 0) or 0)

    if tokenizer_limit and tokenizer_limit < 1000000:
        candidates.append(tokenizer_limit)

    model_limit = int(
        getattr(getattr(model, "config", None), "max_position_embeddings", 0) or 0
    )

    if model_limit:
        candidates.append(model_limit)

    return max(1, min(candidates))


def _load_local_nli_runtime(spec: EvaluatorSpec) -> tuple[Any, Any, str]:
    cache_key = (spec.local_model_path, _resolve_local_nli_device(spec))

    if cache_key in _LOCAL_NLI_RUNTIME_CACHE:
        return _LOCAL_NLI_RUNTIME_CACHE[cache_key]

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = _resolve_local_nli_device(spec)
    tokenizer = AutoTokenizer.from_pretrained(spec.local_model_path)
    model = AutoModelForSequenceClassification.from_pretrained(spec.local_model_path)
    model = model.to(device).eval()
    _LOCAL_NLI_RUNTIME_CACHE[cache_key] = (tokenizer, model, device)
    return tokenizer, model, device


def _predict_local_nli_label(
    spec: EvaluatorSpec,
    *,
    premise: str,
    hypothesis: str,
) -> tuple[str, str]:
    import torch

    tokenizer, model, device = _load_local_nli_runtime(spec)

    effective_max_length = _compute_local_nli_max_length(
        spec,
        tokenizer=tokenizer,
        model=model,
    )

    inputs = tokenizer(
        premise,
        hypothesis,
        return_tensors="pt",
        truncation=True,
        max_length=effective_max_length,
    )

    inputs = {name: value.to(device) for name, value in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits[0]

    pred_idx = int(logits.argmax().item())
    raw_label = str(model.config.id2label[pred_idx])
    return _normalize_local_nli_label(raw_label), raw_label
