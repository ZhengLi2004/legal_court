"""Diagnose Claim 2 evaluator availability at network and probe layers.

This script is intentionally read-only: it does not score any cases and does
not modify existing Claim 2 / Claim 3 artifacts. It helps distinguish between:

1. Environment/configuration issues
2. DNS / TCP / HTTP reachability issues
3. OpenAI-compatible evaluator contract issues (for example, empty outputs)
4. Local NLI model probe failures
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from openai import OpenAI

from benchmarks.experiments.eval.claim2_evaluators import (
    CLAIM2_NLI_MAX_OUTPUT_TOKENS,
    CLAIM2_NLI_SYSTEM_PROMPT,
    EvaluatorProfile,
    EvaluatorSpec,
    _normalize_raw_label,
    load_evaluator_profile,
    probe_evaluator,
)

REPO_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_PROFILE_PATH = (
    REPO_ROOT
    / "benchmarks/experiments/artifacts/evaluator_profiles/claim2_primary_qwen35_mdeberta_current.template.json"
)

DEFAULT_REPORTS_ROOT = REPO_ROOT / "reports" / "diagnostics"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--profile-path",
        default=str(DEFAULT_PROFILE_PATH),
        help="Path to the evaluator profile JSON.",
    )

    parser.add_argument(
        "--attempts",
        type=int,
        default=3,
        help="Number of repeated contract probes per evaluator.",
    )

    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=15.0,
        help="Timeout for DNS/TCP/HTTP/probe calls.",
    )

    parser.add_argument(
        "--output-dir",
        default="",
        help="Optional output directory. Defaults to reports/diagnostics/<timestamp>.",
    )

    return parser.parse_args()


def _now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_json_dumps(payload), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _models_url(base_url: str) -> str:
    return base_url.rstrip("/") + "/models"


def _probe_dns(spec: EvaluatorSpec) -> dict[str, Any]:
    parsed = urllib.parse.urlparse(spec.base_url)
    hostname = parsed.hostname or ""
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    started_at = time.perf_counter()

    if not hostname:
        return {
            "ok": False,
            "hostname": hostname,
            "port": port,
            "latency_ms": 0.0,
            "addresses": [],
            "error_type": "ValueError",
            "error_detail": "Missing hostname in base_url.",
        }

    try:
        infos = socket.getaddrinfo(hostname, port, proto=socket.IPPROTO_TCP)
        addresses = sorted({row[4][0] for row in infos})

        return {
            "ok": True,
            "hostname": hostname,
            "port": port,
            "latency_ms": round((time.perf_counter() - started_at) * 1000.0, 3),
            "addresses": addresses,
            "error_type": "",
            "error_detail": "",
        }

    except Exception as exc:
        return {
            "ok": False,
            "hostname": hostname,
            "port": port,
            "latency_ms": round((time.perf_counter() - started_at) * 1000.0, 3),
            "addresses": [],
            "error_type": type(exc).__name__,
            "error_detail": str(exc),
        }


def _probe_tcp(spec: EvaluatorSpec, timeout_seconds: float) -> dict[str, Any]:
    parsed = urllib.parse.urlparse(spec.base_url)
    hostname = parsed.hostname or ""
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    started_at = time.perf_counter()

    if not hostname:
        return {
            "ok": False,
            "hostname": hostname,
            "port": port,
            "latency_ms": 0.0,
            "error_type": "ValueError",
            "error_detail": "Missing hostname in base_url.",
        }

    try:
        with socket.create_connection((hostname, port), timeout=timeout_seconds):
            pass

        return {
            "ok": True,
            "hostname": hostname,
            "port": port,
            "latency_ms": round((time.perf_counter() - started_at) * 1000.0, 3),
            "error_type": "",
            "error_detail": "",
        }

    except Exception as exc:
        return {
            "ok": False,
            "hostname": hostname,
            "port": port,
            "latency_ms": round((time.perf_counter() - started_at) * 1000.0, 3),
            "error_type": type(exc).__name__,
            "error_detail": str(exc),
        }


def _probe_http_models(
    spec: EvaluatorSpec,
    *,
    timeout_seconds: float,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    api_key = os.getenv(spec.api_key_env, "")
    url = _models_url(spec.base_url)

    if not api_key:
        return {
            "ok": False,
            "url": url,
            "status_code": 0,
            "latency_ms": 0.0,
            "response_excerpt": "",
            "error_type": "ValueError",
            "error_detail": f"Missing API key environment variable: {spec.api_key_env}",
        }

    request = urllib.request.Request(
        url=url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            "User-Agent": "claim2-evaluator-check/1.0",
        },
        method="GET",
    )

    try:
        context = ssl.create_default_context()

        with urllib.request.urlopen(
            request,
            timeout=timeout_seconds,
            context=context,
        ) as response:
            body = response.read(512).decode("utf-8", errors="replace")

            return {
                "ok": True,
                "url": url,
                "status_code": int(getattr(response, "status", 200)),
                "latency_ms": round((time.perf_counter() - started_at) * 1000.0, 3),
                "response_excerpt": body[:200],
                "error_type": "",
                "error_detail": "",
            }

    except urllib.error.HTTPError as exc:
        body = exc.read(512).decode("utf-8", errors="replace")

        return {
            "ok": False,
            "url": url,
            "status_code": int(exc.code),
            "latency_ms": round((time.perf_counter() - started_at) * 1000.0, 3),
            "response_excerpt": body[:200],
            "error_type": type(exc).__name__,
            "error_detail": str(exc),
        }

    except Exception as exc:
        return {
            "ok": False,
            "url": url,
            "status_code": 0,
            "latency_ms": round((time.perf_counter() - started_at) * 1000.0, 3),
            "response_excerpt": "",
            "error_type": type(exc).__name__,
            "error_detail": str(exc),
        }


def _probe_openai_contract_once(
    spec: EvaluatorSpec,
    *,
    timeout_seconds: float,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    api_key = os.getenv(spec.api_key_env, "")

    if not api_key:
        return {
            "ok": False,
            "latency_ms": 0.0,
            "raw_response": "",
            "normalized_label": "",
            "error_type": "ValueError",
            "error_detail": f"Missing API key environment variable: {spec.api_key_env}",
        }

    try:
        client = OpenAI(
            base_url=spec.base_url,
            api_key=api_key,
            timeout=timeout_seconds,
        )

        response = client.chat.completions.create(
            model=spec.model_name,
            messages=[
                {"role": "system", "content": CLAIM2_NLI_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "Assertion:\n测试断言\n\n"
                        "Evidence Section:\n测试\n\n"
                        "Evidence Window:\n测试证据\n\n"
                        "只输出 E"
                    ),
                },
            ],
            max_tokens=CLAIM2_NLI_MAX_OUTPUT_TOKENS,
            temperature=0.0,
        )

        raw_response = str(response.choices[0].message.content or "").strip()
        normalized_label = _normalize_raw_label(raw_response)

        return {
            "ok": normalized_label == "E",
            "latency_ms": round((time.perf_counter() - started_at) * 1000.0, 3),
            "raw_response": raw_response,
            "normalized_label": normalized_label,
            "error_type": "",
            "error_detail": "",
        }

    except Exception as exc:
        return {
            "ok": False,
            "latency_ms": round((time.perf_counter() - started_at) * 1000.0, 3),
            "raw_response": "",
            "normalized_label": "",
            "error_type": type(exc).__name__,
            "error_detail": str(exc),
        }


def _probe_contract_repeated(
    spec: EvaluatorSpec,
    *,
    attempts: int,
    timeout_seconds: float,
) -> list[dict[str, Any]]:
    if spec.backend == "local_nli":
        rows: list[dict[str, Any]] = []

        for attempt_idx in range(1, max(1, attempts) + 1):
            row = probe_evaluator(spec)
            row["attempt"] = attempt_idx
            rows.append(row)

        return rows

    rows: list[dict[str, Any]] = []

    for attempt_idx in range(1, max(1, attempts) + 1):
        row = _probe_openai_contract_once(spec, timeout_seconds=timeout_seconds)
        row["attempt"] = attempt_idx
        rows.append(row)

    return rows


def _build_evaluator_report(
    spec: EvaluatorSpec,
    *,
    attempts: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    base: dict[str, Any] = {
        "name": spec.name,
        "cohort": spec.cohort,
        "backend": spec.backend,
        "model_name": spec.model_name,
        "base_url": spec.base_url,
        "api_key_env": spec.api_key_env,
        "api_key_set": bool(os.getenv(spec.api_key_env, ""))
        if spec.api_key_env
        else False,
        "endpoint_version": spec.endpoint_version,
        "max_input_tokens": spec.max_input_tokens,
    }

    if spec.backend == "local_nli":
        repeated = _probe_contract_repeated(
            spec,
            attempts=attempts,
            timeout_seconds=timeout_seconds,
        )

        base.update(
            {
                "dns": None,
                "tcp": None,
                "http_models": None,
                "contract_attempts": repeated,
                "contract_success_count": sum(1 for row in repeated if row.get("ok")),
                "overall_ok": all(bool(row.get("ok")) for row in repeated),
            }
        )

        return base

    dns = _probe_dns(spec)
    tcp = _probe_tcp(spec, timeout_seconds=timeout_seconds)
    http_models = _probe_http_models(spec, timeout_seconds=timeout_seconds)

    repeated = _probe_contract_repeated(
        spec,
        attempts=attempts,
        timeout_seconds=timeout_seconds,
    )

    base.update(
        {
            "dns": dns,
            "tcp": tcp,
            "http_models": http_models,
            "contract_attempts": repeated,
            "contract_success_count": sum(1 for row in repeated if row.get("ok")),
            "overall_ok": (
                bool(dns.get("ok"))
                and bool(tcp.get("ok"))
                and bool(http_models.get("ok"))
                and any(bool(row.get("ok")) for row in repeated)
            ),
        }
    )

    return base


def _render_markdown(
    *,
    profile: EvaluatorProfile,
    profile_path: Path,
    attempts: int,
    timeout_seconds: float,
    rows: list[dict[str, Any]],
) -> str:
    lines: list[str] = []
    lines.append(f"# Claim 2 Evaluator Availability Check")
    lines.append("")
    lines.append(f"- profile: `{profile.profile_name}`")
    lines.append(f"- profile_path: `{profile_path}`")
    lines.append(f"- attempts_per_evaluator: `{attempts}`")
    lines.append(f"- timeout_seconds: `{timeout_seconds}`")
    lines.append("")
    lines.append("## Summary")
    lines.append("")

    lines.append(
        "| evaluator | backend | overall_ok | dns | tcp | http_models | contract_success |"
    )

    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")

    for row in rows:
        dns_ok = "-" if row["dns"] is None else str(bool(row["dns"]["ok"]))
        tcp_ok = "-" if row["tcp"] is None else str(bool(row["tcp"]["ok"]))

        http_ok = (
            "-" if row["http_models"] is None else str(bool(row["http_models"]["ok"]))
        )

        lines.append(
            f"| {row['name']} | {row['backend']} | {row['overall_ok']} | "
            f"{dns_ok} | {tcp_ok} | {http_ok} | "
            f"{row['contract_success_count']}/{len(row['contract_attempts'])} |"
        )

    lines.append("")
    lines.append("## Details")
    lines.append("")

    for row in rows:
        lines.append(f"### {row['name']}")
        lines.append("")
        lines.append(f"- backend: `{row['backend']}`")
        lines.append(f"- model_name: `{row['model_name']}`")

        if row["base_url"]:
            lines.append(f"- base_url: `{row['base_url']}`")

        if row["api_key_env"]:
            lines.append(f"- api_key_env: `{row['api_key_env']}`")
            lines.append(f"- api_key_set: `{row['api_key_set']}`")

        if row["dns"] is not None:
            lines.append(f"- dns_ok: `{row['dns']['ok']}`")

            if row["dns"].get("addresses"):
                lines.append(f"- dns_addresses: `{', '.join(row['dns']['addresses'])}`")

            if row["dns"].get("error_detail"):
                lines.append(f"- dns_error: `{row['dns']['error_detail']}`")

        if row["tcp"] is not None:
            lines.append(f"- tcp_ok: `{row['tcp']['ok']}`")

            if row["tcp"].get("error_detail"):
                lines.append(f"- tcp_error: `{row['tcp']['error_detail']}`")

        if row["http_models"] is not None:
            lines.append(f"- http_models_ok: `{row['http_models']['ok']}`")
            lines.append(f"- http_models_status: `{row['http_models']['status_code']}`")

            if row["http_models"].get("response_excerpt"):
                lines.append(
                    f"- http_models_excerpt: `{row['http_models']['response_excerpt']}`"
                )

            if row["http_models"].get("error_detail"):
                lines.append(
                    f"- http_models_error: `{row['http_models']['error_detail']}`"
                )

        lines.append(f"- contract_success_count: `{row['contract_success_count']}`")

        for attempt in row["contract_attempts"]:
            attempt_idx = attempt.get("attempt", "?")

            lines.append(
                f"- contract_attempt_{attempt_idx}: "
                f"`ok={attempt.get('ok')}, latency_ms={attempt.get('latency_ms')}, "
                f"label={attempt.get('normalized_label', '')}, "
                f"raw={attempt.get('raw_response', '')}, "
                f"error={attempt.get('error_detail', '')}`"
            )

        lines.append("")

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    profile_path = Path(args.profile_path).resolve()
    profile = load_evaluator_profile(profile_path)

    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else DEFAULT_REPORTS_ROOT / f"claim2_evaluator_availability_{_now_stamp()}"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        _build_evaluator_report(
            spec,
            attempts=args.attempts,
            timeout_seconds=args.timeout_seconds,
        )
        for spec in (*profile.primary_evaluators, *profile.supplementary_evaluators)
    ]

    payload = {
        "profile_name": profile.profile_name,
        "profile_path": str(profile_path),
        "attempts": int(args.attempts),
        "timeout_seconds": float(args.timeout_seconds),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
        "evaluators": rows,
    }

    markdown = _render_markdown(
        profile=profile,
        profile_path=profile_path,
        attempts=args.attempts,
        timeout_seconds=args.timeout_seconds,
        rows=rows,
    )

    _write_json(output_dir / "availability.json", payload)
    _write_text(output_dir / "availability.md", markdown)
    print(_json_dumps(payload))
    print(f"\nWrote: {output_dir / 'availability.json'}")
    print(f"Wrote: {output_dir / 'availability.md'}")


if __name__ == "__main__":
    main()
