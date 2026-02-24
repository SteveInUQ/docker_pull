#!/usr/bin/env python3
"""LLM streaming benchmark tool based on OpenAI-compatible APIs."""

import argparse
import copy
import dataclasses
import datetime as dt
import json
import math
import os
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import openai
from openai import OpenAI
import yaml


@dataclasses.dataclass
class ModelConfig:
    id: str
    name: str
    base_url: str
    model: str
    api_key: Optional[str] = None
    api_key_env: Optional[str] = None
    request_overrides: Optional[Dict[str, Any]] = None


@dataclasses.dataclass
class RunConfig:
    warmup_runs: int
    runs_per_case: int
    timeout_s: float
    retry: int
    concurrency: List[int]


@dataclasses.dataclass
class BenchConfig:
    version: int
    run: RunConfig
    request_defaults: Dict[str, Any]
    test_cases: List[Dict[str, Any]]
    models: List[ModelConfig]


class ConfigError(Exception):
    pass


def _quantile(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    arr = sorted(values)
    pos = (len(arr) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return arr[int(pos)]
    return arr[lo] + (arr[hi] - arr[lo]) * (pos - lo)


def _safe_mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


def classify_error(exc: Exception) -> str:
    if isinstance(exc, (openai.APITimeoutError, TimeoutError)):
        return "timeout"
    if isinstance(exc, openai.RateLimitError):
        return "http_429"
    if isinstance(exc, openai.APIStatusError):
        if exc.status_code == 429:
            return "http_429"
        if exc.status_code >= 500:
            return "http_5xx"
        return f"http_{exc.status_code}"
    return "other"


def validate_and_build_config(raw: Dict[str, Any]) -> BenchConfig:
    if not isinstance(raw, dict):
        raise ConfigError("YAML root must be a mapping")

    version = raw.get("version", 1)
    global_cfg = raw.get("global")
    if not isinstance(global_cfg, dict):
        raise ConfigError("Missing required field: global")

    run_raw = global_cfg.get("run")
    if not isinstance(run_raw, dict):
        raise ConfigError("Missing required field: global.run")

    required_run_fields = ["warmup_runs", "runs_per_case", "timeout_s", "retry", "concurrency"]
    for field in required_run_fields:
        if field not in run_raw:
            raise ConfigError(f"Missing required field: global.run.{field}")

    concurrency = run_raw["concurrency"]
    if not isinstance(concurrency, list) or not concurrency:
        raise ConfigError("global.run.concurrency must be a non-empty list")
    if any((not isinstance(c, int) or c <= 0) for c in concurrency):
        raise ConfigError("global.run.concurrency must contain positive integers")

    request_defaults = global_cfg.get("request_defaults", {})
    if not isinstance(request_defaults, dict):
        raise ConfigError("global.request_defaults must be a mapping")

    test_cases = raw.get("test_cases")
    if not isinstance(test_cases, list) or not test_cases:
        raise ConfigError("test_cases must be a non-empty list")
    for idx, case in enumerate(test_cases):
        if not isinstance(case, dict):
            raise ConfigError(f"test_cases[{idx}] must be a mapping")
        for required in ("id", "messages"):
            if required not in case:
                raise ConfigError(f"Missing required field: test_cases[{idx}].{required}")
        if not isinstance(case["messages"], list) or not case["messages"]:
            raise ConfigError(f"test_cases[{idx}].messages must be a non-empty list")

    models_raw = raw.get("models")
    if not isinstance(models_raw, list) or not models_raw:
        raise ConfigError("models must be a non-empty list")

    models: List[ModelConfig] = []
    seen_ids = set()
    for idx, model_raw in enumerate(models_raw):
        if not isinstance(model_raw, dict):
            raise ConfigError(f"models[{idx}] must be a mapping")
        for required in ("id", "name", "base_url", "model"):
            if required not in model_raw:
                raise ConfigError(f"Missing required field: models[{idx}].{required}")

        model_id = model_raw["id"]
        if model_id in seen_ids:
            raise ConfigError(f"Duplicate model id: {model_id}")
        seen_ids.add(model_id)

        api_key = model_raw.get("api_key")
        api_key_env = model_raw.get("api_key_env")
        if not api_key and not api_key_env:
            raise ConfigError(f"models[{idx}] needs api_key or api_key_env")

        models.append(
            ModelConfig(
                id=model_id,
                name=model_raw["name"],
                base_url=model_raw["base_url"],
                model=model_raw["model"],
                api_key=api_key,
                api_key_env=api_key_env,
                request_overrides=model_raw.get("request_overrides", {}),
            )
        )

    run = RunConfig(
        warmup_runs=int(run_raw["warmup_runs"]),
        runs_per_case=int(run_raw["runs_per_case"]),
        timeout_s=float(run_raw["timeout_s"]),
        retry=int(run_raw["retry"]),
        concurrency=concurrency,
    )

    return BenchConfig(
        version=version,
        run=run,
        request_defaults=request_defaults,
        test_cases=test_cases,
        models=models,
    )


def resolve_api_key(model_cfg: ModelConfig) -> str:
    if model_cfg.api_key:
        return model_cfg.api_key
    assert model_cfg.api_key_env is not None
    api_key = os.getenv(model_cfg.api_key_env)
    if not api_key:
        raise ConfigError(f"Environment variable not set: {model_cfg.api_key_env}")
    return api_key


def run_one_stream(
    client: OpenAI,
    model_cfg: ModelConfig,
    case: Dict[str, Any],
    request_defaults: Dict[str, Any],
    timeout_s: float,
    retry: int,
) -> Dict[str, Any]:
    payload = dict(request_defaults)
    payload.update(model_cfg.request_overrides or {})
    payload["stream"] = True

    last_exc: Optional[Exception] = None

    for attempt in range(retry + 1):
        t0 = time.perf_counter()
        t_first = None
        t_end = None
        output_tokens = 0

        try:
            stream = client.with_options(timeout=timeout_s).chat.completions.create(
                model=model_cfg.model,
                messages=case["messages"],
                **payload,
            )
            for chunk in stream:
                choices = getattr(chunk, "choices", None) or []
                for choice in choices:
                    delta = getattr(choice, "delta", None)
                    content = getattr(delta, "content", None) if delta else None
                    if content:
                        if t_first is None:
                            t_first = time.perf_counter()
                        if isinstance(content, str):
                            output_tokens += 1
                        elif isinstance(content, list):
                            output_tokens += len(content)
                        else:
                            output_tokens += 1
            t_end = time.perf_counter()

            if t_first is None:
                raise RuntimeError("stream finished without content token")

            ttft = t_first - t0
            output_duration = max(0.0, t_end - t_first)
            tpot = output_duration / max(output_tokens, 1)
            return {
                "success": True,
                "ttft_s": ttft,
                "tpot_s": tpot,
                "output_tokens": output_tokens,
                "latency_s": t_end - t0,
                "error_type": None,
                "error_message": None,
                "attempt": attempt + 1,
            }
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < retry:
                time.sleep(0.3 * (attempt + 1))
                continue

    assert last_exc is not None
    return {
        "success": False,
        "ttft_s": None,
        "tpot_s": None,
        "output_tokens": 0,
        "latency_s": None,
        "error_type": classify_error(last_exc),
        "error_message": str(last_exc),
        "attempt": retry + 1,
    }


def execute_benchmark(config: BenchConfig, raw_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    all_runs: List[Dict[str, Any]] = []
    aggregate_rows: List[Dict[str, Any]] = []

    for model_cfg in config.models:
        api_key = resolve_api_key(model_cfg)
        client = OpenAI(base_url=model_cfg.base_url, api_key=api_key)

        for conc in config.run.concurrency:
            def build_jobs() -> List[Tuple[str, Dict[str, Any], int]]:
                jobs: List[Tuple[str, Dict[str, Any], int]] = []
                for case in config.test_cases:
                    for run_index in range(config.run.runs_per_case):
                        jobs.append((case["id"], case, run_index + 1))
                return jobs

            warmup_jobs = []
            for _ in range(config.run.warmup_runs):
                for case in config.test_cases:
                    warmup_jobs.append(case)

            if warmup_jobs:
                with ThreadPoolExecutor(max_workers=conc) as executor:
                    futures = [
                        executor.submit(
                            run_one_stream,
                            client,
                            model_cfg,
                            case,
                            config.request_defaults,
                            config.run.timeout_s,
                            config.run.retry,
                        )
                        for case in warmup_jobs
                    ]
                    for future in as_completed(futures):
                        future.result()

            jobs = build_jobs()
            run_counter = 0
            counter_lock = threading.Lock()

            def do_job(case_id: str, case: Dict[str, Any], run_index: int) -> Dict[str, Any]:
                nonlocal run_counter
                with counter_lock:
                    run_counter += 1
                    run_id = run_counter
                result = run_one_stream(
                    client,
                    model_cfg,
                    case,
                    config.request_defaults,
                    config.run.timeout_s,
                    config.run.retry,
                )
                result.update(
                    {
                        "model_id": model_cfg.id,
                        "model_name": model_cfg.name,
                        "concurrency": conc,
                        "case_id": case_id,
                        "run_index": run_index,
                        "run_id": run_id,
                        "warmup": False,
                    }
                )
                return result

            with ThreadPoolExecutor(max_workers=conc) as executor:
                futures = [executor.submit(do_job, case_id, case, run_idx) for case_id, case, run_idx in jobs]
                for future in as_completed(futures):
                    all_runs.append(future.result())

            scoped = [r for r in all_runs if r["model_id"] == model_cfg.id and r["concurrency"] == conc and not r["warmup"]]
            success_rows = [r for r in scoped if r["success"]]
            ttft_values = [r["ttft_s"] for r in success_rows if r["ttft_s"] is not None]
            tpot_values = [r["tpot_s"] for r in success_rows if r["tpot_s"] is not None]
            failures = [r for r in scoped if not r["success"]]
            error_counts = dict(Counter(r["error_type"] for r in failures))

            total = len(scoped)
            succ = len(success_rows)
            fail = len(failures)

            aggregate_rows.append(
                {
                    "model_id": model_cfg.id,
                    "model_name": model_cfg.name,
                    "concurrency": conc,
                    "total_runs": total,
                    "success_runs": succ,
                    "failed_runs": fail,
                    "success_rate": (succ / total) if total else 0.0,
                    "error_rate": (fail / total) if total else 0.0,
                    "error_counts": error_counts,
                    "ttft_mean_s": _safe_mean(ttft_values),
                    "ttft_p50_s": _quantile(ttft_values, 0.5),
                    "ttft_p95_s": _quantile(ttft_values, 0.95),
                    "tpot_mean_s": _safe_mean(tpot_values),
                    "tpot_p50_s": _quantile(tpot_values, 0.5),
                    "tpot_p95_s": _quantile(tpot_values, 0.95),
                }
            )

    return {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "config_snapshot": raw_snapshot,
        "run_details": sorted(all_runs, key=lambda x: (x["model_id"], x["concurrency"], x["case_id"], x["run_index"], x["run_id"])),
        "aggregates": sorted(aggregate_rows, key=lambda x: (x["concurrency"], x["model_id"])),
    }


def mask_snapshot(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    masked = copy.deepcopy(snapshot)
    for model in masked.get("models", []):
        if "api_key" in model and model["api_key"]:
            model["api_key"] = "***"
    return masked


def _fmt(num: Optional[float]) -> str:
    if num is None:
        return "-"
    return f"{num:.4f}"


def build_markdown_report(result: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# LLM 流式输出性能评测报告")
    lines.append("")
    lines.append(f"- 生成时间（UTC）: {result['generated_at']}")
    lines.append("")

    by_conc: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in result["aggregates"]:
        by_conc[row["concurrency"]].append(row)

    for conc in sorted(by_conc.keys()):
        rows = by_conc[conc]
        lines.append(f"## 并发档位: {conc}")
        lines.append("")
        lines.append("| Model | Runs | Success Rate | TTFT Mean(s) | TTFT P50 | TTFT P95 | TPOT Mean(s) | TPOT P50 | TPOT P95 | Errors |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        for r in sorted(rows, key=lambda x: (x["ttft_mean_s"] is None, x["ttft_mean_s"] if x["ttft_mean_s"] is not None else float("inf"))):
            err = ", ".join(f"{k}:{v}" for k, v in sorted(r["error_counts"].items())) or "-"
            lines.append(
                f"| {r['model_name']} ({r['model_id']}) | {r['total_runs']} | {r['success_rate']:.2%} | {_fmt(r['ttft_mean_s'])} | {_fmt(r['ttft_p50_s'])} | {_fmt(r['ttft_p95_s'])} | {_fmt(r['tpot_mean_s'])} | {_fmt(r['tpot_p50_s'])} | {_fmt(r['tpot_p95_s'])} | {err} |"
            )
        lines.append("")

        valid_ttft = [r for r in rows if r["ttft_mean_s"] is not None]
        valid_tpot = [r for r in rows if r["tpot_mean_s"] is not None]
        if valid_ttft:
            best_ttft = min(valid_ttft, key=lambda x: x["ttft_mean_s"])
            lines.append(f"- TTFT 最优: **{best_ttft['model_name']} ({best_ttft['model_id']})** = {_fmt(best_ttft['ttft_mean_s'])}s")
        if valid_tpot:
            best_tpot = min(valid_tpot, key=lambda x: x["tpot_mean_s"])
            lines.append(f"- TPOT 最优: **{best_tpot['model_name']} ({best_tpot['model_id']})** = {_fmt(best_tpot['tpot_mean_s'])}s")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark streaming performance of LLMs via OpenAI-compatible APIs")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--output-dir", default="benchmark_reports", help="Directory for output reports")
    parser.add_argument("--output-prefix", default="llm_stream_benchmark", help="Output filename prefix")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.exists():
        raise SystemExit(f"Config file does not exist: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    try:
        config = validate_and_build_config(raw)
    except ConfigError as exc:
        raise SystemExit(f"Invalid config: {exc}") from exc

    snapshot = mask_snapshot(raw)
    result = execute_benchmark(config, snapshot)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = output_dir / f"{args.output_prefix}_{timestamp}.json"
    md_path = output_dir / f"{args.output_prefix}_{timestamp}.md"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    with md_path.open("w", encoding="utf-8") as f:
        f.write(build_markdown_report(result))

    print(f"JSON report: {json_path}")
    print(f"Markdown report: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
