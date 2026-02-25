#!/usr/bin/env python3
"""Run streaming benchmark for all models declared in configs/llm_models.yaml.

Project layout:
- coder-eva-service/tools/configs/benchmark.yaml
- coder-eva-service/tools/data/llm_bench_cases.json
- coder-eva-service/configs/llm_models.yaml
- coder-eva-service/configs/.secret
- coder-eva-service/results/
"""

import argparse
import asyncio
import datetime as dt
import json
import math
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import openai
from openai import AsyncOpenAI
import yaml


class ConfigError(Exception):
    """Raised when config/model/case files are invalid."""


def load_key_value_file(path: Path) -> Dict[str, str]:
    """Parse simple KEY=VALUE files like .env/.secret."""
    out: Dict[str, str] = {}
    if not path.exists():
        return out

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key.strip()] = value.strip().strip('"').strip("'")
    return out


def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"YAML file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ConfigError(f"YAML root must be mapping: {path}")
    return data


def load_cases(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise ConfigError(f"Case file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(data, dict):
        cases = data.get("test_cases") or data.get("cases")
    else:
        cases = data

    if not isinstance(cases, list) or not cases:
        raise ConfigError("cases must be a non-empty list (or {test_cases:[...]})")

    for idx, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ConfigError(f"case[{idx}] must be object")
        if "id" not in case or "messages" not in case:
            raise ConfigError(f"case[{idx}] requires id and messages")
        if not isinstance(case["messages"], list) or not case["messages"]:
            raise ConfigError(f"case[{idx}].messages must be non-empty list")
    return cases


def parse_model_key(model_key: str) -> Tuple[str, str]:
    """Parse 'model_name@platform' into tuple. Platform defaults to 'default'."""
    if "@" not in model_key:
        return model_key, "default"
    model_name, platform = model_key.rsplit("@", 1)
    return model_name.strip(), platform.strip()


def resolve_api_key(model_cfg: Dict[str, Any], secrets: Dict[str, str]) -> str:
    """Resolve api key from explicit value or a field name in .secret/.env.

    Priority:
    1) api_key_field: a key name in .secret/.env (recommended)
    2) api_key_env: env var name
    3) api_key: literal value OR env var name if matches existing env/secrets
    """
    field = model_cfg.get("api_key_field")
    if field:
        if field in secrets and secrets[field]:
            return secrets[field]
        if os.getenv(field):
            return os.getenv(field, "")
        raise ConfigError(f"api_key_field not found in .secret/.env/env: {field}")

    env_name = model_cfg.get("api_key_env")
    if env_name:
        value = os.getenv(env_name) or secrets.get(env_name)
        if value:
            return value
        raise ConfigError(f"api_key_env not found: {env_name}")

    key = model_cfg.get("api_key")
    if not key:
        raise ConfigError("model requires api_key_field/api_key_env/api_key")

    # If api_key looks like a field name, resolve from secret/env first.
    maybe = os.getenv(str(key)) or secrets.get(str(key))
    if maybe:
        return maybe
    return str(key)


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


def quantile(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    arr = sorted(values)
    pos = (len(arr) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return arr[int(pos)]
    return arr[lo] + (arr[hi] - arr[lo]) * (pos - lo)


def safe_mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


async def run_one_stream(
    client: AsyncOpenAI,
    model_name: str,
    messages: List[Dict[str, Any]],
    request_defaults: Dict[str, Any],
    request_overrides: Dict[str, Any],
    timeout_s: float,
    retry: int,
) -> Dict[str, Any]:
    payload = dict(request_defaults)
    payload.update(request_overrides or {})
    payload["stream"] = True
    so = dict(payload.get("stream_options", {}) or {})
    so["include_usage"] = True
    payload["stream_options"] = so

    last_exc: Optional[Exception] = None

    for attempt in range(retry + 1):
        t0 = time.perf_counter()
        t_first = None
        output_chunks = 0
        completion_tokens: Optional[int] = None

        try:
            stream = await client.with_options(timeout=timeout_s).chat.completions.create(
                model=model_name,
                messages=messages,
                **payload,
            )

            async for chunk in stream:
                usage = getattr(chunk, "usage", None)
                if usage is not None:
                    completion_tokens = getattr(usage, "completion_tokens", None)

                for choice in getattr(chunk, "choices", None) or []:
                    delta = getattr(choice, "delta", None)
                    content = getattr(delta, "content", None) if delta else None
                    if content:
                        if t_first is None:
                            t_first = time.perf_counter()
                        if isinstance(content, list):
                            output_chunks += len(content)
                        else:
                            output_chunks += 1

            t_end = time.perf_counter()
            if t_first is None:
                raise RuntimeError("stream finished without content")

            output_duration = max(0.0, t_end - t_first)
            tpot_s = None
            if completion_tokens and completion_tokens > 0:
                tpot_s = output_duration / completion_tokens

            return {
                "success": True,
                "ttft_s": t_first - t0,
                "tpot_s": tpot_s,
                "output_tokens": completion_tokens,
                "output_chunks": output_chunks,
                "token_source": "usage.completion_tokens" if completion_tokens is not None else "unknown",
                "latency_s": t_end - t0,
                "error_type": None,
                "error_message": None,
                "attempt": attempt + 1,
            }
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < retry:
                await asyncio.sleep(0.3 * (attempt + 1))

    assert last_exc is not None
    return {
        "success": False,
        "ttft_s": None,
        "tpot_s": None,
        "output_tokens": None,
        "output_chunks": 0,
        "token_source": None,
        "latency_s": None,
        "error_type": classify_error(last_exc),
        "error_message": str(last_exc),
        "attempt": retry + 1,
    }


def parse_models(models_yaml: Dict[str, Any]) -> List[Dict[str, Any]]:
    models = models_yaml.get("models")
    if not isinstance(models, dict) or not models:
        raise ConfigError("llm_models.yaml requires non-empty models mapping")

    parsed: List[Dict[str, Any]] = []
    for key, cfg in models.items():
        if not isinstance(cfg, dict):
            raise ConfigError(f"model '{key}' config must be mapping")
        if "api_url" not in cfg:
            raise ConfigError(f"model '{key}' missing api_url")

        model_name, platform = parse_model_key(str(key))
        parsed.append(
            {
                "id": str(key),
                "name": model_name,
                "platform": platform,
                "api_url": cfg["api_url"],
                "api_key": cfg.get("api_key"),
                "api_key_env": cfg.get("api_key_env"),
                "api_key_field": cfg.get("api_key_field"),
                "request_overrides": cfg.get("request_overrides", {}),
            }
        )
    return parsed


async def execute(
    models: List[Dict[str, Any]],
    cases: List[Dict[str, Any]],
    benchmark_cfg: Dict[str, Any],
    secrets: Dict[str, str],
) -> Dict[str, Any]:
    run_cfg = ((benchmark_cfg.get("global") or {}).get("run") or {})
    request_defaults = ((benchmark_cfg.get("global") or {}).get("request_defaults") or {})

    runs_per_case = int(run_cfg.get("runs_per_case", 1))
    timeout_s = float(run_cfg.get("timeout_s", 60))
    retry = int(run_cfg.get("retry", 0))
    conc_levels = run_cfg.get("concurrency", [1])

    if not isinstance(conc_levels, list) or not conc_levels:
        raise ConfigError("global.run.concurrency must be non-empty list")

    all_runs: List[Dict[str, Any]] = []
    aggregates: List[Dict[str, Any]] = []

    for model in models:
        api_key = resolve_api_key(model, secrets)
        client = AsyncOpenAI(base_url=model["api_url"], api_key=api_key)

        for conc in conc_levels:
            jobs: List[Tuple[str, List[Dict[str, Any]], int]] = []
            for case in cases:
                for i in range(runs_per_case):
                    jobs.append((case["id"], case["messages"], i + 1))

            semaphore = asyncio.Semaphore(int(conc))

            async def one_job(job_id: int, case_id: str, messages: List[Dict[str, Any]], run_index: int) -> Dict[str, Any]:
                async with semaphore:
                    result = await run_one_stream(
                        client=client,
                        model_name=model["name"],
                        messages=messages,
                        request_defaults=request_defaults,
                        request_overrides=model.get("request_overrides") or {},
                        timeout_s=timeout_s,
                        retry=retry,
                    )
                result.update(
                    {
                        "model_id": model["id"],
                        "model_name": model["name"],
                        "platform": model["platform"],
                        "concurrency": int(conc),
                        "case_id": case_id,
                        "run_index": run_index,
                        "run_id": job_id,
                    }
                )
                return result

            scoped = await asyncio.gather(
                *(one_job(i + 1, case_id, msgs, run_idx) for i, (case_id, msgs, run_idx) in enumerate(jobs))
            )
            all_runs.extend(scoped)

            success = [r for r in scoped if r["success"]]
            failures = [r for r in scoped if not r["success"]]
            ttft_values = [r["ttft_s"] for r in success if r["ttft_s"] is not None]
            tpot_values = [r["tpot_s"] for r in success if r["tpot_s"] is not None]

            total = len(scoped)
            succ = len(success)
            fail = len(failures)

            aggregates.append(
                {
                    "model_id": model["id"],
                    "model_name": model["name"],
                    "platform": model["platform"],
                    "concurrency": int(conc),
                    "total_runs": total,
                    "success_runs": succ,
                    "failed_runs": fail,
                    "success_rate": succ / total if total else 0.0,
                    "error_rate": fail / total if total else 0.0,
                    "error_counts": dict(Counter(r["error_type"] for r in failures)),
                    "ttft_mean_s": safe_mean(ttft_values),
                    "ttft_p50_s": quantile(ttft_values, 0.5),
                    "ttft_p95_s": quantile(ttft_values, 0.95),
                    "tpot_mean_s": safe_mean(tpot_values),
                    "tpot_p50_s": quantile(tpot_values, 0.5),
                    "tpot_p95_s": quantile(tpot_values, 0.95),
                }
            )

    return {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "config_snapshot": {
            "benchmark": benchmark_cfg,
            "models": [
                {
                    "id": m["id"],
                    "name": m["name"],
                    "platform": m["platform"],
                    "api_url": m["api_url"],
                    "api_key": "***",
                    "request_overrides": m.get("request_overrides") or {},
                }
                for m in models
            ],
        },
        "run_details": sorted(all_runs, key=lambda x: (x["model_id"], x["concurrency"], x["case_id"], x["run_index"], x["run_id"])),
        "aggregates": sorted(aggregates, key=lambda x: (x["concurrency"], x["model_id"])),
    }


def fmt(value: Optional[float]) -> str:
    return "-" if value is None else f"{value:.4f}"


def to_markdown(report: Dict[str, Any]) -> str:
    lines = ["# LLM Streaming Benchmark Report", "", f"- Generated at (UTC): {report['generated_at']}", ""]

    grouped: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in report["aggregates"]:
        grouped[row["concurrency"]].append(row)

    for conc in sorted(grouped):
        rows = grouped[conc]
        lines.append(f"## Concurrency {conc}")
        lines.append("")
        lines.append("| Model | Platform | Runs | Success Rate | TTFT Mean | TTFT P50 | TTFT P95 | TPOT Mean | TPOT P50 | TPOT P95 | Errors |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")

        for r in rows:
            err = ", ".join(f"{k}:{v}" for k, v in sorted(r["error_counts"].items())) or "-"
            lines.append(
                f"| {r['model_name']} | {r['platform']} | {r['total_runs']} | {r['success_rate']:.2%} |"
                f" {fmt(r['ttft_mean_s'])} | {fmt(r['ttft_p50_s'])} | {fmt(r['ttft_p95_s'])} |"
                f" {fmt(r['tpot_mean_s'])} | {fmt(r['tpot_p50_s'])} | {fmt(r['tpot_p95_s'])} | {err} |"
            )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Run LLM streaming benchmark for all models")
    parser.add_argument("--benchmark-config", default=str(root / "tools" / "configs" / "benchmark.yaml"))
    parser.add_argument("--models-config", default=str(root / "configs" / "llm_models.yaml"))
    parser.add_argument("--cases", default=str(root / "tools" / "data" / "llm_bench_cases.json"))
    parser.add_argument("--secret-file", default=str(root / "configs" / ".secret"))
    parser.add_argument("--env-file", default=str(root / "configs" / ".env"))
    parser.add_argument("--results-dir", default=str(root / "results"))
    parser.add_argument("--output-prefix", default="llm_stream_benchmark")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    benchmark_cfg = load_yaml(Path(args.benchmark_config))
    models_cfg = load_yaml(Path(args.models_config))
    cases = load_cases(Path(args.cases))

    secrets = {}
    secrets.update(load_key_value_file(Path(args.env_file)))
    secrets.update(load_key_value_file(Path(args.secret_file)))

    models = parse_models(models_cfg)
    report = asyncio.run(execute(models=models, cases=cases, benchmark_cfg=benchmark_cfg, secrets=secrets))

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = results_dir / f"{args.output_prefix}_{ts}.json"
    md_path = results_dir / f"{args.output_prefix}_{ts}.md"

    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(to_markdown(report), encoding="utf-8")

    print(f"JSON report: {json_path}")
    print(f"Markdown report: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
