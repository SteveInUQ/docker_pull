#!/usr/bin/env python3
"""code eva service 大模型流式性能评测脚本。

说明：
1. 模型配置来自 `coder-eva-service/configs/llm_models.yaml`。
2. `api_key` 字段不是明文 key，而是 `.secret` 里的字段名（例如 DASHSCOPE_API_KEY）。
3. 真实 key 从 `coder-eva-service/configs/.secret` 读取。
4. 结果输出到 `coder-eva-service/results/`。
"""

import argparse
import asyncio
import datetime as dt
import json
import logging
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import openai
from openai import AsyncOpenAI
import yaml


logger = logging.getLogger("llm_bench")


class ConfigError(Exception):
    """配置异常。"""


def setup_logging(level: str) -> None:
    """初始化日志。"""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"YAML 文件不存在: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ConfigError(f"YAML 根节点必须是对象: {path}")
    return data


def load_secret_file(path: Path) -> Dict[str, str]:
    """读取 .secret 文件（KEY=VALUE）。"""
    if not path.exists():
        raise ConfigError(f".secret 文件不存在: {path}")

    secrets: Dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        secrets[k.strip()] = v.strip().strip('"').strip("'")
    return secrets


def load_cases(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise ConfigError(f"测试用例文件不存在: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    cases = data.get("test_cases") if isinstance(data, dict) else data
    if not isinstance(cases, list) or not cases:
        raise ConfigError("测试用例必须是非空数组")

    for idx, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ConfigError(f"case[{idx}] 必须是对象")
        if "id" not in case or "messages" not in case:
            raise ConfigError(f"case[{idx}] 缺少 id/messages")
    return cases


def parse_model_key(model_key: str) -> Tuple[str, str]:
    """解析 `模型名@平台`。"""
    if "@" not in model_key:
        return model_key, "default"
    name, platform = model_key.rsplit("@", 1)
    return name.strip(), platform.strip()


def parse_models(models_yaml: Dict[str, Any], secrets: Dict[str, str]) -> List[Dict[str, Any]]:
    """解析模型配置并把 api_key 字段映射到 .secret 真值。"""
    models = models_yaml.get("models")
    if not isinstance(models, dict) or not models:
        raise ConfigError("llm_models.yaml 中 models 必须是非空对象")

    out: List[Dict[str, Any]] = []
    for raw_key, cfg in models.items():
        if not isinstance(cfg, dict):
            raise ConfigError(f"模型 {raw_key} 配置必须是对象")
        if "api_url" not in cfg or "api_key" not in cfg:
            raise ConfigError(f"模型 {raw_key} 缺少 api_url/api_key")

        # 注意：这里 api_key 是 .secret 的字段名，不是明文 key。
        key_field = str(cfg["api_key"]).strip()
        real_key = secrets.get(key_field)
        if not real_key:
            raise ConfigError(f"模型 {raw_key} 的 api_key 字段 `{key_field}` 未在 .secret 中找到")

        model_name, platform = parse_model_key(str(raw_key))
        out.append(
            {
                "id": str(raw_key),
                "name": model_name,
                "platform": platform,
                "api_url": str(cfg["api_url"]),
                "api_key": real_key,
                "api_key_field": key_field,
                "request_overrides": cfg.get("request_overrides", {}),
            }
        )
    return out


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


def safe_mean(values: List[float]) -> Optional[float]:
    return (sum(values) / len(values)) if values else None


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


async def preflight_model_api(client: AsyncOpenAI, model_name: str, timeout_s: float) -> Tuple[bool, Optional[str], float]:
    """压测前预检：确认 API 可调。"""
    t0 = time.perf_counter()
    try:
        await client.with_options(timeout=timeout_s).chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "ping"}],
            stream=False,
            max_tokens=8,
            temperature=0,
        )
        return True, None, time.perf_counter() - t0
    except Exception as exc:  # noqa: BLE001
        return False, str(exc), time.perf_counter() - t0


async def run_one_stream(
    client: AsyncOpenAI,
    model_name: str,
    messages: List[Dict[str, Any]],
    request_defaults: Dict[str, Any],
    request_overrides: Dict[str, Any],
    timeout_s: float,
    retry: int,
) -> Dict[str, Any]:
    """执行一次流式请求并采集 TTFT/TPOT。"""
    payload = dict(request_defaults)
    payload.update(request_overrides or {})
    payload["stream"] = True
    stream_options = dict(payload.get("stream_options", {}) or {})
    stream_options["include_usage"] = True
    payload["stream_options"] = stream_options

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
                        output_chunks += len(content) if isinstance(content, list) else 1

            t_end = time.perf_counter()
            if t_first is None:
                raise RuntimeError("流式请求未返回内容")

            output_duration = max(0.0, t_end - t_first)
            tpot_s = (output_duration / completion_tokens) if completion_tokens and completion_tokens > 0 else None
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


async def execute(models: List[Dict[str, Any]], cases: List[Dict[str, Any]], benchmark_cfg: Dict[str, Any], enable_preflight: bool = True) -> Dict[str, Any]:
    run_cfg = ((benchmark_cfg.get("global") or {}).get("run") or {})
    request_defaults = ((benchmark_cfg.get("global") or {}).get("request_defaults") or {})

    runs_per_case = int(run_cfg.get("runs_per_case", 1))
    timeout_s = float(run_cfg.get("timeout_s", 60))
    retry = int(run_cfg.get("retry", 0))
    conc_levels = run_cfg.get("concurrency", [1])

    all_runs: List[Dict[str, Any]] = []
    aggregates: List[Dict[str, Any]] = []

    for model in models:
        client = AsyncOpenAI(base_url=model["api_url"], api_key=model["api_key"])
        logger.info("开始评测 model=%s platform=%s", model["name"], model["platform"])

        if enable_preflight:
            ok, err, elapsed = await preflight_model_api(client, model["name"], timeout_s)
            if not ok:
                logger.error("预检失败 model=%s platform=%s elapsed=%.3fs err=%s", model["name"], model["platform"], elapsed, err)
                for conc in conc_levels:
                    aggregates.append(
                        {
                            "model_id": model["id"],
                            "model_name": model["name"],
                            "platform": model["platform"],
                            "concurrency": int(conc),
                            "total_runs": 0,
                            "success_runs": 0,
                            "failed_runs": 0,
                            "success_rate": 0.0,
                            "error_rate": 1.0,
                            "error_counts": {"preflight_failed": 1},
                            "ttft_mean_s": None,
                            "ttft_p50_s": None,
                            "ttft_p95_s": None,
                            "tpot_mean_s": None,
                            "tpot_p50_s": None,
                            "tpot_p95_s": None,
                        }
                    )
                continue
            logger.info("预检通过 model=%s platform=%s elapsed=%.3fs", model["name"], model["platform"], elapsed)

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

            scoped = await asyncio.gather(*(one_job(i + 1, cid, msgs, ridx) for i, (cid, msgs, ridx) in enumerate(jobs)))
            all_runs.extend(scoped)

            success = [r for r in scoped if r["success"]]
            failures = [r for r in scoped if not r["success"]]
            ttft_values = [r["ttft_s"] for r in success if r["ttft_s"] is not None]
            tpot_values = [r["tpot_s"] for r in success if r["tpot_s"] is not None]

            total = len(scoped)
            succ = len(success)
            fail = len(failures)
            logger.info("完成 model=%s conc=%s success=%s/%s", model["name"], conc, succ, total)

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
                    "api_key_field": m["api_key_field"],
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
    parser = argparse.ArgumentParser(description="运行 code eva service 的流式评测")
    parser.add_argument("--benchmark-config", default=str(root / "tools" / "configs" / "benchmark.yaml"))
    parser.add_argument("--models-config", default=str(root / "configs" / "llm_models.yaml"))
    parser.add_argument("--secret-file", default=str(root / "configs" / ".secret"))
    parser.add_argument("--cases", default=str(root / "tools" / "data" / "llm_bench_cases.json"))
    parser.add_argument("--results-dir", default=str(root / "results"))
    parser.add_argument("--output-prefix", default="llm_stream_benchmark")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--skip-preflight", action="store_true", help="跳过模型 API 预检")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)

    logger.info("读取 benchmark 配置: %s", args.benchmark_config)
    benchmark_cfg = load_yaml(Path(args.benchmark_config))
    logger.info("读取 model 配置: %s", args.models_config)
    models_cfg = load_yaml(Path(args.models_config))
    logger.info("读取 .secret: %s", args.secret_file)
    secrets = load_secret_file(Path(args.secret_file))
    logger.info("读取用例: %s", args.cases)
    cases = load_cases(Path(args.cases))

    models = parse_models(models_cfg, secrets)
    logger.info("共加载模型 %s 个，用例 %s 条", len(models), len(cases))

    report = asyncio.run(execute(models=models, cases=cases, benchmark_cfg=benchmark_cfg, enable_preflight=not args.skip_preflight))

    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"{args.output_prefix}_{ts}.json"
    md_path = out_dir / f"{args.output_prefix}_{ts}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(to_markdown(report), encoding="utf-8")

    logger.info("JSON 报告: %s", json_path)
    logger.info("Markdown 报告: %s", md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
